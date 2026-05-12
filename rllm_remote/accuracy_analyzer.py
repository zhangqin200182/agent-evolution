"""
Accuracy analyzer for remote NPU training.
Fetches ALL accuracy-relevant TensorBoard tags and generates structured analysis.

Usage:
  python -m rllm_remote.accuracy_analyzer <run_id> --ssh-password "<pw>"
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from rllm_remote.config import RemoteTrainConfig
from rllm_remote.ssh import RemoteExecutor

# ============================================================
# Constants: TB tag groups
# ============================================================

TAG_GROUPS = {
    "reward": [
        "critic/rewards/mean", "critic/rewards/max", "critic/rewards/min",
        "critic/score/mean", "critic/score/max", "critic/score/min",
    ],
    "advantage_return": [
        "critic/advantages/mean", "critic/advantages/max", "critic/advantages/min",
        "critic/returns/mean", "critic/returns/max", "critic/returns/min",
    ],
    "actor": [
        "actor/pg_loss", "actor/entropy", "actor/grad_norm",
        "actor/kl_loss", "actor/ppo_kl",
        "actor/pg_clipfrac", "actor/pg_clipfrac_lower", "actor/kl_coef",
    ],
    "rollout_divergence": [
        "rollout_corr/kl", "rollout_corr/k3_kl",
        "rollout_corr/chi2_token", "rollout_corr/chi2_seq",
        "rollout_corr/ppl_ratio",
        "rollout_corr/log_ppl_diff", "rollout_corr/log_ppl_abs_diff",
        "rollout_corr/log_ppl_diff_max", "rollout_corr/log_ppl_diff_min",
        "rollout_corr/rollout_is_catastrophic_token_fraction",
        "rollout_corr/rollout_is_veto_fraction",
        "rollout_corr/rollout_log_ppl", "rollout_corr/rollout_ppl",
        "rollout_corr/training_log_ppl", "rollout_corr/training_ppl",
    ],
    "probability_distribution": [
        "training/rollout_actor_probs_pearson_corr",
        "training/rollout_probs_diff_mean", "training/rollout_probs_diff_std",
        "training/rollout_probs_diff_max", "training/rollout_probs_diff_valid",
    ],
    "response_quality": [
        "response/aborted_ratio", "response_length/mean",
    ],
    "metadata": [
        "training/epoch", "training/global_step",
    ],
}

ALL_ACCURACY_TAGS = [tag for group in TAG_GROUPS.values() for tag in group]

# ============================================================
# Data structures
# ============================================================


@dataclass
class Anomaly:
    id: str
    severity: str  # critical / warning / info
    description: str
    evidence: str  # specific data values that triggered this


@dataclass
class Diagnosis:
    id: str
    severity: str
    detail: str
    impact: str
    config_issues: list[dict] = field(default_factory=list)
    # Each config_issue: {"param": str, "current_value": Any, "problem": str}


@dataclass
class Suggestion:
    id: int
    priority: str  # critical / high / medium / low
    param_path: str
    current_value: Any
    suggested_value: Any
    data_evidence: str
    reason: str
    expected_impact: str
    risk: str


# ============================================================
# Algorithm detection
# ============================================================


def identify_algorithm(config: RemoteTrainConfig) -> str:
    """Identify which RL algorithm variant is being used.

    Returns one of: grpo, gae-ppo, dapo, gspo, gpg, unknown
    """
    # Check verl config section if available
    verl_cfg = getattr(config, "verl_conf", None) or {}

    # adv_estimator from verl config
    algo = verl_cfg.get("algorithm", {}) if isinstance(verl_cfg, dict) else {}
    adv_estimator = algo.get("adv_estimator", "grpo")

    # loss_mode from actor config
    actor_cfg = (verl_cfg.get("actor_rollout_ref", {}).get("actor", {})
                 if isinstance(verl_cfg, dict) else {})
    policy_loss = actor_cfg.get("policy_loss", {}) if isinstance(actor_cfg, dict) else {}
    loss_mode = policy_loss.get("loss_mode", "vanilla") if isinstance(policy_loss, dict) else "vanilla"

    # reward manager
    reward_manager = (verl_cfg.get("reward_manager", {}) if isinstance(verl_cfg, dict) else {})
    rm_type = reward_manager.get("type", "naive") if isinstance(reward_manager, dict) else "naive"

    # Check msrl_conf for mindspeed_rl configs
    msrl_cfg = getattr(config, "msrl_conf", None) or {}
    if isinstance(msrl_cfg, dict):
        rl_cfg = msrl_cfg.get("rl_config", {})
        if isinstance(rl_cfg, dict):
            msrl_adv = rl_cfg.get("adv_estimator", "")
            msrl_kl = rl_cfg.get("kl_penalty", "")
        else:
            msrl_adv = ""
            msrl_kl = ""
    else:
        msrl_adv = ""
        msrl_kl = ""

    # Determine algorithm
    if rm_type == "dapo":
        return "dapo"
    if loss_mode == "gspo":
        return "gspo"
    if loss_mode == "gpg" or adv_estimator == "gpg" or msrl_adv == "gpg":
        return "gpg"
    if adv_estimator in ("gae",) or msrl_adv in ("gae",):
        return "gae-ppo"
    # Default: GRPO family
    if adv_estimator in ("grpo", "grpo_vectorized", "grpo_passk",
                         "rloo", "rloo_vectorized", "opo",
                         "reinforce_plus_plus", "reinforce_plus_plus_baseline",
                         "remax", "") or msrl_adv in ("grpo", "group_norm", ""):
        return "grpo"

    return "grpo"  # default fallback


# ============================================================
# TB data fetching
# ============================================================


def fetch_all_accuracy_tags(executor: RemoteExecutor, run_id: str,
                            timeout: int = 120) -> dict[str, list[tuple[int, float]]]:
    """Fetch ALL accuracy-relevant TB tags with full step history.

    Runs a Python script inside the remote container that uses EventAccumulator.
    """
    tb_dir = (
        f"{executor.config.remote_agent_sdk_dir}/tensorboard_log/"
        f"rllm_remote/{run_id}"
    )

    # Build tag list string
    tags_str = '", "'.join(ALL_ACCURACY_TAGS)

    script = f'''
import sys
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator("{tb_dir}")
    ea.Reload()
    tags = ["{tags_str}"]
    for tag in tags:
        if tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            vals = [f"{{e.step}}:{{e.value:.6f}}" for e in events]
            print(f"{{tag}}|{{','.join(vals)}}")
        else:
            print(f"{{tag}}|MISSING")
except Exception as e:
    print(f"ERROR:{{e}}")
'''

    stdout = executor.run_script(script, timeout=timeout)

    history: dict[str, list[tuple[int, float]]] = {}
    for line in stdout.strip().split("\n"):
        if "|" not in line:
            continue
        tag, vals_str = line.split("|", 1)
        if vals_str == "MISSING":
            history[tag] = []
            continue
        if tag.startswith("ERROR:"):
            print(f"[WARN] TB fetch error: {tag}", file=sys.stderr)
            continue
        pairs = []
        for v in vals_str.split(","):
            parts = v.split(":")
            if len(parts) == 2:
                try:
                    pairs.append((int(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        if pairs:
            history[tag] = pairs

    return history


# ============================================================
# Trend computation
# ============================================================


def compute_trends(tag_data: dict[str, list[tuple[int, float]]]) -> dict:
    """Compute trends, statistics, and derived metrics for all tag groups."""
    trends = {}

    for group_name, tags in TAG_GROUPS.items():
        group_trends = {}
        for tag in tags:
            pairs = tag_data.get(tag, [])
            if not pairs or len(pairs) < 2:
                group_trends[tag] = {"available": False, "n_steps": len(pairs)}
                continue

            values = [v for _, v in pairs]
            n = len(values)
            half = n // 2

            first_half_avg = sum(values[:half]) / half if half > 0 else values[0]
            second_half_avg = sum(values[half:]) / (n - half) if n - half > 0 else values[-1]

            # Trend direction
            if second_half_avg > first_half_avg * 1.05:
                direction = "rising"
            elif second_half_avg < first_half_avg * 0.95:
                direction = "falling"
            else:
                direction = "flat"

            # Linear regression slope (simple)
            mean_x = (n - 1) / 2
            mean_y = sum(values) / n
            num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
            den = sum((i - mean_x) ** 2 for i in range(n))
            slope = num / den if den != 0 else 0

            # Variance
            variance = sum((v - mean_y) ** 2 for v in values) / n

            group_trends[tag] = {
                "available": True,
                "n_steps": n,
                "first": values[0],
                "last": values[-1],
                "min": min(values),
                "max": max(values),
                "mean": mean_y,
                "std": variance ** 0.5,
                "direction": direction,
                "slope": slope,
                "first_half_avg": first_half_avg,
                "second_half_avg": second_half_avg,
                "values": values,
            }

        trends[group_name] = group_trends

    # Compute derived metrics
    trends["_derived"] = _compute_derived_metrics(trends, tag_data)

    return trends


def _compute_derived_metrics(trends: dict, tag_data: dict) -> dict:
    """Compute cross-metric derived indicators."""
    derived = {}

    # Reward variance health
    r_mean = trends.get("reward", {}).get("critic/rewards/mean", {})
    r_min = trends.get("reward", {}).get("critic/rewards/min", {})
    r_max = trends.get("reward", {}).get("critic/rewards/max", {})

    if r_min.get("available") and r_max.get("available"):
        r_min_vals = r_min.get("values", [])
        r_max_vals = r_max.get("values", [])
        # Check if min==max in any step (all rewards identical = no gradient)
        all_same_steps = sum(1 for mn, mx in zip(r_min_vals, r_max_vals) if mn == mx)
        derived["reward_all_same_steps"] = all_same_steps
        derived["reward_all_same_ratio"] = all_same_steps / len(r_min_vals) if r_min_vals else 0

    # Epoch boundaries
    epoch_tag = trends.get("metadata", {}).get("training/epoch", {})
    if epoch_tag.get("available"):
        epoch_vals = epoch_tag.get("values", [])
        boundaries = []
        for i in range(1, len(epoch_vals)):
            if epoch_vals[i] != epoch_vals[i - 1]:
                boundaries.append(i)
        derived["epoch_boundaries"] = boundaries
        derived["num_epochs"] = int(epoch_vals[-1]) + 1 if epoch_vals else 1

    # Epoch-level reward analysis
    if r_mean.get("available") and epoch_tag.get("available"):
        boundaries = derived.get("epoch_boundaries", [])
        r_vals = r_mean.get("values", [])
        epoch_rewards = []
        prev = 0
        for b in boundaries:
            epoch_vals_slice = r_vals[prev:b] if prev < len(r_vals) else []
            epoch_rewards.append({
                "epoch": len(epoch_rewards),
                "steps": f"{prev}-{b-1}",
                "avg": sum(epoch_vals_slice) / len(epoch_vals_slice) if epoch_vals_slice else 0,
            })
            prev = b
        # Last epoch
        last_slice = r_vals[prev:]
        epoch_rewards.append({
            "epoch": len(epoch_rewards),
            "steps": f"{prev}-{len(r_vals)-1}",
            "avg": sum(last_slice) / len(last_slice) if last_slice else 0,
        })

        # Catastrophic forgetting: epoch N+1 avg < epoch N avg * 0.3
        drops = []
        for i in range(1, len(epoch_rewards)):
            if epoch_rewards[i]["avg"] < epoch_rewards[i - 1]["avg"] * 0.3:
                drops.append({
                    "from_epoch": i - 1,
                    "to_epoch": i,
                    "from_avg": epoch_rewards[i - 1]["avg"],
                    "to_avg": epoch_rewards[i]["avg"],
                    "drop_pct": (1 - epoch_rewards[i]["avg"] / epoch_rewards[i - 1]["avg"]) * 100,
                })
        derived["epoch_rewards"] = epoch_rewards
        derived["epoch_drops"] = drops

    # Grad norm spike detection
    gn = trends.get("actor", {}).get("actor/grad_norm", {})
    if gn.get("available"):
        gn_vals = gn.get("values", [])
        gn_mean = gn.get("mean", 0)
        gn_std = gn.get("std", 0)
        spike_threshold = gn_mean + 3 * gn_std if gn_std > 0 else gn_mean * 3
        spikes = [(i, v) for i, v in enumerate(gn_vals) if v > spike_threshold]
        derived["grad_norm_spikes"] = [{"step": s, "value": v, "ratio": v / gn_mean if gn_mean else 0} for s, v in spikes]

    # KL epoch jump detection
    kl = trends.get("actor", {}).get("actor/kl_loss", {})
    if kl.get("available") and derived.get("epoch_boundaries"):
        kl_vals = kl.get("values", [])
        jumps = []
        for b in derived["epoch_boundaries"]:
            if b > 0 and b < len(kl_vals):
                jump_ratio = kl_vals[b] / kl_vals[b - 1] if kl_vals[b - 1] != 0 else float("inf")
                if jump_ratio > 2.0:
                    jumps.append({
                        "step": b,
                        "from_val": kl_vals[b - 1],
                        "to_val": kl_vals[b],
                        "ratio": jump_ratio,
                    })
        derived["kl_epoch_jumps"] = jumps

    # Entropy decay rate
    ent = trends.get("actor", {}).get("actor/entropy", {})
    if ent.get("available"):
        derived["entropy_decay_per_step"] = (
            (ent["last"] - ent["first"]) / (ent["n_steps"] - 1) if ent["n_steps"] > 1 else 0
        )

    # Catastrophic token detection
    cat_tag = trends.get("rollout_divergence", {}).get("rollout_corr/rollout_is_catastrophic_token_fraction", {})
    if cat_tag.get("available"):
        cat_vals = cat_tag.get("values", [])
        derived["catastrophic_token_max"] = max(cat_vals) if cat_vals else 0
        derived["catastrophic_token_detected"] = any(v > 0 for v in cat_vals)

    return derived


# ============================================================
# Anomaly detection
# ============================================================


def detect_anomalies(trends: dict) -> list[Anomaly]:
    """Run 15 universal anomaly detection rules."""
    anomalies = []
    derived = trends.get("_derived", {})

    # 1. REWARD_SATURATION_ALL_ONE
    r_min = trends.get("reward", {}).get("critic/rewards/min", {})
    r_max = trends.get("reward", {}).get("critic/rewards/max", {})
    if r_min.get("available") and r_max.get("available"):
        if all(mn == mx == 1.0 for mn, mx in zip(r_min.get("values", []), r_max.get("values", []))):
            anomalies.append(Anomaly(
                id="REWARD_SATURATION_ALL_ONE",
                severity="critical",
                description="所有 step 的 batch 内所有样本 reward 均为 1.0，任务太简单",
                evidence=f"rewards/min == rewards/max == 1.0 全程 ({r_min['n_steps']} steps)",
            ))

    # 2. REWARD_SATURATION_ALL_ZERO
    if r_min.get("available") and r_max.get("available"):
        if all(mn == mx == 0.0 for mn, mx in zip(r_min.get("values", []), r_max.get("values", []))):
            anomalies.append(Anomaly(
                id="REWARD_SATURATION_ALL_ZERO",
                severity="critical",
                description="所有 step 的 batch 内所有样本 reward 均为 0.0，任务太难或 agent/tool 配置错误",
                evidence=f"rewards/min == rewards/max == 0.0 全程 ({r_min['n_steps']} steps)",
            ))

    # 3. REWARD_EPOCH_DROP
    drops = derived.get("epoch_drops", [])
    for d in drops:
        anomalies.append(Anomaly(
            id="REWARD_EPOCH_DROP",
            severity="critical",
            description=f"Epoch {d['from_epoch']}→{d['to_epoch']} reward 断崖下降 {d['drop_pct']:.0f}% ({d['from_avg']:.3f}→{d['to_avg']:.3f})",
            evidence=f"epoch boundary reward drop: {d['drop_pct']:.0f}%",
        ))

    # 4. ADVANTAGE_ZERO
    adv_max = trends.get("advantage_return", {}).get("critic/advantages/max", {})
    adv_min = trends.get("advantage_return", {}).get("critic/advantages/min", {})
    if adv_max.get("available") and adv_min.get("available"):
        if all(mx == mn == 0 for mx, mn in zip(adv_max.get("values", []), adv_min.get("values", []))):
            anomalies.append(Anomaly(
                id="ADVANTAGE_ZERO",
                severity="critical",
                description="Advantage 全程为 0。组内 reward 方差为零，GRPO 完全无法产生学习梯度",
                evidence="advantages/max == advantages/min == 0 全程",
            ))

    # 5. ENTROPY_COLLAPSE
    ent = trends.get("actor", {}).get("actor/entropy", {})
    if ent.get("available") and ent.get("last", 1) < 0.01:
        anomalies.append(Anomaly(
            id="ENTROPY_COLLAPSE",
            severity="critical",
            description=f"策略熵坍缩至 {ent['last']:.4f}，所有生成几乎相同",
            evidence=f"entropy: {ent['first']:.4f} → {ent['last']:.4f}",
        ))

    # 6. ENTROPY_RISING
    if ent.get("available") and ent.get("direction") == "rising" and ent.get("slope", 0) > 0.001:
        anomalies.append(Anomaly(
            id="ENTROPY_RISING",
            severity="warning",
            description="策略熵不降反升（entropy_coeff=0 时表示训练不稳定）",
            evidence=f"entropy: {ent['first']:.4f} → {ent['last']:.4f}, slope={ent.get('slope', 0):.4f}",
        ))

    # 7. GRADIENT_SPIKE
    spikes = derived.get("grad_norm_spikes", [])
    if spikes:
        worst = max(spikes, key=lambda s: s["ratio"])
        anomalies.append(Anomaly(
            id="GRADIENT_SPIKE",
            severity="info",
            description=f"grad_norm 检测到 {len(spikes)} 次突刺，最严重在 step {worst['step']} (比值 {worst['ratio']:.1f}x)",
            evidence=f"grad_norm spikes at steps: {[s['step'] for s in spikes]}",
        ))

    # 8. GRADIENT_ALWAYS_CLIPPED
    gn = trends.get("actor", {}).get("actor/grad_norm", {})
    if gn.get("available"):
        gn_vals = gn.get("values", [])
        if gn_vals and all(v >= 0.95 for v in gn_vals):  # All close to or above typical clip_grad
            anomalies.append(Anomaly(
                id="GRADIENT_ALWAYS_CLIPPED",
                severity="warning",
                description="所有 step 的 grad_norm 均在较高水平，梯度裁剪可能一直在生效",
                evidence=f"grad_norm min={gn['min']:.2f}, mean={gn['mean']:.2f}",
            ))

    # 9. KL_DIVERGENCE
    kl = trends.get("actor", {}).get("actor/kl_loss", {})
    if kl.get("available") and kl.get("last", 0) > 0.1:
        anomalies.append(Anomaly(
            id="KL_DIVERGENCE",
            severity="critical",
            description=f"KL 散度超过 0.1（{kl['last']:.4f}），策略严重偏离参考模型，存在发散或 reward hacking 风险",
            evidence=f"kl_loss: {kl['first']:.4f} → {kl['last']:.4f}",
        ))

    # 10. KL_EPOCH_JUMP
    kl_jumps = derived.get("kl_epoch_jumps", [])
    for j in kl_jumps:
        anomalies.append(Anomaly(
            id="KL_EPOCH_JUMP",
            severity="warning",
            description=f"kl_loss 在 step {j['step']}（epoch 边界）跳跃 {j['ratio']:.1f}x ({j['from_val']:.4f}→{j['to_val']:.4f})",
            evidence=f"step {j['step']}: {j['from_val']:.4f} → {j['to_val']:.4f}",
        ))

    # 11. DISTRIBUTION_SHIFT
    chi2 = trends.get("rollout_divergence", {}).get("rollout_corr/chi2_seq", {})
    if chi2.get("available"):
        chi2_vals = chi2.get("values", [])
        # Check if chi2_seq > 0.5 for majority of steps
        high_steps = sum(1 for v in chi2_vals if v > 0.5)
        if high_steps > len(chi2_vals) * 0.5:
            anomalies.append(Anomaly(
                id="DISTRIBUTION_SHIFT",
                severity="warning",
                description=f"chi2_seq 在 {high_steps}/{len(chi2_vals)} 步超过 0.5，rollout-training 分布持续不匹配",
                evidence=f"chi2_seq > 0.5 in {high_steps}/{len(chi2_vals)} steps",
            ))

    # 12. CATASTROPHIC_FORGETTING
    if derived.get("catastrophic_token_detected"):
        anomalies.append(Anomaly(
            id="CATASTROPHIC_FORGETTING",
            severity="critical",
            description=f"检测到灾难性遗忘！catastrophic_token_fraction 最大值为 {derived['catastrophic_token_max']:.6f}",
            evidence=f"catastrophic_token_fraction max={derived['catastrophic_token_max']:.6f}",
        ))

    # 13. REWARD_HACKING
    rollout_ppl = trends.get("rollout_divergence", {}).get("rollout_corr/rollout_ppl", {})
    r_mean = trends.get("reward", {}).get("critic/rewards/mean", {})
    if (rollout_ppl.get("available") and r_mean.get("available")
            and rollout_ppl.get("direction") == "rising" and r_mean.get("direction") == "rising"):
        anomalies.append(Anomaly(
            id="REWARD_HACKING",
            severity="critical",
            description="reward 上升但 rollout_ppl 也在上升——模型可能通过语言退化来获取更高 reward",
            evidence=f"rollout_ppl: {rollout_ppl['first']:.3f}→{rollout_ppl['last']:.3f}, reward: {r_mean['first']:.3f}→{r_mean['last']:.3f}",
        ))

    # 14. PEARSON_DECOUPLING
    pearson = trends.get("probability_distribution", {}).get("training/rollout_actor_probs_pearson_corr", {})
    if pearson.get("available") and pearson.get("last", 1) < 0.95:
        anomalies.append(Anomaly(
            id="PEARSON_DECOUPLING",
            severity="warning",
            description=f"rollout-actor 概率 Pearson 相关系数降至 {pearson['last']:.4f}（<0.95），概率分布严重不一致",
            evidence=f"pearson_corr: {pearson['first']:.4f} → {pearson['last']:.4f}",
        ))

    # 15. ABORTED_GENERATION
    aborted = trends.get("response_quality", {}).get("response/aborted_ratio", {})
    if aborted.get("available") and aborted.get("max", 0) > 0:
        anomalies.append(Anomaly(
            id="ABORTED_GENERATION",
            severity="warning",
            description=f"检测到生成截断: aborted_ratio 最大为 {aborted['max']:.4f}，max_response_length 可能不够",
            evidence=f"aborted_ratio max={aborted['max']:.4f}",
        ))

    return anomalies


# ============================================================
# Health scoring
# ============================================================


def compute_health_scores(trends: dict, anomalies: list[Anomaly], algorithm: str) -> dict:
    """Compute 5-dimension health scores (0-1, higher is better)."""
    anomaly_ids = {a.id for a in anomalies}
    derived = trends.get("_derived", {})

    # Learning signal
    learning_signal = 1.0
    if "ADVANTAGE_ZERO" in anomaly_ids:
        learning_signal = 0.0
    elif "REWARD_SATURATION_ALL_ONE" in anomaly_ids or "REWARD_SATURATION_ALL_ZERO" in anomaly_ids:
        learning_signal = 0.1
    else:
        pg_loss = trends.get("actor", {}).get("actor/pg_loss", {})
        if pg_loss.get("available") and pg_loss.get("max", 0) == 0 and pg_loss.get("min", 0) == 0:
            if algorithm == "gpg":
                learning_signal = 0.8  # GPG often has zero pg_loss normally
            else:
                # pg_loss=0 but advantages exist? Check
                adv_max = trends.get("advantage_return", {}).get("critic/advantages/max", {})
                if adv_max.get("max", 0) > 0.5:
                    learning_signal = 0.2  # signal exists but not flowing through
                else:
                    learning_signal = 0.3

    # Stability
    stability = 1.0
    if "GRADIENT_SPIKE" in anomaly_ids:
        stability -= 0.2
    if "GRADIENT_ALWAYS_CLIPPED" in anomaly_ids:
        stability -= 0.2
    if "ENTROPY_COLLAPSE" in anomaly_ids:
        stability -= 0.3
    if "ENTROPY_RISING" in anomaly_ids:
        stability -= 0.2
    stability = max(0.0, stability)

    # Divergence risk
    divergence_risk = 1.0
    kl = trends.get("actor", {}).get("actor/kl_loss", {})
    if "KL_DIVERGENCE" in anomaly_ids:
        divergence_risk = 0.0
    elif kl.get("available"):
        final_kl = kl.get("last", 0)
        if final_kl > 0.05:
            divergence_risk = 0.5
        elif final_kl > 0.02:
            divergence_risk = 0.7
    if "KL_EPOCH_JUMP" in anomaly_ids:
        divergence_risk -= 0.2
    if "DISTRIBUTION_SHIFT" in anomaly_ids:
        divergence_risk -= 0.3
    divergence_risk = max(0.0, divergence_risk)

    # Catastrophic forgetting risk
    forgetting_risk = 1.0
    if "CATASTROPHIC_FORGETTING" in anomaly_ids:
        forgetting_risk = 0.0
    if "REWARD_EPOCH_DROP" in anomaly_ids:
        forgetting_risk -= 0.5
    if derived.get("catastrophic_token_max", 0) > 0:
        forgetting_risk = 0.0
    forgetting_risk = max(0.0, forgetting_risk)

    # Overall (weighted average)
    overall = (
        learning_signal * 0.35
        + stability * 0.20
        + divergence_risk * 0.25
        + forgetting_risk * 0.20
    )

    return {
        "learning_signal": round(learning_signal, 2),
        "stability": round(stability, 2),
        "divergence_risk": round(divergence_risk, 2),
        "catastrophic_forgetting_risk": round(forgetting_risk, 2),
        "overall_health": round(overall, 2),
    }


# ============================================================
# Suggestion generation
# ============================================================


def generate_suggestions(trends: dict, anomalies: list[Anomaly],
                         algorithm: str, config: RemoteTrainConfig) -> list[Suggestion]:
    """Map diagnoses to structured tuning suggestions."""
    suggestions = []
    anomaly_ids = {a.id for a in anomalies}
    derived = trends.get("_derived", {})
    sug_id = 0

    # Helper
    def add(priority, param_path, current, suggested, data_evidence, reason, expected, risk):
        nonlocal sug_id
        sug_id += 1
        # Round suggested value to avoid floating point artifacts
        if isinstance(suggested, float) and abs(suggested) < 0.001:
            suggested = float(f"{suggested:.1e}")
        elif isinstance(suggested, float):
            suggested = round(suggested, 6)
        suggestions.append(Suggestion(
            id=sug_id, priority=priority,
            param_path=param_path, current_value=current, suggested_value=suggested,
            data_evidence=data_evidence, reason=reason, expected_impact=expected, risk=risk,
        ))

    # --- Critical level ---

    if "CATASTROPHIC_FORGETTING" in anomaly_ids:
        add("critical", "kl_coef", config.kl_coef, config.kl_coef * 5,
            f"catastrophic_token_fraction max={derived.get('catastrophic_token_max', 0):.6f}",
            "灾难性遗忘已发生，必须大幅增大 KL 约束以将策略锚定在参考模型附近",
            "策略回退到参考模型附近，遗忘缓解",
            "reward 可能短期下降")
        add("critical", "total_epochs", config.num_epochs, max(1, config.num_epochs // 2),
            f"catastrophic_token_fraction > 0",
            "减少训练 epoch 以降低过拟合风险",
            "减少数据重复，缓解遗忘",
            "训练总步数减少")

    if "REWARD_HACKING" in anomaly_ids:
        add("critical", "kl_coef", config.kl_coef, config.kl_coef * 10,
            "rollout_ppl 上升 + reward 上升",
            "Reward hacking 检测：模型通过语言退化获取高 reward，需大幅增强 KL 约束",
            "模型停止退化，真实能力提升",
            "reward 可能短期下降，需要接受")
        add("critical", "kl_loss_coef", 0.001, 0.01,
            "rollout_ppl 上升 + reward 上升",
            "Actor 内部 KL loss 也需要增大以双重约束策略",
            "额外的 KL 约束防止策略走偏",
            "收敛速度变慢")

    # --- High level ---

    pg_loss = trends.get("actor", {}).get("actor/pg_loss", {})
    if pg_loss.get("available") and pg_loss.get("max", 0) == 0 and pg_loss.get("min", 0) == 0:
        if algorithm not in ("gpg",):
            add("high", "lr", config.lr, config.lr * 5,
                f"pg_loss 在所有 {pg_loss['n_steps']} 步均为 0，pg_clipfrac=0",
                "pg_loss 全程为 0 说明 PPO 裁剪或 lr 过小导致策略梯度未生效。增大 lr 让梯度更新进入有效范围",
                "pg_loss 出现非零值，策略梯度开始贡献学习信号",
                "KL 可能同步快速上升，需要监控并考虑同步增大 kl_coef")
            add("medium", "entropy_coeff", 0.0, 0.001,
                f"pg_loss=0 + entropy 在下降 ({trends.get('actor', {}).get('actor/entropy', {}).get('first', 0):.3f}→{trends.get('actor', {}).get('actor/entropy', {}).get('last', 0):.3f})",
                "添加轻量熵奖励增加探索多样性",
                "策略保持探索能力，可能打破 pg_loss=0 的僵局",
                "过大熵奖励会导致不稳定")

    if "REWARD_SATURATION_ALL_ONE" in anomaly_ids:
        add("high", "train_data_path", config.train_data_path, "mixed_difficulty_dataset",
            "rewards/min == rewards/max == 1.0",
            "任务太简单，所有样本全对。需要增加难度或切换到混合难度数据集",
            "reward 方差恢复，advantage 信号恢复",
            "如果数据不可用，考虑调整 verifier 的评分标准")

    if "REWARD_SATURATION_ALL_ZERO" in anomaly_ids:
        add("high", "temperature", config.temperature, config.temperature * 1.3,
            "rewards/min == rewards/max == 0.0",
            "所有样本全错。增大采样温度增加生成多样性，可能产生正确答案",
            "可能产生得分样本",
            "生成质量可能下降")

    if "ENTROPY_COLLAPSE" in anomaly_ids:
        add("high", "temperature", config.temperature, min(2.0, config.temperature * 1.5),
            f"entropy 坍缩至 {trends.get('actor', {}).get('actor/entropy', {}).get('last', 0):.4f}",
            "策略坍缩，所有生成趋同。增大采样温度是最直接的恢复手段",
            "生成多样性恢复",
            "reward 可能短期波动")
        add("high", "n_samples_per_prompt", config.n_samples_per_prompt, config.n_samples_per_prompt * 2,
            f"entropy < 0.01",
            "增大组内样本数提供更多样的 reward 信号",
            "advantage 方差增大，学习信号增强",
            "计算成本增加")

    if "KL_DIVERGENCE" in anomaly_ids:
        add("high", "kl_coef", config.kl_coef, config.kl_coef * 5,
            f"kl_loss 升至 {trends.get('actor', {}).get('actor/kl_loss', {}).get('last', 0):.4f}",
            "KL 散度过大（>0.1），策略严重偏离参考模型。增大 KL 约束系数",
            "策略回退到更合理的位置",
            "reward 提升速度可能减慢")

    # --- Medium level ---

    if "KL_EPOCH_JUMP" in anomaly_ids:
        jumps = derived.get("kl_epoch_jumps", [])
        worst_ratio = max((j["ratio"] for j in jumps), default=0)
        add("medium", "kl_coef", config.kl_coef, config.kl_coef * max(2.0, min(5.0, worst_ratio / 2)),
            f"kl_loss epoch 边界跳跃 {worst_ratio:.1f}x",
            "Epoch 边界 KL 跳跃说明 epoch 间数据分布差异或模型过拟合。增大 KL 系数平滑 epoch 过渡",
            "Epoch 间切换更平滑",
            "可能略微降低 epoch 2 的学习效率")
        add("low", "kl_ctrl_type", "fixed", "adaptive",
            "kl_loss epoch 边界跳跃",
            "切换到 adaptive KL control 可以自动根据当前 KL 动态调整系数",
            "KL 自动维持在 target_kl 附近",
            "adaptive 模式调参更复杂（需设 target_kl 和 horizon）")

    if "GRADIENT_SPIKE" in anomaly_ids:
        spikes = derived.get("grad_norm_spikes", [])
        if spikes:
            add("medium", "lr", config.lr, config.lr * 0.5,
                f"grad_norm 突刺 {len(spikes)} 次，最大比值 {max(s['ratio'] for s in spikes):.1f}x",
                "梯度突刺说明某些 step 更新过激。减小 lr 降低单步更新幅度",
                "梯度稳定，减少对已学策略的破坏",
                "收敛速度变慢")

    if "GRADIENT_ALWAYS_CLIPPED" in anomaly_ids:
        gn = trends.get("actor", {}).get("actor/grad_norm", {})
        add("medium", "clip_grad", 1.0, 2.0,
            f"grad_norm min={gn.get('min', 0):.2f} > clip_grad",
            "梯度始终被裁剪，clip_grad 可能过小。增大裁剪阈值让合理的梯度通过",
            "梯度更新更充分",
            "如果梯度本身过大，增大阈值可能导致不稳定")

    if "DISTRIBUTION_SHIFT" in anomaly_ids:
        # Check if rollout correction is already enabled
        add("medium", "rollout_correction", "disabled", "decoupled_seq_is",
            "chi2_seq 持续 > 0.5，rollout-training 分布严重不匹配",
            "启用 Rollout Correction (Sequence IS) 纠正 off-policy 分布偏移",
            "分布偏移被 IS 权重纠正，训练更有效",
            "IS 权重增加 variance，可能需要调整 threshold")

    # --- Low level ---

    if "ENTROPY_RISING" in anomaly_ids:
        add("low", "entropy_coeff", 0.0, 0.0,
            "entropy 在 entropy_coeff=0 时不降反升",
            "观察 entropy 趋势。如果持续上升超过 5 steps，考虑减小 lr",
            "保持现状观察",
            "若不处理可能导致训练不稳定")

    if "PEARSON_DECOUPLING" in anomaly_ids:
        add("low", "lr", config.lr, config.lr * 0.5,
            "pearson_corr < 0.95",
            "Rollout-actor 概率分布不一致。减小 lr 降低每步策略变化幅度",
            "pearson_corr 回升至 >0.99",
            "收敛速度变慢")

    if "ABORTED_GENERATION" in anomaly_ids:
        add("low", "max_response_length", config.max_response_length, config.max_response_length * 2,
            f"aborted_ratio max={trends.get('response_quality', {}).get('response/aborted_ratio', {}).get('max', 0):.4f}",
            "生成被截断。增大 max_response_length",
            "不再截断",
            "每步计算成本增加")

    return suggestions


# ============================================================
# Main analyzer class
# ============================================================


class AccuracyAnalyzer:
    """Remote training accuracy analyzer."""

    def __init__(self, run_id: str, ssh_password: str | None = None):
        self.run_id = run_id
        self.ssh_password = ssh_password
        self.config: Optional[RemoteTrainConfig] = None
        self.executor: Optional[RemoteExecutor] = None
        self.algorithm: str = "unknown"
        self.tag_data: dict = {}
        self.trends: dict = {}
        self.anomalies: list[Anomaly] = []
        self.health_scores: dict = {}
        self.suggestions: list[Suggestion] = []

    def _connect(self):
        """Load config and create executor."""
        paths = [
            os.path.join("rllm_remote", "output", "runs", self.run_id, "config.json"),
        ]
        config_path = None
        for p in paths:
            if os.path.exists(p):
                config_path = p
                break
        if config_path is None:
            raise FileNotFoundError(
                f"Config not found for run_id={self.run_id}. "
                f"Tried: {paths}"
            )
        self.config = RemoteTrainConfig.from_json(config_path)
        if self.ssh_password:
            self.config.ssh_password = self.ssh_password
        self.executor = RemoteExecutor(self.config)

    def analyze(self) -> str:
        """Run full analysis pipeline. Returns path to report file."""
        print(f"[1/6] Connecting to remote server...")
        self._connect()

        print(f"[2/6] Fetching accuracy TB tags (44 tags)...")
        self.tag_data = fetch_all_accuracy_tags(self.executor, self.run_id)

        tag_count = sum(1 for pairs in self.tag_data.values() if pairs)
        print(f"  ... fetched {tag_count} tags with step data")

        print(f"[3/6] Computing trends and detecting anomalies...")
        self.algorithm = identify_algorithm(self.config)
        print(f"  ... detected algorithm: {self.algorithm}")
        self.trends = compute_trends(self.tag_data)
        self.anomalies = detect_anomalies(self.trends)
        print(f"  ... found {len(self.anomalies)} anomalies")

        print(f"[4/6] Computing health scores and generating suggestions...")
        self.health_scores = compute_health_scores(self.trends, self.anomalies, self.algorithm)
        self.suggestions = generate_suggestions(self.trends, self.anomalies, self.algorithm, self.config)
        print(f"  ... generated {len(self.suggestions)} suggestions")

        print(f"[5/6] Writing analysis.json...")
        json_path = self.write_analysis_json()
        print(f"  ... {json_path}")

        print(f"[6/6] Writing accuracy report...")
        report_path = self.write_report()
        print(f"  ... {report_path}")

        print(f"\nAnalysis complete!")
        print(f"  Algorithm: {self.algorithm}")
        print(f"  Overall health: {self.health_scores.get('overall_health', 'N/A')}")
        print(f"  Anomalies: {len(self.anomalies)}")
        print(f"  Suggestions: {len(self.suggestions)}")

        return report_path

    def write_analysis_json(self) -> str:
        """Write structured analysis.json."""
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "rllm_remote", "output", "runs", self.run_id,
        )
        output_dir = os.path.normpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Build summary
        summary = {
            "total_steps": len(self.tag_data.get("training/global_step", [])),
            "total_epochs": self.trends.get("_derived", {}).get("num_epochs", 1),
            "critical_findings": sum(1 for a in self.anomalies if a.severity == "critical"),
            "warnings": sum(1 for a in self.anomalies if a.severity == "warning"),
            "info": sum(1 for a in self.anomalies if a.severity == "info"),
        }

        # Build full analysis dict
        analysis = {
            "version": 2,
            "run_id": self.run_id,
            "backend": "remote",
            "analysis_type": "accuracy",
            "algorithm": self.algorithm,
            "model": self.config.model_name_or_path if self.config else "unknown",
            "summary": summary,
            "health_scores": self.health_scores,
            "anomalies": [
                {"id": a.id, "severity": a.severity, "description": a.description, "evidence": a.evidence}
                for a in self.anomalies
            ],
            "suggestions": [
                {
                    "id": s.id, "priority": s.priority,
                    "param_path": s.param_path,
                    "current_value": s.current_value, "suggested_value": s.suggested_value,
                    "data_evidence": s.data_evidence,
                    "reason": s.reason,
                    "expected_impact": s.expected_impact,
                    "risk": s.risk,
                }
                for s in self.suggestions
            ],
            # Include trend summaries for key metrics
            "key_metrics": {},
        }

        # Add key metric trends
        for group in ["reward", "actor", "rollout_divergence", "probability_distribution"]:
            analysis["key_metrics"][group] = {}
            group_trends = self.trends.get(group, {})
            for tag, tdata in group_trends.items():
                if tdata.get("available"):
                    analysis["key_metrics"][group][tag] = {
                        "first": tdata["first"],
                        "last": tdata["last"],
                        "min": tdata["min"],
                        "max": tdata["max"],
                        "mean": tdata["mean"],
                        "direction": tdata["direction"],
                        "values": tdata.get("values", []),
                    }

        # Add epoch analysis
        derived = self.trends.get("_derived", {})
        analysis["epoch_analysis"] = {
            "boundaries": derived.get("epoch_boundaries", []),
            "epoch_rewards": derived.get("epoch_rewards", []),
            "drops": derived.get("epoch_drops", []),
            "kl_jumps": derived.get("kl_epoch_jumps", []),
        }

        path = os.path.join(output_dir, "analysis.json")
        with open(path, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        return path

    def write_report(self) -> str:
        """Generate and write accuracy_report.md."""
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "rllm_remote", "output", "runs", self.run_id,
        )
        output_dir = os.path.normpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        lines = []
        lines.append(f"# 精度分析报告 — {self.run_id}\n")

        # ---- Part 1: Analysis Process ----
        lines.append("## 第一部分：分析过程\n")
        lines.append("### 数据来源")
        lines.append(f"- 远程服务器: {self.config.ssh_host if self.config else 'N/A'}, 容器 {self.config.container_name if self.config else 'N/A'}")
        lines.append(f"- TB 事件路径: `.../tensorboard_log/rllm_remote/{self.run_id}/`")
        lines.append(f"- 配置文件: `rllm_remote/output/runs/{self.run_id}/config.json`")
        lines.append("")
        lines.append("### 分析方法")
        lines.append("1. 从远程 TB events 中读取全部 83 个 scalar tags，提取其中 44 个精度相关指标")
        n_steps = len(self.tag_data.get("training/global_step", []))
        lines.append(f"2. 获取每个指标的完整 step-by-step 历史序列（共 {n_steps} 步）")
        lines.append("3. 计算每个指标的趋势（首半段 vs 后半段对比、线性回归斜率）")
        lines.append("4. 进行异常检测：极值分析、方差分析、step间变化率分析、epoch边界效应分析")
        lines.append(f"5. 将检测到的异常模式与当前算法（{self.algorithm.upper()}）的已知失效模式进行匹配")
        lines.append("6. 基于匹配结果生成诊断和调优建议")
        lines.append("")
        lines.append("### 分析覆盖的指标组")
        lines.append("| 指标组 | 标签数 | 关键指标 |")
        lines.append("|--------|--------|---------|")
        lines.append("| Reward & Score | 6 | rewards/mean, score/mean |")
        lines.append("| Advantage & Return | 6 | advantages/max, advantages/min |")
        lines.append("| Actor 训练指标 | 8 | pg_loss, entropy, grad_norm, kl_loss, pg_clipfrac |")
        lines.append("| Rollout-Training 分布差异 | 15 | kl, chi2_seq, ppl_ratio, catastrophic_token_fraction |")
        lines.append("| 概率分布对齐 | 5 | pearson_corr, probs_diff_mean |")
        lines.append("| Response 质量 | 2 | aborted_ratio, response_length/mean |")
        lines.append("| 训练元信息 | 2 | epoch, global_step |")
        lines.append("")

        # Step-by-step key metrics
        lines.append("### 关键指标的 Step-by-Step 趋势\n")
        lines.extend(self._format_step_table("Rewards", [
            ("mean", "critic/rewards/mean"),
            ("max", "critic/rewards/max"),
            ("min", "critic/rewards/min"),
        ]))
        lines.extend(self._format_step_table("Advantages", [
            ("max", "critic/advantages/max"),
            ("min", "critic/advantages/min"),
            ("mean", "critic/advantages/mean"),
        ]))
        lines.extend(self._format_step_table("Actor 训练指标", [
            ("pg_loss", "actor/pg_loss"),
            ("entropy", "actor/entropy"),
            ("grad_norm", "actor/grad_norm"),
            ("kl_loss", "actor/kl_loss"),
        ]))
        lines.extend(self._format_step_table("Rollout-Training 分布差异", [
            ("corr_kl", "rollout_corr/kl"),
            ("chi2_seq", "rollout_corr/chi2_seq"),
            ("ppl_ratio", "rollout_corr/ppl_ratio"),
            ("catastrophic", "rollout_corr/rollout_is_catastrophic_token_fraction"),
            ("veto", "rollout_corr/rollout_is_veto_fraction"),
        ]))
        lines.extend(self._format_step_table("概率分布对齐", [
            ("pearson_corr", "training/rollout_actor_probs_pearson_corr"),
            ("probs_diff_mean", "training/rollout_probs_diff_mean"),
        ]))

        # ---- Part 2: Configuration Problems ----
        lines.append("## 第二部分：当前配置问题诊断\n")
        if self.anomalies:
            for i, a in enumerate(self.anomalies):
                severity_label = {"critical": "🔴", "warning": "⚠️", "info": "🟡"}.get(a.severity, "")
                lines.append(f"### {severity_label} 问题 {i+1}：{a.id}")
                lines.append(f"- **严重度**: {a.severity}")
                lines.append(f"- **数据证据**: {a.evidence}")
                lines.append(f"- **诊断结论**: {a.description}")

                # Find related suggestions for config issues
                related = [s for s in self.suggestions if s.data_evidence and a.id.lower() in s.data_evidence.lower()
                          or any(kw in s.reason.lower() for kw in a.id.lower().split("_"))]
                if not related:
                    # Fallback: match by anomaly id keywords
                    related = [s for s in self.suggestions if any(
                        kw in s.param_path.lower() or kw in s.reason.lower()
                        for kw in a.id.lower().split("_")[:2]
                    )]
                if related:
                    lines.append("")
                    lines.append("**相关配置问题**：")
                    for s in related[:3]:
                        lines.append(f"- `{s.param_path}` = {s.current_value} → 建议改为 {s.suggested_value}（{s.reason}）")
                lines.append("")
        else:
            lines.append("未检测到异常。训练状态健康。\n")

        # ---- Part 3: Health Scores ----
        lines.append("## 第三部分：健康评分\n")
        lines.append("| 维度 | 分数 | 依据 |")
        lines.append("|------|------|------|")
        hs = self.health_scores
        score_labels = {
            "learning_signal": "学习信号",
            "stability": "训练稳定性",
            "divergence_risk": "发散风险",
            "catastrophic_forgetting_risk": "遗忘风险",
            "overall_health": "综合健康",
        }
        anomaly_msgs = [f"{a.id}" for a in self.anomalies]
        for key, label in score_labels.items():
            score = hs.get(key, 0)
            emoji = "✅" if score > 0.8 else ("🟡" if score > 0.5 else "⚠️")
            if key == "overall_health":
                basis = f"加权综合评分 ({len(self.anomalies)} 项异常)" if self.anomalies else "所有维度正常"
            elif score > 0.8:
                basis = "指标在健康范围内"
            elif score > 0.5:
                basis = "存在轻微异常，需关注"
            else:
                basis = "存在严重问题，需调整"
            lines.append(f"| {emoji} {label} | {score:.1f}/1.0 | {basis} |")
        lines.append("")

        # ---- Part 4: Tuning Suggestions ----
        lines.append("## 第四部分：调优建议\n")
        if self.suggestions:
            for s in self.suggestions:
                priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(s.priority, "")
                lines.append(f"### 建议 #{s.id} [{priority_emoji} {s.priority}]")
                lines.append(f"- **参数路径**: `{s.param_path}`")
                lines.append(f"- **当前值**: {s.current_value} → **建议值**: {s.suggested_value}")
                lines.append(f"- **数据依据**: {s.data_evidence}")
                lines.append(f"- **理由**: {s.reason}")
                lines.append(f"- **预期效果**: {s.expected_impact}")
                lines.append(f"- **风险**: {s.risk}")
                lines.append("")
        else:
            lines.append("当前训练状态健康，无需调整参数。\n")

        path = os.path.join(output_dir, "accuracy_report.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    def _format_step_table(self, title: str, metrics: list[tuple[str, str]]) -> list[str]:
        """Format a step-by-step trend table for given metrics."""
        lines = []
        lines.append(f"#### {title}")
        lines.append("```")

        # Collect all unique steps
        all_steps = set()
        for _, tag in metrics:
            pairs = self.tag_data.get(tag, [])
            all_steps.update(s[0] for s in pairs)
        steps = sorted(all_steps)

        if not steps:
            lines.append("  (无数据)")
            lines.append("```")
            lines.append("")
            return lines

        # Header
        header = "Step:  " + "  ".join(f"{s:>7}" for s in steps)
        lines.append(header)

        # Each metric row
        for name, tag in metrics:
            pairs = self.tag_data.get(tag, [])
            step_map = {s: v for s, v in pairs}
            vals_str = "  ".join(f"{step_map.get(s, float('nan')):>7.3f}" for s in steps)
            lines.append(f"{name:>6}:  {vals_str}")

        lines.append("```")
        lines.append("")
        return lines

    def apply_suggestions(self, approved_ids: list[int]) -> str:
        """Apply approved suggestions to config. Returns summary of changes."""
        if not self.config:
            raise RuntimeError("No config loaded. Run analyze() first.")

        approved = [s for s in self.suggestions if s.id in approved_ids]
        if not approved:
            return "No suggestions to apply."

        # Backup
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "rllm_remote", "output", "runs", self.run_id,
        )
        output_dir = os.path.normpath(output_dir)
        config_path = os.path.join(output_dir, "config.json")
        backup_path = f"{config_path}.bak.{int(time.time())}"

        with open(config_path) as f:
            original = json.load(f)
        with open(backup_path, "w") as f:
            json.dump(original, f, indent=2, ensure_ascii=False)

        # Apply changes
        changed = []
        for s in approved:
            if s.param_path in original:
                old_val = original[s.param_path]
                original[s.param_path] = s.suggested_value
                changed.append(f"  {s.param_path}: {old_val} → {s.suggested_value}")
            else:
                changed.append(f"  {s.param_path}: (not found in config, skipped)")

        # Write updated config
        with open(config_path, "w") as f:
            json.dump(original, f, indent=2, ensure_ascii=False)

        summary = f"Applied {len(changed)} changes to {config_path}\n"
        summary += f"Backup: {backup_path}\n"
        summary += "\n".join(changed)
        return summary


# ============================================================
# CLI entry
# ============================================================


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Remote NPU training accuracy analyzer")
    parser.add_argument("run_id", help="Training run ID")
    parser.add_argument("--ssh-password", help="SSH password for remote server")
    parser.add_argument("--apply", nargs="*", type=int, default=None,
                        help="Apply specific suggestion IDs (space-separated)")
    parser.add_argument("--apply-all", action="store_true",
                        help="Apply all suggestions without interactive approval")
    args = parser.parse_args()

    analyzer = AccuracyAnalyzer(args.run_id, ssh_password=args.ssh_password)

    if args.apply_all and analyzer.suggestions:
        analyzer.analyze()
        result = analyzer.apply_suggestions([s.id for s in analyzer.suggestions])
        print(result)
    elif args.apply is not None:
        # Apply specific suggestions from cached analysis (no SSH needed)
        analyzer._connect()  # only loads config, no SSH
        # Load suggestions from cached analysis.json
        analysis_json_path = os.path.join(
            analyzer.config.local_output_dir, args.run_id, "analysis.json"
        )
        if not os.path.exists(analysis_json_path):
            print(f"No cached analysis found at {analysis_json_path}. Run without --apply first.")
            sys.exit(1)
        with open(analysis_json_path) as f:
            cached = json.load(f)
        # Rebuild suggestions from cached data
        analyzer.suggestions = [
            Suggestion(
                id=s["id"], priority=s["priority"],
                param_path=s["param_path"],
                current_value=s["current_value"], suggested_value=s["suggested_value"],
                data_evidence=s.get("data_evidence", ""),
                reason=s["reason"], expected_impact=s["expected_impact"], risk=s["risk"],
            )
            for s in cached.get("suggestions", [])
        ]
        result = analyzer.apply_suggestions(args.apply)
        print(result)
    else:
        report_path = analyzer.analyze()
        print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
