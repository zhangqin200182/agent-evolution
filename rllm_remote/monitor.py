"""
Remote training monitor combining training log + TensorBoard events.

Usage:
    python -m rllm_remote.monitor <run_id>            # single report
    python -m rllm_remote.monitor <run_id> --watch     # continuous polling
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any


class RemoteMonitor:
    """Monitor remote NPU training by combining log tail + TensorBoard parsing."""

    # Key TB tags to surface
    TB_TAGS = [
        # (tag, label, format)
        ("training/global_step", "Step", "{}"),
        ("critic/rewards/mean", "Reward mean", "{:.3f}"),
        ("critic/rewards/max", "Reward max", "{:.3f}"),
        ("critic/rewards/min", "Reward min", "{:.3f}"),
        ("actor/pg_loss", "PG loss", "{:.4f}"),
        ("actor/grad_norm", "Grad norm", "{:.2f}"),
        ("actor/kl_loss", "KL loss", "{:.4f}"),
        ("actor/entropy", "Entropy", "{:.3f}"),
        ("actor/lr", "LR", "{:.2e}"),
        ("perf/time_per_step", "Time/step", "{:.1f}s"),
        ("perf/throughput", "Throughput", "{:.1f} tok/s"),
        ("perf/mfu/actor", "MFU", "{:.1%}"),
        ("timing_s/gen", "Gen time", "{:.1f}s"),
        ("timing_s/update_actor", "Train time", "{:.1f}s"),
        ("training/epoch", "Epoch", "{:.2f}"),
    ]

    # Log patterns for agent activity
    REWARD_PATTERN = re.compile(r"final reward:\s*([\d.]+)")
    LLM_TIME_PATTERN = re.compile(r"total_llm_time:([\d.]+)")
    TOKENS_PATTERN = re.compile(r"total_prompt_tokens:(\d+),\s*total_response_tokens:(\d+)")

    # Error patterns
    ERROR_PATTERNS = [
        (re.compile(r"ConfigAttributeError"), "配置字段缺失"),
        (re.compile(r"Missing key"), "配置字段缺失"),
        (re.compile(r"Traceback"), "进程异常"),
        (re.compile(r"Engine core proc.*died"), "vLLM 引擎崩溃"),
        (re.compile(r"NPU out of memory|Ascend OOM"), "NPU 显存不足"),
        (re.compile(r"HCCL timeout"), "NPU 互联超时"),
        (re.compile(r"RayTaskError"), "Ray 任务错误"),
        (re.compile(r"Error processing"), "样本处理错误"),
    ]

    def __init__(self, run_id: str, ssh_password: str | None = None):
        self.run_id = run_id

        # Load config (must be called from project root)
        from rllm_remote.config import RemoteTrainConfig

        config_path = f"rllm_remote/output/runs/{run_id}/config.json"
        if not os.path.exists(config_path):
            # Try local_output_dir from default
            from rllm_remote.config import RemoteTrainConfig as RTC
            base = RTC().local_output_dir
            config_path = os.path.join(base, run_id, "config.json")

        self.config = RemoteTrainConfig.from_json(config_path)
        if ssh_password:
            self.config.ssh_password = ssh_password

        from rllm_remote.ssh import RemoteExecutor

        self.executor = RemoteExecutor(self.config)
        self._total_steps = self._estimate_total_steps()

    def _estimate_total_steps(self) -> int:
        """Estimate total training steps from actual dataset row count."""
        batch = self.config.train_batch_size
        epochs = self.config.num_epochs
        # Try to get actual row count from remote parquet
        try:
            script = (
                f"import pandas as pd; "
                f"df = pd.read_parquet('{self.config.train_data_path}'); "
                f"print(len(df))"
            )
            stdout = self.executor.run_script(script, timeout=30)
            ds_size = int(stdout.strip())
        except Exception:
            ds_size = 7500  # fallback
        return max(1, (ds_size // batch) * epochs)

    def check_alive(self, pid: str) -> bool:
        return self.executor.check_pid(pid)

    def fetch_log(self, lines: int = 50) -> str:
        log_path = f"{self.config.remote_output_dir}/{self.run_id}/training_log.txt"
        return self.executor.tail_log(log_path, lines=lines)

    def fetch_tb_metrics(self) -> dict[str, tuple[int, float]]:
        """Parse remote TensorBoard events. Returns {tag: (step, value)}."""
        tb_dir = (
            f"{self.config.remote_agent_sdk_dir}/tensorboard_log/"
            f"rllm_remote/{self.run_id}"
        )
        script = f'''
import os, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

d = "{tb_dir}"
if not os.path.isdir(d):
    print("TB_DIR_NOT_FOUND")
    exit(0)
ea = EventAccumulator(d)
ea.Reload()
for tag in sorted(ea.Tags().get("scalars", [])):
    events = ea.Scalars(tag)
    if events:
        print(f"{{tag}}|{{events[-1].step}}|{{events[-1].value}}")
'''
        try:
            stdout = self.executor.run_script(script, timeout=30)
        except Exception as e:
            return {"_error": (0, str(e))}

        if "TB_DIR_NOT_FOUND" in stdout:
            return {}

        metrics: dict[str, tuple[int, float]] = {}
        for line in stdout.strip().split("\n"):
            parts = line.split("|")
            if len(parts) == 3:
                try:
                    metrics[parts[0]] = (int(parts[1]), float(parts[2]))
                except ValueError:
                    pass
        return metrics

    def scan_log(self, log: str) -> dict[str, Any]:
        """Extract info from training log."""
        # Count rewards
        rewards = [float(m) for m in self.REWARD_PATTERN.findall(log)]
        total_trajs = len(rewards)
        success = sum(1 for r in rewards if r >= 0.5)

        # LLM time
        llm_times = [float(m) for m in self.LLM_TIME_PATTERN.findall(log)]
        avg_llm_time = sum(llm_times) / len(llm_times) if llm_times else 0

        # Anomalies
        anomalies = []
        for pattern, desc in self.ERROR_PATTERNS:
            matches = pattern.findall(log)
            if matches:
                anomalies.append(f"{desc}({len(matches)})")

        return {
            "total_trajs": total_trajs,
            "success_rate": success / total_trajs if total_trajs > 0 else 0,
            "avg_llm_time": avg_llm_time,
            "anomalies": anomalies,
        }

    def compute_progress(self, metrics: dict) -> dict:
        """Compute progress from TB metrics."""
        step_tag = "training/global_step"
        time_tag = "perf/time_per_step"

        current_step = 0
        time_per_step = 0

        if step_tag in metrics:
            current_step = metrics[step_tag][1]
        if time_tag in metrics:
            time_per_step = metrics[time_tag][1]

        pct = min(current_step / self._total_steps * 100, 100) if self._total_steps else 0
        remaining_steps = max(self._total_steps - int(current_step), 0)
        eta_seconds = remaining_steps * time_per_step if time_per_step > 0 else 0

        return {
            "current_step": int(current_step),
            "total_steps": self._total_steps,
            "pct": pct,
            "time_per_step": time_per_step,
            "eta_seconds": eta_seconds,
        }

    def format_eta(self, seconds: float) -> str:
        if seconds <= 0:
            return "-"
        h, m = divmod(int(seconds), 3600)
        m, s = divmod(m, 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        elif m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def assess(self, metrics: dict, log_info: dict, progress: dict, alive: bool) -> str:
        """Generate overall training assessment and suggestions."""
        lines = []

        if not alive:
            lines.append("判断: 训练进程已退出，检查日志确认是正常完成还是异常退出。")
            return "\n".join(lines)

        if progress["current_step"] < 2:
            lines.append("判断: 训练启动阶段，样本不足，暂无法评估趋势。")
            lines.append("建议: 继续等待，首步含模型编译，后续会加速。")
            return "\n".join(lines)

        reward = metrics.get("critic/rewards/mean", (0, 0))[1]
        grad = metrics.get("actor/grad_norm", (0, 0))[1]
        kl = metrics.get("actor/kl_loss", (0, 0))[1]
        mfu = metrics.get("perf/mfu/actor", (0, 0))[1]
        time_per_step = metrics.get("perf/time_per_step", (0, 0))[1]

        # Reward judgment
        if reward > 0.7:
            lines.append(f"判断: Reward 良好 ({reward:.2f})，模型正在学习有效策略。")
        elif reward > 0.4:
            lines.append(f"判断: Reward 中等 ({reward:.2f})，有学习信号但尚未收敛。")
        else:
            lines.append(f"判断: Reward 偏低 ({reward:.2f})，可能需要调整超参。")

        # Health checks
        warnings = []
        if grad > 10:
            warnings.append(f"梯度偏大 ({grad:.1f})，训练可能不稳定")
        if kl > 0.05:
            warnings.append(f"KL 偏高 ({kl:.4f})，策略偏离参考模型")
        if progress["time_per_step"] > 600:
            warnings.append("每步耗时 >10min")
        if mfu > 0 and mfu < 0.08:
            warnings.append(f"MFU 偏低 ({mfu:.1%})")

        if warnings:
            lines.append("注意: " + "; ".join(warnings))

        # ETA
        eta_h = progress["eta_seconds"] / 3600
        lines.append(f"预计剩余: {self.format_eta(progress['eta_seconds'])} ({eta_h:.1f}h)")

        # Suggestions
        suggestions = []
        if reward < 0.3 and progress["current_step"] >= 5:
            suggestions.append("reward 持续偏低，考虑增大 lr 或 epochs")
        if kl > 0.05:
            suggestions.append("增大 kl_coef 到 0.005 限制策略偏移")
        if progress["time_per_step"] > 600:
            suggestions.append("减小 batch_size 或 max_response_length 加速训练")

        if suggestions:
            lines.append("建议: " + "; ".join(suggestions))
        elif progress["current_step"] >= 3:
            lines.append("建议: 训练正常，继续观察。")

        return "\n".join(lines)

    def report(self, pid: str) -> str:
        """Generate a full monitoring report."""
        alive = self.check_alive(pid)
        log = self.fetch_log()
        metrics = self.fetch_tb_metrics()
        log_info = self.scan_log(log)
        progress = self.compute_progress(metrics)

        reward = metrics.get("critic/rewards/mean", (0, 0))[1]
        reward_max = metrics.get("critic/rewards/max", (0, 0))[1]
        reward_min = metrics.get("critic/rewards/min", (0, 0))[1]
        pg_loss = metrics.get("actor/pg_loss", (0, 0))[1]
        grad = metrics.get("actor/grad_norm", (0, 0))[1]
        kl = metrics.get("actor/kl_loss", (0, 0))[1]
        entropy = metrics.get("actor/entropy", (0, 0))[1]
        time_step = metrics.get("perf/time_per_step", (0, 0))[1]
        gen_t = metrics.get("timing_s/gen", (0, 0))[1]
        train_t = metrics.get("timing_s/update_actor", (0, 0))[1]
        throughput = metrics.get("perf/throughput", (0, 0))[1]
        mfu = metrics.get("perf/mfu/actor", (0, 0))[1]

        step = progress["current_step"]
        total = progress["total_steps"]
        pct = progress["pct"]
        eta = self.format_eta(progress["eta_seconds"])

        # Summary line
        if step > 0:
            summary = (
                f"Step {step}/{total} ({pct:.1f}%)，"
                f"reward {reward:.2f}，"
                f"{time_step:.0f}s/step，"
                f"ETA {eta}"
            )
        else:
            summary = "初始化中，暂无 step 数据"

        lines = []
        lines.append(f"⏺ {summary}")
        lines.append(f"  远程训练监控报告")
        lines.append(f"  ================")
        lines.append(f"  Run ID:      {self.run_id}")
        status = "RUNNING" if alive else "STOPPED"
        lines.append(f"  Status:      {status} (PID={pid})")
        lines.append(f"  Progress:    Step {step}/{total} ({pct:.1f}%)  ETA: {eta}")
        lines.append(f"  Reward:      mean={reward:.3f}  max={reward_max:.3f}  min={reward_min:.3f}")
        lines.append(f"  Actor:       loss={pg_loss:.4f}  grad={grad:.2f}  kl={kl:.4f}  entropy={entropy:.3f}")
        lines.append(f"  Perf:        {time_step:.0f}s/step  gen:{gen_t:.0f}s  train:{train_t:.0f}s  {throughput:.0f}tok/s  MFU:{mfu:.1%}")

        if log_info["anomalies"]:
            lines.append(f"  Anomalies:   {', '.join(log_info['anomalies'])}")
        else:
            lines.append(f"  Anomalies:   无")

        lines.append("")
        lines.append(f"  {self.assess(metrics, log_info, progress, alive)}")
        return "\n".join(lines)

    def fetch_tb_history(self) -> dict[str, list[tuple[int, float]]]:
        """Get full step-by-step history from TB for key metrics."""
        tb_dir = (
            f"{self.config.remote_agent_sdk_dir}/tensorboard_log/"
            f"rllm_remote/{self.run_id}"
        )
        key_tags = [
            "training/global_step", "critic/rewards/mean", "critic/rewards/max",
            "critic/rewards/min", "actor/pg_loss", "actor/grad_norm",
            "actor/kl_loss", "actor/entropy", "perf/time_per_step",
            "perf/throughput", "perf/mfu/actor", "timing_s/gen",
            "timing_s/update_actor", "training/epoch",
        ]
        tags_str = '", "'.join(key_tags)
        script = f'''
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator("{tb_dir}")
ea.Reload()
tags = ["{tags_str}"]
for tag in tags:
    if tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        vals = [f"{{e.step}}:{{e.value:.6f}}" for e in events]
        print(f"{{tag}}|{{','.join(vals)}}")
'''
        try:
            stdout = self.executor.run_script(script, timeout=30)
        except Exception:
            return {}

        history: dict[str, list[tuple[int, float]]] = {}
        for line in stdout.strip().split("\n"):
            if "|" not in line:
                continue
            tag, vals_str = line.split("|", 1)
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

    def fetch_trajectories(self) -> list[dict]:
        """Download and parse trajectory files for this run.

        AgentSDK writes trajectory files to outputs/ (not outputs/<run_id>/).
        We find files by matching timestamps in the filename against our run.
        """
        # 1. Find trajectory files
        find_cmd = f"ls {self.config.remote_agent_sdk_dir}/outputs/rollout_trajectories_*.json 2>/dev/null"
        try:
            stdout = self.executor.run(find_cmd, timeout=10).stdout
        except Exception:
            return []
        all_files = sorted(stdout.strip().split("\n"))

        # 2. Find which files belong to this run by checking iteration -> run_id mapping
        # The trajectory file contains a timestamp; we parse 3 files to sample
        trajs = []
        for f in all_files[-5:]:  # sample last 5 files
            try:
                content = self.executor.run(f"cat {f}", timeout=15).stdout
                data = json.loads(content)
                trajs.extend(data.get("trajectories", []))
            except Exception:
                continue
        return trajs

    def analyze_trajectories(self, trajs: list[dict]) -> dict:
        """Analyze trajectory data and return summary."""
        if not trajs:
            return {"count": 0, "note": "未找到轨迹文件"}

        rewards = []
        llm_times = []
        env_times = []
        total_times = []
        steps_counts = []
        correct = 0
        total = 0

        for t in trajs:
            try:
                r = float(t.get("trajectory_reward", 0))
                rewards.append(r)
                if r >= 0.5:
                    correct += 1
                total += 1

                metrics = t.get("metrics", {})
                if metrics:
                    llm_times.append(float(metrics.get("llm_time", 0)))
                    env_times.append(float(metrics.get("env_time", 0)))
                    total_times.append(float(metrics.get("total_time", 0)))
                    steps_counts.append(int(metrics.get("steps", 0)))
            except (ValueError, TypeError):
                continue

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        accuracy = correct / total if total > 0 else 0

        return {
            "count": total,
            "avg_reward": avg_reward,
            "accuracy": accuracy,
            "correct": correct,
            "avg_llm_time": sum(llm_times) / len(llm_times) if llm_times else 0,
            "avg_env_time": sum(env_times) / len(env_times) if env_times else 0,
            "avg_total_time": sum(total_times) / len(total_times) if total_times else 0,
            "avg_steps": sum(steps_counts) / len(steps_counts) if steps_counts else 0,
            "reward_distribution": {
                "0.0": rewards.count(0.0),
                "1.0": rewards.count(1.0),
            },
        }

    def analyze(self) -> str:
        """Generate analysis.json for a completed training run."""
        log = self.fetch_log(lines=100)
        log_info = self.scan_log(log)
        history = self.fetch_tb_history()

        if not history:
            return "ERROR: No TB data found. Training may not have completed any steps."

        # Extract step-level data
        rewards = [v for _, v in history.get("critic/rewards/mean", [])]
        losses = [v for _, v in history.get("actor/pg_loss", [])]
        grads = [v for _, v in history.get("actor/grad_norm", [])]
        kls = [v for _, v in history.get("actor/kl_loss", [])]
        times = [v for _, v in history.get("perf/time_per_step", [])]
        throughputs = [v for _, v in history.get("perf/throughput", [])]
        mfus = [v for _, v in history.get("perf/mfu/actor", [])]
        gen_times = [v for _, v in history.get("timing_s/gen", [])]
        train_times = [v for _, v in history.get("timing_s/update_actor", [])]

        total_steps = len(rewards)
        reward_start = rewards[0] if rewards else 0
        reward_end = rewards[-1] if rewards else 0
        reward_max = max(rewards) if rewards else 0
        reward_min = min(rewards) if rewards else 0

        # Trend: compare first half vs second half
        half = total_steps // 2
        first_half_avg = sum(rewards[:half]) / half if half > 0 else 0
        second_half_avg = sum(rewards[half:]) / (total_steps - half) if total_steps > half else 0
        if second_half_avg > first_half_avg * 1.05:
            trend = "rising"
        elif second_half_avg < first_half_avg * 0.95:
            trend = "falling"
        else:
            trend = "flat"

        # Performance
        avg_time = sum(times) / len(times) if times else 0
        avg_gen = sum(gen_times) / len(gen_times) if gen_times else 0
        avg_train = sum(train_times) / len(train_times) if train_times else 0
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
        avg_mfu = sum(mfus) / len(mfus) if mfus else 0
        total_time = sum(times) if times else 0

        # Suggestions
        suggestions = []
        if trend == "falling" and reward_end < reward_start * 0.5:
            suggestions.append({"param": "lr", "old": self.config.lr, "new": self.config.lr / 2,
                              "priority": "high", "reason": "reward 下降超过50%，降低学习率"})
        if trend == "flat" and reward_max < 0.5:
            suggestions.append({"param": "num_epochs", "old": self.config.num_epochs, "new": self.config.num_epochs * 2,
                              "priority": "high", "reason": "reward 停滞在低位，增加训练量"})
        if kls and kls[-1] > 0.05:
            suggestions.append({"param": "kl_coef", "old": self.config.kl_coef, "new": self.config.kl_coef * 2,
                              "priority": "medium", "reason": f"KL 偏高 ({kls[-1]:.4f})，限制策略偏移"})

        # Trajectory analysis
        trajs = self.fetch_trajectories()
        traj_analysis = self.analyze_trajectories(trajs)

        analysis = {
            "run_id": self.run_id,
            "backend": "remote",
            "model": self.config.model_name_or_path,
            "total_steps": total_steps,
            "total_epochs": self.config.num_epochs,
            "trajectory_analysis": traj_analysis,
            "reward": {
                "start": reward_start, "end": reward_end, "max": reward_max, "min": reward_min,
                "trend": trend, "first_half_avg": first_half_avg, "second_half_avg": second_half_avg,
                "values": rewards,
            },
            "loss": {"values": losses, "avg_grad_norm": sum(grads) / len(grads) if grads else 0},
            "kl": {"start": kls[0] if kls else 0, "end": kls[-1] if kls else 0, "values": kls},
            "performance": {
                "total_time_s": total_time,
                "avg_time_per_step": avg_time,
                "avg_gen_time": avg_gen,
                "avg_train_time": avg_train,
                "avg_throughput": avg_throughput,
                "avg_mfu": avg_mfu,
            },
            "agent": {
                "trajectories_in_log": log_info["total_trajs"],
                "success_rate": log_info["success_rate"],
            },
            "anomalies": log_info["anomalies"],
            "suggestions": suggestions,
        }

        import json as _json
        local_dir = os.path.join(self.config.local_output_dir, self.run_id)
        os.makedirs(local_dir, exist_ok=True)
        analysis_path = os.path.join(local_dir, "analysis.json")
        with open(analysis_path, "w") as f:
            _json.dump(analysis, f, indent=2, ensure_ascii=False)

        # Also download training log
        try:
            remote_log = f"{self.config.remote_output_dir}/{self.run_id}/training_log.txt"
            self.executor.download_file(remote_log, os.path.join(local_dir, "training_log.txt"))
        except Exception:
            pass

        return f"analysis.json written to {analysis_path}\n{_json.dumps(analysis, indent=2, ensure_ascii=False)}"

    def summary_line(self, pid: str) -> str:
        """Single-line summary for watch mode."""
        alive = self.check_alive(pid)
        if not alive:
            return f"[STOPPED] {self.run_id}"

        metrics = self.fetch_tb_metrics()
        progress = self.compute_progress(metrics)

        step = progress["current_step"]
        pct = progress["pct"]
        eta = self.format_eta(progress["eta_seconds"])

        reward_str = "-"
        if "critic/rewards/mean" in metrics:
            reward_str = f"{metrics['critic/rewards/mean'][1]:.3f}"

        time_str = "-"
        if "perf/time_per_step" in metrics:
            time_str = f"{metrics['perf/time_per_step'][1]:.0f}s"

        return (
            f"Step {step}/{progress['total_steps']} ({pct:.1f}%)  "
            f"reward={reward_str}  "
            f"{time_str}/step  "
            f"ETA={eta}"
        )


def main():
    parser = argparse.ArgumentParser(description="Remote NPU training monitor")
    parser.add_argument("run_id", help="Training run ID")
    parser.add_argument("--pid", default=None, help="Remote process PID")
    parser.add_argument("--ssh-password", default=None, help="SSH password")
    parser.add_argument("--watch", action="store_true", help="Continuous polling mode")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds (watch mode)")
    parser.add_argument("--analyze", action="store_true", help="Generate analysis.json from completed run")
    args = parser.parse_args()

    monitor = RemoteMonitor(args.run_id, ssh_password=args.ssh_password)

    # --analyze mode: generate analysis.json (no PID needed for completed runs)
    if args.analyze:
        print(monitor.analyze())
        return

    # Auto-detect PID from saved info or ask
    pid = args.pid
    if not pid:
        info_path = os.path.join(
            monitor.config.local_output_dir, monitor.run_id, "run_info.json"
        )
        if os.path.exists(info_path):
            with open(info_path) as f:
                pid = json.load(f).get("pid", "")
        if not pid:
            print("No PID provided. Use --pid <PID>")
            sys.exit(1)

    if args.watch:
        print(f"Monitoring {args.run_id} (PID={pid}), interval={args.interval}s")
        print()
        try:
            while True:
                print(monitor.report(pid), flush=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        print(monitor.report(pid))


if __name__ == "__main__":
    main()
