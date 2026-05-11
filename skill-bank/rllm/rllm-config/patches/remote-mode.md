---
id: "remote-mode"
target_section: "output"
action: append
description: "Add remote NPU training config generation mode. When backend=remote, generates RemoteTrainConfig JSON instead of TrainingConfig."
source: "2026-05-11 Remote NPU training feature"
created: "2026-05-11"

depends_on: []
conflicts_with: []

status: active
superseded_by: ""
---

### Remote 模式配置生成

当 args 中包含 `backend=remote` 时，进入远程配置模式。

#### 远程配置生成流程

1. 读取 `rllm_remote/config.py` 了解 RemoteTrainConfig 参数
2. 根据需求摘要设置参数：

| 需求参数 | RemoteTrainConfig 字段 | 说明 |
|---|---|---|
| model_name | model_name_or_path | 使用服务器路径而非 HF 名称 |
| num_epochs | num_epochs | 直接映射，远程可设更大值 |
| batch_size | train_batch_size | 远程 NPU 可设更大 (默认 32) |
| num_generations | n_samples_per_prompt | 直接映射 |
| learning_rate | lr | 直接映射 |
| temperature | temperature | 直接映射 |
| max_agent_steps | max_agent_steps | 直接映射 |

3. 服务器路径默认值：

```python
model_name_or_path = "/opt/DPC/models/l00619320/code/AGENTIC_RL_WS/AgenticRL_Binary_Files/models/Qwen2.5-7B-Instruct"
train_data_path = "/opt/DPC/models/l00619320/code/VERL_NPU_WS/data/gsm8k/train.parquet"
val_data_path = "/opt/DPC/models/l00619320/code/VERL_NPU_WS/data/gsm8k/test.parquet"
ssh_host = "<server-ip>"
container_name = "agent5.0.0_qjy"
```

4. 写配置文件：

```bash
python -c "
from rllm_remote.config import RemoteTrainConfig
config = RemoteTrainConfig(
    model_name_or_path='<model_path>',
    num_epochs=<n>,
    train_batch_size=<n>,
    lr=<lr>,
    ...
)
config.to_json('rllm_remote/output/runs/<run_id>/config.json')
print('OK: config.json written')
print(config.summary())
"
```

#### NPU 参数安全范围

| 参数 | 最小值 | 最大值 | 默认值 | 说明 |
|---|---|---|---|---|
| train_batch_size | 8 | 128 | 32 | NPU 8卡可开大 batch |
| ppo_mini_batch_size | 1 | 32 | 8 | 必须能整除 train_batch_size |
| lr | 1e-7 | 1e-4 | 1e-6 | NPU 训练建议较低 lr |
| num_epochs | 10 | 500 | 100 | 远程可长时间训练 |
| n_samples_per_prompt | 2 | 16 | 8 | 受 NPU 显存限制 |
| tensor_parallel_size | 1 | 8 | 4 | 不超过 n_npus_per_node |
| max_response_length | 256 | 4096 | 2048 | 受 NPU 显存限制 |
| kl_coef | 0.0 | 0.1 | 0.001 | 防止 policy 偏离 |

#### 调参规则（远程模式）

与本地调参规则不同，远程模式有更大调整空间：

| 问题 | 远程调参策略 |
|---|---|
| Reward 低 + Loss 不收敛 | lr * 2, 增加 epochs |
| Reward 低 + Loss 下降 | 增加 epochs, 增大 n_samples_per_prompt |
| Reward 震荡 | lr / 2, kl_coef * 2 |
| NPU OOM | 减小 batch_size, max_response_length, tp_size |
| 训练太慢 | 增大 batch_size, 减小 epochs |
