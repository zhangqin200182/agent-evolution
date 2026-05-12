---
description: SSH connection management for remote NPU training server. Standardizes
  connection setup, verification, and troubleshooting across all remote skills.
metadata:
  categories:
  - infrastructure
  - remote
  version: 1.0.0
name: rllm-remote-connect
---


# 远程连接管理

你是远程 NPU 训练服务器的 SSH 连接专家。服务器信息：

- **地址**: <server-ip>:22
- **用户**: root
- **容器**: agent5.0.0_qjy（所有命令通过 `docker exec` 执行）
- **AgentSDK 路径**: `/home/qjy/code/AgentSDK/master/AgentSDK`

其他远程 skill（rllm-remote-run, rllm-remote-monitor, rllm-analyze-accuracy）都依赖你提供的连接方式。

# 连接方式

## 方式一：使用 connect() 工厂（推荐）

```python
from rllm_remote import connect
config, executor = connect("<run_id>", ssh_password="<password>")
```

`connect()` 自动查找 `rllm_remote/output/runs/<run_id>/config.json` 并创建 `RemoteExecutor`。

## 方式二：手动构造

```python
from rllm_remote.config import RemoteTrainConfig
from rllm_remote.ssh import RemoteExecutor

config = RemoteTrainConfig(
    ssh_host="<server-ip>",
    ssh_port=22,
    ssh_user="root",
    ssh_password="<password>",  # 不序列化到config.json
    container_name="agent5.0.0_qjy",
)
executor = RemoteExecutor(config)
```

## RemoteExecutor 核心方法

| 方法 | 用途 | 示例 |
|------|------|------|
| `executor.run(cmd)` | 执行命令，返回 result 对象（.stdout/.stderr） | `executor.run("npu-smi info")` |
| `executor.run_script(py_code)` | 上传并执行 Python 脚本，返回 stdout 字符串 | `executor.run_script("print(1+1)")` |
| `executor.run_host(cmd)` | 在宿主机执行（不进入容器） | `executor.run_host("docker ps")` |

## 执行模型

所有 `run()` 和 `run_script()` 内部等价于：

```bash
ssh root@<server-ip> "docker exec agent5.0.0_qjy bash -c '<command>'"
```

# 密码安全

密码通过 SSH_ASKPASS 机制传递，不会出现在进程列表或命令行参数中：

1. 密码写入临时文件 `/tmp/rllm_askpass_*.sh`（权限 0700）
2. 通过环境变量 `SSH_ASKPASS` + `SSH_ASKPASS_REQUIRE=force` + `DISPLAY=""` 强制 SSH 使用脚本
3. RemoteExecutor 析构时自动清理临时文件

**重要规则**：
- 密码绝不写入 config.json（`to_json()` 自动排除 `ssh_password` 字段）
- 密码不在日志或 stdout 中输出
- 每次连接都需要重新提供密码（或使用 SSH key 作为替代）

# 连接验证

连接建立后应验证以下内容：

```python
# 1. SSH 连通性
executor.run_host("echo ok")

# 2. 容器运行状态
executor.run("echo container_alive")

# 3. AgentSDK 存在
executor.run(f"test -f {config.remote_agent_sdk_dir}/run_start_in_local.sh && echo EXISTS")

# 4. 训练数据存在
executor.run(f"test -f {config.train_data_path} && echo EXISTS")

# 5. 模型存在
executor.run(f"test -d {config.model_name_or_path} && echo EXISTS")

# 6. NPU 可用
executor.run("npu-smi info | grep -c 'NPU'")
```

# 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `Permission denied` | SSH 密码错误或 key 未授权 | 检查密码，或使用 `ssh-keygen` 生成 key 并添加到服务器 |
| `Connection timeout` | 网络不通或防火墙 | `ping <server-ip>`，检查 VPN |
| `container not running` | 容器已停止 | `docker start agent5.0.0_qjy` |
| `No such file` | AgentSDK 路径不对 | 检查 `remote_agent_sdk_dir` 配置 |
| `npu-smi: command not found` | 不在 NPU 节点上 | 确认容器运行在 NPU 节点 |
| TB events 为空 | 训练未开启 tensorboard | 检查 `use_tensorboard: true` |
