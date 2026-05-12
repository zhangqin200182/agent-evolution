"""Remote NPU training backend for agent RL via AgentSDK on Huawei Ascend servers."""

import os


def connect(run_id: str, ssh_password: str | None = None):
    """Load config and create RemoteExecutor for a given run_id.

    Canonical entry point for remote server connections.
    Returns (RemoteTrainConfig, RemoteExecutor).

    Usage:
        config, executor = connect("remote_1778504118", ssh_password="...")
    """
    from rllm_remote.config import RemoteTrainConfig
    from rllm_remote.ssh import RemoteExecutor

    paths = [
        os.path.join("rllm_remote", "output", "runs", run_id, "config.json"),
    ]
    default = RemoteTrainConfig()
    paths.append(os.path.join(default.local_output_dir, run_id, "config.json"))

    config_path = None
    for p in paths:
        if os.path.exists(p):
            config_path = p
            break

    if config_path is None:
        raise FileNotFoundError(
            f"No config.json found for run_id={run_id}. Tried: {paths}"
        )

    config = RemoteTrainConfig.from_json(config_path)
    if ssh_password:
        config.ssh_password = ssh_password
    executor = RemoteExecutor(config)
    return config, executor
