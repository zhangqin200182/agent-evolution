"""
SSH-based remote command execution for NPU training server.
Uses bare subprocess (ssh + docker exec) — zero additional Python dependencies.
"""

import os
import subprocess
import sys
import time

from rllm_remote.config import RemoteTrainConfig


class RemoteExecutor:
    """Execute commands on the remote NPU server via SSH + docker exec.

    Supports both key-based and password-based authentication.
    Password mode uses SSH_ASKPASS (built into OpenSSH) — no external
    dependencies. The password is written to a permission-restricted temp
    script, never exposed in process listings.
    """

    def __init__(self, config: RemoteTrainConfig):
        self.config = config
        self._use_password = bool(config.ssh_password)
        self._askpass_script = None
        self._ssh_base = self._build_ssh_base()

    def _build_ssh_base(self) -> list[str]:
        ssh_args = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-p", str(self.config.ssh_port),
        ]
        if self._use_password:
            # PreferredAuthentications=password skips key-based auth attempts
            ssh_args += ["-o", "PreferredAuthentications=password"]
        else:
            key_path = os.path.expanduser(self.config.ssh_key_path)
            ssh_args += ["-i", key_path]
        cmd = ["ssh"] + ssh_args
        cmd.append(f"{self.config.ssh_user}@{self.config.ssh_host}")
        return cmd

    def _ssh_env(self) -> dict:
        """Environment variables for SSH subprocess.

        For password auth: uses SSH_ASKPASS with a temp script that echoes
        the password. SSH_ASKPASS_REQUIRE=force + DISPLAY= + stdin=/dev/null
        forces SSH to use the askpass program instead of prompting the TTY.
        """
        env = os.environ.copy()
        if self._use_password:
            if self._askpass_script is None:
                self._askpass_script = self._write_askpass_script()
            env["SSH_ASKPASS"] = self._askpass_script
            env["SSH_ASKPASS_REQUIRE"] = "force"
            env["DISPLAY"] = ""  # triggers askpass path on macOS
        return env

    def _write_askpass_script(self) -> str:
        """Write a temporary askpass script that echoes the password."""
        import tempfile
        fd, path = tempfile.mkstemp(prefix="rllm_askpass_", suffix=".sh")
        script = f"#!/bin/bash\necho '{self.config.ssh_password}'"
        os.write(fd, script.encode())
        os.close(fd)
        os.chmod(path, 0o700)
        return path

    def __del__(self):
        if self._askpass_script and os.path.exists(self._askpass_script):
            os.unlink(self._askpass_script)

    def _docker_cmd(self, command: str) -> str:
        """Wrap a command to run inside the container via docker exec."""
        escaped = command.replace("'", "'\\''")
        return f"docker exec {self.config.container_name} bash -c '{escaped}'"

    def _scp_base(self) -> list[str]:
        """Build the scp command prefix."""
        scp_args = [
            "-o", "StrictHostKeyChecking=no",
            "-P", str(self.config.ssh_port),
        ]
        if self._use_password:
            scp_args += ["-o", "PreferredAuthentications=password"]
        else:
            key_path = os.path.expanduser(self.config.ssh_key_path)
            scp_args += ["-i", key_path]
        return ["scp"] + scp_args

    def _run_kwargs(self) -> dict:
        """Common kwargs for subprocess.run/Popen."""
        kwargs: dict = {"env": self._ssh_env()}
        if self._use_password:
            kwargs["stdin"] = subprocess.DEVNULL
        return kwargs

    def run(self, command: str, timeout: int = 3600, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command inside the container and return the result."""
        full_cmd = self._ssh_base + [self._docker_cmd(command)]
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout,
                                **self._run_kwargs())
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Remote command failed (exit={result.returncode}):\n"
                f"  stderr: {result.stderr.strip()}\n"
                f"  stdout: {result.stdout.strip()}"
            )
        return result

    def run_quiet(self, command: str, timeout: int = 3600) -> subprocess.CompletedProcess:
        """Run a command without raising on non-zero exit."""
        return self.run(command, timeout=timeout, check=False)

    def run_host(self, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a command on the host (not inside container)."""
        full_cmd = self._ssh_base + [command]
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout,
                                **self._run_kwargs())
        if result.returncode != 0:
            raise RuntimeError(
                f"Host command failed (exit={result.returncode}):\n  stderr: {result.stderr.strip()}"
            )
        return result

    def is_container_running(self) -> bool:
        """Check if the target container is running on the server (host-level check)."""
        result = self.run_host(
            f"docker ps --filter name={self.config.container_name} --format '{{{{.Names}}}}'"
        )
        return self.config.container_name in result.stdout

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the container via scp + docker cp."""
        basename = os.path.basename(local_path)
        host_tmp = f"/tmp/rllm_upload_{basename}"

        scp_cmd = self._scp_base() + [
            local_path,
            f"{self.config.ssh_user}@{self.config.ssh_host}:{host_tmp}",
        ]
        subprocess.run(scp_cmd, capture_output=True, text=True, check=True,
                       **self._run_kwargs())

        remote_dir = os.path.dirname(remote_path)
        self.run(f"mkdir -p {remote_dir}")
        self.run_host(f"docker cp {host_tmp} {self.config.container_name}:{remote_path}")

    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a file from the container to local via docker cp + scp."""
        basename = os.path.basename(remote_path)
        host_tmp = f"/tmp/rllm_download_{basename}"

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.run_host(f"docker cp {self.config.container_name}:{remote_path} {host_tmp}")

        scp_cmd = self._scp_base() + [
            f"{self.config.ssh_user}@{self.config.ssh_host}:{host_tmp}",
            local_path,
        ]
        subprocess.run(scp_cmd, capture_output=True, text=True, check=True,
                       **self._run_kwargs())

    def tail_log(self, remote_path: str, lines: int = 30) -> str:
        """Return the last N lines of a remote log file."""
        result = self.run(f"tail -{lines} {remote_path}", timeout=30)
        return result.stdout

    def stream_log(self, remote_path: str) -> subprocess.Popen:
        """Start streaming a remote log file. Returns a Popen for the caller to manage."""
        cmd = self._ssh_base + [
            self._docker_cmd(f"tail -f {remote_path}")
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                **self._run_kwargs())

    def run_background(self, command: str, log_file: str) -> str:
        """Launch a command in the background inside the container via nohup.

        Prepends required container environment setup:
        1. VLLM_ASCEND_ENABLE_NZ=0 — required by Ascend NPU vLLM backend
        2. /etc/hosts entry mapping 127.0.0.1 to hostname — required by Ray

        Returns the PID of the background process.
        """
        log_dir = os.path.dirname(log_file)
        self.run(f"mkdir -p {log_dir}")

        # Container initialization (mandatory for NPU training, idempotent)
        init_cmds = (
            "export VLLM_ASCEND_ENABLE_NZ=0 && "
            'grep -q "127.0.0.1 $(hostname)" /etc/hosts 2>/dev/null || '
            'echo "127.0.0.1 $(hostname)" >> /etc/hosts'
        )

        full_cmd = (
            f"{init_cmds} && "
            f"cd {self.config.remote_agent_sdk_dir} && "
            f"{command} > {log_file} 2>&1"
        )
        result = self.run(
            f"nohup bash -c '{full_cmd}' > /dev/null 2>&1 & echo $!",
            timeout=30,
        )
        pid = result.stdout.strip()
        # Small wait to detect immediate failures
        time.sleep(2)
        if not self.check_pid(pid):
            tail = self.tail_log(log_file, lines=20)
            raise RuntimeError(
                f"Process {pid} died immediately after launch.\n"
                f"Log tail ({log_file}):\n{tail}"
            )
        return pid

    def check_pid(self, pid: str) -> bool:
        """Check if a process with given PID is still running."""
        result = self.run_quiet(f"kill -0 {pid} 2>/dev/null")
        return result.returncode == 0

    def check_process(self, run_id: str) -> bool:
        """Check if a training process by run_id is still running."""
        result = self.run(f"ps aux | grep {run_id} | grep -v grep | wc -l")
        return int(result.stdout.strip()) > 0

    def kill_pid(self, pid: str) -> None:
        """Kill a background process by PID."""
        self.run_quiet(f"kill {pid} 2>/dev/null")

    def run_script(self, script: str, timeout: int = 60) -> str:
        """Write a Python script to /tmp in the container and execute it.

        Uses base64 encoding to avoid heredoc quoting conflicts with bash -c.
        Returns stdout of the script.
        """
        import base64
        import uuid
        script_path = f"/tmp/rllm_script_{uuid.uuid4().hex[:8]}.py"
        encoded = base64.b64encode(script.encode()).decode()
        self.run(f"echo {encoded} | base64 -d > {script_path}")
        result = self.run(f"python3 {script_path} 2>&1", timeout=timeout)
        self.run_quiet(f"rm -f {script_path}")
        return result.stdout

    def verify_connectivity(self) -> dict:
        """Run a full connectivity check. Returns a dict with status of each check."""
        results = {}

        # SSH connectivity
        try:
            self.run_host("echo ok", timeout=15)
            results["ssh"] = True
        except Exception as e:
            results["ssh"] = False
            results["ssh_error"] = str(e)
            return results  # can't check further

        # Container running
        results["container"] = self.is_container_running()

        # AgentSDK launch script exists
        try:
            r = self.run(
                f"test -f {self.config.remote_agent_sdk_dir}/run_start_in_local.sh && echo EXISTS"
            )
            results["agentsdk_script"] = "EXISTS" in r.stdout
        except Exception:
            results["agentsdk_script"] = False

        # Training data exists
        try:
            r = self.run(f"test -f {self.config.train_data_path} && echo EXISTS")
            results["train_data"] = "EXISTS" in r.stdout
        except Exception:
            results["train_data"] = False

        # Model exists
        try:
            r = self.run(f"test -f {self.config.model_name_or_path}/config.json && echo EXISTS")
            results["model"] = "EXISTS" in r.stdout
        except Exception:
            results["model"] = False

        return results
