---
id: traj-20260503-091950-round-signal
target_section: round-signal
action: append
description: 修复 session_id 快照差分法，增加目录预创建和 fallback 逻辑
status: proposed
source: trajectory-analysis
source_sessions: ["b4d588ba-052e-4153-9c8b-5681a8850d9f"]
---

### session_id 快照差分法改进

Phase 0 记录快照时，先确保目录存在:
```python
import os
raw_dir = "traj_opt/output/rllm/raw/"
os.makedirs(raw_dir, exist_ok=True)
existing = set(os.listdir(raw_dir))
```

Phase 6.5 差分时，增加容错:
```python
current = set(os.listdir(raw_dir)) if os.path.exists(raw_dir) else set()
new_sessions = current - existing
if new_sessions:
    session_id = sorted(new_sessions)[-1]
else:
    # fallback: 使用最近修改的目录
    import pathlib
    dirs = sorted(pathlib.Path(raw_dir).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    session_id = dirs[0].name if dirs else "unknown"
```
