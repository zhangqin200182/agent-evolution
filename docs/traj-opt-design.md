# traj_opt 优化后端设计

> Skill 自动优化的 Python 后端。提供轨迹捕获、存储、分割、分析基础设施和 patch 生成能力。

> 命名说明：文档中 `traj_opt` 指代优化后端（代码目录 `traj_opt/`）。跨模块设计（架构、隔离、双 CLI）见 `system-overview.md`。

## 1. 概述

traj_opt 是 traj-xx skills 的 Python 后端，提供五层基础设施：

```
捕获 (hooks/) → 适配 (adapter/) → 存储 (store/) → 分割 (segmenter/) → 分析/优化 (analyzer/ + optimizer/)
```

核心智能在 traj-xx skills 的 SKILL.md 中（分析策略、领域知识），Python 代码只提供基础设施。这样分析策略可以通过 skill-bank patch 快速迭代，不需要改 Python 代码。

## 2. 数据模型

### 原始数据层

数据层次：Session > Conversation > Turn > ToolCall

```python
@dataclass
class Session:
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    conversations: List[Conversation]

@dataclass
class Conversation:
    conversation_id: str
    parent_conversation_id: Optional[str]
    turns: List[Turn]
    is_subagent: bool

@dataclass
class Turn:
    turn_index: int
    tool_calls: List[ToolCall]
    timestamp: datetime

@dataclass
class ToolCall:
    tool_name: str          # Bash, Read, Edit, Write, Skill, Agent, ...
    tool_input: Dict[str, Any]
    tool_response: Optional[Dict[str, Any]]
    timestamp: datetime
    success: bool
    files_touched: List[str]
```

### 分析数据层

Trajectory 是叠加在原始层次上的分析概念，由 Segmenter 生成：

```python
@dataclass
class Trajectory:
    trajectory_id: str
    session_id: str
    conversation_id: str
    trajectory_type: str        # "skill" | "free"
    skill_name: Optional[str]
    skill_args: Optional[str]
    tool_calls: List[ToolCall]
    nested_conversations: List[Conversation]
    start_time: datetime
    end_time: datetime
    duration_ms: float
    files_touched: List[str]
    intent_tags: List[str]      # exploration, implementation, testing, debugging
    outcome: str                # success, failure, partial, abandoned

@dataclass
class SkillOptimizationSuggestion:
    skill_name: str
    target_section: str
    action: str                 # replace | append | prepend | insert_after
    description: str
    rationale: str
    priority: str               # P0 | P1 | P2
    patch_content: str
    source_sessions: List[str]
```

## 3. 存储格式

所有数据使用 JSONL 格式，按 layer 隔离存储：

```
traj_opt/output/
├── rllm/                          # Layer 1 轨迹
│   ├── raw/{session_id}/events.jsonl
│   ├── trajectories/{session_id}/trajectories.jsonl
│   └── reports/
├── traj/                          # Layer 2 轨迹
│   └── (同上结构)
├── rounds/                        # 轮次协调
│   └── round_{n}/status.json
└── index.jsonl                    # 全局索引（标注 layer）
```

Layer 检测：hooks 根据 skill 名称前缀自动标注（rllm-* → rllm, traj-* → traj）。

## 4. 模块结构

```
trajectory/
├── hooks/                  # Hook 入口脚本
│   ├── post_tool.py        # PostToolUse — 实时记录工具调用
│   └── on_stop.py          # Stop — turn/session 结束处理
├── adapter/                # Hooks JSON → 内部格式
│   ├── schema.py           # 内部数据模型
│   └── hooks_adapter.py    # Hooks stdin JSON → TrajectoryEvent
├── segmenter/              # 轨迹分割（可插拔策略）
│   ├── base.py             # SegmenterStrategy 接口
│   ├── skill_segmenter.py  # Skill 轨迹识别
│   ├── free_segmenter.py   # 自由轨迹分割
│   └── registry.py         # 策略注册与链式执行
├── store/                  # 存储层
│   ├── writer.py           # 事件写入 JSONL
│   ├── reader.py           # 轨迹查询与读取
│   └── index.py            # 索引管理
├── analyzer/               # 分析层基础设施
│   ├── base.py             # AnalyzerBase（轨迹读取、训练数据提取）
│   └── report.py           # ReportWriter
├── optimizer/              # 优化层基础设施
│   ├── patch_generator.py  # PatchGenerator（生成 + 校验 + 激活）
│   └── compiler_bridge.py  # CompilerBridge（调用 compile.py）
├── round_state.py          # 轮次协调（status.json 读写）
└── config.py               # TrajectoryConfig
```

## 5. 捕获层

### Hooks 配置

由 traj-setup skill 写入 `.claude/settings.json`：

```json
{
  "hooks": {
    "PostToolUse": [{"matcher": ".*", "hooks": [{"type": "command", "command": "python traj_opt/hooks/post_tool.py"}]}],
    "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "python traj_opt/hooks/on_stop.py"}]}],
    "SubagentStop": [{"matcher": "", "hooks": [{"type": "command", "command": "python traj_opt/hooks/on_stop.py --subagent"}]}]
  }
}
```

### 处理流程

PostToolUse：读取 stdin JSON → HooksAdapter 转换 → Writer 追加到 events.jsonl。失败时静默。

过滤：只存储包含工具调用的 turn，纯文本讨论跳过。

## 6. 适配器层

```python
class TrajectoryEvent:
    """适配器输出 — 下游所有模块只依赖此格式"""
    event_type: str         # "tool_call" | "turn_end" | "session_end"
    session_id: str
    conversation_id: str
    timestamp: datetime
    tool_name: Optional[str]
    tool_input: Optional[Dict]
    tool_response: Optional[Dict]
    success: Optional[bool]
    files_touched: List[str]
    raw_hook_data: Dict

class HooksAdapter:
    """Claude Code Hooks JSON → TrajectoryEvent — 唯一的 schema 耦合点"""
    def adapt(self, hook_type: str, stdin_json: dict) -> TrajectoryEvent: ...
```

Hooks schema 变化时只需修改 HooksAdapter，下游不受影响。

Conversation ID 推断：主对话用 session_id，子 agent 检测 `tool_name == "Agent"` 创建新 conversation。

## 7. 分割层

### 策略接口

```python
class SegmenterStrategy(ABC):
    @abstractmethod
    def segment(self, events: List[TrajectoryEvent]) -> List[Trajectory]: ...
```

### Skill Segmenter

以 `tool_name == "Skill"` 为锚点，收集后续工具调用直到下一个 Skill 调用或 turn 边界。输出 `Trajectory(type="skill")`。

### Free Segmenter

处理 Skill Segmenter 未覆盖的事件。按 turn 边界切分，turn 内按文件亲和性聚合，打意图标签（exploration/implementation/testing/debugging）。

### Registry

```python
class SegmenterRegistry:
    def segment(self, events) -> List[Trajectory]:
        # 先 Skill Segmenter，剩余用 Free Segmenter
```

## 8. 分析层

架构分离：SKILL.md 定义领域知识和分析策略，Python 代码提供基础设施。

```
traj-analyze-rllm (SKILL.md)        trajectory/analyzer/ (Python)
┌──────────────────────────┐        ┌────────────────────────┐
│ 领域知识:                 │        │ 基础设施:               │
│ - 训练动态理解            │  调用   │ - 读取轨迹文件          │
│ - 失败模式识别            │ ─────→ │ - 训练数据提取          │
│ - 优化建议生成策略         │        │ - 输出报告              │
└──────────────────────────┘        └────────────────────────┘
```

### AnalyzerBase

```python
class AnalyzerBase:
    def get_rllm_trajectories(self, days=None, session_id=None) -> List[Trajectory]
    def get_available_training_data(self, session_id=None) -> List[dict]
    def summarize_trajectory(self, traj) -> dict
```

## 9. 优化层

### PatchGenerator

生成 skill-bank patch 文件，含三项校验：

```python
class PatchGenerator:
    ALLOWED_TARGET_GROUPS = {"rllm"}

    def generate_patch(self, suggestion) -> Path:
        group = self._find_group(suggestion.skill_name)
        self._validate_target_group(suggestion.skill_name, group)
        self._validate_target_section(suggestion.skill_name, group, suggestion.target_section)
        # ... 生成 patch 文件 ...
        self._activate_patch(skill_dir, patch_id)
        return patch_path
```

### CompilerBridge

调用 `skill-bank/compile.py` 编译 skill。

## 10. 配置

```python
@dataclass
class TrajectoryConfig:
    output_dir: str = "traj_opt/output"
    capture_all_tools: bool = True
    default_segmenter: str = "default"
    analysis_lookback_days: int = 7
    min_trajectories_for_analysis: int = 5
```

## 11. 注意事项

- **性能**: Hook 脚本 < 1s（只做文件追加）。分割和分析是离线后处理
- **容错**: Hook 失败静默。事件写入追加模式。分割和分析可重复执行（幂等）
- **隐私**: `output/` 加入 `.gitignore`。不存储纯文本讨论

## 12. 扩展性

- **新增分析场景**: 在 `skill-bank/traj/` 下创建新分析 skill，复用 `trajectory/analyzer/` 基础设施
- **新增分割策略**: 实现 `SegmenterStrategy` 接口，注册到 `SegmenterRegistry`
- **Meta-optimization**: 分析结果可反向优化分析 skill 自身、分割策略、捕获过滤规则
