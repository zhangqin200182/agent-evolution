"""
Self-contained chat template and tool parsers inlined from rllm.
Provides QwenChatTemplateParser, QwenToolParser, and helper functions
for converting messages to tokens and masks.

Re-exports from rllm_common for backward compatibility.
"""

from rllm_common.parsers import (  # noqa: F401
    ChatTemplateParser,
    QwenChatTemplateParser,
    QwenToolParser,
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
