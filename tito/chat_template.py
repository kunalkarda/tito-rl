from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase


def is_prefix_preserving_for_tools(tokenizer: PreTrainedTokenizerBase) -> bool:
    dummy_tool_calls = [{"type": "function", "function": {"name": "dummy", "arguments": {}}}]
    messages1 = [
        {"role": "user", "content": "dummy"},
        {"role": "assistant", "content": "", "tool_calls": dummy_tool_calls},
    ]
    messages2 = messages1 + [{"role": "tool", "name": "dummy", "content": "dummy"}]
    ids1 = tokenizer.apply_chat_template(messages1, tokenize=True, return_dict=False)
    ids2 = tokenizer.apply_chat_template(messages2, tokenize=True, return_dict=False, add_generation_prompt=True)
    return ids2[: len(ids1)] == ids1


def check_prefix_preservation(tokenizer: PreTrainedTokenizerBase) -> bool:
    return is_prefix_preserving_for_tools(tokenizer)
