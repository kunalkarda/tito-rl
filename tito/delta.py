from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase

def compute_delta(
    tokenizer: PreTrainedTokenizerBase,
    prefix_messages: List[Dict[str, Any]],
    tool_response_messages: List[Dict[str, Any]],
    add_generation_prompt: bool = True,
) -> List[int]:
    prefix_ids = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
    )
    full_messages = prefix_messages + tool_response_messages
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=add_generation_prompt,
    )
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Chat template is not prefix-preserving for tool messages")
    return full_ids[len(prefix_ids) :]
