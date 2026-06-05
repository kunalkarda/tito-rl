import json
import re
from typing import Any, Callable, Dict, List, Optional

from .buffer import TokenBuffer
from .delta import compute_delta


def parse_assistant_response(text: str) -> Dict[str, Any]:
    tool_calls: List[Dict[str, Any]] = []
    patterns = [
        r"<tool_call>(.*?)</tool_call>",
        r"```(?:json)?\s*(.*?)```",
        r"(?:tool_call|call tool|invoke tool)\s*(.*)",
    ]
    for pat in patterns:
        for m in re.findall(pat, text, re.DOTALL | re.IGNORECASE):
            candidate = m.strip()
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
            except Exception:
                try:
                    data = json.loads("{" + candidate + "}")
                except Exception:
                    continue
            if not isinstance(data, dict):
                continue
            if "name" in data:
                args = data.get("arguments", data.get("args", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                tool_calls.append({
                    "type": "function",
                    "function": {"name": data["name"], "arguments": args},
                })
            elif "function" in data and isinstance(data["function"], dict):
                tool_calls.append(data)
    content = text
    for pat in patterns:
        content = re.sub(pat, "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"\s+", " ", content).strip()
    return {"content": content, "tool_calls": tool_calls}


def execute_tool_call(tools: Dict[str, Callable], tc: Dict[str, Any]) -> Any:
    fn = tc.get("function", {})
    name = fn.get("name")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if name in tools:
        try:
            if isinstance(args, dict):
                return tools[name](**args)
            return tools[name](args)
        except Exception as e:
            return f"ERROR: {e}"
    return f"UNKNOWN_TOOL: {name}"


def collect_rollout(
    tokenizer: Any,
    initial_messages: List[Dict[str, Any]],
    generate_fn: Callable[[List[int]], Any],
    tools: Optional[Dict[str, Callable]] = None,
    parser: Optional[Callable[[str], Dict[str, Any]]] = None,
    max_turns: int = 8,
    max_new_tokens_per_turn: int = 512,
    stop_on_no_tool: bool = True,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    if tools is None:
        tools = {}
    if parser is None:
        parser = parse_assistant_response
    prompt_ids = tokenizer.apply_chat_template(
        initial_messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    buffer = TokenBuffer(initial_tokens=prompt_ids)
    messages = list(initial_messages)
    turns = 0
    while turns < max_turns:
        current = buffer.get_input_ids()
        if max_length is not None and len(current) >= max_length:
            buffer.mark_truncated()
            break
        out = generate_fn(current)
        new_ids: List[int] = []
        logprobs: Optional[List[float]] = None
        if isinstance(out, tuple) and len(out) == 2:
            new_ids, logprobs = out
        elif isinstance(out, dict):
            new_ids = out.get("ids") or out.get("token_ids") or out.get("new_ids") or []
            logprobs = out.get("logprobs")
        else:
            new_ids = list(out) if out else []
        if not new_ids:
            break
        buffer.append_assistant(new_ids, logprobs)
        turns += 1
        if max_length is not None and len(buffer) > max_length:
            buffer.mark_truncated()
            break
        gen_text = tokenizer.decode(new_ids, skip_special_tokens=False)
        parsed = parser(gen_text)
        tool_calls = parsed.get("tool_calls", [])
        if tool_calls:
            asst = {
                "role": "assistant",
                "content": parsed.get("content", ""),
                "tool_calls": tool_calls,
            }
            messages.append(asst)
            for tc in tool_calls:
                result = execute_tool_call(tools, tc)
                tname = tc.get("function", {}).get("name", "tool")
                tmsg = {"role": "tool", "name": tname, "content": str(result)}
                delta = compute_delta(tokenizer, messages, [tmsg])
                buffer.append_tool(delta)
                messages.append(tmsg)
                if max_length is not None and len(buffer) > max_length:
                    buffer.mark_truncated()
                    break
        else:
            asst = {
                "role": "assistant",
                "content": parsed.get("content", gen_text),
            }
            messages.append(asst)
            if stop_on_no_tool:
                break
    return {
        "buffer": buffer,
        "messages": messages,
        "turns": turns,
        "final_tokens": buffer.get_input_ids(),
        "loss_mask": buffer.get_loss_mask(),
        "logprobs": buffer.get_logprobs(),
        "assistant_logprobs": buffer.get_assistant_logprobs(),
        "truncated": buffer.truncated,
        "rewrite_points": buffer.get_rewrite_points(),
    }


def make_hf_generate_fn(model: Any, tokenizer: Any, return_logprobs: bool = False, **default_kwargs: Any) -> Callable:
    def generate(current_ids: List[int], max_new_tokens: int = None, **kwargs: Any):
        import torch
        gen_kwargs = dict(default_kwargs)
        gen_kwargs.update(kwargs)
        mnt = max_new_tokens if max_new_tokens is not None else gen_kwargs.get("max_new_tokens", 256)
        gen_kwargs["max_new_tokens"] = mnt
        inp = torch.tensor([current_ids], dtype=torch.long)
        if hasattr(model, "device"):
            inp = inp.to(model.device)
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id or eos
        if return_logprobs:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True
        out = model.generate(
            inp,
            eos_token_id=eos,
            pad_token_id=pad,
            **gen_kwargs,
        )
        if return_logprobs and hasattr(out, "sequences"):
            seq = out.sequences[0]
            new_tokens = seq[len(current_ids):].tolist()
            scores = out.scores
            lps = []
            for i, t in enumerate(new_tokens):
                if i < len(scores):
                    s = scores[i][0]
                    lp = torch.log_softmax(s, dim=-1)[t].item()
                    lps.append(lp)
            return new_tokens, lps
        if hasattr(out, "sequences"):
            new_part = out.sequences[0, len(current_ids):].tolist()
        else:
            new_part = out[0, len(current_ids):].tolist()
        return new_part
    return generate
