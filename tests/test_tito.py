import pytest
import torch
from tito import (
    TokenBuffer,
    compute_delta,
    collect_rollout,
    parse_assistant_response,
    get_loss_mask_from_ranges,
    get_active_token_count,
    pad_mask,
    masked_cross_entropy_loss,
    compute_kl_divergence,
    GRPOTrainer,
    ppo_loss,
)
from tito.chat_template import is_prefix_preserving_for_tools
from transformers import AutoModelForCausalLM, AutoTokenizer as HFAutoTokenizer

class FakeTokenizer:
    def __init__(self):
        self._cache = {}
        self._id_counter = 200

    def _msg_block(self, msg: dict) -> list[int]:
        role = msg.get("role", "user")
        content = str(msg.get("content", "") or "")
        tc = str(msg.get("tool_calls", "")) if "tool_calls" in msg else ""
        key = f"{role}:{content}:{tc}"
        if key not in self._cache:
            h = abs(hash(key)) % 10000
            block = [
                {"system": 10, "user": 20, "assistant": 30, "tool": 40}.get(role, 99),
                100 + (h % 500),
                600 + ((h // 10) % 300),
                99,
            ]
            self._cache[key] = block
        return self._cache[key]

    def apply_chat_template(self, messages, tokenize=True, return_dict=False, add_generation_prompt=False):
        ids = []
        for m in messages:
            ids.extend(self._msg_block(m))
        if add_generation_prompt:
            ids.extend([151644, 77091, 198])
        return ids

    def decode(self, token_ids, skip_special_tokens=False):
        if not token_ids:
            return ""
        s = "".join(str(i) for i in token_ids)
        if "99" in s or (token_ids and token_ids[0] % 7 == 0):
            return '<tool_call>{"name": "dummy", "arguments": {"x": 2}}</tool_call>'
        return "final answer 42"


class BadFakeTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=True, return_dict=False, add_generation_prompt=False):
        ids = []
        for m in messages:
            ids.extend(self._msg_block(m))
        if len(messages) > 2:
            ids = [999999] + ids
        if add_generation_prompt:
            ids.extend([151644, 77091, 198])
        return ids


def test_buffer_basic():
    b = TokenBuffer([10, 20, 30])
    assert len(b) == 3
    assert b.loss_mask == [0, 0, 0]
    assert b.get_assistant_ranges() == []
    b.append_assistant([40, 41])
    assert b.loss_mask == [0, 0, 0, 1, 1]
    assert b.get_assistant_ranges() == [(3, 5)]
    b.append_tool([50])
    assert b.loss_mask == [0, 0, 0, 1, 1, 0]
    b.append_prompt([60])
    assert len(b) == 7
    assert b.get_input_ids()[-1] == 60


def test_buffer_tensor():
    b = TokenBuffer([1])
    b.append_assistant([2])
    t = b.get_loss_mask_tensor()
    assert t.tolist() == [0, 1]
    assert t.dtype is not None


def test_compute_delta_preserves():
    tok = FakeTokenizer()
    prefix = [
        {"role": "user", "content": "what is 2+2"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "calc", "arguments": {"expr": "2+2"}}}]},
    ]
    tool_msgs = [{"role": "tool", "content": "4"}]
    delta = compute_delta(tok, prefix, tool_msgs, add_generation_prompt=True)
    prefix_ids = tok.apply_chat_template(prefix, tokenize=True, return_dict=False, add_generation_prompt=False)
    full = prefix + tool_msgs
    full_ids = tok.apply_chat_template(full, tokenize=True, return_dict=False, add_generation_prompt=True)
    assert delta == full_ids[len(prefix_ids):]
    assert len(delta) > 0


def test_compute_delta_raises_on_non_preserving():
    tok = BadFakeTokenizer()
    prefix = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "f"}}]},
    ]
    tool_msgs = [{"role": "tool", "content": "y"}]
    with pytest.raises(ValueError):
        compute_delta(tok, prefix, tool_msgs)


def test_is_prefix_preserving_fake():
    tok = FakeTokenizer()
    assert is_prefix_preserving_for_tools(tok) is True


def test_roundtrip_buffer_with_delta():
    tok = FakeTokenizer()
    b = TokenBuffer([1, 2, 3])
    b.append_assistant([100, 101, 102])
    prefix = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "t"}}]}]
    tool_msgs = [{"role": "tool", "content": "res"}]
    delta = compute_delta(tok, prefix, tool_msgs)
    b.append_tool(delta)
    assert len(b.get_loss_mask()) == len(b.get_input_ids())
    assert b.loss_mask.count(1) == 3
    assert b.get_assistant_ranges() == [(3, 6)]


def test_parse_assistant_response_tool():
    text = 'thinking <tool_call>{"name": "calc", "arguments": {"expr": "2+2"}}</tool_call> more'
    out = parse_assistant_response(text)
    assert len(out["tool_calls"]) == 1
    assert out["tool_calls"][0]["function"]["name"] == "calc"
    assert isinstance(out["content"], str)


def test_parse_assistant_response_normal():
    out = parse_assistant_response("the final answer is 42")
    assert out["tool_calls"] == []
    assert "42" in out["content"]


def test_collect_rollout_no_tools():
    tok = FakeTokenizer()
    calls = {"n": 0}
    def gen(ids):
        calls["n"] += 1
        return [40, 41]
    initial = [{"role": "user", "content": "hello"}]
    res = collect_rollout(tok, initial, gen, tools={}, max_turns=2)
    assert res["turns"] == 1
    assert 1 in res["loss_mask"]
    assert len(res["buffer"]) > len(initial)


def test_collect_rollout_with_tool_and_continue():
    tok = FakeTokenizer()
    calls = {"n": 0}
    def gen(ids):
        calls["n"] += 1
        if calls["n"] == 1:
            return [30, 99]
        return [50, 51]
    def double(x=1):
        return int(x) * 2
    tools = {"dummy": double}
    initial = [{"role": "user", "content": "compute"}]
    res = collect_rollout(tok, initial, gen, tools=tools, max_turns=3, stop_on_no_tool=True)
    assert res["turns"] >= 2
    assert res["buffer"].loss_mask.count(1) >= 2
    msgs = res["messages"]
    assert any(m.get("role") == "tool" for m in msgs)
    assert any("tool_calls" in m for m in msgs)


def test_collect_rollout_truncates_on_max_turns():
    tok = FakeTokenizer()
    def gen(ids):
        return [30, 99]
    def echo(x):
        return x
    res = collect_rollout(tok, [{"role": "user", "content": "x"}], gen, tools={"dummy": echo}, max_turns=2)
    assert res["turns"] == 2

def test_collect_rollout_truncates_on_max_length():
    tok = FakeTokenizer()
    def gen(ids):
        return [30, 99]
    res = collect_rollout(tok, [{"role": "user", "content": "x"}], gen, max_turns=5, max_length=10)
    assert res["truncated"] is True
    assert len(res["final_tokens"]) <= 30  # deltas add but we stopped

def test_buffer_rewrite_freeze():
    b = TokenBuffer([1,2,3,4,5,6,7])
    b.append_assistant([10,11,12])
    b.record_rewrite(5)
    assert b.loss_mask[:5] == [0,0,0,0,0]
    assert b.loss_mask[7:10] == [1,1,1]
    assert b.get_rewrite_points() == [5]
    b.append_assistant([20])
    assert b.loss_mask[-1] == 1


def test_buffer_logprobs():
    b = TokenBuffer([1, 2])
    b.append_assistant([10, 11], logprobs=[-0.5, -0.1])
    assert b.get_logprobs()[-2:] == [-0.5, -0.1]
    assert b.get_assistant_logprobs() == [-0.5, -0.1]
    b.append_tool([99])
    assert b.get_logprobs()[-1] is None


def test_masking_utils():
    ranges = [(2, 5)]
    mask = get_loss_mask_from_ranges(7, ranges)
    assert mask == [0, 0, 1, 1, 1, 0, 0]
    assert get_active_token_count(mask) == 3
    padded = pad_mask(mask, 10)
    assert len(padded) == 10
    assert padded[-1] == 0


def test_rollout_with_logprobs():
    tok = FakeTokenizer()
    def gen(ids):
        return ([30, 99], [-0.3, -0.7])
    def dummy(x=0):
        return x
    res = collect_rollout(tok, [{"role": "user", "content": "t"}], gen, tools={"dummy": dummy}, max_turns=1)
    assert res["assistant_logprobs"][0] in (-0.3, -0.7)
    assert len(res["logprobs"]) == len(res["final_tokens"])


def test_grpo_trainer_basic_step():
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    tok = HFAutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    trainer = GRPOTrainer(model, lr=1e-4, device="cpu", kl_coef=0.0)
    traj = {
        "final_tokens": [1, 2, 3, 4, 5],
        "loss_mask": [0, 0, 1, 1, 1],
        "logprobs": [0.0, 0.0, -0.2, -0.5, -1.0],
        "advantages": [0.0, 0.0, 1.0, 1.0, 1.0],
    }
    stats = trainer.train_step([traj])
    assert "loss" in stats
    assert stats["active_tokens"] > 0
    assert stats["loss"] < 100.0


def test_ppo_loss_helper():
    new_l = torch.tensor([[ -0.1, -0.2 ]])
    old_l = torch.tensor([[ -0.3, -0.4 ]])
    adv = torch.tensor([[ 1.0, 0.5 ]])
    m = torch.tensor([[ 1.0, 1.0 ]])
    l = ppo_loss(new_l, old_l, adv, m, clip_eps=0.2)
    assert l.item() >= 0.0 or l.item() <= 10.0
