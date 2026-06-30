# tito-rl

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/kunalkarda/tito-rl)

**Token-In, Token-Out (TITO) RL Training Framework for tool-using LLM agents.**

A clean, reusable, production-oriented library that enforces correct token handling for reinforcement learning (PPO/GRPO) of agents that use tools.

---

## Why TITO?

When training agents that call tools, the natural approach is to keep a list of messages, re-render the conversation on every turn, and re-tokenize everything for the loss. This is **MITO** (Message-In, Token-Out) and it silently violates a critical invariant:

> **RL must train on the exact tokens the model sampled.**

Decoding + re-encoding is not a no-op. BPE merges, JSON whitespace, special token handling, and template conditionals can all produce different token IDs. The result: gradients land on sequences the policy never generated. You get weird loss spikes, shape errors, and unreliable training.

**TITO rule:** never re-encode tokens you have decoded.

- The running `TokenBuffer` is the source of truth.
- We only decode for routing decisions (e.g., "should I call a tool?").
- Tool responses are injected using precise chat-template deltas (render twice, take the suffix).
- The loss mask is built incrementally — only assistant-generated tokens ever get loss=1.
- The only template operation in the loop is the tool-response delta.

The only requirement is that the chat template must be **prefix-preserving for tool messages** (the vast majority of modern templates are).

Reference: [Agentic RL: Token-In, Token-Out Done Right](https://huggingface.co/blog/huggingface/tito)

## Features

- Exact token buffer with incremental loss mask and segments
- `compute_delta` for safe tool-response injection
- Full rollout loop with tool calling (supports any inference engine that can take token ids)
- GRPO trainer (with basic PPO loss helper)
- Support for history rewriting / compaction (freeze prefix loss)
- Prefix-preservation checker for chat templates
- Works with real models (Qwen, Llama, etc.)

## Installation

```bash
pip install -e .
# with dev / test dependencies
pip install -e ".[dev]"
```

Requires Python ≥ 3.10, `transformers`, and `torch`.

## Quick Start

```python
from transformers import AutoTokenizer
from tito import TokenBuffer, compute_delta

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 1. Start with tokenized prompt
initial = [{"role": "user", "content": "What is 2+2?"}]
prompt_ids = tok.apply_chat_template(initial, add_generation_prompt=True)
buf = TokenBuffer(initial_tokens=prompt_ids)

# 2. Model generates (you pass exact token ids to your inference engine)
# new_ids = your_generate(buf.get_input_ids())

# 3. When the model emits a tool call, append its tokens
buf.append_assistant(new_ids)   # loss = 1

# 4. Run the tool, then inject response via delta (no re-encoding!)
prefix_msgs = [initial[0], {"role": "assistant", "content": "", "tool_calls": [...] }]
tool_msgs = [{"role": "tool", "content": "4"}]
delta = compute_delta(tok, prefix_msgs, tool_msgs)
buf.append_tool(delta)          # loss = 0

# 5. Continue...
print("Trainable tokens:", buf.get_loss_mask().count(1))
```

See `examples/` for full training and evaluation scripts that use real models.

## Core API

### TokenBuffer

```python
from tito import TokenBuffer

buf = TokenBuffer([1, 2, 3])
buf.append_assistant([10, 11], logprobs=[-0.3, -0.1])
buf.append_tool([99])
print(buf.get_assistant_ranges())
print(buf.get_loss_mask())
```

Supports `record_rewrite()` for history compaction / `clear_thinking`.

### compute_delta

```python
delta = compute_delta(
    tokenizer,
    prefix_messages,           # up to and including the assistant tool call
    tool_response_messages,
    add_generation_prompt=True
)
```

### Rollouts

```python
from tito import collect_rollout, make_hf_generate_fn

gen = make_hf_generate_fn(model, tok, return_logprobs=True)
traj = collect_rollout(
    tok,
    initial_messages,
    gen,
    tools={"calc": lambda a, b: a + b},
    max_turns=4,
    max_length=2048,
)
```

Returns a dict with `buffer`, `loss_mask`, `logprobs`, `truncated`, `rewrite_points`, etc.

### Training

```python
from tito import GRPOTrainer

trainer = GRPOTrainer(model, lr=5e-6)
stats = trainer.train_step(trajectories, rewards=rewards, group_size=4)
```

## Examples

```bash
# End-to-end training with real model + GRPO
python examples/train_simple_agent.py

# Run inference traces with real model
python examples/evaluate_agent.py
```

## Honest Edges (from the HF article)

- **History rewriting** (compaction, `clear_thinking`, sub-agent summaries): call `buf.record_rewrite(index)` at the last rewrite point. Everything before becomes prompt (loss=0).
- **Truncation**: just stop. Pass `max_length` to `collect_rollout`. No special close-token synthesis needed.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format .
```

## Roadmap

- ✅ Core TITO primitives (buffer, delta, prefix check)
- ✅ Rollout collection with tools
- ✅ GRPO trainer + logprob support
- ✅ Real-model examples
- Polish, more tests, comparison demo vs naive re-tokenization

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

The sacred rule: **the buffer is the source of truth**. Never re-encode model-generated tokens.

## License

MIT © 2026 Kunal Karda
