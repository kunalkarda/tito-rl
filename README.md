# tito-rl

Token-In Token-Out (TITO) RL Training Framework for tool-using LLM agents.

A clean, reusable, production-oriented library that enforces correct token handling for RL (PPO/GRPO) of agents that call tools.

## Why TITO

When training agents with tools:

- The model generates tokens (assistant turns).
- You parse them to detect tool calls.
- You run tools and inject results.
- The next model turn continues from there.

The natural approach (keep a list of messages, re-render with `apply_chat_template` each turn, tokenize the whole history at loss time) silently violates the invariant:

**Train only on the exact tokens the model sampled.**

Re-encoding after decode is not a no-op (BPE merges, JSON whitespace, special token rendering, template conditionals). The ids you compute loss on can differ from the ids that were actually generated under the policy. Gradients become invalid. Shape mismatches and weird loss spikes follow.

TITO rule: **never re-encode tokens you have decoded.**

- Keep a running buffer of token ids. It is the source of truth.
- Parse decoded text only for routing (decide whether to call a tool).
- Insert tool responses by computing a precise delta from the chat template (two renders, subtract).
- Build the loss mask incrementally: only tokens from assistant generations get loss=1. Tool responses and prompt tokens get 0.
- The only template operation in the hot loop is the tool-response delta.

The sole requirement on the chat template is that it is *prefix-preserving for tool messages*: appending a tool result must extend the previous render token-for-token.

```text
render([user, asst_tool_call, tool]) == render([user, asst_tool_call]) + delta
```

The vast majority of modern templates satisfy this (Qwen2.5, Llama-3.1/3.2/4, DeepSeek, Gemma, GLM, etc.). Qwen3 needs a one-line Jinja patch.

Reference: https://huggingface.co/blog/huggingface/tito

## Project Structure

```
tito-rl/
├── tito/
│   ├── __init__.py
│   ├── buffer.py          # TokenBuffer
│   ├── delta.py           # compute_delta
│   ├── chat_template.py   # prefix preservation test
│   ├── rollout.py         # (phase 2+)
│   ├── masking.py
│   └── trainer.py
├── examples/
│   ├── train_simple_agent.py
│   └── evaluate_agent.py
├── tests/
│   └── test_tito.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -e .
# or with test extras
pip install -e ".[test]"
```

Requires Python >=3.10, transformers, torch.

## Phase 1 Core API (done)

### TokenBuffer

Maintains exact token ids + loss mask. Never mutates prior content.

```python
from tito import TokenBuffer

buf = TokenBuffer(initial_tokens=[1, 2, 3])  # prompt ids, loss=0
buf.append_assistant([10, 11, 12])           # model sampled, loss=1
buf.append_tool([20, 21])                    # tool delta ids, loss=0

print(len(buf))
print(buf.get_loss_mask())
print(buf.get_assistant_ranges())
ids = buf.get_input_ids()
```

### compute_delta

Inject tool response by ids only.

```python
from tito import compute_delta
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

prefix_messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "calculator", "arguments": {"expr": "2+2"}}}
    ]},
]
tool_messages = [{"role": "tool", "content": "4"}]

delta_ids = compute_delta(tok, prefix_messages, tool_messages, add_generation_prompt=True)
# delta_ids are exactly the tokens to append to your buffer for the tool turn + next assistant start
```

Under the hood it does the two renders and slices. It raises if the template is not prefix-preserving.

### Prefix preservation check

```python
from tito import is_prefix_preserving_for_tools
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
print(is_prefix_preserving_for_tools(tok))  # True
```

Run the same on any model the day it ships.

## Rollout collection

```python
from tito import collect_rollout, make_hf_generate_fn, parse_assistant_response

# generate_fn must accept the current exact token ids (the TITO buffer)
# and return ONLY the newly generated continuation ids (no re-encode of prefix)
def my_generate(current_token_ids):
    # plug vLLM, SGLang, HF, etc. here. Must be token-in/token-out.
    ...

tools = {
    "calculator": lambda expr: str(eval(expr)),
}

initial_messages = [
    {"role": "system", "content": "You use tools."},
    {"role": "user", "content": "What is 2+2?"},
]

traj = collect_rollout(
    tokenizer,
    initial_messages,
    generate_fn=my_generate,
    tools=tools,
    max_turns=6,
    max_new_tokens_per_turn=256,
)

buf = traj["buffer"]
print("assistant loss tokens:", buf.get_loss_mask().count(1))
print("total length:", len(buf))
print("turns:", traj["turns"])
```

The built-in parser handles common tool call formats (XML blocks, JSON code fences, etc.). Pass a custom `parser=your_fn` if needed.

HF convenience:

```python
hf_generate = make_hf_generate_fn(model, tokenizer, temperature=0.6, do_sample=True)
traj = collect_rollout(tokenizer, initial_messages, hf_generate, tools=tools)
```

All tool responses are injected via `compute_delta`; sampled assistant tokens go verbatim into the buffer. The returned buffer + loss_mask is directly usable for RL objectives.

## Loss masking and RL training (Phase 3)

The buffer carries the loss mask and (optionally) old logprobs captured during rollout.

```python
from tito import GRPOTrainer, collect_rollout, make_hf_generate_fn

# after rollout with logprob-aware generate
# gen = make_hf_generate_fn(model, tok, return_logprobs=True, ...)
# traj = collect_rollout(..., generate_fn=gen)
# user computes reward for the traj e.g. from final answer correctness
traj["rewards"] = 1.0   # or list for groups

trainer = GRPOTrainer(model, ref_model=ref, lr=5e-7, kl_coef=0.01)
stats = trainer.train_step([traj], group_size=4)  # relative advantages within group
print(stats["loss"])
```

Masking helpers (for custom loops):

```python
from tito import masked_cross_entropy_loss, compute_token_level_advantages
# ...
loss = masked_cross_entropy_loss(logits, labels, mask_tensor)
advs = compute_token_level_advantages(rewards, masks, group_size=4)
```

`ppo_loss` helper is also exported for custom PPO implementations with optional value head.

## Examples

Run the end-to-end demo (uses Qwen/Qwen2.5-0.5B-Instruct for real model behavior and generations in rollouts + GRPO; first run downloads weights):

```bash
python examples/train_simple_agent.py
```

It performs several collect+train iterations, prints per-step avg reward and loss, saves a checkpoint.

Evaluation (real model inference traces):

```bash
python examples/evaluate_agent.py
```

These demonstrate the complete TITO pipeline: exact token buffers, optional logprobs, tool rollouts (parser + compute_delta + loss=0 on tool responses), reward assignment, and training.

Sample inference result from running the eval (or the demo section in train):

```
=== TITO Inference Result ===
Total tokens in buffer: 89
Loss mask 1s (assistant tokens): 30

Conversation trace:
User: Use the calc tool to add 2 and 3.
Assistant: [tool call] calc({'a': 2, 'b': 3})
Tool (calc): 5
Assistant: The result is 5.

Tool was used in rollout: True
=== End ===
```

(The assistant turns have loss=1; tool response injected via delta has loss=0.)

## Usage in a TITO rollout loop (sketch)

```python
# 1. tokenize the initial user prompt once
prompt_ids = tok.apply_chat_template([{"role": "user", "content": "..."}], add_generation_prompt=True)

buf = TokenBuffer(initial_tokens=prompt_ids)

while not done:
    # 2. generate continuation using *exact* current token prefix (pass buf.tokens to vllm/sglang etc)
    new_ids = model.generate(input_ids=buf.tokens, ...)  # only the newly generated ids returned
    buf.append_assistant(new_ids)

    # 3. decode only to decide routing (do not feed back into tokens)
    text = tok.decode(new_ids)
    tool_call = parse_tool_call(text)  # your parser

    if tool_call:
        result = run_tool(tool_call)
        # 4. compute delta using the messages bookkeeping (for routing only)
        #    prefix_messages ends at the assistant tool_call turn
        delta = compute_delta(tok, current_messages_up_to_tool_call, [{"role": "tool", "content": result}])
        buf.append_tool(delta)
        # append to your messages list for future delta construction and final answer parsing
    else:
        done = True
        final = text

# 5. now buf.tokens and buf.loss_mask are ready for RL loss
#    loss only on assistant segments
```

The messages list is only for constructing the next `prefix_messages` for `compute_delta` and for your application logic. It is never re-tokenized into the training sequence.

## Renderer alternative

The article contrasts TITO with per-model "renderers" (e.g. PrimeIntellect renderers lib). Renderers are useful when you *don't* control the inference endpoint and only speak messages (not raw token ids). TITO is the simpler, correct choice when you *do* control tokens (vLLM/SGLang/HF with input_ids, etc.). This library implements the TITO path.

## Honest edges (from the article)

The article notes two places where reality pushes back even with correct TITO:

**History rewriting** (compaction, `clear_thinking`, sub-agent summaries, etc.): these replace past sampled tokens with something else. This breaks the "train on exact tokens the model produced" rule for the rewritten part. The workaround (supported here):

```python
buf.record_rewrite(at_token_index=last_rewrite_pos)
# zeros loss_mask before that point; only post-rewrite tail carries loss
# trainer will only optimize the genuine sampled tail
```

Call it at the last rewrite point in the rollout. Everything before becomes prompt (loss=0). Multiple calls are allowed; the last one determines the trainable tail. See `TokenBuffer.record_rewrite`, `get_rewrite_points`.

**Truncation** (hitting `max_seq_len` mid-turn): under TITO this is a non-event. Just stop. No need to synthesize close tokens or fall back to re-render (as renderers must). Dangling structures are fine in the buffer; the loss mask only cares about what was actually sampled.

```python
traj = collect_rollout(..., max_length=4096)
if traj["truncated"]:
    # buffer simply ends with what was generated
    ...
```

`collect_rollout` also accepts `max_length` and sets `truncated` + stops.

These features make the framework handle the cases called out in the source article without violating TITO where possible.

## Running tests

```bash
python -m pytest tests/ -q
```

## Roadmap (per build plan)

- Phase 1: TokenBuffer + compute_delta + prefix checks + tests. ✅
- Phase 2: Rollout collection + tool integration (collect_rollout, parser, HF adapter). ✅
- Phase 3: Loss masking utilities + basic RL loop (GRPOTrainer + logprobs support in buffer/rollout). ✅
- Phase 4: Working example + training script (train_simple_agent.py + evaluate). ✅
- Phase 5: Polish, more tests, docs, comparison vs MITO drift demo

## Contributing

Keep the TITO rule: the buffer is sacred. If a change would require re-tokenizing sampled tokens, it does not belong here.

## License

MIT
