import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tito import collect_rollout, make_hf_generate_fn, GRPOTrainer

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_name)

    def calc(a=0, b=0):
        return a + b

    tools = {"calc": calc}

    # Clean illustrative TITO demo trace (always nice, shows full mechanics)
    tool_str = "<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"a\": 2, \"b\": 3}}\n</tool_call>"
    t_ids = tok.encode(tool_str, add_special_tokens=False)
    t_lps = [-0.1] * len(t_ids)
    final_str = "The result is 5."
    f_ids = tok.encode(final_str, add_special_tokens=False)
    f_lps = [-0.2] * len(f_ids)
    dst = {"t": 0}
    def tgen(c):
        dst["t"] += 1
        if dst["t"] == 1:
            return t_ids, t_lps
        return f_ids, f_lps

    t_traj = collect_rollout(tok, [{"role": "user", "content": "Use the calc tool to add 2 and 3."}], tgen, tools=tools, max_turns=2, stop_on_no_tool=True)
    print("=== Illustrative TITO Tool Inference Demo (mechanics) ===")
    for m in t_traj["messages"]:
        role = m.get("role", "?")
        if role == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]["function"]
            print(f"Assistant: [tool call] {tc['name']}({tc.get('arguments', {})})")
        elif role == "tool":
            print(f"Tool ({m.get('name')}): {m.get('content')}")
        else:
            content = m.get("content", "")
            print(f"{role.capitalize()}: {content}")
    print("Tool used:", any(m.get("role") == "tool" for m in t_traj["messages"]))
    print("=== End illustrative ===\n")

    # Real model for training rollouts
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gen = make_hf_generate_fn(
        model, tok,
        return_logprobs=True,
        max_new_tokens=12,
        do_sample=True,
        temperature=0.7,
    )

    trainer = GRPOTrainer(
        model, lr=5e-6, clip_eps=0.2, kl_coef=0.0, device="cpu", max_grad_norm=1.0
    )

    base_prompts = [
        "You use tools. Use the calc tool to add 2 and 3. Output in the <tool_call> format.",
        "Use calc to compute 1 plus 4.",
        "Call the calc tool for 5 plus 2.",
        "Add 8 and 1 using the tool.",
    ]

    for step in range(3):
        trajectories = []
        rewards = []
        for p in base_prompts:
            initial = [{"role": "user", "content": p}]
            traj = collect_rollout(tok, initial, gen, tools=tools, max_turns=2, stop_on_no_tool=True)
            trajectories.append(traj)
            used_tool = any(m.get("role") == "tool" for m in traj["messages"])
            r = 1.0 if used_tool else 0.0
            rewards.append(r)

        stats = trainer.train_step(trajectories, rewards=rewards, group_size=4)
        avg_r = sum(rewards) / len(rewards)
        print(f"step {step} avg_reward {avg_r:.2f} loss {stats['loss']:.4f} active {stats['active_tokens']}")

    torch.save(model.state_dict(), "/tmp/tito_real_demo.pt")
    print("saved checkpoint to /tmp/tito_real_demo.pt")
    print("training complete (real Qwen2.5-0.5B usage)")

if __name__ == "__main__":
    main()
