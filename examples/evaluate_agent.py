from transformers import AutoModelForCausalLM, AutoTokenizer
from tito import collect_rollout, make_hf_generate_fn

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_name)

    # Illustrative clean TITO trace (mechanics, no model weights needed)
    tool_str = "<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"a\": 2, \"b\": 3}}\n</tool_call>"
    tool_ids = tok.encode(tool_str, add_special_tokens=False)
    final_str = "The result is 5."
    final_ids = tok.encode(final_str, add_special_tokens=False)

    state = {"turn": 0}
    def forced_gen(current):
        state["turn"] += 1
        if state["turn"] == 1:
            return tool_ids, [-0.1] * len(tool_ids)
        return final_ids, [-0.2] * len(final_ids)

    def calc(a=0, b=0):
        return a + b

    tools = {"calc": calc}

    print("=== Illustrative TITO Tool Inference Trace (mechanics) ===")
    initial = [{"role": "user", "content": "Use the calc tool to add 2 and 3."}]
    traj = collect_rollout(tok, initial, forced_gen, tools=tools, max_turns=2, stop_on_no_tool=True)
    print("Total tokens in buffer:", len(traj["final_tokens"]))
    print("Assistant loss tokens:", traj["loss_mask"].count(1))
    for m in traj["messages"]:
        role = m.get("role", "?")
        if role == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]["function"]
            print(f"Assistant: [tool call] {tc['name']}({tc.get('arguments', {})})")
        elif role == "tool":
            print(f"Tool ({m.get('name')}): {m.get('content')}")
        else:
            print(f"{role.capitalize()}: {m.get('content', '')}")
    print("Tool was used:", any(m.get("role") == "tool" for m in traj["messages"]))
    print("=== End illustrative ===\n")

    # Now real model usage
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gen = make_hf_generate_fn(
        model, tok,
        return_logprobs=False,
        max_new_tokens=16,
        do_sample=True,
        temperature=0.2,
    )

    prompt = (
        "You are a helpful assistant that uses tools. "
        "To use a tool, output exactly in this format:\n"
        "<tool_call>\n"
        "{\"name\": \"calc\", \"arguments\": {\"a\": 2, \"b\": 3}}\n"
        "</tool_call>\n"
        "Then the system will give you the tool result and you answer.\n\n"
        "Question: Use the calc tool to add 2 and 3."
    )

    initial = [{"role": "user", "content": prompt}]

    traj = collect_rollout(
        tok, initial, gen,
        tools=tools,
        max_turns=3,
        stop_on_no_tool=False,
    )

    print("=== Real Model TITO Inference Result ===")
    print("Model:", model_name)
    print("Total tokens:", len(traj["final_tokens"]))
    print("Assistant loss tokens:", traj["loss_mask"].count(1))
    print("Truncated:", traj.get("truncated", False))
    print()
    print("Conversation trace:")
    for m in traj["messages"]:
        role = m.get("role", "?")
        if role == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]["function"]
            print(f"Assistant: [tool call] {tc['name']}({tc.get('arguments', {})})")
        elif role == "tool":
            print(f"Tool ({m.get('name', '')}): {m.get('content')}")
        else:
            content = m.get("content", "")[:100]
            print(f"{role.capitalize()}: {content}")
    print()
    print("Tool was used:", any(m.get("role") == "tool" for m in traj["messages"]))
    print("=== End real ===")

if __name__ == "__main__":
    main()
