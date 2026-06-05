from transformers import AutoTokenizer
from tito import collect_rollout

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def make_tool_ids():
    s = '<tool_call>\n{"name": "calc", "arguments": {"expr": "2+2"}}\n</tool_call>'
    return tok.encode(s, add_special_tokens=False)

tool_ids = make_tool_ids()
final_ids = tok.encode("The answer is 4.", add_special_tokens=False)[:4]

state = {"step": 0}
def generate(current):
    state["step"] += 1
    if state["step"] == 1:
        return tool_ids
    return final_ids

def calc(expr=""):
    return str(eval(expr))

initial = [{"role": "user", "content": "use the calc tool for 2+2"}]
traj = collect_rollout(tok, initial, generate, tools={"calc": calc}, max_turns=4)

buf = traj["buffer"]
print("turns:", traj["turns"])
print("len:", len(buf))
print("loss_positions:", [i for i, m in enumerate(buf.get_loss_mask()) if m])
print("assistant_ranges:", buf.get_assistant_ranges())
print("demo complete")
