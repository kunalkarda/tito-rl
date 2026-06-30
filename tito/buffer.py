from typing import List, Dict, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


class TokenBuffer:
    __slots__ = (
        "tokens",
        "loss_mask",
        "logprobs",
        "segments",
        "rewrite_points",
        "truncated",
        "_loss_mask_tensor",
        "_tensor_dirty",
    )

    def __init__(self, initial_tokens: Optional[List[int]] = None):
        self.tokens: List[int] = list(initial_tokens) if initial_tokens is not None else []
        self.loss_mask: List[int] = [0] * len(self.tokens)
        self.logprobs: List[Optional[float]] = [None] * len(self.tokens)
        self.segments: List[Dict] = []
        self.rewrite_points: List[int] = []
        self.truncated: bool = False
        self._loss_mask_tensor = None
        self._tensor_dirty = True
        if self.tokens:
            self.segments.append({"start": 0, "end": len(self.tokens), "type": "prompt", "loss": False})

    def _append(self, ids: List[int], seg_type: str, loss: bool, logprobs: Optional[List[float]] = None):
        if not ids:
            return
        start = len(self.tokens)
        self.tokens.extend(ids)
        self.loss_mask.extend([1 if loss else 0 for _ in ids])
        if loss and logprobs is not None and len(logprobs) == len(ids):
            self.logprobs.extend(logprobs)
        else:
            self.logprobs.extend([None] * len(ids))
        self.segments.append({"start": start, "end": len(self.tokens), "type": seg_type, "loss": loss})
        self._tensor_dirty = True

    def append_prompt(self, ids: List[int]):
        self._append(ids, "prompt", False)

    def append_assistant(self, ids: List[int], logprobs: Optional[List[float]] = None):
        self._append(ids, "assistant", True, logprobs)

    def append_tool(self, ids: List[int]):
        self._append(ids, "tool", False)

    def get_input_ids(self) -> List[int]:
        return self.tokens

    def get_loss_mask(self) -> List[int]:
        return self.loss_mask

    def get_loss_mask_tensor(self, device: Optional[str] = None):
        if torch is None:
            raise ImportError("torch is required for tensor conversion")
        if self._tensor_dirty or self._loss_mask_tensor is None:
            self._loss_mask_tensor = torch.tensor(self.loss_mask, dtype=torch.long, device=device)
            self._tensor_dirty = False
        elif device is not None:
            target_device = torch.device(device)
            if self._loss_mask_tensor.device != target_device:
                self._loss_mask_tensor = self._loss_mask_tensor.to(target_device)
        return self._loss_mask_tensor

    def get_assistant_ranges(self) -> List[Tuple[int, int]]:
        return [(s["start"], s["end"]) for s in self.segments if s.get("type") == "assistant" and s.get("loss")]

    def get_logprobs(self) -> List[Optional[float]]:
        return self.logprobs

    def get_assistant_logprobs(self) -> List[Optional[float]]:
        return [
            self.logprobs[i]
            for s in self.segments
            if s.get("type") == "assistant" and s.get("loss")
            for i in range(s["start"], s["end"])
        ]

    def record_rewrite(self, at_token_index: int):
        if at_token_index < 0 or at_token_index > len(self.tokens):
            raise ValueError("invalid rewrite point")
        self.rewrite_points.append(at_token_index)
        if at_token_index > 0:
            self.loss_mask[:at_token_index] = [0] * at_token_index
        self._tensor_dirty = True

    def get_rewrite_points(self) -> List[int]:
        return self.rewrite_points

    def mark_truncated(self):
        self.truncated = True

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]
