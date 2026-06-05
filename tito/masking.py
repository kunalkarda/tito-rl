from typing import List, Optional, Tuple
import torch


def get_loss_mask_from_ranges(seq_len: int, ranges: List[Tuple[int, int]]) -> List[int]:
    mask = [0] * seq_len
    for start, end in ranges:
        for i in range(start, min(end, seq_len)):
            mask[i] = 1
    return mask


def get_active_token_count(mask: List[int]) -> int:
    return sum(1 for m in mask if m == 1)


def pad_mask(mask: List[int], target_len: int, pad_value: int = 0) -> List[int]:
    if len(mask) >= target_len:
        return mask[:target_len]
    return mask + [pad_value] * (target_len - len(mask))


def compute_token_level_advantages(rewards: List[float], masks: List[List[int]], group_size: int) -> List[List[float]]:
    advantages: List[List[float]] = []
    for i in range(0, len(rewards), group_size):
        group = rewards[i : i + group_size]
        if not group:
            continue
        mean_r = sum(group) / len(group)
        var = sum((r - mean_r) ** 2 for r in group) / max(len(group) - 1, 1)
        std = (var + 1e-8) ** 0.5
        for r in group:
            adv = (r - mean_r) / std
            advantages.append([adv] * len(masks[len(advantages)]))
    return advantages


def shift_labels_and_mask(labels: torch.Tensor, loss_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return labels[:, 1:], loss_mask[:, 1:]


def masked_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size), labels.view(-1), reduction="none"
    )
    loss = loss * loss_mask.view(-1)
    denom = loss_mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def compute_kl_divergence(logps: torch.Tensor, ref_logps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kl = (logps - ref_logps).exp() - (logps - ref_logps) - 1
    kl = kl * mask
    return kl.sum() / mask.sum().clamp(min=1.0)
