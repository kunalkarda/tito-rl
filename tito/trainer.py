from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.nn.utils import clip_grad_norm_
from .buffer import TokenBuffer
from .masking import masked_cross_entropy_loss, compute_kl_divergence, compute_token_level_advantages


class GRPOTrainer:
    def __init__(
        self,
        model: Any,
        ref_model: Optional[Any] = None,
        lr: float = 1e-6,
        clip_eps: float = 0.2,
        kl_coef: float = 0.0,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def _prepare_batch(self, trajectories: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(len(t["final_tokens"]) for t in trajectories)
        lp_len = max(0, max_len - 1)
        batch_ids = []
        batch_masks = []
        batch_old_lps = []
        batch_advs = []
        batch_ref_lps = []
        for t in trajectories:
            ids = t["final_tokens"]
            n = len(ids)
            mask = t.get("loss_mask", [0] * n)
            old_lps = t.get("logprobs") or t.get("assistant_logprobs") or [0.0] * n
            adv = t.get("advantages", [0.0] * n)
            old_lps = [(x if x is not None else 0.0) for x in old_lps]
            if len(old_lps) < n:
                old_lps = old_lps + [0.0] * (n - len(old_lps))
            if len(adv) < n:
                adv = adv + [0.0] * (n - len(adv))
            pad = max_len - n
            batch_ids.append(ids + [0] * pad)
            lp_pad = lp_len - max(0, n - 1)
            msk_lp = mask[1: n] if n > 1 else []
            old_lp = old_lps[1: n] if n > 1 else []
            adv_lp = adv[1: n] if n > 1 else []
            batch_masks.append(msk_lp + [0] * lp_pad)
            batch_old_lps.append(old_lp + [0.0] * lp_pad)
            batch_advs.append(adv_lp + [0.0] * lp_pad)
            if self.ref_model is not None:
                ref_lps = self._compute_logprobs(ids, mask)
                rpad = lp_len - len(ref_lps)
                batch_ref_lps.append(ref_lps + [0.0] * rpad)
            else:
                batch_ref_lps.append([0.0] * lp_len)
        ids_t = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        mask_t = torch.tensor(batch_masks, dtype=torch.float, device=self.device)
        old_t = torch.tensor(batch_old_lps, dtype=torch.float, device=self.device)
        adv_t = torch.tensor(batch_advs, dtype=torch.float, device=self.device)
        ref_t = torch.tensor(batch_ref_lps, dtype=torch.float, device=self.device)
        return ids_t, mask_t, old_t, adv_t, ref_t

    def _compute_logprobs(self, ids: List[int], mask: List[int]) -> List[float]:
        if not ids:
            return []
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(inp).logits
        logps = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = inp[:, 1:]
        gathered = torch.gather(logps, -1, targets.unsqueeze(-1)).squeeze(-1)
        m = torch.tensor([mask], dtype=torch.float, device=self.device)[:, 1:]
        gathered = gathered * m
        return gathered[0].tolist()

    def compute_advantages_from_rewards(self, rewards: List[float], group_size: int = 1) -> List[float]:
        adv_list = []
        for i in range(0, len(rewards), group_size):
            group = rewards[i:i + group_size]
            if not group:
                continue
            mean_r = sum(group) / len(group)
            var = sum((r - mean_r) ** 2 for r in group) / max(len(group) - 1, 1)
            std = (var + 1e-8) ** 0.5
            for r in group:
                adv_list.append((r - mean_r) / std)
        return adv_list

    def train_step(self, trajectories: List[Dict[str, Any]], rewards: Optional[List[float]] = None, group_size: int = 1) -> Dict[str, float]:
        if rewards is not None:
            adv_scalars = self.compute_advantages_from_rewards(rewards, group_size)
            for i, t in enumerate(trajectories):
                t = dict(t)
                scalar = adv_scalars[i] if i < len(adv_scalars) else 0.0
                t["advantages"] = [scalar] * len(t["final_tokens"])
                trajectories[i] = t
        ids, masks, old_lps, advs, ref_lps = self._prepare_batch(trajectories)
        self.model.train()
        outputs = self.model(ids)
        logits = outputs.logits
        logps = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = ids[:, 1:]
        new_lps = torch.gather(logps, -1, targets.unsqueeze(-1)).squeeze(-1)
        m = masks
        ratio = torch.exp(new_lps - old_lps)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advs
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * m).sum() / m.sum().clamp(min=1.0)
        total_loss = policy_loss
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.ref_model is not None and self.kl_coef > 0:
            with torch.no_grad():
                ref_logits = self.ref_model(ids).logits
            ref_logps = torch.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_g = torch.gather(ref_logps, -1, targets.unsqueeze(-1)).squeeze(-1)
            kl_loss = compute_kl_divergence(new_lps, ref_g, m)
            total_loss = total_loss + self.kl_coef * kl_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item() if self.kl_coef > 0 else 0.0,
            "active_tokens": int(m.sum().item()),
        }

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def ppo_loss(new_logps: torch.Tensor, old_logps: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor, clip_eps: float = 0.2, value_pred: Optional[torch.Tensor] = None, returns: Optional[torch.Tensor] = None, vf_coef: float = 0.1) -> torch.Tensor:
    ratio = torch.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    pi_loss = -torch.min(surr1, surr2)
    pi_loss = (pi_loss * mask).sum() / mask.sum().clamp(min=1)
    if value_pred is not None and returns is not None:
        v_loss = ((value_pred - returns) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        return pi_loss + vf_coef * v_loss
    return pi_loss
