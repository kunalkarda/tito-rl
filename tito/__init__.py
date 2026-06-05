from .buffer import TokenBuffer
from .delta import compute_delta
from .chat_template import is_prefix_preserving_for_tools, check_prefix_preservation
from .rollout import collect_rollout, make_hf_generate_fn, parse_assistant_response
from .masking import (
    get_loss_mask_from_ranges,
    get_active_token_count,
    pad_mask,
    compute_token_level_advantages,
    shift_labels_and_mask,
    masked_cross_entropy_loss,
    compute_kl_divergence,
)
from .trainer import GRPOTrainer, ppo_loss

__version__ = "0.1.0"
__all__ = [
    "TokenBuffer",
    "compute_delta",
    "is_prefix_preserving_for_tools",
    "check_prefix_preservation",
    "collect_rollout",
    "make_hf_generate_fn",
    "parse_assistant_response",
    "get_loss_mask_from_ranges",
    "get_active_token_count",
    "pad_mask",
    "compute_token_level_advantages",
    "shift_labels_and_mask",
    "masked_cross_entropy_loss",
    "compute_kl_divergence",
    "GRPOTrainer",
    "ppo_loss",
]
