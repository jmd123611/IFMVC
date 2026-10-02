import torch
import torch.nn.functional as F


EPS = 1e-8


def stage1_kl_loss(q, target, eps=EPS):
    """KL(one_hot(target) || q) for an already-normalized probability q."""
    one_hot_target = F.one_hot(target, num_classes=q.size(1)).to(dtype=q.dtype)
    log_q = torch.log(q.clamp_min(eps))
    return F.kl_div(log_q, one_hot_target, reduction="batchmean")


def stage2_cross_entropy_loss(q, target, eps=EPS):
    """One-hot cross entropy for an already-normalized probability q."""
    log_q = torch.log(q.clamp_min(eps))
    return F.nll_loss(log_q, target)
