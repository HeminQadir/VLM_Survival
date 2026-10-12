"""Piecewise exponential survival likelihood: handles censoring inside intervals."""
import torch
from torch.nn import functional as F


def integrated_hazards(logits):
    """Positive cumulative hazard increment for each complete interval."""
    return F.softplus(logits.float()).clamp_min(1e-8)


def survival_curves(logits):
    """Survival at each right interval boundary. S(0)=1 is omitted."""
    return torch.exp(-integrated_hazards(logits).cumsum(dim=-1))


def survival_nll(logits, times, events, edges, reduction='mean'):
    """Continuous event density in days; administrative censoring at final edge.

    Each logit parameterizes an integrated hazard H_k. The constant daily hazard
    in interval k is H_k / width_k. No fabricated labels beyond censor time.
    """
    edges = torch.as_tensor(edges, device=logits.device, dtype=torch.float32)
    times, events = times.float(), events.float()
    if edges.numel() != logits.shape[-1] + 1 or edges[0] != 0 or not torch.all(edges.diff() > 0):
        raise ValueError('Edges must start at zero, increase, and match logits.')
    if not torch.all(torch.isfinite(times) & (times > 0)):
        raise ValueError('Follow-up times must be finite and positive.')
    if not torch.all((events == 0) | (events == 1)):
        raise ValueError('Event indicators must be 0 or 1.')
    widths = edges.diff()
    exposure = ((times[:, None] - edges[:-1]) / widths).clamp(0, 1)
    increments = integrated_hazards(logits)
    cumulative = (increments * exposure).sum(dim=-1)
    interval = torch.bucketize(times.contiguous(), edges[1:], right=False).clamp_max(len(widths)-1)
    daily_rate = increments.gather(1, interval[:, None]).squeeze(1) / widths[interval]
    observed_event = events * (times <= edges[-1])
    losses = cumulative - observed_event * daily_rate.log()
    return losses.mean() if reduction == 'mean' else losses
