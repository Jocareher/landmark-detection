"""Optional read-only gradient probes for selecting fixed training loss weights."""
import csv
from pathlib import Path

import torch


def gradient_norm(loss, parameters):
    """Unscaled L2 gradient norm on the same trainable parameter set for all terms."""
    if not loss.requires_grad or not parameters:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squared = [g.detach().float().square().sum() for g in gradients if g is not None]
    return float(torch.stack(squared).sum().sqrt()) if squared else 0.0


def training_loss_probe(losses, model, pca_weight, mahalanobis_weight,
                        reconstruction_weight=None, coefficient_weight=None):
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    from .losses import resolve_pca_training_weights
    wr, wh = resolve_pca_training_weights(pca_weight, mahalanobis_weight,
                                        reconstruction_weight, coefficient_weight)
    other = losses['total_loss'] - losses['weighted_pca_loss']
    gr = gradient_norm(losses['pca_reconstruction_loss'], parameters)
    gh = gradient_norm(losses['pca_mahalanobis_penalty'], parameters)
    gp = gradient_norm(losses['pca_loss'], parameters)
    gw = gradient_norm(losses['weighted_pca_loss'], parameters)
    go = gradient_norm(other, parameters)
    return dict(raw_reconstruction_grad_norm=gr, raw_mahalanobis_grad_norm=gh,
                raw_pca_grad_norm=gp, other_losses_grad_norm=go,
                weighted_reconstruction_grad_norm=wr*gr,
                weighted_mahalanobis_grad_norm=wh*gh,
                weighted_pca_grad_norm=gw)


def append_probe(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
