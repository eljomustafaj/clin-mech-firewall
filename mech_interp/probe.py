"""Fit and apply a linear probe in the original activation coordinates."""

import numpy as np


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


def fit_probe(activations, labels, l2=0.1):
    """Standardize on training rows only; fold the transform into the final weights."""
    import torch

    x = np.asarray(activations, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0 or y.shape != (len(x),):
        raise ValueError("Provide one label per activation vector")
    if set(y) != {0.0, 1.0} or not np.isfinite(x).all() or not np.isfinite(l2) or l2 <= 0:
        raise ValueError("Training requires finite activations, both binary labels, and positive L2")
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    features = torch.from_numpy((x - center) / scale)
    targets = torch.from_numpy(y)
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, intercept], line_search_fn="strong_wolfe", max_iter=300,
        tolerance_grad=1e-9, tolerance_change=1e-13,
    )

    def closure():
        optimizer.zero_grad()
        logits = features @ weight + intercept
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        objective = loss + l2 * weight.square().sum() / 2
        objective.backward()
        return objective

    optimizer.step(closure)
    objective = float(closure().detach())
    grad = max(float(weight.grad.abs().max()), float(intercept.grad.abs()))
    if not np.isfinite(objective) or grad > 1e-5:
        raise RuntimeError(f"Probe optimization did not converge (maximum gradient {grad})")
    raw_weight = weight.detach().numpy() / scale
    raw_bias = intercept.item() - np.dot(center, raw_weight)
    return {
        "weights": raw_weight.tolist(), "bias": float(raw_bias),
        "fit": {"l2": float(l2), "objective": objective, "max_abs_gradient": grad,
                "optimizer": "full-batch CPU float64 L-BFGS; standardized train features"},
    }


def score(activations, probe):
    x = np.asarray(activations, dtype=np.float64)
    w = np.asarray(probe["weights"], dtype=np.float64)
    bias = float(probe["bias"])
    if w.ndim != 1 or x.ndim not in (1, 2) or x.shape[-1] != len(w):
        raise ValueError("Activation width must match the probe weight vector")
    if not (np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(bias)):
        raise ValueError("Probe scoring requires finite inputs")
    return sigmoid(x @ w + bias)


def classification_metrics(labels, scores, threshold):
    """Positive is unsupported. Exact ties get half credit in AUROC."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or y.shape != s.shape or set(y) != {0, 1}:
        raise ValueError("Metrics require aligned scores and both labels")
    if not np.isfinite(s).all() or not ((s >= 0) & (s <= 1)).all():
        raise ValueError("Scores must be finite and in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    pred = s >= threshold
    tp, fp = int(sum(pred & (y == 1))), int(sum(pred & (y == 0)))
    fn, tn = int(sum(~pred & (y == 1))), int(sum(~pred & (y == 0)))
    positives, negatives = s[y == 1], np.sort(s[y == 0])
    lower = np.searchsorted(negatives, positives, side="left")
    upper = np.searchsorted(negatives, positives, side="right")
    auc = np.mean((lower + upper) / (2 * len(negatives)))
    return {
        "n": len(y), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "auroc": float(auc),
        "f1": 2 * tp / (2 * tp + fp + fn), "false_positive_rate": fp / (fp + tn),
        "recall": tp / (tp + fn), "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
    }


def choose_threshold(labels, scores):
    """Maximize validation F1; break ties toward fewer false positives, then larger threshold."""
    def key(threshold):
        metrics = classification_metrics(labels, scores, threshold)
        return metrics["f1"], -metrics["fp"], threshold

    return float(max(np.unique(np.append(scores, 1.0)), key=key))
