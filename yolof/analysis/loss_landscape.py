"""
2D Loss Landscape Analysis — filter-normalised random directions.

Based on:  Li et al., "Visualizing the Loss Landscape of Neural Nets", NeurIPS 2018.
           https://arxiv.org/abs/1712.09913

Usage (standalone)::

    from yolof.analysis.loss_landscape import LossLandscape
    ll = LossLandscape(model, dataloader, device)
    surface = ll.compute(grid_size=21, radius=1.0)   # dict with 'Z', 'X', 'Y'
    ll.save(surface, output_dir / "landscape.npz")
    ll.plot(surface, output_dir / "landscape.png")
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Filter normalisation helpers
# --------------------------------------------------------------------------- #

def _filter_norm_direction(reference: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Generate one random, filter-normalised perturbation direction.

    For each weight tensor W with shape (n_filters, ...), we sample a
    random direction d of the same shape and scale each filter-slice so
    its norm equals the norm of the corresponding filter in W.  This
    makes the landscape scale-invariant with respect to BN weight folding.
    """
    direction: Dict[str, torch.Tensor] = {}
    for name, param in reference.named_parameters():
        d = torch.randn_like(param.data)
        if param.data.dim() > 1:          # Conv / Linear — normalise per filter
            for i in range(d.shape[0]):
                w_norm = param.data[i].norm()
                d_norm = d[i].norm()
                if d_norm.item() > 1e-10:
                    d[i] = d[i] * (w_norm / d_norm)
        else:                              # 1-D (bias, BN) — scalar normalisation
            w_norm = param.data.norm()
            d_norm = d.norm()
            if d_norm.item() > 1e-10:
                d = d * (w_norm / d_norm)
        direction[name] = d
    return direction


def _apply_direction(
    base_state: Dict[str, torch.Tensor],
    dir1: Dict[str, torch.Tensor],
    dir2: Dict[str, torch.Tensor],
    delta1: float,
    delta2: float,
) -> Dict[str, torch.Tensor]:
    perturbed: Dict[str, torch.Tensor] = {}
    for k, v in base_state.items():
        if k in dir1 and torch.is_floating_point(v):
            perturbed[k] = v + delta1 * dir1[k] + delta2 * dir2[k]
        else:
            perturbed[k] = v
    return perturbed


# --------------------------------------------------------------------------- #
# Loss evaluation helper (re-uses mode_connectivity.evaluate_loss_on_dataset)
# --------------------------------------------------------------------------- #

def _eval_loss(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> float:
    from yolof.analysis.mode_connectivity import evaluate_loss_on_dataset
    result = evaluate_loss_on_dataset(
        model, dataloader, device,
        return_val_loss=True, max_samples=max_samples,
    )
    return result["loss_total"]


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #

class LossLandscape:
    """
    Compute and plot a 2-D filter-normalised loss landscape around a model.

    Args:
        model:             YOLOF model (already loaded with target checkpoint weights).
        dataloader:        Evaluation dataloader.
        device:            torch.device.
        max_eval_samples:  Cap samples per grid point to keep runtime manageable.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        max_eval_samples: int = 300,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.max_eval_samples = max_eval_samples

        # Freeze base weights so we can restore after each perturbation
        self._base_state: Dict[str, torch.Tensor] = {
            k: v.clone().cpu() for k, v in model.state_dict().items()
        }

    # ---------------------------------------------------------------------- #
    def compute(
        self,
        grid_size: int = 21,
        radius: float = 1.0,
        seed: int = 0,
    ) -> Dict:
        """
        Compute the (grid_size x grid_size) loss surface.

        Args:
            grid_size:  Number of grid points along each axis.
            radius:     Half-width of the perturbation range (in filter-norm units).
            seed:       RNG seed for reproducible directions.

        Returns:
            dict with keys: 'X', 'Y' (meshgrid coords), 'Z' (loss values),
            'grid_size', 'radius'.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        logger.info("[LossLandscape] Generating filter-normalised directions ...")
        dir1 = _filter_norm_direction(self.model)
        dir2 = _filter_norm_direction(self.model)

        coords = np.linspace(-radius, radius, grid_size)
        X, Y = np.meshgrid(coords, coords)
        Z = np.zeros_like(X)

        total = grid_size * grid_size
        logger.info(
            "[LossLandscape] Evaluating %d grid points (grid=%d, radius=%.2f) ...",
            total, grid_size, radius,
        )

        for i in range(grid_size):
            for j in range(grid_size):
                d1, d2 = float(X[i, j]), float(Y[i, j])

                perturbed = _apply_direction(
                    self._base_state, dir1, dir2, d1, d2
                )
                self.model.load_state_dict(perturbed, strict=False)
                self.model.to(self.device)

                loss = _eval_loss(
                    self.model, self.dataloader,
                    self.device, self.max_eval_samples,
                )
                Z[i, j] = loss

                done = i * grid_size + j + 1
                if done % max(1, total // 10) == 0:
                    logger.info(
                        "[LossLandscape] %d/%d -- loss=%.4f", done, total, loss
                    )

        # Restore original weights
        self.model.load_state_dict(self._base_state, strict=False)

        return {
            "X": X, "Y": Y, "Z": Z,
            "grid_size": grid_size, "radius": radius,
        }

    # ---------------------------------------------------------------------- #
    @staticmethod
    def save(surface: Dict, path: Path) -> None:
        """Save surface to a compressed .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            X=surface["X"], Y=surface["Y"], Z=surface["Z"],
            grid_size=surface["grid_size"], radius=surface["radius"],
        )
        logger.info("[LossLandscape] Surface saved -> %s", path)

    # ---------------------------------------------------------------------- #
    @staticmethod
    def plot(surface: Dict, path: Path, title: str = "Loss Landscape") -> None:
        """
        Plot 2-D surface (contour + 3-D surface) and save to PNG.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("[LossLandscape] matplotlib not available -- skipping plot.")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        X, Y, Z = surface["X"], surface["Y"], surface["Z"]

        fig = plt.figure(figsize=(14, 6))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Contour plot
        ax1 = fig.add_subplot(1, 2, 1)
        cs = ax1.contourf(X, Y, Z, levels=30, cmap="coolwarm")
        fig.colorbar(cs, ax=ax1, label="Loss")
        ax1.set_xlabel("Direction 1 (d1)")
        ax1.set_ylabel("Direction 2 (d2)")
        ax1.set_title("Contour")
        ax1.axhline(0, color="k", lw=0.5, ls="--")
        ax1.axvline(0, color="k", lw=0.5, ls="--")

        # 3-D surface
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0, antialiased=True, alpha=0.85)
        ax2.set_xlabel("d1")
        ax2.set_ylabel("d2")
        ax2.set_zlabel("Loss")
        ax2.set_title("Surface")

        plt.tight_layout()
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("[LossLandscape] Plot saved -> %s", path)
