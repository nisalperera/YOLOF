"""
Custom Detectron2 EventWriter: Logs all metrics + images to W&B.
Mimics TensorboardXWriter structure exactly.
"""

import os

try:
    import wandb
    wndb_init = True
except ImportError:
    wndb_init = False

from functools import cached_property
from detectron2.utils.events import EventWriter, get_event_storage

from yolof.config import get_cfg
from .wandb import get_latest_wandb_run


class WANDBWriter(EventWriter):
    """
    Write all scalars and images to Weights & Biases.
    """

    def __init__(self, log_dir: str = None, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): ignored (W&B uses cloud); for compatibility
            window_size (int): the scalars will be median-smoothed by this window size before logging to W&B
            kwargs: ignored (passed to wandb.init if needed)
        """
        self._window_size = window_size
        self._writer_args = {"dir": log_dir, **kwargs}
        self._last_write = -1
        self._cfg = get_cfg()

    @cached_property
    def _writer(self):
        """Lazily initialize W&B run on first write."""
        if wndb_init:
            if wandb.run is None:
                return self._auto_init_wandb()
            return wandb.run  # Use active run as "writer"
        else:
            return None

    def _auto_init_wandb(self):
        """Auto-init/resume from env vars if WANDB_RESUME="must"."""
        resume = os.getenv("WANDB_RESUME", "never")
        if resume == "must":
            # project = os.getenv("WANDB_PROJECT")
            # run_id = os.getenv("WANDB_RUN_ID")
            latest = get_latest_wandb_run()
            if not latest["run_id"]:
                raise ValueError("WANDB_RUN_ID required when WANDB_RESUME='must'")
            return wandb.init(
                project=self._writer_args.get("project", "default_project"),
                id=latest["run_id"],
                resume="must",
                **self._writer_args
            )
        else:
            # Fallback init if no env vars
            return wandb.init(**self._writer_args)

    def _is_eval_iter(self, iter: int) -> bool:
        """Check if current iteration is evaluation (matches EvalHook)."""
        eval_period = self._cfg.TEST.EVAL_PERIOD
        return eval_period > 0 and ((iter + 1) % eval_period == 0 or iter == self._cfg.SOLVER.MAX_ITER - 1)

    def write(self):

        if self._writer is None:
            return  # No writer available, skip logging

        storage = get_event_storage()
        iter = storage.iter
        new_last_write = self._last_write

        latest = storage.latest() if self._window_size <= 0 else storage.latest_with_smoothing_hint(self._window_size)

        # Separate training vs eval metrics
        training_scalars = {}
        eval_scalars = {}

        # Log all smoothed scalars (exact TensorBoardXWriter logic)
        for k, (v, _) in latest.items():
            if iter > self._last_write:

                if self._is_eval_iter(iter) and ("val_loss_" in k or "bbox" in k):
                    eval_scalars[k] = v
                else:
                    training_scalars[k] = v

        if training_scalars:
            self._writer.log(training_scalars, step=iter)
            new_last_write = max(new_last_write, iter)

        if eval_scalars:
            self._writer.log(eval_scalars, step=iter)

        self._last_write = new_last_write

        # Log images from storage._vis_data (exact logic)
        if len(storage._vis_data) >= 1:
            imgs = []
            for img_name, img, _ in storage._vis_data:
                imgs.append(wandb.Image(img, caption=img_name))
            if imgs:
                self._writer.log({"predictions": imgs}, step=iter)

        # Log histograms if any
        if len(storage._histograms) >= 1:
            for params in storage._histograms:
                self._writer.log({"histogram": wandb.Histogram(params["data"])}, step=iter)

    def close(self):
        if self._writer is None:
            return  # No writer available, skip closing
        if wandb.run is not None:
            wandb.run.finish()
