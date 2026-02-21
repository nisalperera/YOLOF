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

from .wandb import get_latest_wandb_run
from functools import cached_property
from detectron2.utils.events import EventWriter, get_event_storage


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

    def write(self):

        if self._writer is None:
            return  # No writer available, skip logging

        storage = get_event_storage()
        iter = storage.iter
        new_last_write = self._last_write
        
        # Log all smoothed scalars (exact TensorBoardXWriter logic)
        for k, (v, _) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                self._writer.log({k: v}, step=iter)
                new_last_write = max(new_last_write, iter)
        
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
