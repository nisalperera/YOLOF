import os
import json
from pathlib import Path
from datetime import datetime

def get_latest_wandb_run(wandb_dir="./wandb"):
    """Returns dict: {'run_id': str, 'path': str, 'config': dict, 'summary': dict}"""
    wandb_path = Path(wandb_dir)
    if not wandb_path.exists():
        raise FileNotFoundError(f"wandb dir not found: {wandb_path}")
    
    # Find latest-run symlink or max timestamp run folder
    latest_sym = wandb_path / "latest-run"
    if latest_sym.exists():
        latest_folder = latest_sym.resolve().name
    else:
        runs = [p for p in wandb_path.glob("run-*") if p.is_dir()]
        latest_folder = max(runs, key=os.path.getmtime).name
    
    run_folder = wandb_path / latest_folder
    run_id = latest_folder.split("-")[-1]  # Last part after final -
    
    # Parse metadata
    files_path = run_folder / "files"
    config = json.load(open(files_path / "config.json")) if (files_path / "config.json").exists() else {}
    summary = json.load(open(files_path / "wandb-summary.json")) if (files_path / "wandb-summary.json").exists() else {}
    
    return {
        "run_id": run_id,
        "local_path": str(run_folder),
        "config": config,
        "summary": summary,
        "latest_folder": latest_folder
    }
