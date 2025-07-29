import os
import random
from pathlib import Path
import numpy as np
import torch

def find_project_root(start_path: str) -> str:
    """Finds the project root by searching upwards for a .git directory."""
    path = Path(start_path).resolve()
    while path.parent != path:
        if (path / '.git').is_dir():
            return str(path)
        path = path.parent
    # Fallback to current working directory if .git is not found
    print("Warning: .git directory not found. Using current working directory as project root.")
    return os.getcwd()

def get_git_commit_hash() -> str:
    """Gets the current git commit hash of the repository."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        print("Warning: Could not get git commit hash.")
        return "git_not_found"

def set_seed(seed_value: int):
    """Sets the random seeds for reproducibility across all relevant libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"  - Random seed set to: {seed_value}")

def get_device() -> torch.device:
    """Determines and returns the appropriate torch.device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  - Using CUDA GPU: {torch.cuda.get_device_name(device)}")
        return device
    else:
        print("  - CUDA not available. Using CPU.")
        return torch.device("cpu")