"""
Model Foundry - Controlled Rearing Study Framework

This package provides a complete experimental framework for investigating
grammatical rule acquisition in language models through controlled dataset
manipulations and systematic evaluation.
"""

__version__ = "0.1.0"
__author__ = "Model Foundry Team"

from .config import ExperimentConfig
from .model import create_model
from .trainer import Trainer
from .data import create_data_processor
from .utils import find_project_root, set_seed, get_device, get_git_commit_hash

__all__ = [
    "ExperimentConfig",
    "create_model", 
    "Trainer",
    "create_data_processor",
    "find_project_root",
    "set_seed",
    "get_device",
    "get_git_commit_hash"
] 