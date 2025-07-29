from pydantic import BaseModel, Field
from typing import List, Optional

# Nested Pydantic models for better organization

class DataConfig(BaseModel):
    source_corpus: str
    training_corpus: str
    batch_size: int = Field(..., gt=0)
    max_sequence_length: int = Field(..., gt=0)

class TokenizerConfig(BaseModel):
    output_dir: str
    vocab_size: int = Field(..., gt=0)

class ModelConfig(BaseModel):
    layers: int = Field(..., gt=0)
    embedding_size: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    intermediate_hidden_size: int = Field(..., gt=0)
    attention_heads: int = Field(..., gt=0)
    activation_function: str
    dropout: float = Field(..., ge=0.0, lt=1.0)
    attention_dropout: float = Field(..., ge=0.0, lt=1.0)

class TrainingConfig(BaseModel):
    output_dir: str
    learning_rate: float = Field(..., gt=0)
    adam_beta1: float = Field(..., ge=0.0, lt=1.0)
    adam_beta2: float = Field(..., ge=0.0, lt=1.0)
    adam_epsilon: float = Field(..., gt=0)
    warmup_steps: int = Field(..., ge=0)
    train_steps: int = Field(..., gt=0)
    epochs: int = Field(..., gt=0)
    checkpointing_strategy: Optional[str] = None
    checkpoint_schedule: Optional[List[int]] = []

class LoggingConfig(BaseModel):
    use_wandb: bool = False
    wandb_project: Optional[str] = None

# The main configuration model that brings everything together

class ExperimentConfig(BaseModel):
    experiment_name: str
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    random_seed: int