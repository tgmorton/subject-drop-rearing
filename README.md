# Just drop the subject: A controlled rearing study of Subject Drop in English

This repository contains the complete codebase for a series of controlled-rearing studies on Large Language Models (LLMs). The project investigates how different forms of linguistic evidence affect a model's acquisition of grammatical rules, specifically focusing on subject-drop phenomena in English.

The core of this project is a modular, configuration-driven framework that allows for fully reproducible experimental pipelines, from data ablation and tokenizer training to model training and evaluation.

## Core Architecture

This framework is built on a simple principle: **every experiment is defined by a single, declarative `.yaml` configuration file**. The codebase reads this file and executes the specified steps, ensuring that each experimental run is transparent, version-controllable, and precisely replicable.

-----

## Installation

To set up the environment, first install the required Python packages. It is highly recommended to use a virtual environment.

**1. Install PyTorch:**
PyTorch must be installed separately to ensure compatibility with your specific CUDA toolkit. Please follow the official instructions at [pytorch.org](https://pytorch.org). For example:

```sh
# Example for CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. Install Package Requirements:**
Install all other required packages from the `requirements.txt` file.

```sh
pip install -r requirements.txt
```

The requirements are:

```
# Core ML & Training
transformers==4.41.2
datasets==2.19.2
sentencepiece==0.2.0
protobuf==4.25.3
# For experiment tracking
wandb==0.17.0

# Linguistic Preprocessing (Ablations)
spacy==3.7.5

# Data Validation & CLI
pydantic==2.7.4
typer[all]==0.12.3
pyyaml==6.0.1

# Utilities
tqdm==4.66.4
numpy==1.26.4
nltk==3.8.1
```

**3. Download Linguistic Models:**
The data ablation scripts depend on `spaCy` models. Download the necessary English model after installation. [cite\_start]A transformer-based model is recommended for higher accuracy on the coreference tasks mentioned in the proposal[cite: 97].

```sh
python -m spacy download en_core_web_trf
```

-----

## Filesystem Overview

The project is organized into logical components to separate configuration, source code, and experimental artifacts.

```
.
├── configs/
│   └── experiment_0_baseline.yaml
├── data/
│   ├── raw/
│   └── processed/
├── evaluation/
│   ├── surprisal.py
│   └── run_blimp.py
├── model_foundry/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── trainer.py
│   └── tokenizer/
│       └── train_tokenizer.py
├── models/
├── preprocessing/
│   ├── remove_expletives.py
│   └── ... (other ablation scripts)
├── tokenizers/
├── .gitignore
├── pyproject.toml
└── README.md
```

  * **`configs/`**: Contains `.yaml` files, with each file defining a complete, self-contained experiment.
  * [cite\_start]**`data/`**: Holds the `raw` source corpora (e.g., BabyLM [cite: 10]), the `processed` (ablated) datasets, and fixed evaluation stimuli.
  * **`preprocessing/`**: Contains a script for each type of linguistic manipulation (e.g., `remove_expletives.py`). These are called by the `preprocess` CLI command.
  * **`model_foundry/`**: The core Python package for the project. [cite\_start]It contains the model architecture [cite: 43][cite\_start], trainer loop[cite: 39], data loaders, and the main CLI.
  * [cite\_start]**`tokenizers/`**: Stores the trained `SentencePiece` tokenizer models[cite: 40], with one subdirectory per experiment.
  * **`models/`**: Stores the output model checkpoints from training runs.
  * [cite\_start]**`evaluation/`**: Holds scripts for post-training analysis, such as calculating surprisal [cite: 59, 62] [cite\_start]and running the BLIMP benchmark[cite: 69].

-----

## Experimental Workflow

The codebase is designed to be run in modular stages, which is ideal for managing compute resources. Each command is driven by the same configuration file.

#### Step 1: Define an Experiment

Create a new `.yaml` file in the `configs/` directory. Copy the `TEMPLATE.yaml` (see below) and modify the parameters to design your experiment.

#### Step 2: Preprocess the Dataset

This step runs the linguistic ablation pipeline defined in your config file.

  * **Command:**
    ```sh
    python -m model_foundry.cli preprocess configs/your_experiment.yaml
    ```
  * **Action:** Reads the `dataset_manipulation` section of the config and executes the corresponding scripts from `preprocessing/` to create the final training corpus.

#### Step 3: Train the Tokenizer

This step trains a new `SentencePiece` tokenizer on the (potentially ablated) corpus.

  * **Command:**
    ```sh
    python -m model_foundry.cli train-tokenizer configs/your_experiment.yaml
    ```
  * **Action:** Reads the `tokenizer` and `data` sections and saves a new tokenizer model to the `tokenizers/` directory.

#### Step 4: Tokenize the Dataset

This step tokenizes the training corpus using the trained tokenizer.

  * **Command:**
    ```sh
    python -m model_foundry.cli tokenize-dataset configs/your_experiment.yaml
    ```
  * **Action:** Loads the training corpus, tokenizes it using the SentencePiece model, and saves the tokenized dataset to disk.

#### Step 5: Preprocess Data (NEW)

This step converts the tokenized dataset into fixed-length chunks for efficient training.

  * **Command:**
    ```sh
    python -m model_foundry.cli preprocess-data configs/your_experiment.yaml
    ```
  * **Action:** Loads the tokenized dataset, creates fixed-length chunks, and saves the processed dataset to disk.

#### Step 6: Train the Model

This step runs the main training loop.

  * **Command:**
    ```sh
    python -m model_foundry.cli run configs/your_experiment.yaml
    ```
  * **Action:** Reads all sections of the config, loads the processed data and tokenizer, builds the model, and begins training, saving checkpoints to the `models/` directory.

#### Step 5: Evaluate the Model

These scripts are run independently on saved model checkpoints.

  * **Command:**
    ```sh
    # Example for surprisal evaluation
    python -m evaluation.surprisal --checkpoint_path models/your_experiment/checkpoint-1000/ --stimuli_file data/evaluation_stimuli/minimal_pairs.csv
    ```
  * [cite\_start]**Action:** Loads a specific model checkpoint and runs the specified evaluation (e.g., surprisal analysis [cite: 59] [cite\_start]or BLIMP [cite: 69]).