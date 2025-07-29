import argparse
import os
import yaml
import sentencepiece as spm
from datasets import load_dataset, disable_progress_bar
from pathlib import Path
import glob

# Correctly disable the progress bars from the datasets library
disable_progress_bar()


def find_project_root(start_path: str) -> str:
    """Finds the project root by searching upwards for a .git directory."""
    path = Path(start_path).resolve()
    while path.parent != path:
        if (path / '.git').is_dir():
            return str(path)
        path = path.parent
    print("Warning: .git directory not found. Falling back to current working directory as project root.")
    return os.getcwd()


def tokenize_dataset_from_config(config_path: str):
    """
    Loads a text corpus, tokenizes it using the experiment-specific SentencePiece
    model, and saves the tokenized dataset to disk.
    """
    print(f"--- Tokenizing Dataset from Config: {config_path} ---")

    base_dir = find_project_root(__file__)
    print(f"  - Project Root (Base Directory): {base_dir}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    experiment_name = config.get('experiment_name', 'default_experiment')
    corpus_path_from_config = config['data']['training_corpus']
    tokenizer_dir_from_config = config['tokenizer']['output_dir']

    corpus_path = corpus_path_from_config if os.path.isabs(corpus_path_from_config) else os.path.join(base_dir,
                                                                                                      corpus_path_from_config)
    tokenizer_dir = tokenizer_dir_from_config if os.path.isabs(tokenizer_dir_from_config) else os.path.join(base_dir,
                                                                                                            tokenizer_dir_from_config)
    tokenized_data_dir = os.path.join(base_dir, "data", "tokenized", experiment_name)
    tokenizer_model_path = os.path.join(tokenizer_dir, 'tokenizer.model')

    if not os.path.exists(corpus_path):
        print(f"FATAL ERROR: Training corpus path not found at '{corpus_path}'.")
        return
    if not os.path.exists(tokenizer_model_path):
        print(f"FATAL ERROR: Tokenizer model not found at '{tokenizer_model_path}'.")
        return

    print(f"  - Searching for .train files in '{corpus_path}'...")
    search_pattern = os.path.join(corpus_path, '**', '*.train')
    training_files = glob.glob(search_pattern, recursive=True)

    if not training_files:
        print(f"FATAL ERROR: No .train files found in '{corpus_path}'.")
        return

    print(f"  - Experiment:          {experiment_name}")
    print(f"  - Tokenizer Model:     {tokenizer_model_path}")
    print(f"  - Output Directory:    {tokenized_data_dir}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_model_path)
    print(f"  - Successfully loaded tokenizer with vocab size: {tokenizer.vocab_size()}")

    print(f"\n  - Loading {len(training_files)} training file(s)...")
    raw_dataset = load_dataset('text', data_files={'train': training_files}, split='train')
    print(f"  - Found {len(raw_dataset):,} total lines in the training corpus.")

    def tokenize_function(examples):
        return {'input_ids': tokenizer.encode(examples['text'], out_type=int)}

    print("  - Tokenizing dataset (this may take a while)...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=['text']
    )
    print("  - Tokenization complete.")

    print(f"  - Saving tokenized dataset to '{tokenized_data_dir}'...")
    os.makedirs(tokenized_data_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(tokenized_data_dir)
    print("\n----- Dataset Tokenization Complete -----")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize a dataset for a specific experiment using its trained tokenizer."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the experiment's .yaml configuration file."
    )
    args = parser.parse_args()

    project_root = find_project_root(__file__)
    absolute_config_path = args.config_path if os.path.isabs(args.config_path) else os.path.join(project_root,
                                                                                                 args.config_path)

    if not os.path.exists(absolute_config_path):
        print(f"FATAL ERROR: Configuration file not found at '{absolute_config_path}'")
        return

    tokenize_dataset_from_config(absolute_config_path)


if __name__ == '__main__':
    main()