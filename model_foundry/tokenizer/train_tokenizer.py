import argparse
import os
import yaml
import sentencepiece as spm
from pathlib import Path
import glob

def find_project_root(start_path: str) -> str:
    """Finds the project root by searching upwards for a .git directory."""
    path = Path(start_path).resolve()
    while path.parent != path:
        if (path / '.git').is_dir():
            return str(path)
        path = path.parent
    print("Warning: .git directory not found. Falling back to current working directory as project root.")
    return os.getcwd()

def train_tokenizer_from_config(config_path: str):
    """
    Trains a SentencePiece tokenizer using parameters from a .yaml experiment file.
    Resolves paths relative to the project's root directory.
    """
    print(f"--- Training Tokenizer from Config: {config_path} ---")

    base_dir = find_project_root(__file__)
    print(f"  - Project Root (Base Directory): {base_dir}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    training_corpus_path_from_config = config['data']['training_corpus']
    output_dir_from_config = config['tokenizer']['output_dir']

    training_corpus_path = training_corpus_path_from_config if os.path.isabs(training_corpus_path_from_config) else os.path.join(base_dir, training_corpus_path_from_config)
    output_dir = output_dir_from_config if os.path.isabs(output_dir_from_config) else os.path.join(base_dir, output_dir_from_config)

    vocab_size = config['tokenizer']['vocab_size']
    experiment_name = config.get('experiment_name', 'tokenizer')

    if not os.path.exists(training_corpus_path):
        print(f"FATAL ERROR: Path not found at '{training_corpus_path}'.")
        return

    # --- UPDATED: Search for both .train and .test files ---
    if os.path.isdir(training_corpus_path):
        print(f"  - Path '{training_corpus_path}' is a directory. Searching for .train and .test files...")
        train_files = glob.glob(os.path.join(training_corpus_path, '**', '*.train'), recursive=True)
        test_files = glob.glob(os.path.join(training_corpus_path, '**', '*.test'), recursive=True)
        input_files = train_files + test_files

        if not input_files:
            print(f"FATAL ERROR: No .train or .test files found in '{training_corpus_path}'.")
            return

        training_input_arg = ",".join(input_files)
        print(f"  - Found {len(input_files)} total files for training the tokenizer.")
    else:
        print(f"  - Path '{training_corpus_path}' is a single file.")
        training_input_arg = training_corpus_path
    # --- END UPDATED SECTION ---

    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, 'tokenizer')

    print(f"  - Output Dir:   {output_dir}")
    print(f"  - Vocab Size:   {vocab_size}")

    spm_args = {
        'input': training_input_arg,
        'model_prefix': model_prefix,
        'vocab_size': vocab_size,
        'model_type': 'unigram',
        'max_sentence_length': 8192,
        'character_coverage': 1.0,
        'hard_vocab_limit': 'false',
    }
    arg_string = ' '.join([f'--{key}={value}' for key, value in spm_args.items()])

    print(f"\n  - Starting SentencePiece training for experiment: '{experiment_name}'...")
    spm.SentencePieceTrainer.train(arg_string)

    print(f"  - Successfully trained and saved tokenizer to '{output_dir}'.\n")
    print("----- Tokenizer Training Complete -----")


def main():
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer for a specific experiment."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the experiment's .yaml configuration file."
    )
    args = parser.parse_args()

    # Resolve the config path relative to the project root as well
    project_root = find_project_root(__file__)
    absolute_config_path = args.config_path if os.path.isabs(args.config_path) else os.path.join(project_root, args.config_path)

    if not os.path.exists(absolute_config_path):
        print(f"FATAL ERROR: Configuration file not found at '{absolute_config_path}'")
        return

    train_tokenizer_from_config(absolute_config_path)


if __name__ == '__main__':
    main()