import argparse
import os
import glob
from tqdm import tqdm
import math


def count_words_in_file(file_path):
    """Counts the total words in a single text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(len(line.split()) for line in f)


def main():
    parser = argparse.ArgumentParser(
        description="Split multiple .train files proportionally into main and pool sets, preserving directory structure."
    )
    parser.add_argument("--source_dir", type=str, required=True, help="Root directory containing source .train files.")
    parser.add_argument("--main_output_dir", type=str, required=True,
                        help="Output directory for the main training sets.")
    parser.add_argument("--pool_output_dir", type=str, required=True,
                        help="Output directory for the replacement pool sets.")
    parser.add_argument("--pool_words_total", type=int, default=10000000,
                        help="Total approximate number of words for the entire replacement pool.")
    args = parser.parse_args()

    # --- Step 1: Scan and get stats for all source files ---
    search_pattern = os.path.join(args.source_dir, '**', '*.train')
    source_files = glob.glob(search_pattern, recursive=True)

    if not source_files:
        raise FileNotFoundError(f"No '.train' files found in {args.source_dir}")

    print(f"Found {len(source_files)} source files. Analyzing word counts...")
    file_stats = {path: count_words_in_file(path) for path in tqdm(source_files, desc="Analyzing files")}
    total_corpus_words = sum(file_stats.values())

    if total_corpus_words == 0:
        raise ValueError("Source files are empty. Cannot proceed.")

    print(f"Total words in source corpus: {total_corpus_words:,}")

    # --- Step 2: Calculate the global proportion for the replacement pool ---
    pool_proportion = args.pool_words_total / total_corpus_words
    print(
        f"Targeting ~{args.pool_words_total:,} words for the pool, which is {pool_proportion:.2%} of the total corpus.")

    # --- Step 3: Process each file individually ---
    for file_path, total_words in tqdm(file_stats.items(), desc="Splitting files"):
        if total_words == 0:
            continue

        # --- Read the file content ---
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # --- Determine how many words to pull from this specific file ---
        words_to_pull_for_pool = math.ceil(total_words * pool_proportion)

        # --- Split the lines for this file ---
        pool_lines = []
        pool_word_count = 0
        for line in reversed(lines):
            if pool_word_count >= words_to_pull_for_pool:
                break
            pool_lines.append(line)
            pool_word_count += len(line.split())

        pool_lines.reverse()
        main_lines = lines[:-len(pool_lines)]

        # --- Create corresponding output directories ---
        relative_path = os.path.relpath(file_path, args.source_dir)

        main_dest_path = os.path.join(args.main_output_dir, relative_path)
        pool_dest_path = os.path.join(args.pool_output_dir, relative_path)

        os.makedirs(os.path.dirname(main_dest_path), exist_ok=True)
        os.makedirs(os.path.dirname(pool_dest_path), exist_ok=True)

        # --- Write the split files ---
        with open(main_dest_path, 'w', encoding='utf-8') as f:
            f.writelines(main_lines)

        with open(pool_dest_path, 'w', encoding='utf-8') as f:
            f.writelines(pool_lines)

    print("\nCorpus preparation complete.")
    print(f"Main training files written to: {args.main_output_dir}")
    print(f"Replacement pool files written to: {args.pool_output_dir}")


if __name__ == "__main__":
    main()