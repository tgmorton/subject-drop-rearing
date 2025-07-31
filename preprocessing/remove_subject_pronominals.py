"""
Remove Subject Pronominals Ablation Script

This script removes all pronouns identified as nominal subjects (nsubj) from the corpus.
This ablation tests how the model learns subject-drop patterns when explicit subject
pronouns are removed from the training data.

Procedure 6: RemoveSubjectPronominals(text)
1. Load spaCy NLP model with POS tagger and dependency parser
2. Initialize modified_parts an empty list
3. doc = process(text, NLP model)
4. for each token in doc do
5.   is_subj_pronoun = token.pos_ == 'PRON' and token.dep_ == 'nsubj'
6.   if not is_subj_pronoun:
7.     append token.text_with_ws to modified_parts
8.   end if
9. end for
10. result = join(modified_parts)
11. return result
"""

import spacy
import argparse
from pathlib import Path
import json
from tqdm import tqdm


def remove_subject_pronominals(text: str, nlp) -> str:
    """
    Remove all pronouns identified as nominal subjects (nsubj) from the text.
    
    Args:
        text: Input text to process
        nlp: spaCy NLP model with POS tagger and dependency parser
        
    Returns:
        Modified text with subject pronouns removed
    """
    doc = nlp(text)
    modified_parts = []
    
    for token in doc:
        # Check if token is a pronoun functioning as a nominal subject
        is_subj_pronoun = token.pos_ == 'PRON' and token.dep_ == 'nsubj'
        
        if not is_subj_pronoun:
            # Preserve original spacing
            modified_parts.append(token.text_with_ws)
    
    result = ''.join(modified_parts)
    return result


def process_file(input_path: Path, output_path: Path, nlp, stats: dict):
    """
    Process a single file and remove subject pronominals.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        nlp: spaCy NLP model
        stats: Dictionary to track statistics
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Process the text
        modified_text = remove_subject_pronominals(text, nlp)
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modified_text)
        
        # Update statistics
        stats['files_processed'] += 1
        stats['original_chars'] += len(text)
        stats['modified_chars'] += len(modified_text)
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        stats['errors'] += 1


def main():
    """Main function to run the subject pronominals removal ablation."""
    parser = argparse.ArgumentParser(
        description="Remove subject pronominals from corpus files"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing input corpus files"
    )
    parser.add_argument(
        "output_dir", 
        type=str,
        help="Directory to save modified corpus files"
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model to use for NLP processing"
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.txt",
        help="File pattern to match (default: *.txt)"
    )
    
    args = parser.parse_args()
    
    # Load spaCy model
    print(f"Loading spaCy model: {args.spacy_model}")
    try:
        nlp = spacy.load(args.spacy_model)
    except OSError:
        print(f"Model {args.spacy_model} not found. Installing...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", args.spacy_model])
        nlp = spacy.load(args.spacy_model)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    # Find all matching files
    input_files = list(input_dir.glob(args.file_pattern))
    if not input_files:
        print(f"No files found matching pattern '{args.file_pattern}' in {input_dir}")
        return 1
    
    print(f"Found {len(input_files)} files to process")
    
    # Initialize statistics
    stats = {
        'files_processed': 0,
        'original_chars': 0,
        'modified_chars': 0,
        'errors': 0
    }
    
    # Process each file
    for input_file in tqdm(input_files, desc="Processing files"):
        # Create corresponding output file path
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path
        
        process_file(input_file, output_file, nlp, stats)
    
    # Print statistics
    print("\n=== Processing Complete ===")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Errors: {stats['errors']}")
    print(f"Original characters: {stats['original_chars']:,}")
    print(f"Modified characters: {stats['modified_chars']:,}")
    
    if stats['original_chars'] > 0:
        reduction = (stats['original_chars'] - stats['modified_chars']) / stats['original_chars'] * 100
        print(f"Text reduction: {reduction:.2f}%")
    
    # Save statistics
    stats_file = output_dir / "removal_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_file}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 