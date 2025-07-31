import argparse
import spacy
import random
import os
import glob
from tqdm import tqdm
import math


def get_spacy_device():
    """
    Detects and returns the best available spaCy device.
    Checks for Apple Silicon (MPS), then CUDA, and defaults to CPU.
    """
    # Debug: Check environment variables
    import os
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Check for Apple Silicon (MPS)
        if torch.backends.mps.is_available():
            print("Apple Silicon (MPS) device detected. Using GPU.")
            return "mps"
        
        # Check for CUDA
        if torch.cuda.is_available():
            print(f"NVIDIA CUDA device detected. Using GPU.")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            return "cuda"
        else:
            print("PyTorch CUDA not available")
            
    except ImportError as e:
        print(f"PyTorch import error: {e}")
    except Exception as e:
        print(f"GPU detection error: {e}")

    print("Warning: No compatible GPU detected. spaCy will run on CPU, which may be slow.")
    return "cpu"


def count_tokens(text):
    """Count tokens using a more sophisticated approach that better matches training tokenization."""
    import re
    # Use regex to split on whitespace, handling punctuation better
    tokens = re.findall(r'\S+', text)
    return len(tokens)


def find_and_confirm_expletives(doc, nlp_coref):
    """
    Implements Procedure 1 & 2 on a single spaCy Doc object.
    Finds potential dummy pronouns and uses a coreference model on a localized context.
    """
    indices_to_remove = set()
    potential_dummies = [
        tok for tok in doc if tok.dep_ == 'expl' and tok.head.pos_ == 'VERB'
    ]

    for token in potential_dummies:
        current_sent = token.sent
        prev_sent = None
        if current_sent.start > 0:
            if token.doc is doc:
                prev_token = doc[current_sent.start - 1]
                prev_sent = prev_token.sent

        context_text = prev_sent.text + " " + current_sent.text if prev_sent else current_sent.text
        coref_doc = nlp_coref(context_text)

        is_referential = False
        if 'coref' in coref_doc.spans:
            for cluster in coref_doc.spans['coref']:
                for mention in cluster:
                    if token.text.lower() == mention.text.lower():
                        is_referential = True
                        break
                if is_referential:
                    break

        if not is_referential:
            indices_to_remove.add(token.i)

    return indices_to_remove


def ablate_doc(doc, nlp_coref):
    """
    Performs the ablation on a single spaCy Doc object.
    Returns the ablated text for that doc.
    """
    indices_to_remove = find_and_confirm_expletives(doc, nlp_coref)

    if not indices_to_remove:
        return doc.text_with_ws

    new_tokens = [tok.text_with_ws for i, tok in enumerate(doc) if i not in indices_to_remove]
    return "".join(new_tokens)


def validate_expletive_removal(original_text, ablated_text, nlp):
    """
    Check that expletives were actually removed.
    Returns True if expletives were reduced, False otherwise.
    """
    original_doc = nlp(original_text)
    ablated_doc = nlp(ablated_text)
    
    original_expletives = [tok for tok in original_doc if tok.dep_ == 'expl']
    ablated_expletives = [tok for tok in ablated_doc if tok.dep_ == 'expl']
    
    print(f"  Original expletives: {len(original_expletives)}")
    print(f"  Remaining expletives: {len(ablated_expletives)}")
    
    if len(original_expletives) > 0:
        removal_rate = (len(original_expletives) - len(ablated_expletives)) / len(original_expletives)
        print(f"  Expletive removal rate: {removal_rate:.2%}")
        return removal_rate > 0
    else:
        print("  No expletives found in original text")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Ablate expletives from a directory of .train files while maintaining dataset size."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing source .train files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the ablated .train files.")
    parser.add_argument("--replacement_pool_dir", required=True,
                        help="Directory containing corresponding replacement pool files.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of lines to process in each chunk.")
    args = parser.parse_args()

    # --- Device and Model Loading ---
    spacy_device = get_spacy_device()
    spacy.prefer_gpu(spacy_device)

    print("Loading spaCy models...")
    # Use the transformer model for better accuracy in dependency parsing and coreference
    nlp = spacy.load("en_core_web_trf")
    
    # Use the same model for coreference resolution
    nlp_coref = nlp

    print("Models loaded successfully.")

    # --- Find all source files ---
    search_pattern = os.path.join(args.input_dir, '**', '*.train')
    source_files = glob.glob(search_pattern, recursive=True)
    if not source_files:
        raise FileNotFoundError(f"No '.train' files found in {args.input_dir}")

    # --- Process each file individually ---
    for source_path in tqdm(source_files, desc="Processing Corpus Files"):

        relative_path = os.path.relpath(source_path, args.input_dir)
        pool_path = os.path.join(args.replacement_pool_dir, relative_path)
        output_path = os.path.join(args.output_dir, relative_path)

        if not os.path.exists(pool_path):
            print(f"\nWarning: No matching pool file found for {source_path}. Skipping.")
            continue

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(source_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Calculate target token count from the full original text
            target_token_count = count_tokens("".join(lines))

        if target_token_count == 0:
            open(output_path, 'w').close()
            continue

        with open(pool_path, 'r', encoding='utf-8') as f:
            replacement_pool_sentences = f.readlines()

        # --- Ablate the main file in chunks with a correct progress bar ---
        ablated_text = ""
        original_text = "".join(lines)
        
        with tqdm(
                total=len(lines), desc=f"  Ablating {os.path.basename(source_path)}", leave=False
        ) as pbar:
            # Manually create batches to have precise control over the progress update
            for i in range(0, len(lines), args.chunk_size):
                chunk = lines[i:i + args.chunk_size]
                docs = nlp.pipe(chunk)
                for doc in docs:
                    ablated_text += ablate_doc(doc, nlp_coref)
                pbar.update(len(chunk))  # Update by the actual number of lines in the chunk
        
        # Validate that expletives were actually removed
        print(f"\n  Validating expletive removal for {os.path.basename(source_path)}:")
        validation_success = validate_expletive_removal(original_text, ablated_text, nlp)
        if not validation_success:
            print(f"  ⚠️  Warning: No expletives were removed from {os.path.basename(source_path)}")

        # --- Iterative Replacement Loop ---
        current_token_count = count_tokens(ablated_text)
        with tqdm(
                total=target_token_count, initial=current_token_count, desc="  Rebuilding to size", leave=False
        ) as pbar_rebuild:
            while current_token_count < target_token_count:
                tokens_needed = target_token_count - current_token_count

                text_to_add_chunk = ""
                while count_tokens(text_to_add_chunk) < tokens_needed * 1.2 and replacement_pool_sentences:
                    text_to_add_chunk += replacement_pool_sentences.pop(
                        random.randint(0, len(replacement_pool_sentences) - 1))

                if not text_to_add_chunk:
                    print(f"\nWarning: Replacement pool for {source_path} exhausted.")
                    break

                ablated_chunk_to_add = ""
                # Also process the replacement chunk with nlp.pipe
                for doc in nlp.pipe(text_to_add_chunk.splitlines()):
                    ablated_chunk_to_add += ablate_doc(doc, nlp_coref)

                ablated_text += ablated_chunk_to_add
                new_token_count = count_tokens(ablated_text)
                pbar_rebuild.update(new_token_count - current_token_count)
                current_token_count = new_token_count

        # --- Finalization ---
        # Preserve line structure by processing line by line
        ablated_lines = []
        current_line_tokens = 0
        
        for line in lines:
            if current_line_tokens >= target_token_count:
                break
                
            doc = nlp(line)
            ablated_line = ablate_doc(doc, nlp_coref)
            ablated_lines.append(ablated_line)
            current_line_tokens += count_tokens(ablated_line)
        
        # Write the ablated text preserving line structure
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(ablated_lines)

    print("\nAblation complete for all files.")


if __name__ == "__main__":
    main()