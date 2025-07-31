import pathlib
from typing import Generator

import spacy
import typer
from tqdm import tqdm

def _validate_spacy_model(model_name: str) -> spacy.Language:
    """Validate that the spaCy model is available and return the nlp object."""
    try:
        return spacy.load(model_name)
    except OSError:
        print(
            f"spaCy model '{model_name}' not found. "
            f"Please run 'python -m spacy download {model_name}'"
        )
        raise typer.Exit(1) from None

def _process_text(text: str, nlp: spacy.Language) -> str:
    """Replace all determiners with 'the'."""
    doc = nlp(text)
    modified_tokens = []
    for token in doc:
        if token.pos_ == "DET":
            modified_tokens.append("the" + token.whitespace_)
        else:
            modified_tokens.append(token.text_with_ws)
    return "".join(modified_tokens)

def _read_lines(file_path: pathlib.Path) -> Generator[str, None, None]:
    """Yield lines from a file."""
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()

def main(
    input_path: pathlib.Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the input text file.",
    ),
    output_path: pathlib.Path = typer.Argument(
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        help="Path to save the processed text file.",
    ),
    spacy_model: str = typer.Option(
        "en_core_web_sm",
        "--spacy-model",
        "-s",
        help="Name of the spaCy model to use for POS tagging.",
    ),
):
    """
    Impoverish determiners in a text file by replacing them all with 'the'.
    """
    nlp = _validate_spacy_model(spacy_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = list(_read_lines(input_path))
    processed_lines = nlp.pipe(
        lines,
        as_tuples=False,
        n_process=-1,
        batch_size=1000,
    )

    with output_path.open("w", encoding="utf-8") as f:
        for doc in tqdm(processed_lines, total=len(lines), desc="Processing lines"):
            modified_tokens = []
            for token in doc:
                if token.pos_ == "DET":
                    modified_tokens.append("the" + token.whitespace_)
                else:
                    modified_tokens.append(token.text_with_ws)
            f.write("".join(modified_tokens) + "\n")

    print(f"✅ Successfully saved impoverished text to {output_path}")

if __name__ == "__main__":
    typer.run(main)
