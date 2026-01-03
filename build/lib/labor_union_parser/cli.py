"""Command-line interface for labor union parser."""

import csv
import itertools
import json
import sys

import click
from tqdm import tqdm

from .extractor import Extractor, lookup_fnum


@click.command()
@click.argument("input_file", type=click.File("r"), default="-")
@click.option(
    "-c",
    "--column",
    help="Column name containing union text (required if CSV has multiple columns)",
)
@click.option(
    "-o",
    "--output",
    type=click.File("w"),
    default="-",
    help="Output file (default: stdout)",
)
@click.option(
    "--batch-size",
    type=int,
    default=256,
    help="Batch size for extraction (default: 256)",
)
@click.option(
    "--no-header",
    is_flag=True,
    help="Input has no header row (only valid for single-column input)",
)
def main(input_file, column: str, output, batch_size: int, no_header: bool):
    """
    Extract affiliation and designation from union names in a CSV file.

    Reads CSV from INPUT_FILE (or stdin if not specified) and appends columns:
    pred_aff, pred_desig, pred_prob, pred_fnum, pred_fnum_multiple

    Examples:

        labor-union-parser unions.csv -c union_name -o results.csv

        cat unions.csv | labor-union-parser -c union_name > results.csv

        labor-union-parser single_column.csv > results.csv
    """
    # Handle --no-header mode
    if no_header:
        if column is not None:
            click.echo("Error: --no-header cannot be used with -c/--column", err=True)
            sys.exit(1)

        # Read lines directly as single-column data
        texts = [line.rstrip("\n\r") for line in input_file]
        if not texts:
            click.echo("Error: Empty input", err=True)
            sys.exit(1)

        fieldnames = None
        rows = None
    else:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames

        if not fieldnames:
            click.echo("Error: Empty CSV or no header row", err=True)
            sys.exit(1)

        # Determine which column to use
        if len(fieldnames) == 1:
            text_column = fieldnames[0]
        elif column is None:
            click.echo(
                f"Error: CSV has multiple columns ({', '.join(fieldnames)}). "
                f"Use -c/--column to specify which column contains the union text.",
                err=True,
            )
            sys.exit(1)
        elif column not in fieldnames:
            click.echo(
                f"Error: Column '{column}' not found. "
                f"Available columns: {', '.join(fieldnames)}",
                err=True,
            )
            sys.exit(1)
        else:
            text_column = column

        # Read all rows
        rows = list(reader)
        texts = [row[text_column] for row in rows]

    # Extract in batches (progress to stderr)
    extractor = Extractor()
    results = []
    with tqdm(total=len(texts), desc="Extracting", file=sys.stderr) as pbar:
        for batch in itertools.batched(texts, batch_size):
            batch_results = extractor.extract_batch(list(batch))
            results.extend(batch_results)
            pbar.update(len(batch))

    # Look up fnums and build output
    pred_fields = [
        "pred_aff",
        "pred_desig",
        "pred_prob",
        "pred_fnum",
        "pred_fnum_multiple",
    ]

    if no_header:
        # No header mode: input has no header, but output includes header
        output_fieldnames = ["text"] + pred_fields
        writer = csv.DictWriter(output, fieldnames=output_fieldnames)
        writer.writeheader()

        for text, result in zip(texts, results):
            aff = result["affiliation"]
            desig = result["designation"]
            fnums = lookup_fnum(aff, desig) if aff and desig else []

            if len(fnums) == 0:
                fnum_str = ""
                fnum_multiple = ""
            elif len(fnums) == 1:
                fnum_str = str(fnums[0])
                fnum_multiple = False
            else:
                fnum_str = json.dumps(fnums)
                fnum_multiple = True

            writer.writerow(
                {
                    "text": text,
                    "pred_aff": aff,
                    "pred_desig": desig,
                    "pred_prob": f"{result['confidence']:.4f}",
                    "pred_fnum": fnum_str,
                    "pred_fnum_multiple": fnum_multiple,
                }
            )
    else:
        output_fieldnames = fieldnames + pred_fields
        writer = csv.DictWriter(output, fieldnames=output_fieldnames)
        writer.writeheader()

        for row, result in zip(rows, results):
            aff = result["affiliation"]
            desig = result["designation"]
            fnums = lookup_fnum(aff, desig) if aff and desig else []

            if len(fnums) == 0:
                fnum_str = ""
                fnum_multiple = ""
            elif len(fnums) == 1:
                fnum_str = str(fnums[0])
                fnum_multiple = False
            else:
                fnum_str = json.dumps(fnums)
                fnum_multiple = True

            row["pred_aff"] = aff
            row["pred_desig"] = desig
            row["pred_prob"] = f"{result['confidence']:.4f}"
            row["pred_fnum"] = fnum_str
            row["pred_fnum_multiple"] = fnum_multiple

            writer.writerow(row)


if __name__ == "__main__":
    main()
