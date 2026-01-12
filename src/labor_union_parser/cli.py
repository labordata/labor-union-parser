"""Command-line interface for labor union parser."""

import csv
import itertools
import json
import sys

import click
from tqdm import tqdm

from .extractor import Extractor, lookup_fnum

PRED_FIELDS = [
    "pred_is_union",
    "pred_aff",
    "pred_unknown",
    "pred_desig",
    "pred_union_score",
    "pred_fnum",
    "pred_fnum_multiple",
]


def validate_no_header(ctx, param, value):
    """Callback to validate --no-header isn't used with --column."""
    if value and ctx.params.get("column") is not None:
        raise click.BadParameter("cannot be used with -c/--column")
    return value


def validate_column(ctx, param, value):
    """Callback to validate --column isn't used with --no-header."""
    if value is not None and ctx.params.get("no_header"):
        raise click.BadParameter("cannot be used with --no-header")
    return value


def make_row_stream(input_file, column, no_header):
    """Create a row stream and return (stream, output_fieldnames, text_column)."""
    if no_header:

        def stream():
            for line in input_file:
                text = line.rstrip("\n\r")
                yield text, {"text": text}

        return stream(), ["text"] + PRED_FIELDS

    # Read and parse header line
    header_line = input_file.readline()
    fieldnames = next(csv.reader([header_line]))
    if not fieldnames:
        raise click.ClickException("Empty CSV or no header row")

    # Determine text column
    if len(fieldnames) == 1:
        text_column = fieldnames[0]
    elif column is None:
        raise click.ClickException(
            f"CSV has multiple columns ({', '.join(fieldnames)}). "
            f"Use -c/--column to specify which column contains the union text."
        )
    elif column not in fieldnames:
        raise click.ClickException(
            f"Column '{column}' not found. Available columns: {', '.join(fieldnames)}"
        )
    else:
        text_column = column

    def stream():
        reader = csv.DictReader(input_file, fieldnames=fieldnames)
        for row in reader:
            yield row[text_column], row

    return stream(), fieldnames + PRED_FIELDS


def format_fnum(fnums):
    """Format fnum lookup result for output."""
    if len(fnums) == 0:
        return "", ""
    elif len(fnums) == 1:
        return str(fnums[0]), False
    else:
        return json.dumps(fnums), True


def build_pred_row(result):
    """Build prediction fields from extraction result."""
    aff = result["affiliation"] or ""
    desig = result["designation"] or ""
    fnums = lookup_fnum(aff, desig) if aff and desig else []
    fnum_str, fnum_multiple = format_fnum(fnums)

    return {
        "pred_is_union": result["is_union"],
        "pred_aff": aff,
        "pred_unknown": result["affiliation_unrecognized"],
        "pred_desig": desig,
        "pred_union_score": f"{result['union_score']:.4f}",
        "pred_fnum": fnum_str,
        "pred_fnum_multiple": fnum_multiple,
    }


@click.command()
@click.argument("input_file", type=click.File("r"), default="-")
@click.option(
    "--no-header",
    is_flag=True,
    is_eager=True,
    callback=validate_no_header,
    help="Input has no header row (only valid for single-column input)",
)
@click.option(
    "-c",
    "--column",
    callback=validate_column,
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
def main(input_file, column, output, batch_size, no_header):
    """
    Extract affiliation and designation from union names in a CSV file.

    Reads CSV from INPUT_FILE (or stdin if not specified) and appends columns:
    pred_is_union, pred_aff, pred_unknown, pred_desig, pred_union_score, pred_fnum, pred_fnum_multiple

    Examples:

        labor-union-parser unions.csv -c union_name -o results.csv

        cat unions.csv | labor-union-parser -c union_name > results.csv

        labor-union-parser single_column.csv > results.csv
    """
    rows, fieldnames = make_row_stream(input_file, column, no_header)

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    extractor = Extractor()

    with tqdm(
        desc="Extracting",
        file=sys.stderr,
        unit=" rows",
        disable=not sys.stderr.isatty(),
    ) as pbar:
        for batch in itertools.batched(rows, batch_size):
            texts, row_data = zip(*batch)
            results = extractor.extract_batch(list(texts))

            for row, result in zip(row_data, results):
                writer.writerow({**row, **build_pred_row(result)})

            pbar.update(len(batch))


if __name__ == "__main__":
    main()
