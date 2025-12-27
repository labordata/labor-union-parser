"""Command-line interface for labor union parser."""

import json

import click

from . import extract


@click.command()
@click.argument("text")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def main(text: str, as_json: bool):
    """
    Extract affiliation and designation from a union name.

    TEXT is the union name to parse (e.g., "SEIU Local 1199").

    Examples:

        labor-union-parser "SEIU Local 1199"

        labor-union-parser "Teamsters Local 705" --json
    """
    result = extract(text)

    if as_json:
        click.echo(json.dumps(result))
    else:
        click.echo(f"Affiliation: {result['affiliation']}")
        click.echo(f"Designation: {result['designation']}")


if __name__ == "__main__":
    main()
