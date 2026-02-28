#!/usr/bin/env python3
"""Database research tools for labeled data auditing.

Subcommands:
  setup    Load gazetteer + labeled_data into opdr.db tables
  check    Run consistency checks on labeled_data
  resolve  Search for candidate (union_name, f_num) matches for a text string
"""

import csv
import json
import re
import sqlite3
from pathlib import Path

import click

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "opdr.db"
GAZETTEER_PATH = DATA_DIR / "gazetteer.json"
LABELED_DATA_PATH = DATA_DIR / "labeled_data.csv"


def get_db():
    return sqlite3.connect(DB_PATH)


def clean_fnum(fnum):
    """Normalize f_num by stripping .0 suffix."""
    if fnum and fnum.endswith(".0"):
        return fnum[:-2]
    return fnum


# ---------------------------------------------------------------------------
# Setup: load gazetteer + labeled_data into SQLite
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """Database research tools for labeled data auditing."""
    pass


@cli.command()
def setup():
    """Load gazetteer and labeled_data into opdr.db tables."""
    db = get_db()

    # -- Gazetteer table --
    db.execute("DROP TABLE IF EXISTS gazetteer")
    db.execute(
        """
        CREATE TABLE gazetteer (
            f_num INTEGER,
            union_name TEXT,
            desig_name TEXT,
            desig_num INTEGER,
            prefix INTEGER,
            suffix TEXT,
            unit_id TEXT
        )
    """
    )
    with open(GAZETTEER_PATH) as f:
        gaz = json.load(f)

    rows = []
    for f_num, records in gaz.items():
        for rec in records:
            rows.append(
                (
                    int(f_num),
                    rec["union_name"],
                    rec.get("desig_name", ""),
                    rec.get("desig_num", 0),
                    rec.get("prefix", 0),
                    rec.get("suffix", ""),
                    rec.get("unit_id", ""),
                )
            )
    db.executemany("INSERT INTO gazetteer VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    db.execute("CREATE INDEX idx_gaz_fnum ON gazetteer(f_num)")
    db.execute("CREATE INDEX idx_gaz_union ON gazetteer(union_name)")
    print(f"Loaded {len(rows)} gazetteer records")

    # -- Labeled data table (normalize f_num on load) --
    db.execute("DROP TABLE IF EXISTS labeled_data")
    db.execute(
        """
        CREATE TABLE labeled_data (
            text TEXT,
            union_name TEXT,
            desig_num TEXT,
            f_num TEXT,
            reason_missing_fnum TEXT,
            split TEXT
        )
    """
    )
    with open(LABELED_DATA_PATH) as f:
        reader = csv.DictReader(f)
        ld_rows = []
        for row in reader:
            ld_rows.append(
                (
                    row["text"],
                    row["union_name"],
                    row["desig_num"],
                    clean_fnum(row["f_num"]),
                    row["reason_missing_fnum"],
                    row["split"],
                )
            )
    db.executemany("INSERT INTO labeled_data VALUES (?, ?, ?, ?, ?, ?)", ld_rows)
    db.execute("CREATE INDEX idx_ld_fnum ON labeled_data(f_num)")
    db.execute("CREATE INDEX idx_ld_union ON labeled_data(union_name)")
    db.execute("CREATE INDEX idx_ld_split ON labeled_data(split)")
    print(f"Loaded {len(ld_rows)} labeled_data records")

    # -- Abbreviations table --
    db.execute("DROP TABLE IF EXISTS abbreviations")
    db.execute(
        """
        CREATE TABLE abbreviations (
            acronym TEXT,
            full_name TEXT,
            count INTEGER,
            confidence TEXT,
            example TEXT
        )
    """
    )
    abbr_path = DATA_DIR / "acronym_to_fullname.csv"
    if abbr_path.exists():
        with open(abbr_path) as f:
            reader = csv.DictReader(f)
            abbr_rows = []
            for row in reader:
                abbr_rows.append(
                    (
                        row["acronym"],
                        row["full_name"],
                        int(row["count"]) if row["count"] else 0,
                        row["confidence"],
                        row.get("example", ""),
                    )
                )
        db.executemany("INSERT INTO abbreviations VALUES (?, ?, ?, ?, ?)", abbr_rows)
        db.execute("CREATE INDEX idx_abbr_acronym ON abbreviations(acronym)")
        db.execute("CREATE INDEX idx_abbr_fullname ON abbreviations(full_name)")
        print(f"Loaded {len(abbr_rows)} abbreviation mappings")

    # -- FTS table (if missing) --
    existing = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lm_fts'"
    ).fetchone()
    if not existing:
        print("Creating lm_fts FTS5 table...")
        db.execute(
            """
            CREATE VIRTUAL TABLE lm_fts USING fts5(
                f_num UNINDEXED,
                union_name, aff_abbr, unit_name, desig_name,
                desig_num, desiq_pre, desig_suf, city, state,
                tokenize = 'trigram'
            )
        """
        )
        db.execute(
            """
            INSERT INTO lm_fts
            SELECT DISTINCT f_num, union_name, aff_abbr, unit_name,
                   desig_name, desig_num, desiq_pre, desig_suf, city, state
            FROM lm_data
        """
        )
        count = db.execute("SELECT COUNT(*) FROM lm_fts").fetchone()[0]
        print(f"Created lm_fts with {count} rows")
    else:
        print("lm_fts already exists")

    db.commit()
    db.close()
    print("Done.")


# ---------------------------------------------------------------------------
# Check: consistency checks
# ---------------------------------------------------------------------------


@cli.command()
def check():
    """Run consistency checks on labeled_data."""
    db = get_db()

    # Check that gazetteer/labeled_data tables exist
    tables = [
        r[0]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    if "gazetteer" not in tables or "labeled_data" not in tables:
        click.echo("Run 'setup' first to load tables into the database.")
        db.close()
        return

    issues_found = 0

    # 1. f_num set but union_name doesn't match gazetteer
    click.echo("\n=== union_name doesn't match gazetteer for f_num ===")
    mismatches = db.execute(
        """
        SELECT DISTINCT ld.text, ld.union_name, ld.f_num,
               g.union_name as gaz_union_name
        FROM labeled_data ld
        JOIN gazetteer g ON CAST(ld.f_num AS INTEGER) = g.f_num
        WHERE ld.f_num != ''
          AND ld.union_name != ''
          AND ld.union_name != g.union_name
          AND ld.split IN ('train', 'test')
    """
    ).fetchall()
    for text, union, fnum, gaz_union in mismatches:
        click.echo(f"  {text[:70]}")
        click.echo(f"    labeled: {union}  |  gazetteer: {gaz_union}  |  f={fnum}")
        issues_found += 1
    if not mismatches:
        click.echo("  None found.")

    # 2. Same text with different union_name (in train/test)
    click.echo("\n=== Same text, different union_name ===")
    dupes = db.execute(
        """
        SELECT text, GROUP_CONCAT(DISTINCT union_name) as names,
               COUNT(DISTINCT union_name) as n
        FROM labeled_data
        WHERE split IN ('train', 'test') AND union_name != ''
        GROUP BY text
        HAVING n > 1
    """
    ).fetchall()
    for text, names, _n in dupes:
        click.echo(f"  {text[:70]}")
        click.echo(f"    unions: {names}")
        issues_found += 1
    if not dupes:
        click.echo("  None found.")

    # 3. Text/union_name keyword mismatch
    # Skip entries where the f_num confirms the label via gazetteer
    click.echo("\n=== Text/union_name keyword mismatch ===")
    # (keyword, expected_union, also_ok) — also_ok lists unions where
    # the keyword legitimately appears due to mergers or dual-branding
    keyword_checks = [
        ("boilermaker", "BOILERMAKERS AFL-CIO", set()),
        (
            "steelworker",
            "STEELWORKERS, AFL-CIO",
            {
                "PACE, AFL-CIO",  # PACE merged into USW
                "GLASS MOLDERS PLASTICS AFL-CIO",  # GMP merged into USW
            },
        ),
        (
            "teamster",
            "TEAMSTERS",
            {
                "PRINTING PACKAGING & PRODUCTION WORKERS UNION OF N",  # GCC/IBT
                "MAINTENANCE OF WAY EMPLS, IBT",  # BMWED/IBT
                "BAKERY WORKERS AFL-CIO",  # BCTGM/IBT
            },
        ),
        (
            "laborer",
            "LABORERS",
            {
                "POSTAL MAIL HANDLERS, LIUNA",  # Mail Handlers are LIUNA division
            },
        ),
        (
            "machinists",
            "MACHINISTS AFL-CIO",
            {
                "TRANSPORTATION COMMUNICATIONS UNION/IAM, AFL-CIO",  # TCU/IAM
                "FED EMPL NFFE, DIST 1, IAM, AFL-CIO",  # NFFE/IAM
            },
        ),
        (
            "ibew",
            "ELECTRICAL WORKERS IBEW AFL-CIO",
            {
                "METAL TRADES DEPT AFL-CIO",  # MTD councils list member unions
            },
        ),
        ("plumber", "PLUMBERS AFL-CIO", set()),
        ("carpenters", "CARPENTERS IND", set()),
        ("iron ?worker", "IRON WORKERS AFL-CIO", set()),
        (
            "fire ?fighter",
            "FIRE FIGHTERS AFL-CIO",
            {
                "SECURITY POLICE, FIRE PROF, IND",  # SPFPA is separate from IAFF
            },
        ),
        (
            "longshoremen",
            "LONGSHOREMENS ASN AFL-CIO",
            {
                "LONGSHORE AND WAREHOUSE UNION",  # ILWU is separate from ILA
            },
        ),
        (
            "seiu",
            "SERVICE EMPLOYEES",
            {
                "WORKERS UNITED, SEIU",  # WU affiliated with SEIU
                "GOVERNMENT EMPLS NAGE SEIU AFL-CIO",  # NAGE/SEIU
                "LAUNDRY AND DRY CLEAN, SEIU, AFL-CIO",
            },
        ),
        (
            "ufcw",
            "FOOD AND COMMERCIAL WKRS",
            {
                "RETAIL WHOLESALE, DC, UFCW",  # RWDSU is UFCW-affiliated
                "TEXTILE PROCESSORS, UFCW, AFL-CIO",
            },
        ),
        ("painters", "PAINTERS AFL-CIO", set()),
        ("roofer", "ROOFERS AFL-CIO", set()),
        ("bricklayer", "BRICKLAYERS AFL-CIO", set()),
        ("operating engineer", "ENGINEERS, OPERATING, AFL-CIO", set()),
    ]

    # Build a set of (f_num, union_name) confirmed by gazetteer
    confirmed = set()
    for row in db.execute(
        """
        SELECT DISTINCT ld.f_num, ld.union_name
        FROM labeled_data ld
        JOIN gazetteer g ON CAST(ld.f_num AS INTEGER) = g.f_num
        WHERE ld.f_num != ''
          AND ld.union_name = g.union_name
          AND ld.split IN ('train', 'test')
    """
    ).fetchall():
        confirmed.add((row[0], row[1]))

    all_rows = db.execute(
        """
        SELECT text, union_name, f_num, reason_missing_fnum
        FROM labeled_data
        WHERE split IN ('train', 'test')
          AND union_name != ''
    """
    ).fetchall()

    seen = set()
    for keyword, expected_union, also_ok in keyword_checks:
        pattern = re.compile(keyword, re.IGNORECASE)
        for text, union_name, fnum, reason in all_rows:
            if pattern.search(text) and union_name != expected_union:
                if reason == "multi-union":
                    continue
                if union_name in also_ok:
                    continue
                # Skip if gazetteer confirms the label
                if (fnum, union_name) in confirmed:
                    continue
                key = (text, union_name)
                if key in seen:
                    continue
                seen.add(key)
                click.echo(f"  '{keyword}' in text but union={union_name}")
                click.echo(f"    {text[:80]}")
                click.echo(f"    expected: {expected_union}  |  f={fnum}")
                issues_found += 1

    if not seen:
        click.echo("  None found.")

    # 4. Potential multi-union filings not marked as such
    click.echo("\n=== Potential unmarked multi-union filings ===")
    # Look for entries with union_name set and text containing patterns
    # that suggest multiple unions: ";" between words, "and" between
    # capitalized union-like names
    multi_rows = db.execute(
        """
        SELECT text, union_name, f_num, reason_missing_fnum
        FROM labeled_data
        WHERE split IN ('train', 'test')
          AND union_name != ''
          AND reason_missing_fnum != 'multi-union'
          AND (text LIKE '%;%' OR text LIKE '%&%')
    """
    ).fetchall()

    # Only flag if the text has multiple distinct union *groups*
    # Group aliases together so IAM/machinists or IBT/teamsters
    # don't trigger false positives
    union_keyword_groups = [
        ({"teamster", "ibt"}, "Teamsters"),
        ({"ibew"}, "IBEW"),
        ({"seiu"}, "SEIU"),
        ({"afscme"}, "AFSCME"),
        ({"ufcw"}, "UFCW"),
        ({"iuoe", "operating engineer"}, "IUOE"),
        ({"boilermaker"}, "Boilermakers"),
        ({"steelworker", "usw"}, "Steelworkers"),
        ({"laborer", "liuna"}, "Laborers"),
        ({"machinists", "iam", "iamaw"}, "IAM"),
        ({"carpenters", "ubc"}, "Carpenters"),
        ({"iron worker", "ironworker"}, "Iron Workers"),
        ({"plumber", "pipefitter"}, "Plumbers"),
        ({"painters"}, "Painters"),
        ({"roofer"}, "Roofers"),
        ({"iatse"}, "IATSE"),
        ({"unite here"}, "Unite Here"),
        ({"cwa"}, "CWA"),
        ({"smart", "sheet metal"}, "SMART"),
    ]
    multi_seen = set()
    for text, union_name, fnum, reason in multi_rows:
        text_lower = text.lower()
        matched_groups = set()
        for keywords, group_name in union_keyword_groups:
            if any(kw in text_lower for kw in keywords):
                matched_groups.add(group_name)
        if len(matched_groups) >= 2:
            key = text
            if key in multi_seen:
                continue
            multi_seen.add(key)
            click.echo(f"  {text[:80]}")
            click.echo(f"    union={union_name}  |  groups: {sorted(matched_groups)}")
            issues_found += 1

    if not multi_seen:
        click.echo("  None found.")

    # 5. f_num set but not in gazetteer (and reason is empty)
    click.echo("\n=== f_num set but not in gazetteer (no reason given) ===")
    missing_gaz = db.execute(
        """
        SELECT ld.text, ld.union_name, ld.f_num
        FROM labeled_data ld
        LEFT JOIN gazetteer g ON CAST(ld.f_num AS INTEGER) = g.f_num
        WHERE ld.f_num != ''
          AND g.f_num IS NULL
          AND ld.reason_missing_fnum = ''
          AND ld.split IN ('train', 'test')
    """
    ).fetchall()
    for text, union, fnum in missing_gaz[:20]:
        click.echo(f"  {text[:70]}  |  {union}  |  f={fnum}")
        issues_found += 1
    if len(missing_gaz) > 20:
        click.echo(f"  ... and {len(missing_gaz) - 20} more")
    if not missing_gaz:
        click.echo("  None found.")

    click.echo(f"\n{'='*60}")
    click.echo(f"Total issues: {issues_found}")
    db.close()


# ---------------------------------------------------------------------------
# Resolve: find candidate matches for a text string
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("text")
def resolve(text):
    """Search for candidate (union_name, f_num) matches for a text string."""
    db = get_db()

    # Check FTS table exists
    existing = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lm_fts'"
    ).fetchone()
    if not existing:
        click.echo("Run 'setup' first to create FTS table.")
        db.close()
        return

    # Load abbreviation map
    abbr_table = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='abbreviations'"
    ).fetchone()
    abbr_map = {}
    if abbr_table:
        for acronym, full_name in db.execute(
            "SELECT acronym, full_name FROM abbreviations"
        ).fetchall():
            abbr_map[acronym.upper()] = full_name

    # Extract searchable tokens from the text
    stop_words = {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "in",
        "for",
        "to",
        "with",
        "inc",
        "llc",
        "corp",
        "co",
        "no",
        "num",
        "local",
        "union",
        "international",
        "affiliated",
        "afl",
        "cio",
        "clc",
        "a/w",
    }
    tokens = re.findall(r"[a-zA-Z]+", text)
    search_tokens = [t for t in tokens if t.lower() not in stop_words and len(t) > 1]

    # Try abbreviation expansion first — if found, search expanded form
    # before falling back to original tokens
    abbr_results = []
    for token in search_tokens:
        if token.upper() in abbr_map:
            full = abbr_map[token.upper()]
            click.echo(f"Abbreviation: {token} => {full}")
            # Search the full name directly
            try:
                hits = db.execute(
                    """
                    SELECT DISTINCT f.f_num, f.union_name, f.unit_name
                    FROM lm_fts f
                    WHERE lm_fts MATCH ?
                    LIMIT 10
                    """,
                    (f'"{full}"',),
                ).fetchall()
                for fnum, union_name, unit_name in hits:
                    gaz = db.execute(
                        "SELECT union_name FROM gazetteer WHERE f_num = ? LIMIT 1",
                        (int(fnum),),
                    ).fetchone()
                    in_gaz = "YES" if gaz else "no"
                    gaz_name = gaz[0] if gaz else ""
                    abbr_results.append(
                        (
                            fnum,
                            gaz_name or union_name,
                            unit_name,
                            in_gaz,
                            full,
                        )
                    )
            except sqlite3.OperationalError:
                pass

    click.echo(f"Search tokens: {search_tokens}")
    click.echo()

    # If abbreviation search found results, show those first
    results = abbr_results

    # Also do phrase search on original tokens
    for window in range(len(search_tokens), 0, -1):
        for i in range(len(search_tokens) - window + 1):
            phrase = " ".join(search_tokens[i : i + window])
            if len(phrase) < 3:
                continue
            try:
                hits = db.execute(
                    """
                    SELECT DISTINCT f.f_num, f.union_name, f.unit_name
                    FROM lm_fts f
                    WHERE lm_fts MATCH ?
                    LIMIT 10
                    """,
                    (f'"{phrase}"',),
                ).fetchall()
            except sqlite3.OperationalError:
                continue

            for fnum, union_name, unit_name in hits:
                gaz = db.execute(
                    "SELECT union_name FROM gazetteer WHERE f_num = ? LIMIT 1",
                    (int(fnum),),
                ).fetchone()
                in_gaz = "YES" if gaz else "no"
                gaz_name = gaz[0] if gaz else ""
                key = (fnum, gaz_name or union_name)
                if key not in [(r[0], r[1]) for r in results]:
                    results.append(
                        (
                            fnum,
                            gaz_name or union_name,
                            unit_name,
                            in_gaz,
                            phrase,
                        )
                    )

        # Stop at the longest matching window (but only for non-abbr results)
        if len(results) > len(abbr_results):
            break

    if results:
        click.echo(
            f"{'f_num':<10} {'in_gaz':<7} {'union_name':<45} "
            f"{'unit_name':<30} {'matched'}"
        )
        click.echo("-" * 130)
        for fnum, union_name, unit_name, in_gaz, phrase in results[:20]:
            click.echo(
                f"{fnum:<10} {in_gaz:<7} {union_name:<45} "
                f'{unit_name:<30} "{phrase}"'
            )
    else:
        click.echo("No matches found.")

    db.close()


if __name__ == "__main__":
    cli()
