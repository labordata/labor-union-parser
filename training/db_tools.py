#!/usr/bin/env python3
"""Database research tools for labeled data auditing.

Subcommands:
  setup         Load gazetteer + labeled_data into opdr.db tables
  check         Run consistency checks on labeled_data
  resolve       Search for candidate (union_name, f_num) matches for a text string
  resolve-batch Analyze all unresolved entries for a union (read-only)
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

# ---------------------------------------------------------------------------
# Union merger map: text keywords -> DB union names to search under.
# When resolving entries labeled as the "parent" union, also search these
# DB union names if the text contains the keyword.
# ---------------------------------------------------------------------------
MERGER_MAP = {
    # IUE merged into CWA; filed under both names in DB
    "iue": ["ELECTRICAL WORKERS IUE AFL-CIO", "COMMUNICATIONS WORKERS AFL-CIO"],
    "electronic": ["ELECTRICAL WORKERS IUE AFL-CIO", "COMMUNICATIONS WORKERS AFL-CIO"],
    # NABET merged into CWA; filed under CWA with NABET in unit_name
    "nabet": ["COMMUNICATIONS WORKERS AFL-CIO"],
    "broadcast": ["COMMUNICATIONS WORKERS AFL-CIO"],
    # TNG (Newspaper Guild) merged into CWA
    "newspaper guild": ["COMMUNICATIONS WORKERS AFL-CIO"],
    "newsguild": ["COMMUNICATIONS WORKERS AFL-CIO"],
    "tng": ["COMMUNICATIONS WORKERS AFL-CIO"],
    # ITU merged into CWA (via Printing Packaging, but locals filed under CWA)
    "typograph": ["COMMUNICATIONS WORKERS AFL-CIO"],
    "itu": ["COMMUNICATIONS WORKERS AFL-CIO"],
    # GCC merged into IBT; gazetteer kept PRINTING PACKAGING for most locals
    "graphic comm": ["PRINTING PACKAGING & PRODUCTION WORKERS UNION OF N", "TEAMSTERS"],
    "gcc": ["PRINTING PACKAGING & PRODUCTION WORKERS UNION OF N", "TEAMSTERS"],
    "gciu": ["PRINTING PACKAGING & PRODUCTION WORKERS UNION OF N", "TEAMSTERS"],
    # PACE merged into USW
    "pace": ["PACE, AFL-CIO", "STEELWORKERS, AFL-CIO"],
    # RWDSU affiliated with UFCW
    "rwdsu": ["RETAIL WHOLESALE, DC, UFCW", "FOOD AND COMMERCIAL WKRS"],
}


def get_db():
    return sqlite3.connect(DB_PATH)


def clean_fnum(fnum):
    """Normalize f_num by stripping .0 suffix."""
    if fnum and fnum.endswith(".0"):
        return fnum[:-2]
    return fnum


def detect_merger_keywords(text):
    """Return list of (keyword, db_union_names) for merger keywords found in text."""
    t = text.lower()
    matches = []
    for keyword, db_unions in MERGER_MAP.items():
        if keyword == "itu":
            if re.search(r"\bitu\b", t):
                matches.append((keyword, db_unions))
        elif keyword in t:
            matches.append((keyword, db_unions))
    return matches


def find_fnum_for_desig_num(
    db, desig_num, union_names, gaz_fnums, unit_name_pattern=None
):
    """Search for f_num by desig_num under the given union names.

    Returns list of f_nums found in gazetteer.
    If unit_name_pattern is given, also tries matching unit_name LIKE pattern.
    """
    results = set()
    for union in union_names:
        rows = db.execute(
            """SELECT DISTINCT f_num FROM lm_data
               WHERE union_name = ? AND desig_num = ?""",
            (union, desig_num),
        ).fetchall()
        results.update(str(f[0]) for f in rows if str(f[0]) in gaz_fnums)

    if not results and unit_name_pattern:
        for union in union_names:
            rows = db.execute(
                """SELECT DISTINCT f_num FROM lm_data
                   WHERE union_name = ? AND unit_name LIKE ?""",
                (union, unit_name_pattern),
            ).fetchall()
            results.update(str(f[0]) for f in rows if str(f[0]) in gaz_fnums)

    return sorted(results)


def pick_best_fnum(matches, gaz_union, preferred_union):
    """From a list of f_nums, prefer one whose gazetteer union_name matches preferred_union.

    When a union merged (e.g. GCC into IBT), some f_nums ended up under
    the new parent while others kept the old name.  Prefer the f_num that
    matches the label we already have.
    """
    # Try exact match on current label first
    same_union = [f for f in matches if gaz_union.get(f) == preferred_union]
    if same_union:
        return same_union[0]
    return matches[0]


def pick_smart_fnum(db, matches, gaz, text):
    """Disambiguate SMART f_nums using text keywords and filing metadata.

    SMART has collisions between:
    - plain_LU: sheet metal locals (LU, no unit_name or no 'TRANSPORT' in unit_name)
    - transport_LU: former UTU locals (LU, 'TRANSPORTATION DIVISION' in unit_name)
    - SLB: state legislative boards (state names in unit_name)
    - GCA: general committees of adjustment (railroad names in unit_name)

    Returns (fnum, confidence) where confidence is 'high' or 'low'.
    """
    t = text.lower()

    # Classify each f_num
    classified = []
    for fnum in matches:
        if fnum not in gaz:
            continue
        g = gaz[fnum][0]
        desig_name = g.get("desig_name", "")
        unit_row = db.execute(
            "SELECT DISTINCT unit_name FROM lm_data "
            "WHERE f_num = ? AND unit_name != ''",
            (int(fnum),),
        ).fetchone()
        unit = unit_row[0] if unit_row else ""
        has_transport = "TRANSPORT" in unit.upper()

        if desig_name == "SLB":
            label = "SLB"
        elif desig_name == "GCA":
            label = "GCA"
        elif has_transport:
            label = "transport_LU"
        else:
            label = "plain_LU"
        classified.append((fnum, label, unit))

    # Match text keywords to organizational type
    is_sheet_metal = any(
        kw in t for kw in ["sheet metal", "sheetmetal", "smw", "s.m.w", "smwia", "tin"]
    )
    is_transport = any(
        kw in t
        for kw in [
            "transport",
            "utu",
            "trainmen",
            "conductor",
            "railroad",
            "railway",
            "transit",
        ]
    ) or re.search(r"\btd\b", t)
    is_legislative = any(kw in t for kw in ["legislative", "slb"])
    is_gca = any(kw in t for kw in ["general committee", "gca"])

    # Pick based on keywords
    if is_sheet_metal and not is_transport:
        cands = [f for f, lbl, _ in classified if lbl == "plain_LU"]
        if cands:
            return cands[0], "high"
    if is_transport and not is_sheet_metal:
        cands = [f for f, lbl, _ in classified if lbl == "transport_LU"]
        if cands:
            return cands[0], "high"
    if is_legislative:
        cands = [f for f, lbl, _ in classified if lbl == "SLB"]
        if cands:
            return cands[0], "high"
    if is_gca:
        cands = [f for f, lbl, _ in classified if lbl == "GCA"]
        if cands:
            return cands[0], "high"

    # Generic "SMART" — if there's only one plain_LU, pick it (most common case)
    plain = [f for f, lbl, _ in classified if lbl == "plain_LU"]
    if len(plain) == 1:
        return plain[0], "low"

    return None, None


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

    # -- FTS tables (if missing) --
    # Trigram FTS for fuzzy/substring matching
    existing = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lm_fts'"
    ).fetchone()
    if not existing:
        print("Creating lm_fts FTS5 table (trigram)...")
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

    # Word-level FTS for BM25-ranked search
    existing_bm25 = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lm_fts_bm25'"
    ).fetchone()
    if not existing_bm25:
        print("Creating lm_fts_bm25 FTS5 table (word tokenizer)...")
        db.execute(
            """
            CREATE VIRTUAL TABLE lm_fts_bm25 USING fts5(
                f_num UNINDEXED,
                union_name, aff_abbr, unit_name, desig_name,
                desig_num, desiq_pre, desig_suf, city, state
            )
        """
        )
        db.execute(
            """
            INSERT INTO lm_fts_bm25
            SELECT DISTINCT f_num, union_name, aff_abbr, unit_name,
                   desig_name, desig_num, desiq_pre, desig_suf, city, state
            FROM lm_data
        """
        )
        count = db.execute("SELECT COUNT(*) FROM lm_fts_bm25").fetchone()[0]
        print(f"Created lm_fts_bm25 with {count} rows")
    else:
        print("lm_fts_bm25 already exists")

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
                if reason in ("multi-union", "multi-local"):
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
    """Search for candidate (union_name, f_num) matches for a text string.

    Uses BM25-ranked word search first, then falls back to trigram fuzzy matching.
    """
    db = get_db()

    # Check FTS tables exist
    has_bm25 = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lm_fts_bm25'"
    ).fetchone()
    has_trigram = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lm_fts'"
    ).fetchone()
    if not has_bm25 and not has_trigram:
        click.echo("Run 'setup' first to create FTS tables.")
        db.close()
        return

    # Check for merger keywords
    merger_hits = detect_merger_keywords(text)
    if merger_hits:
        for kw, db_unions in merger_hits:
            click.echo(f"Merger keyword '{kw}' detected — also searching: {db_unions}")

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

    # Extract searchable tokens
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
    }
    tokens = re.findall(r"[a-zA-Z]+", text)
    search_tokens = [t for t in tokens if t.lower() not in stop_words and len(t) > 1]

    click.echo(f"Search tokens: {search_tokens}")

    # Expand abbreviations
    expanded_tokens = []
    for token in search_tokens:
        if token.upper() in abbr_map:
            full = abbr_map[token.upper()]
            click.echo(f"Abbreviation: {token} => {full}")
            expanded_tokens.extend(re.findall(r"[a-zA-Z]+", full))
        else:
            expanded_tokens.append(token)

    click.echo()

    def lookup_gaz(fnum):
        row = db.execute(
            "SELECT union_name FROM gazetteer WHERE f_num = ? LIMIT 1",
            (int(fnum),),
        ).fetchone()
        return ("YES", row[0]) if row else ("no", "")

    seen_keys = set()
    results = []

    def add_result(fnum, union_name, unit_name, in_gaz, matched_by):
        key = (fnum, union_name)
        if key not in seen_keys:
            seen_keys.add(key)
            results.append((fnum, union_name, unit_name, in_gaz, matched_by))

    # --- Phase 1: BM25-ranked word search ---
    if has_bm25 and expanded_tokens:
        # Build query: all tokens as separate terms (implicit AND in FTS5)
        query = " ".join(expanded_tokens)
        try:
            hits = db.execute(
                """SELECT f_num, union_name, unit_name
                   FROM lm_fts_bm25
                   WHERE lm_fts_bm25 MATCH ?
                   ORDER BY rank
                   LIMIT 20""",
                (query,),
            ).fetchall()
            for fnum, union_name, unit_name in hits:
                in_gaz, gaz_name = lookup_gaz(fnum)
                add_result(
                    fnum,
                    gaz_name or union_name,
                    unit_name or "",
                    in_gaz,
                    f"bm25: {query}",
                )
        except sqlite3.OperationalError:
            pass

        # If full query got nothing, try progressively shorter phrases
        if not results:
            for window in range(len(expanded_tokens), 0, -1):
                for i in range(len(expanded_tokens) - window + 1):
                    phrase = " ".join(expanded_tokens[i : i + window])
                    try:
                        hits = db.execute(
                            """SELECT f_num, union_name, unit_name
                               FROM lm_fts_bm25
                               WHERE lm_fts_bm25 MATCH ?
                               ORDER BY rank
                               LIMIT 10""",
                            (phrase,),
                        ).fetchall()
                    except sqlite3.OperationalError:
                        continue
                    for fnum, union_name, unit_name in hits:
                        in_gaz, gaz_name = lookup_gaz(fnum)
                        add_result(
                            fnum,
                            gaz_name or union_name,
                            unit_name or "",
                            in_gaz,
                            f"bm25: {phrase}",
                        )
                if results:
                    break

    # --- Phase 2: Trigram fuzzy fallback (if BM25 found nothing) ---
    if not results and has_trigram and search_tokens:
        for window in range(len(search_tokens), 0, -1):
            for i in range(len(search_tokens) - window + 1):
                phrase = " ".join(search_tokens[i : i + window])
                if len(phrase) < 3:
                    continue
                try:
                    hits = db.execute(
                        """SELECT DISTINCT f_num, union_name, unit_name
                           FROM lm_fts
                           WHERE lm_fts MATCH ?
                           LIMIT 10""",
                        (f'"{phrase}"',),
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                for fnum, union_name, unit_name in hits:
                    in_gaz, gaz_name = lookup_gaz(fnum)
                    add_result(
                        fnum,
                        gaz_name or union_name,
                        unit_name or "",
                        in_gaz,
                        f"trigram: {phrase}",
                    )
            if results:
                break

    if results:
        click.echo(
            f"{'f_num':<10} {'in_gaz':<7} {'union_name':<45} "
            f"{'unit_name':<30} {'matched'}"
        )
        click.echo("-" * 130)
        for fnum, union_name, unit_name, in_gaz, matched in results[:20]:
            click.echo(
                f"{fnum:<10} {in_gaz:<7} {union_name:<45} "
                f'{unit_name:<30} "{matched}"'
            )
    else:
        click.echo("No matches found.")

    db.close()


@cli.command("resolve-batch")
@click.argument("union_name")
def resolve_batch(union_name):
    """Analyze all unresolved entries for UNION_NAME and show what could be resolved.

    Searches by desig_num under the union's DB names (including merger aliases).
    Detects prefix variants (e.g. IUE 800/84800) that resolve to the same f_num.
    Categorizes entries as: resolvable, prefix-variant, multi-local, reclassify,
    or unresolvable.  Does not modify any files.
    """
    db = get_db()

    # Load full gazetteer for disambiguation (e.g. SMART desig_name)
    with open(GAZETTEER_PATH) as f:
        gaz = json.load(f)
    gaz_fnums = set(gaz.keys())
    gaz_union = {k: v[0]["union_name"] for k, v in gaz.items()}

    # Load unresolved entries for this union
    unresolved = db.execute(
        """SELECT rowid, text, union_name, desig_num, f_num, reason_missing_fnum, split
           FROM labeled_data
           WHERE union_name = ? AND (f_num = '' OR f_num IS NULL)""",
        (union_name,),
    ).fetchall()

    if not unresolved:
        click.echo(f"No unresolved entries for '{union_name}'")
        db.close()
        return

    click.echo(f"Unresolved entries for '{union_name}': {len(unresolved)}")

    # Categorize proposals
    proposals = []  # (action, text, details_dict)

    for rowid, text, uname, desig_num, fnum, reason, split in unresolved:
        desig_num = clean_fnum(desig_num) if desig_num else ""

        # Determine which DB union names to search
        merger_hits = detect_merger_keywords(text)
        if merger_hits:
            search_unions = []
            for _, db_unions in merger_hits:
                search_unions.extend(db_unions)
            search_unions = list(dict.fromkeys(search_unions))
            # Determine correct gazetteer union_name from merger
            merger_kw = merger_hits[0][0]
        else:
            search_unions = [union_name]
            merger_kw = None

        # --- Entries with a desig_num ---
        if desig_num:
            # Build unit_name pattern for NABET-style lookups
            unit_pattern = None
            if merger_kw in ("nabet", "broadcast"):
                unit_pattern = f"NABET LOCAL {desig_num}"

            matches = find_fnum_for_desig_num(
                db, desig_num, search_unions, gaz_fnums, unit_pattern
            )
            if len(matches) == 1:
                resolved_fnum = matches[0]
                resolved_union = gaz_union.get(resolved_fnum, union_name)
                proposals.append(
                    (
                        "resolve",
                        text,
                        {
                            "rowid": rowid,
                            "f_num": resolved_fnum,
                            "union_name": resolved_union,
                            "desig_num": desig_num,
                            "old_reason": reason,
                        },
                    )
                )
            elif len(matches) > 1:
                # Try SMART-specific disambiguation
                smart_fnum, smart_conf = pick_smart_fnum(db, matches, gaz, text)
                if smart_fnum and smart_conf == "high":
                    resolved_union = gaz_union.get(smart_fnum, union_name)
                    proposals.append(
                        (
                            "resolve",
                            text,
                            {
                                "rowid": rowid,
                                "f_num": smart_fnum,
                                "union_name": resolved_union,
                                "desig_num": desig_num,
                                "old_reason": reason,
                            },
                        )
                    )
                else:
                    # Ambiguous — prefer f_num matching current label
                    resolved_fnum = (
                        smart_fnum
                        if smart_fnum
                        else pick_best_fnum(matches, gaz_union, union_name)
                    )
                    resolved_union = gaz_union.get(resolved_fnum, union_name)
                    proposals.append(
                        (
                            "resolve-ambiguous",
                            text,
                            {
                                "rowid": rowid,
                                "f_num": resolved_fnum,
                                "union_name": resolved_union,
                                "desig_num": desig_num,
                                "all_fnums": matches,
                                "old_reason": reason,
                            },
                        )
                    )
            else:
                proposals.append(
                    (
                        "unresolvable",
                        text,
                        {"rowid": rowid, "desig_num": desig_num, "reason": reason},
                    )
                )
            continue

        # --- Entries without desig_num ---
        # Check for multiple numbers (potential multi-local or prefix variant)
        nums = re.findall(r"\d{3,}", text)

        if len(nums) >= 2:
            # Look up each number
            num_to_fnums = {}
            for n in nums:
                matches = find_fnum_for_desig_num(db, n, search_unions, gaz_fnums)
                if matches:
                    num_to_fnums[n] = set(matches)

            if num_to_fnums:
                # Check for prefix variant: all numbers share a common f_num
                common = None
                for fset in num_to_fnums.values():
                    common = fset.copy() if common is None else common & fset

                if common:
                    resolved_fnum = sorted(common)[0]
                    resolved_union = gaz_union.get(resolved_fnum, union_name)
                    shortest_num = min(nums, key=len)
                    proposals.append(
                        (
                            "resolve-prefix",
                            text,
                            {
                                "rowid": rowid,
                                "f_num": resolved_fnum,
                                "union_name": resolved_union,
                                "desig_num": shortest_num,
                                "nums": nums,
                            },
                        )
                    )
                else:
                    proposals.append(
                        (
                            "multi-local",
                            text,
                            {
                                "rowid": rowid,
                                "nums": nums,
                                "num_to_fnums": {
                                    k: sorted(v) for k, v in num_to_fnums.items()
                                },
                            },
                        )
                    )
            else:
                proposals.append(
                    (
                        "multi-local",
                        text,
                        {"rowid": rowid, "nums": nums, "num_to_fnums": {}},
                    )
                )
        elif len(nums) == 1:
            # Single number but no desig_num field — try resolving
            n = nums[0]
            matches = find_fnum_for_desig_num(db, n, search_unions, gaz_fnums)
            if len(matches) == 1:
                resolved_fnum = matches[0]
                resolved_union = gaz_union.get(resolved_fnum, union_name)
                proposals.append(
                    (
                        "resolve",
                        text,
                        {
                            "rowid": rowid,
                            "f_num": resolved_fnum,
                            "union_name": resolved_union,
                            "desig_num": n,
                            "old_reason": reason,
                        },
                    )
                )
            elif len(matches) > 1:
                smart_fnum, smart_conf = pick_smart_fnum(db, matches, gaz, text)
                if smart_fnum and smart_conf == "high":
                    resolved_union = gaz_union.get(smart_fnum, union_name)
                    proposals.append(
                        (
                            "resolve",
                            text,
                            {
                                "rowid": rowid,
                                "f_num": smart_fnum,
                                "union_name": resolved_union,
                                "desig_num": n,
                                "old_reason": reason,
                            },
                        )
                    )
                else:
                    resolved_fnum = (
                        smart_fnum
                        if smart_fnum
                        else pick_best_fnum(matches, gaz_union, union_name)
                    )
                    resolved_union = gaz_union.get(resolved_fnum, union_name)
                    proposals.append(
                        (
                            "resolve-ambiguous",
                            text,
                            {
                                "rowid": rowid,
                                "f_num": resolved_fnum,
                                "union_name": resolved_union,
                                "desig_num": n,
                                "all_fnums": matches,
                                "old_reason": reason,
                            },
                        )
                    )
            else:
                proposals.append(
                    ("unresolvable", text, {"rowid": rowid, "reason": reason})
                )
        else:
            proposals.append(("unresolvable", text, {"rowid": rowid, "reason": reason}))

    # --- Display proposals grouped by action ---
    from collections import Counter

    action_counts = Counter(a for a, _, _ in proposals)

    click.echo(f"\n{'='*60}")
    click.echo("Proposal summary:")
    for action, count in sorted(action_counts.items()):
        click.echo(f"  {action}: {count}")
    click.echo(f"  total: {len(proposals)}")

    for action in [
        "resolve",
        "resolve-prefix",
        "resolve-ambiguous",
        "reclassify",
        "multi-local",
        "unresolvable",
    ]:
        entries = [(t, d) for a, t, d in proposals if a == action]
        if not entries:
            continue

        click.echo(f"\n{'='*60}")
        click.echo(f"=== {action} ({len(entries)}) ===\n")

        for text, details in entries:
            if action in ("resolve", "resolve-ambiguous"):
                amb = ""
                if action == "resolve-ambiguous":
                    amb = f" (ambiguous: {details['all_fnums']})"
                click.echo(
                    f"  dn={details['desig_num']:>6} -> f={details['f_num']}"
                    f"  union={details['union_name']}{amb}"
                )
                click.echo(f"    {text[:90]}")
            elif action == "resolve-prefix":
                click.echo(
                    f"  nums={details['nums']} -> f={details['f_num']}"
                    f"  dn={details['desig_num']}  union={details['union_name']}"
                )
                click.echo(f"    {text[:90]}")
            elif action == "reclassify":
                click.echo(
                    f"  -> union={details['new_union']}"
                    f"  reason={details['new_reason']}"
                )
                click.echo(f"    {text[:90]}")
            elif action == "multi-local":
                fstr = details.get("num_to_fnums", {})
                click.echo(f"  nums={details['nums']}  fnums={fstr}")
                click.echo(f"    {text[:90]}")
            else:
                click.echo(f"  reason={details.get('reason', '?')}")
                click.echo(f"    {text[:90]}")

    db.close()


if __name__ == "__main__":
    cli()
