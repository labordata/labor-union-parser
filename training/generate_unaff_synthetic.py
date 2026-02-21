"""Generate synthetic training data for UNAFF unions."""

import csv
import random
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "opdr.db"
DATA_DIR = Path(__file__).parent / "data"
ACRONYM_MAPPING_PATH = DATA_DIR / "acronym_to_fullname.csv"


def load_acronym_mappings():
    """Load acronym -> full_name mappings from CSV.

    Returns dict mapping full_name (uppercase) -> list of acronyms
    """
    fullname_to_acronyms = defaultdict(list)

    if not ACRONYM_MAPPING_PATH.exists():
        print(f"Warning: {ACRONYM_MAPPING_PATH} not found")
        return fullname_to_acronyms

    with open(ACRONYM_MAPPING_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            acronym = row.get("acronym", "").strip()
            full_name = row.get("full_name", "").strip().upper()

            if acronym and full_name:
                # Handle acronyms with suffixes like "MNA-12" -> base acronym "MNA"
                base_acronym = re.split(r"[-/]", acronym)[0]
                fullname_to_acronyms[full_name].append(acronym)
                if base_acronym != acronym:
                    fullname_to_acronyms[full_name].append(base_acronym)

    # Deduplicate
    for k in fullname_to_acronyms:
        fullname_to_acronyms[k] = list(set(fullname_to_acronyms[k]))

    print(
        f"Loaded {sum(len(v) for v in fullname_to_acronyms.values())} acronym mappings for {len(fullname_to_acronyms)} unions"
    )
    return fullname_to_acronyms


# Global acronym mappings (loaded once)
ACRONYM_MAPPINGS = None

# Word abbreviations - common labor union terminology
WORD_ABBREVS = {
    "INTERNATIONAL": ["INT'L", "INTL", "INT"],
    "ASSOCIATION": ["ASSN", "ASSOC", "ASN"],
    "EMPLOYEES": ["EMPLS", "EMPS", "EES"],
    "WORKERS": ["WKRS", "WRKRS"],
    "UNION": ["UN"],
    "AMERICAN": ["AM", "AMER"],
    "AMERICA": ["AM", "AMER"],
    "UNITED": ["UTD"],
    "FEDERATION": ["FED"],
    "BROTHERHOOD": ["BHD", "BRO"],
    "PROFESSIONAL": ["PROF", "PROFL"],
    "INDEPENDENT": ["IND", "INDEP"],
    "NATIONAL": ["NAT'L", "NATL"],
    "SERVICE": ["SVC", "SERV"],
    "SERVICES": ["SVCS", "SERVS"],
    "COMMUNICATIONS": ["COMM", "COMMS"],
    "GOVERNMENT": ["GOV'T", "GOVT"],
    "TEACHERS": ["TCHRS"],
    "ELECTRICAL": ["ELEC"],
    "ENGINEERS": ["ENGRS", "ENG"],
    "PRODUCTION": ["PROD"],
    "MAINTENANCE": ["MAINT"],
    "HOSPITAL": ["HOSP"],
    "MUNICIPAL": ["MUNI", "MUN"],
    "COUNTY": ["CTY", "CO"],
    "DISTRICT": ["DIST"],
    "DIVISION": ["DIV"],
    "COUNCIL": ["CNCL"],
    "CHAPTER": ["CHAP", "CH"],
    "SECURITY": ["SEC"],
    "PROTECTIVE": ["PROT"],
    "POLICE": ["POL"],
    "FIREFIGHTERS": ["FF", "FFS"],
    "TRANSPORTATION": ["TRANS", "TRANSP"],
    "TEAMSTERS": ["TMSTRS"],
    "MACHINISTS": ["MACH"],
    "CARPENTERS": ["CARP", "CARPS"],
    "LABORERS": ["LABRS"],
    "PLUMBERS": ["PLMBRS"],
    "ELECTRICIANS": ["ELECS"],
    "OPERATING": ["OPER", "OPR"],
    "CONSTRUCTION": ["CONST", "CONSTR"],
    "BUILDING": ["BLDG"],
    "TRADES": ["TRS"],
    # Additional abbreviations found in real data
    "NURSES": ["NRS", "NRSS"],
    "GUILD": ["GLD"],
    "DIRECTORS": ["DIR", "DIRS"],
    "INCORPORATED": ["INC"],
    "HEALTHCARE": ["HLTHCR", "HC"],
    "MEDICAL": ["MED"],
    "STAFF": ["STF"],
    "OFFICERS": ["OFCRS", "OFF"],
    "ORGANIZATION": ["ORG"],
    "COMMITTEE": ["COMM", "CMT", "CMTE"],
    "REPRESENTATIVES": ["REPS"],
    "ORGANIZING": ["ORG"],
    "REGISTERED": ["REG"],
    "CERTIFIED": ["CERT"],
    "LICENSED": ["LIC"],
    "TECHNICAL": ["TECH"],
    "CLERICAL": ["CLER"],
    "OFFICE": ["OFC"],
    "PUBLIC": ["PUB"],
    "EDUCATION": ["ED", "EDUC"],
    "EDUCATIONAL": ["ED", "EDUC"],
}

# Words to potentially drop
DROPABLE_WORDS = {"OF", "THE", "AND", "A", "AN", "FOR", "IN"}

# Words to skip in acronym generation (suffixes, designators)
ACRONYM_SKIP_WORDS = {
    "OF",
    "THE",
    "AND",
    "A",
    "AN",
    "FOR",
    "IN",  # connectors
    "INC",
    "INCORPORATED",
    "LLC",
    "LTD",
    "CORP",
    "CORPORATION",  # business suffixes
    "NHQ",
    "HQ",
    "HEADQUARTERS",  # location designators
    "AFL-CIO",
    "AFL",
    "CIO",
    "AFLCIO",  # federation affiliations
    "LOCAL",
    "LU",
    "UNIT",
    "CHAPTER",
    "BRANCH",
    "DIVISION",  # org unit types
}

# Common umbrella organization suffixes seen in labeled data
UMBRELLA_SUFFIXES = [
    "/PASNAP",
    "/NNU",
    "/National Nurses United",
    ", AFL-CIO",
    " AFL-CIO",
]

# Symbol swaps
SYMBOL_SWAPS = {
    "&": "AND",
    "AND": "&",
}


def abbreviate_word(word):
    """Return an abbreviation for a word, or the word itself."""
    upper = word.upper()
    if upper in WORD_ABBREVS:
        return random.choice(WORD_ABBREVS[upper])
    return word


def abbreviate_text(text, n_abbrevs=None):
    """Randomly abbreviate some words in text.

    Args:
        text: Input text
        n_abbrevs: Number of words to abbreviate. If None, random 1-3.
    """
    words = text.split()

    # Find abbreviatable words
    abbrev_indices = [i for i, w in enumerate(words) if w.upper() in WORD_ABBREVS]

    if not abbrev_indices:
        return text

    # Choose how many to abbreviate
    if n_abbrevs is None:
        n_abbrevs = random.randint(1, min(3, len(abbrev_indices)))

    # Randomly select which words to abbreviate
    to_abbrev = random.sample(abbrev_indices, min(n_abbrevs, len(abbrev_indices)))

    result = []
    for i, word in enumerate(words):
        if i in to_abbrev:
            result.append(abbreviate_word(word))
        else:
            result.append(word)

    return " ".join(result)


def generate_acronym(text):
    """Generate acronym from text (first letter of significant words)."""
    words = text.upper().split()
    # Skip connectors, suffixes, and designators
    significant = [w for w in words if w not in ACRONYM_SKIP_WORDS and len(w) > 1]

    if len(significant) < 2:
        return None

    # Take first letter of each word, handle special chars
    acronym = ""
    for word in significant:
        # Skip if word starts with special char
        if word[0].isalpha():
            acronym += word[0]

    # Require at least 2 chars for acronym to be useful
    if len(acronym) < 2:
        return None

    return acronym


def drop_connectors(text, drop_prob=0.5):
    """Randomly drop connector words."""
    words = text.split()
    result = []
    for word in words:
        if word.upper() in DROPABLE_WORDS:
            if random.random() > drop_prob:
                result.append(word)
        else:
            result.append(word)
    return " ".join(result)


def swap_symbols(text):
    """Swap & <-> and."""
    words = text.split()
    result = []
    for word in words:
        upper = word.upper()
        if upper in SYMBOL_SWAPS:
            result.append(SYMBOL_SWAPS[upper])
        else:
            result.append(word)
    return " ".join(result)


def generate_variations(
    union_name, desig_num=None, unit_name=None, n_variations=10, acronym_mappings=None
):
    """Generate text variations for a UNAFF record.

    Args:
        union_name: The union name from database
        desig_num: Local number (None if no local number)
        unit_name: Unit name from database (optional additional context)
        n_variations: Number of variations to generate
        acronym_mappings: Dict mapping full_name -> list of known acronyms

    Returns:
        List of text variations
    """
    variations = set()

    # Clean up inputs
    union_name = union_name.strip() if union_name else ""
    unit_name = unit_name.strip() if unit_name else ""

    # Combine union_name parts if split across fields
    full_name = union_name
    if unit_name and unit_name not in union_name:
        # unit_name often has additional info
        pass  # We'll use it in some templates

    # Base templates depend on whether we have a local number
    if desig_num:
        templates = [
            f"{full_name} Local {desig_num}",
            f"{full_name} LU {desig_num}",
            f"{full_name} L {desig_num}",
            f"{full_name} #{desig_num}",
            f"{full_name} {desig_num}",
            f"{full_name}, Local {desig_num}",
            f"Local {desig_num}, {full_name}",
            f"Local {desig_num} {full_name}",
            f"LU {desig_num} {full_name}",
            f"{full_name} LOCAL {desig_num}",
            # Dash patterns seen in real data (e.g., "NCFO - 78")
            f"{full_name} - {desig_num}",
            f"{full_name} - Local {desig_num}",
            f"{full_name} - LU {desig_num}",
            # No space patterns
            f"{full_name}-{desig_num}",
            f"{full_name} Local No. {desig_num}",
            f"{full_name} Local Union {desig_num}",
            f"{full_name} Local Union No. {desig_num}",
            # Local # pattern
            f"{full_name} Local #{desig_num}",
            # Letter suffix variations (e.g., Local 621-A)
            f"{full_name} Local {desig_num}-A",
            f"{full_name} Local {desig_num}-B",
            f"{full_name} {desig_num}-A",
            f"{full_name} {desig_num}-B",
        ]
        if unit_name:
            templates.extend(
                [
                    f"{unit_name} Local {desig_num}",
                    f"{unit_name}, {full_name} Local {desig_num}",
                    f"{full_name} {unit_name} Local {desig_num}",
                    f"{unit_name} - {desig_num}",
                    f"{unit_name}/{full_name} Local {desig_num}",
                ]
            )
    else:
        # No local number - just variations of the name
        templates = [
            full_name,
            full_name,  # Duplicate to weight toward original
            full_name,
        ]
        if unit_name:
            templates.extend(
                [
                    f"{full_name}, {unit_name}",
                    f"{unit_name}, {full_name}",
                    f"{full_name} {unit_name}",
                    unit_name,
                ]
            )

    # Generate variations from templates
    for template in templates:
        variations.add(template)

        # Abbreviation variations
        for _ in range(2):
            abbrev = abbreviate_text(template)
            variations.add(abbrev)

        # Drop connectors
        dropped = drop_connectors(template)
        variations.add(dropped)

        # Symbol swaps
        swapped = swap_symbols(template)
        variations.add(swapped)

        # Combined: abbreviate + drop
        combined = abbreviate_text(drop_connectors(template))
        variations.add(combined)

        # Combined: abbreviate + swap
        combined2 = abbreviate_text(swap_symbols(template))
        variations.add(combined2)

    # Acronym variations (much more aggressive now)
    # Collect all acronyms: generated + from mapping file
    all_acronyms = set()

    # Generated acronym from name
    generated_acronym = generate_acronym(full_name)
    if generated_acronym and len(generated_acronym) >= 2:
        all_acronyms.add(generated_acronym)

    # Known acronyms from mapping file
    if acronym_mappings:
        known_acronyms = acronym_mappings.get(full_name.upper(), [])
        all_acronyms.update(known_acronyms)

    # Generate variations for each acronym
    for acronym in all_acronyms:
        if not acronym or len(acronym) < 2:
            continue

        # Always add standalone acronym
        variations.add(acronym)

        if desig_num:
            # Many formats for acronym + local number
            variations.add(f"{acronym} {desig_num}")
            variations.add(f"{acronym} Local {desig_num}")
            variations.add(f"{acronym} LU {desig_num}")
            variations.add(f"{acronym}-{desig_num}")
            variations.add(f"{acronym} #{desig_num}")
            variations.add(f"{acronym}, Local {desig_num}")
            variations.add(f"{acronym} - {desig_num}")
            variations.add(f"{acronym} - Local {desig_num}")
            variations.add(f"Local {desig_num}, {acronym}")
            variations.add(f"Local {desig_num} {acronym}")
            # Letter suffix variations (e.g., Local 621-A)
            for suffix in ["A", "B", "C"]:
                variations.add(f"{acronym} {desig_num}-{suffix}")
                variations.add(f"{acronym} Local {desig_num}-{suffix}")
                variations.add(f"Local {desig_num}-{suffix}, {acronym}")
        else:
            # Add suffix variations for no-local unions
            variations.add(f"{acronym} AFL-CIO")
            variations.add(f"{acronym}, AFL-CIO")
            # Common UNAFF pattern: "ACRONYM - " (bare acronym with dash)
            variations.add(f"{acronym} - ")
            variations.add(f"{acronym} -")
            variations.add(f"{acronym}-")

        # Parenthetical acronym at end of full name - e.g., "California Nurses Association (CNA)"
        variations.add(f"{full_name} ({acronym})")
        if desig_num:
            variations.add(f"{full_name} ({acronym}) Local {desig_num}")

    # Add common suffix variations to full name
    if not desig_num:
        variations.add(f"{full_name} AFL-CIO")
        variations.add(f"{full_name}, AFL-CIO")
        variations.add(f"{full_name} INC")
        if unit_name:
            variations.add(f"{full_name}/{unit_name}")
            variations.add(f"{unit_name}/{full_name}")

    # Add umbrella organization suffixes for nurse-related unions
    full_name_upper = full_name.upper()
    if "NURSE" in full_name_upper or "NURSING" in full_name_upper:
        for suffix in UMBRELLA_SUFFIXES:
            variations.add(f"{full_name}{suffix}")
            if acronym:
                variations.add(f"{full_name}{suffix} ({acronym})")
                variations.add(f"{full_name} ({acronym}){suffix}")

    # Filter out empty/whitespace-only and too-short variations
    variations = {v.strip() for v in variations if v.strip() and len(v.strip()) >= 2}

    # Return requested number (or all if fewer)
    variation_list = list(variations)
    if len(variation_list) > n_variations:
        return random.sample(variation_list, n_variations)
    return variation_list


def load_unaff_records():
    """Load UNAFF records from database.

    Returns (union_name, unit_name, desig_num) tuples that uniquely identify an f_num.
    Including unit_name allows disambiguation of unions like "Michigan Nurses Association"
    which have multiple locals/chapters with different unit_names.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Find (union_name, unit_name, desig_num) tuples that uniquely map to one f_num
    # Filter out negative desig_num (data quality issue)
    cur.execute(
        """
        SELECT union_name,
               COALESCE(unit_name, '') as unit_name,
               COALESCE(desig_num, 0) as desig_num,
               GROUP_CONCAT(DISTINCT f_num) as f_nums,
               COUNT(DISTINCT f_num) as num_fnums
        FROM lm_data
        WHERE aff_abbr = 'UNAFF'
          AND union_name IS NOT NULL
          AND f_num IS NOT NULL
          AND COALESCE(desig_num, 0) >= 0
        GROUP BY union_name, COALESCE(unit_name, ''), COALESCE(desig_num, 0)
        HAVING COUNT(DISTINCT f_num) = 1
    """
    )

    records = []
    for union_name, unit_name, desig_num, f_nums, _ in cur.fetchall():
        records.append(
            {
                "union_name": union_name,
                "unit_name": unit_name,
                "desig_num": desig_num,
                "f_num": f_nums,  # single f_num
            }
        )

    conn.close()

    return records


def generate_synthetic_data(n_variations_per_record=10):
    """Generate synthetic training data for UNAFF unions."""
    print("Loading UNAFF records (filtered to unique f_num mappings)...")
    records = load_unaff_records()
    print(f"Found {len(records)} unique (union_name, desig_num) -> f_num pairs")

    # Load acronym mappings
    acronym_mappings = load_acronym_mappings()

    # Generate variations
    synthetic_data = []

    for record in records:
        union_name = record["union_name"]
        desig_num = record["desig_num"]
        f_num = record["f_num"]

        variations = generate_variations(
            union_name=union_name,
            desig_num=desig_num if desig_num else None,
            unit_name=record["unit_name"],
            n_variations=n_variations_per_record,
            acronym_mappings=acronym_mappings,
        )

        for text in variations:
            synthetic_data.append(
                {
                    "text": text,
                    "union_name": union_name,
                    "desig_num": desig_num,  # 0 for no local number
                    "f_num": f_num,
                }
            )

    print(f"Generated {len(synthetic_data)} synthetic examples")
    return synthetic_data


def main():
    random.seed(42)

    synthetic_data = generate_synthetic_data(n_variations_per_record=50)

    # Save to CSV
    output_path = DATA_DIR / "unaff_synthetic.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "union_name", "desig_num", "f_num"]
        )
        writer.writeheader()
        writer.writerows(synthetic_data)

    print(f"Saved to {output_path}")

    # Print some examples
    print("\nSample generated texts:")
    sample = random.sample(synthetic_data, min(20, len(synthetic_data)))
    for item in sample:
        desig_str = f"desig={item['desig_num']}" if item["desig_num"] else "no desig"
        print(f"  [{desig_str}] {item['text'][:60]} -> f_num={item['f_num']}")


if __name__ == "__main__":
    main()
