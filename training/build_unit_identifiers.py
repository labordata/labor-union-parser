#!/usr/bin/env python3
"""
Build pseudo unit identifiers for f_nums.

Logic: If two f_nums share ANY representation (union_name, desig_name, desig_num,
prefix, suffix) IN THE SAME YEAR, they should have different unit identifiers
(A, B, C, etc.).

Requiring same-year overlap avoids creating spurious conflicts between f_nums that
changed prefixes/suffixes over time but were never actually indistinguishable in
any single filing period.

This is a graph connected components problem:
- Nodes: f_nums
- Edges: f_nums that share a representation in the same year
- Component: group of f_nums that transitively share representations
- Within each component, assign A, B, C... identifiers
"""

import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "opdr.db"
OUTPUT_CSV = Path(__file__).parent / "data" / "fnum_to_unit_identifier.csv"


def normalize_designation(s):
    """Normalize designation strings."""
    if not s:
        return ""
    return s.strip().upper()


def get_fnum_representations():
    """
    Get all unique (year, representation) pairs for each f_num,
    plus unit_names per f_num for cross-year conflict detection.

    Returns:
        fnum_reprs: f_num -> set of (year, union_name, desig_name, desig_num, prefix, suffix)
        fnum_unit_names: f_num -> set of unit_name strings
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get all unique representations per f_num per year
    cur.execute(
        """
        SELECT
            f_num,
            yr_covered,
            union_name,
            desig_name,
            COALESCE(desig_num, 0) as desig_num,
            desiq_pre,
            desig_suf,
            COALESCE(unit_name, '') as unit_name
        FROM lm_data
        WHERE f_num IS NOT NULL
          AND union_name IS NOT NULL
          AND yr_covered IS NOT NULL
        GROUP BY f_num, yr_covered, union_name, desig_name, desig_num,
                 desiq_pre, desig_suf, unit_name
    """
    )

    fnum_reprs = defaultdict(set)
    fnum_unit_names = defaultdict(set)

    for row in cur.fetchall():
        fnum, year, union_name, desig_name, desig_num, prefix, suffix, unit_name = row

        # Normalize
        # Non-numeric prefixes are invisible to the model (mapped to 0),
        # so treat them as empty for conflict detection.
        prefix_norm = str(int(prefix)) if prefix and prefix.strip().isdigit() else ""

        repr_key = (
            year,
            union_name,
            desig_name or "",
            int(desig_num),
            prefix_norm,
            normalize_designation(suffix or ""),
        )

        fnum_reprs[fnum].add(repr_key)
        fnum_unit_names[fnum].add(unit_name.strip().upper())

    conn.close()

    print(f"Loaded {len(fnum_reprs)} unique f_nums")

    # Count representations per f_num
    repr_counts = defaultdict(int)
    for fnum, reprs in fnum_reprs.items():
        repr_counts[len(reprs)] += 1

    print("\nRepresentations per f_num:")
    for count in sorted(repr_counts.keys())[:10]:
        print(f"  {count} repr(s): {repr_counts[count]} f_nums")
    if len(repr_counts) > 10:
        print(f"  ... and {len(repr_counts) - 10} more")

    return fnum_reprs, fnum_unit_names


def build_conflict_graph(fnum_reprs, fnum_unit_names):
    """
    Build a graph where f_nums are connected if:
    1. They share a representation in the same year, OR
    2. They share structured fields across years but have different unit_names
       (indicating they are genuinely different entities, not the same local re-filed).

    Returns:
        dict: f_num -> set of conflicting f_nums
    """
    # Build a reverse index: (year, repr) -> set of f_nums
    repr_to_fnums = defaultdict(set)

    for fnum, reprs in fnum_reprs.items():
        for repr_key in reprs:
            repr_to_fnums[repr_key].add(fnum)

    # Count how many representations are shared by multiple f_nums
    shared_reprs = {
        repr_key: fnums for repr_key, fnums in repr_to_fnums.items() if len(fnums) > 1
    }
    print(f"\nFound {len(shared_reprs)} representations shared by multiple f_nums")

    # Build conflict graph from same-year edges
    conflicts = defaultdict(set)

    for repr_key, fnums in shared_reprs.items():
        fnums_list = list(fnums)
        for i, fnum1 in enumerate(fnums_list):
            for fnum2 in fnums_list[i + 1 :]:
                conflicts[fnum1].add(fnum2)
                conflicts[fnum2].add(fnum1)

    same_year_count = len(conflicts)
    print(f"Same-year conflicts: {same_year_count} f_nums")

    # Also build cross-year edges for f_nums that share structured fields
    # but have different unit_names (different entities needing disambiguation)
    struct_to_fnums = defaultdict(set)
    for fnum, reprs in fnum_reprs.items():
        for repr_key in reprs:
            # Strip year from the key
            struct_key = repr_key[1:]  # (union, desig, num, prefix, suffix)
            struct_to_fnums[struct_key].add(fnum)

    cross_year_edges = 0
    for struct_key, fnums in struct_to_fnums.items():
        if len(fnums) <= 1:
            continue
        fnums_list = list(fnums)
        for i, fnum1 in enumerate(fnums_list):
            for fnum2 in fnums_list[i + 1 :]:
                if fnum2 in conflicts.get(fnum1, set()):
                    continue  # already connected
                # Only add edge if unit_names differ
                names1 = fnum_unit_names.get(fnum1, set())
                names2 = fnum_unit_names.get(fnum2, set())
                if names1 != names2:
                    conflicts[fnum1].add(fnum2)
                    conflicts[fnum2].add(fnum1)
                    cross_year_edges += 1

    print(f"Cross-year edges (different unit_names): {cross_year_edges}")
    print(f"Total conflict graph: {len(conflicts)} f_nums")

    # Count conflict degrees
    conflict_degrees = defaultdict(int)
    for fnum, conflicting_fnums in conflicts.items():
        conflict_degrees[len(conflicting_fnums)] += 1

    print("\nConflict degrees:")
    for degree in sorted(conflict_degrees.keys())[:10]:  # Show first 10
        print(f"  {degree} conflicts: {conflict_degrees[degree]} f_nums")
    if len(conflict_degrees) > 10:
        print(f"  ... and {len(conflict_degrees) - 10} more degrees")

    return conflicts


def find_connected_components(conflicts, all_fnums):
    """
    Find connected components in the conflict graph using DFS.

    Returns:
        list of sets: Each set is a connected component of conflicting f_nums
    """
    visited = set()
    components = []

    def dfs(fnum, component):
        """Depth-first search to find all f_nums in this component."""
        if fnum in visited:
            return
        visited.add(fnum)
        component.add(fnum)

        # Visit all conflicting f_nums
        for neighbor in conflicts.get(fnum, set()):
            dfs(neighbor, component)

    # Find components for all f_nums that have conflicts
    for fnum in conflicts.keys():
        if fnum not in visited:
            component = set()
            dfs(fnum, component)
            components.append(component)

    # F_nums with no conflicts get their own component (identifier "A")
    isolated_fnums = all_fnums - visited
    for fnum in isolated_fnums:
        components.append({fnum})

    print(f"\nFound {len(components)} connected components")

    # Component size distribution
    component_sizes = defaultdict(int)
    for comp in components:
        component_sizes[len(comp)] += 1

    print("\nComponent sizes:")
    for size in sorted(component_sizes.keys())[:15]:
        print(f"  Size {size}: {component_sizes[size]} components")
    if len(component_sizes) > 15:
        print(f"  ... and {len(component_sizes) - 15} more sizes")

    # Show largest component
    largest = max(components, key=len)
    print(f"\nLargest component: {len(largest)} f_nums")

    return components


def assign_identifiers(components):
    """
    Assign unit identifiers (A, B, C, ..., Z, AA, AB, ...) to f_nums within each component.

    Returns:
        dict: f_num -> (component_id, unit_identifier)
    """

    def get_identifier(index):
        """Convert index to identifier: 0->A, 1->B, ..., 25->Z, 26->AA, 27->AB, ..."""
        result = ""
        index += 1  # Make 1-based
        while index > 0:
            index -= 1  # Adjust for 0-based alphabet
            result = chr(65 + (index % 26)) + result
            index //= 26
        return result

    fnum_to_id = {}
    max_component_size = 0

    for comp_idx, component in enumerate(components):
        component_size = len(component)
        max_component_size = max(max_component_size, component_size)

        # Sort f_nums for deterministic assignment
        sorted_fnums = sorted(component)

        for idx, fnum in enumerate(sorted_fnums):
            if component_size == 1:
                # Isolated f_num: no conflicts, unit_id not needed
                # Empty string will map to padding idx and get masked
                identifier = ""
            else:
                identifier = get_identifier(idx)
            fnum_to_id[fnum] = (comp_idx, identifier)

    print(f"\nMax identifiers needed in one component: {max_component_size}")
    print(f"Max identifier: {get_identifier(max_component_size - 1)}")

    # Count identifier frequencies
    id_counts = defaultdict(int)
    for comp_id, identifier in fnum_to_id.values():
        id_counts[identifier] += 1

    print("\nIdentifier frequencies (top 10):")
    for identifier, count in sorted(
        id_counts.items(), key=lambda x: x[1], reverse=True
    )[:10]:
        print(f"  {identifier}: {count} f_nums")

    return fnum_to_id


def save_to_csv(fnum_to_id, output_path):
    """Save f_num to unit_identifier mapping to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["f_num", "component_id", "unit_identifier"])

        for fnum in sorted(fnum_to_id.keys()):
            comp_id, identifier = fnum_to_id[fnum]
            writer.writerow([fnum, comp_id, identifier])

    print(f"\nSaved {len(fnum_to_id)} f_num mappings to {output_path}")


def main():
    """Build unit identifiers for f_nums."""
    print("=" * 80)
    print("Building Unit Identifiers for F_nums")
    print("=" * 80)

    # Step 1: Get all representations per f_num
    fnum_reprs, fnum_unit_names = get_fnum_representations()

    # Step 2: Build conflict graph
    conflicts = build_conflict_graph(fnum_reprs, fnum_unit_names)

    # Step 3: Find connected components
    all_fnums = set(fnum_reprs.keys())
    components = find_connected_components(conflicts, all_fnums)

    # Step 4: Assign identifiers within each component
    fnum_to_id = assign_identifiers(components)

    # Step 5: Save to CSV
    save_to_csv(fnum_to_id, OUTPUT_CSV)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
