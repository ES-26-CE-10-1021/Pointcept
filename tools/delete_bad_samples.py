#!/usr/bin/env python
"""Delete annotation JSONs for samples listed in a bad-samples TSV.

Each TSV data line is whitespace-separated: ``<idx> <root> <sensor> <timestamp>``.
Lines that don't match that shape (banner output, blanks, comments) are skipped.
Only files under ``<root>/<sensor>/annotations/<timestamp>.json`` are deleted —
raw scans are left untouched.
"""

import argparse
import os
import sys


def parse_entries(tsv_path):
    entries = []
    with open(tsv_path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                int(parts[0])
            except ValueError:
                continue
            root, sensor, ts = parts[1], parts[2], parts[3]
            entries.append((lineno, root, sensor, ts))
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tsv", help="Path to bad-samples TSV")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be deleted without removing files")
    args = p.parse_args()

    entries = parse_entries(args.tsv)
    if not entries:
        print(f"No data rows found in {args.tsv}", file=sys.stderr)
        sys.exit(1)

    seen = set()
    deleted = missing = skipped_dup = 0
    for lineno, root, sensor, ts in entries:
        path = os.path.join(root, sensor, "annotations", f"{ts}.json")
        if path in seen:
            skipped_dup += 1
            continue
        seen.add(path)

        if not os.path.isfile(path):
            print(f"MISSING (line {lineno}): {path}")
            missing += 1
            continue

        if args.dry_run:
            print(f"WOULD DELETE: {path}")
        else:
            os.remove(path)
            print(f"DELETED: {path}")
        deleted += 1

    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"\n{verb}: {deleted}  missing: {missing}  duplicates skipped: {skipped_dup}")


if __name__ == "__main__":
    main()
