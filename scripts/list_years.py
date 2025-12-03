#!/usr/bin/env python3
"""
Script đơn giản để list các năm có sẵn trong thư mục raw_sgf.
Không cần dependencies, chỉ cần Python standard library.
"""

import sys
from pathlib import Path
import re
from collections import defaultdict

def list_available_years(sgf_dir):
    """List all available years from SGF files."""
    sgf_dir = Path(sgf_dir)
    
    if not sgf_dir.exists():
        print(f"ERROR: Thu muc khong ton tai: {sgf_dir}", file=sys.stderr)
        return None
    
    sgf_files = list(sgf_dir.glob('*.sgf'))
    
    if not sgf_files:
        print(f"WARNING: Khong tim thay file SGF nao trong: {sgf_dir}", file=sys.stderr)
        return None
    
    years = defaultdict(int)
    for f in sgf_files:
        # Match both formats: YYYY-M-D-X.sgf and YYYY-MM-DD-XX.sgf
        match = re.match(r'(\d{4})-\d{1,2}-\d{1,2}-\d+\.sgf', f.name)
        if match:
            year = int(match.group(1))
            years[year] += 1
        else:
            # Try to extract year from other formats
            year_match = re.search(r'(\d{4})', f.name)
            if year_match:
                year = int(year_match.group(1))
                if 1900 <= year <= 2100:  # Reasonable year range
                    years[year] += 1
    
    return dict(sorted(years.items()))


def main():
    if len(sys.argv) > 1:
        sgf_dir = sys.argv[1]
    else:
        # Default to data/raw_sgf
        project_root = Path(__file__).parent.parent
        sgf_dir = project_root / 'data' / 'raw_sgf'
    
    years = list_available_years(sgf_dir)
    
    if years is None:
        sys.exit(1)
    
    if not years:
        print("WARNING: Khong tim thay nam nao trong ten file", file=sys.stderr)
        sys.exit(1)
    
    # Set UTF-8 encoding for output
    import io
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    print(f"\nCac nam co san trong thu muc: {sgf_dir}")
    print("=" * 60)
    total_files = 0
    for year, count in years.items():
        print(f"  {year}: {count:,} files")
        total_files += count
    print("=" * 60)
    print(f"Tong cong: {len(years)} nam, {total_files:,} files")
    print()
    print("Su dung:")
    print(f"  python scripts/parse_by_year.py --year <YEAR>")
    print(f"  python scripts/parse_by_year.py --year all")


if __name__ == "__main__":
    main()

