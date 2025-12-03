#!/usr/bin/env python3
"""
Script đơn giản để parse SGF files theo năm từ thư mục raw_sgf.

Usage:
    # Parse tất cả các năm
    python scripts/parse_by_year.py --year all
    
    # Parse một năm cụ thể
    python scripts/parse_by_year.py --year 2000
    
    # Parse nhiều năm
    python scripts/parse_by_year.py --year 2000 --year 2001
    
    # Parse và generate labels luôn
    python scripts/parse_by_year.py --year 2000 --generate-labels
"""

import sys
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parse_by_year.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Check dependencies
def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    try:
        import sgf
    except ImportError as e:
        missing.append("sgf")
        logger.debug(f"sgf import error: {e}")
    
    try:
        import numpy
    except ImportError as e:
        missing.append("numpy")
        logger.debug(f"numpy import error: {e}")
    
    try:
        import torch
    except ImportError as e:
        missing.append("torch")
        logger.debug(f"torch import error: {e}")
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Please install them with:")
        logger.error(f"  pip install -r {scripts_dir / 'requirements_local.txt'}")
        logger.error("")
        logger.error("NOTE: Make sure you're using the correct Python interpreter.")
        logger.error("If you're using MSYS2/Git Bash, you may need to use Windows Python:")
        logger.error("  C:\\Users\\Hai\\AppData\\Local\\Programs\\Python\\Python312\\python.exe scripts/parse_by_year.py --year <YEAR>")
        logger.error("Or use the wrapper script: scripts/parse_by_year_wrapper.bat")
        return False
    return True

# Import parse functions
try:
    from parse_sgf_local import process_sgf_directory_by_year
except ImportError as e:
    logger.error(f"Cannot import parse_sgf_local: {e}")
    if "sgf" in str(e).lower():
        logger.error("Please install dependencies: pip install -r scripts/requirements_local.txt")
    sys.exit(1)

# Import label generation
try:
    from generate_labels_local import process_dataset_file
except ImportError as e:
    logger.warning(f"Cannot import generate_labels_local: {e}. Label generation will be skipped.")
    process_dataset_file = None


def list_available_years(sgf_dir):
    """List all available years from SGF files."""
    import re
    from collections import defaultdict
    
    sgf_dir = Path(sgf_dir)
    sgf_files = list(sgf_dir.glob('*.sgf'))
    
    years = defaultdict(int)
    for f in sgf_files:
        match = re.match(r'(\d{4})-\d{1,2}-\d{1,2}-\d+\.sgf', f.name)
        if match:
            year = int(match.group(1))
            years[year] += 1
    
    return dict(sorted(years.items()))


def main():
    parser = argparse.ArgumentParser(
        description='Parse SGF files theo năm từ thư mục raw_sgf',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse tất cả các năm
  python scripts/parse_by_year.py --year all
  
  # Parse một năm cụ thể
  python scripts/parse_by_year.py --year 2000
  
  # Parse nhiều năm
  python scripts/parse_by_year.py --year 2000 --year 2001
  
  # Parse và generate labels luôn
  python scripts/parse_by_year.py --year 2000 --generate-labels
  
  # List các năm có sẵn
  python scripts/parse_by_year.py --list-years
        """
    )
    
    parser.add_argument(
        '--input', type=str, default='data/raw_sgf',
        help='Thư mục chứa SGF files (default: data/raw_sgf)'
    )
    parser.add_argument(
        '--output', type=str, default='data/processed',
        help='Thư mục output cho positions (default: data/processed)'
    )
    parser.add_argument(
        '--year', type=str, nargs='+', default=None,
        help='Năm cần parse (có thể nhiều năm, hoặc "all" cho tất cả)'
    )
    parser.add_argument(
        '--list-years', action='store_true',
        help='List tất cả các năm có sẵn trong thư mục'
    )
    parser.add_argument(
        '--generate-labels', action='store_true',
        help='Generate labels sau khi parse (tích hợp với multi-task labels)'
    )
    parser.add_argument(
        '--labels-output', type=str, default='data/datasets',
        help='Thư mục output cho labeled datasets (default: data/datasets)'
    )
    parser.add_argument(
        '--board-sizes', type=int, nargs='+', default=[9, 13, 19],
        help='Board sizes cần xử lý (default: 9 13 19)'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Số worker processes (default: auto)'
    )
    parser.add_argument(
        '--min-positions', type=int, default=10,
        help='Số positions tối thiểu mỗi game (default: 10)'
    )
    parser.add_argument(
        '--no-filter-handicap', action='store_true',
        help='Không filter handicap positions khi generate labels'
    )
    
    args = parser.parse_args()
    
    # Check dependencies (trừ khi chỉ list years)
    if not args.list_years:
        if not check_dependencies():
            sys.exit(1)
    
    # List years nếu được yêu cầu
    if args.list_years:
        sgf_dir = Path(args.input)
        if not sgf_dir.exists():
            logger.error(f"Thư mục không tồn tại: {sgf_dir}")
            sys.exit(1)
        
        years = list_available_years(sgf_dir)
        if not years:
            logger.warning("Không tìm thấy file SGF nào trong thư mục")
            sys.exit(1)
        
        logger.info("Các năm có sẵn trong thư mục:")
        for year, count in years.items():
            logger.info(f"  {year}: {count} files")
        sys.exit(0)
    
    # Validate input directory
    sgf_dir = Path(args.input)
    if not sgf_dir.exists():
        logger.error(f"Thư mục không tồn tại: {sgf_dir}")
        sys.exit(1)
    
    # Determine years to process
    if args.year is None:
        logger.error("Vui lòng chỉ định năm (--year) hoặc dùng --list-years để xem các năm có sẵn")
        sys.exit(1)
    
    if 'all' in args.year:
        # Process all years
        years = list_available_years(sgf_dir)
        if not years:
            logger.error("Không tìm thấy file SGF nào trong thư mục")
            sys.exit(1)
        years_to_process = list(years.keys())
        logger.info(f"Processing tất cả các năm: {years_to_process}")
    else:
        # Process specified years
        years_to_process = []
        for year_str in args.year:
            try:
                year = int(year_str)
                years_to_process.append(year)
            except ValueError:
                logger.error(f"Invalid year: {year_str}. Phải là số (ví dụ: 2000)")
                sys.exit(1)
    
    # Process each year
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels_output_dir = Path(args.labels_output)
    if args.generate_labels:
        labels_output_dir.mkdir(parents=True, exist_ok=True)
    
    for year in years_to_process:
        logger.info("\n" + "="*60)
        logger.info(f"Processing năm {year}")
        logger.info("="*60)
        
        # Parse SGF files
        try:
            process_sgf_directory_by_year(
                sgf_dir=sgf_dir,
                output_dir=output_dir,
                year=year,
                board_sizes=args.board_sizes,
                num_workers=args.workers,
                min_positions_per_game=args.min_positions
            )
        except Exception as e:
            logger.error(f"Lỗi khi parse năm {year}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Generate labels nếu được yêu cầu
        if args.generate_labels and process_dataset_file:
            logger.info(f"\nGenerating labels cho năm {year}...")
            
            for board_size in args.board_sizes:
                positions_file = output_dir / f'positions_{board_size}x{board_size}_{year}.pt'
                
                if not positions_file.exists():
                    logger.warning(f"Không tìm thấy file positions: {positions_file}")
                    continue
                
                labels_file = labels_output_dir / f'labeled_{board_size}x{board_size}_{year}.pt'
                
                try:
                    process_dataset_file(
                        input_path=positions_file,
                        output_path=labels_file,
                        filter_handicap=not args.no_filter_handicap,
                        num_workers=args.workers,
                        batch_size=5000
                    )
                    logger.info(f"✅ Đã generate labels cho {board_size}x{board_size} năm {year}")
                except Exception as e:
                    logger.error(f"Lỗi khi generate labels cho {board_size}x{board_size} năm {year}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        elif args.generate_labels:
            logger.warning("Không thể generate labels: generate_labels_local không khả dụng")
    
    logger.info("\n" + "="*60)
    logger.info("Hoàn thành!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

