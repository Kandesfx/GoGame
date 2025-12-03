#!/usr/bin/env python3
"""
Script để parse SGF files và tự động chia thành các file output với kích thước hợp lý.
Phù hợp cho các file SGF không có năm trong tên file (ví dụ: 1547679.sgf, 1547692.sgf).

Usage:
    # Parse tất cả file SGF và chia thành chunks
    python scripts/parse_sgf_with_chunking.py \
        --input data/raw_sgf \
        --output data/processed \
        --board-sizes 9
    
    # Tùy chỉnh số positions mỗi chunk
    python scripts/parse_sgf_with_chunking.py \
        --input data/raw_sgf \
        --output data/processed \
        --board-sizes 9 \
        --positions-per-chunk 100000
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
        logging.FileHandler('parse_with_chunking.log', encoding='utf-8'),
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
        logger.error(f"  pip install {' '.join(missing)}")
        logger.error("")
        logger.error("NOTE: Make sure you're using the correct Python interpreter.")
        return False
    return True

# Import parse functions
try:
    from parse_sgf_local import process_sgf_directory_with_chunking
except ImportError as e:
    logger.error(f"Cannot import parse_sgf_local: {e}")
    if "sgf" in str(e).lower():
        logger.error("Please install dependencies: pip install sgf numpy torch")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Parse SGF files and automatically split into chunked output files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse 9x9 files with default chunking (50K positions/chunk)
  python scripts/parse_sgf_with_chunking.py --input data/raw_sgf --output data/processed --board-sizes 9
  
  # Custom chunk size (100K positions/chunk)
  python scripts/parse_sgf_with_chunking.py --input data/raw_sgf --output data/processed --board-sizes 9 --positions-per-chunk 100000
  
  # Multiple board sizes
  python scripts/parse_sgf_with_chunking.py --input data/raw_sgf --output data/processed --board-sizes 9 13 19
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing SGF files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for processed positions'
    )
    parser.add_argument(
        '--board-sizes',
        type=int,
        nargs='+',
        default=[9, 13, 19],
        help='Board sizes to process (default: 9 13 19)'
    )
    parser.add_argument(
        '--positions-per-chunk',
        type=int,
        default=50000,
        help='Number of positions per chunk file (default: 50000)'
    )
    parser.add_argument(
        '--chunk-prefix',
        type=str,
        default='chunk',
        help='Prefix for chunk filenames (default: chunk)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of worker processes (default: auto)'
    )
    parser.add_argument(
        '--min-positions-per-game',
        type=int,
        default=10,
        help='Minimum positions per game to keep (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Validate paths
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate board sizes
    valid_sizes = [9, 13, 19]
    invalid_sizes = [s for s in args.board_sizes if s not in valid_sizes]
    if invalid_sizes:
        logger.error(f"Invalid board sizes: {invalid_sizes}. Valid sizes: {valid_sizes}")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("SGF Parser with Automatic Chunking")
    logger.info("="*60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Board sizes: {args.board_sizes}")
    logger.info(f"Positions per chunk: {args.positions_per_chunk:,}")
    logger.info(f"Chunk prefix: {args.chunk_prefix}")
    logger.info(f"Workers: {args.num_workers or 'auto'}")
    logger.info("="*60)
    logger.info("")
    
    # Process files
    try:
        process_sgf_directory_with_chunking(
            sgf_dir=input_dir,
            output_dir=output_dir,
            board_sizes=args.board_sizes,
            num_workers=args.num_workers,
            min_positions_per_game=args.min_positions_per_game,
            positions_per_chunk=args.positions_per_chunk,
            chunk_prefix=args.chunk_prefix
        )
        
        logger.info("")
        logger.info("="*60)
        logger.info("✅ Parsing completed successfully!")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

