"""
Script để merge datasets từ nhiều năm thành một file.

Tính năng:
- Merge labeled datasets từ nhiều năm
- Validate và filter duplicates
- Statistics và summary
"""

import torch
from pathlib import Path
from collections import defaultdict
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('merge_datasets.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def merge_labeled_datasets(
    input_dir,
    output_path,
    board_size,
    years=None,
    validate=True
):
    """
    Merge labeled datasets từ nhiều năm.
    
    Args:
        input_dir: Directory chứa labeled datasets
        output_path: Path để lưu merged dataset
        board_size: Board size (9, 13, hoặc 19)
        years: List of years to merge (None = all)
        validate: Validate và filter duplicates
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    
    # Find all labeled files for this board size
    pattern = f'labeled_{board_size}x{board_size}_*.pt'
    files = list(input_dir.glob(pattern))
    
    if not files:
        logger.warning(f"No files found matching {pattern}")
        return
    
    logger.info(f"Found {len(files)} files for {board_size}x{board_size}")
    
    # Filter by years if specified
    if years:
        files = [
            f for f in files
            if any(str(year) in f.name for year in years)
        ]
        logger.info(f"Filtered to {len(files)} files for years {years}")
    
    # Load and merge
    all_labeled_data = []
    years_processed = []
    total_before = 0
    
    for file_path in sorted(files):
        logger.info(f"Loading {file_path.name}...")
        
        try:
            data = torch.load(file_path, map_location='cpu')
            labeled_data = data['labeled_data']
            year = data.get('year')
            
            total_before += len(labeled_data)
            all_labeled_data.extend(labeled_data)
            
            if year:
                years_processed.append(year)
            
            logger.info(
                f"  Loaded {len(labeled_data):,} samples "
                f"(year: {year if year else 'unknown'})"
            )
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            continue
    
    logger.info(f"Total samples before validation: {len(all_labeled_data):,}")
    
    # Validate and filter duplicates
    if validate:
        logger.info("Validating and filtering duplicates...")
        
        # Simple duplicate detection (based on features hash)
        seen = set()
        unique_data = []
        duplicates = 0
        
        for sample in all_labeled_data:
            # Create hash from features (simplified)
            features = sample['features']
            features_hash = hash(tuple(features.flatten().tolist()[:100]))  # First 100 values
            
            if features_hash not in seen:
                seen.add(features_hash)
                unique_data.append(sample)
            else:
                duplicates += 1
        
        all_labeled_data = unique_data
        logger.info(f"Removed {duplicates:,} duplicates")
    
    # Save merged dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'labeled_data': all_labeled_data,
        'board_size': board_size,
        'total': len(all_labeled_data),
        'years': sorted(set(years_processed)) if years_processed else None,
        'metadata': {
            'source_files': len(files),
            'total_before_validation': total_before,
            'date_merged': datetime.now().isoformat(),
            'validated': validate
        }
    }, output_path)
    
    logger.info(f"✅ Saved {len(all_labeled_data):,} merged samples to {output_path}")
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("Merge Summary:")
    logger.info(f"  Board size: {board_size}x{board_size}")
    logger.info(f"  Source files: {len(files)}")
    logger.info(f"  Years: {sorted(set(years_processed)) if years_processed else 'all'}")
    logger.info(f"  Total samples: {len(all_labeled_data):,}")
    logger.info(f"  Duplicates removed: {duplicates if validate else 0}")
    logger.info("="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge labeled datasets')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing labeled datasets')
    parser.add_argument('--output', type=str, required=True,
                        help='Output merged dataset file (.pt)')
    parser.add_argument('--board-size', type=int, required=True,
                        choices=[9, 13, 19],
                        help='Board size to merge')
    parser.add_argument('--years', type=int, nargs='+', default=None,
                        help='Specific years to merge (default: all)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip validation and duplicate filtering')
    
    args = parser.parse_args()
    
    merge_labeled_datasets(
        input_dir=args.input,
        output_path=args.output,
        board_size=args.board_size,
        years=args.years,
        validate=not args.no_validate
    )

