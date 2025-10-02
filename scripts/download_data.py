#!/usr/bin/env python3
"""
Kaggleからデータをダウンロード

Usage:
    python -m scripts.download_data --competition titanic --output data/raw
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def download_kaggle_data(competition: str, output_dir: str):
    """Kaggle APIでデータをダウンロード"""
    logger = setup_logging()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {competition} data to {output_dir}")

    try:
        # Kaggle APIでデータダウンロード
        cmd = f"kaggle competitions download -c {competition} -p {output_dir}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Download failed: {result.stderr}")
            sys.exit(1)

        logger.info("Download completed successfully")

        # ZIPファイルを展開
        import zipfile

        zip_files = list(output_path.glob("*.zip"))

        for zip_file in zip_files:
            logger.info(f"Extracting {zip_file}")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(output_path)
            zip_file.unlink()  # ZIP削除

        # ダウンロードされたファイルをリスト
        downloaded_files = list(output_path.glob("*"))
        logger.info(f"Downloaded files: {[f.name for f in downloaded_files]}")

        return True

    except Exception as e:
        logger.error(f"Error during download: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle competition data")
    parser.add_argument("--competition", required=True, help="Competition name")
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()

    success = download_kaggle_data(args.competition, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
