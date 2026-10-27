"""
Script to download real-world sample 10-K financial reports.
SEC EDGAR requires a custom User-Agent in the header (Sample: 'SampleUser user@example.com').
"""

import os
from pathlib import Path
import requests

DATA_RAW_DIR = Path(__file__).resolve().parent / "raw"

# Sample 10-K PDFs hosted on public GitHub repositories
SAMPLES = {
    "apple_10k_2023.pdf": "https://raw.githubusercontent.com/dharavathramdas101/Agentic-financial-document-audit/main/data/aapl-20230930.pdf",
}

HEADERS = {
    "User-Agent": "FinancialRAGPortfolioProject/1.0 (contact: student_ds@example.com)"
}

def download_file(url: str, dest_path: Path):
    print(f"Downloading: {url} -> {dest_path.name}...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Successfully downloaded {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        print("Note: You can also manually drop any corporate financial report (PDF) into data/raw/")

def main():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SAMPLES.items():
        dest = DATA_RAW_DIR / filename
        if not dest.exists():
            download_file(url, dest)
        else:
            print(f"File already exists: {dest.name}")

if __name__ == "__main__":
    main()
