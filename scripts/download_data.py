"""
Script designed to download data from Kaggle using their API
It needs to be have an API key in `$HOME/.kaggle/kaggle.json`.

ref: https://www.kaggle.com/docs/api
"""

import kagglehub
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
COMPETITION = "ieee-fraud-detection"


def download(competition: str = COMPETITION, out_dir: Path | str = RAW_DIR) -> None:
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)

    print(f"Downloading {competition} from Kaggle...")
    cache_path = Path(kagglehub.competition_download(competition))

    print(f"Copying files to {out_dir}...")
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in cache_path.glob("*"):
        dest = out_dir / file.name
        if not dest.exists():
            shutil.copy2(file, dest)
            print(f"  Copied {file.name}")
        else:
            print(f"  Skipped {file.name} (already exists)")

    print("Done.")


if __name__ == "__main__":
    download()
