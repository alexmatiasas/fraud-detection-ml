import kagglehub
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
COMPETITION = "ieee-fraud-detection"


def download() -> None:
    print(f"Downloading {COMPETITION} from Kaggle...")
    cache_path = Path(kagglehub.competition_download(COMPETITION))

    print(f"Copying files to {RAW_DIR}...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for file in cache_path.glob("*"):
        dest = RAW_DIR / file.name
        if not dest.exists():
            shutil.copy2(file, dest)
            print(f"  Copied {file.name}")
        else:
            print(f"  Skipped {file.name} (already exists)")

    print("Done.")


if __name__ == "__main__":
    download()
