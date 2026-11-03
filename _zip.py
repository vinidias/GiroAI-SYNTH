import zipfile
from pathlib import Path

root = Path("D:/Dev/giro-ai/giro-synth")
zip_path = root / "giro-synth-v0.1.0.zip"

include_dirs = ["data/operational", "data/ground_truth", "data/macro"]
include_files = ["README.md", "KAGGLE_DATASET_CARD.md", "LICENSE", "LICENSE-DATA.md",
                 "manifesto.csv", "dicionario.csv", "schema.json", "CHECKSUMS.sha256",
                 "notebooks/01_verify_signals.ipynb"]

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for d in include_dirs:
        for f in sorted((root / d).glob("*.csv")):
            z.write(f, f.relative_to(root))
    for f in include_files:
        p = root / f
        if p.exists():
            z.write(p, f)

print(f"zip: {zip_path}")
print(f"size: {zip_path.stat().st_size / 1e6:.1f} MB")