"""Create single ZIP per course submission instructions."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
root=Path(__file__).resolve().parents[1]
output=root/"submission"/"IMDb_Movie_Reviews_NLP_Submission.zip"
output.parent.mkdir(exist_ok=True)
include=["README.md","pyproject.toml","app.py",".gitignore"]
folders=["src","configs","tests","scripts","report","results",".github"]
files=[root/x for x in include]+[f for folder in folders for f in (root/folder).rglob("*") if f.is_file()]
with ZipFile(output,"w",ZIP_DEFLATED) as z:
    for f in files:
        if "__pycache__" in f.parts or f.suffix==".pyc":continue
        z.write(f,arcname=Path("Analysis_on_Movie_Reviews")/f.relative_to(root))
print(f"Created: {output} ({output.stat().st_size:,} bytes)")