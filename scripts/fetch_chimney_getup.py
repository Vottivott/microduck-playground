"""Download the pinned official recovery dependency and verify its digest."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from huggingface_hub import hf_hub_download

manifest, destination = map(Path, sys.argv[1:3])
dep = json.loads(manifest.read_text())["official_getup"]
source = Path(hf_hub_download(dep["repo_id"], dep["filename"], revision=dep["revision"]))
assert hashlib.sha256(source.read_bytes()).hexdigest() == dep["sha256"]
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, destination)
print(destination)
