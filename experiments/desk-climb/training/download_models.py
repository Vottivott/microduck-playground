from pathlib import Path
import shutil,hashlib,json
from huggingface_hub import hf_hub_download
P=Path(__file__).parent;root=P.parent;dest=P/'models';dest.mkdir(exist_ok=True)
manifest=json.loads((root/'models.json').read_text())
for name,entry in manifest.items():
 src=hf_hub_download('HannesVonEssen/microduck-climb','models/'+name);assert hashlib.sha256(Path(src).read_bytes()).hexdigest()==entry['sha256'];shutil.copy2(src,dest/name)
# The recorded evaluator accepts GETUP_FILE. Keep the canonical name explicit.
print('Verified all four model files in',dest)
