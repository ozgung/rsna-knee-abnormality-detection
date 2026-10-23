"""Lane D2: write P2's payload exactly as P2's notebook bootstrap does (payload sha256 77e65cea..., verified) and check every
pinned asset (codecs, R6 manifest + assets, anatomy, Raptor checkpoints) by sha256.  Uses /content/p2_payload.json."""
import hashlib, json
from pathlib import Path
payload = json.load(open("/content/p2_payload.json"))
assert hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() == \
    "77e65ceaa47624c45578656dc1e600417655577fcd3ee1c99ccb5c83df5a9cad", "P2 payload changed"
project = Path("/kaggle/working/r6flip-project"); code = project / "source"
for name, content in payload["sources"].items():
    rel = Path(name); assert not rel.is_absolute() and ".." not in rel.parts
    p = project / name if name.startswith("knee/") else code / name
    p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(content.encode())
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
I = Path("/kaggle/input/datasets"); cfg = payload["cfg"]; b6 = cfg["r6flip_binding"]
for n, pin in payload["codecs"].items():
    assert sha(I / "goodpjw2008/rsna-knee-2-5d-convnext-reader" / n) == pin, n
r6 = I / "heliosli/rsna-knee-r6-frozen-fe055406-w1"
assert sha(r6 / "manifest.json") == b6["r6"]["manifest_sha256"]
for n, pin in b6["r6"]["asset_sha256"].items():
    assert sha(r6 / n) == pin, n
assert sha(I / "heliosli/rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt") == b6["r6"]["manifest"]["anchor_weights_sha256"]
print("P2_PAYLOAD_WRITTEN sources", len(payload["sources"]), "assets verified")
