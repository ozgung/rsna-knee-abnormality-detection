"""Small private preprocessing QA gallery; no identifiers or report text drawn."""
import json
from pathlib import Path
from knee.runtime import competition_root, prepare_codecs

prepare_codecs("/kaggle/working/wheels", download=True)
import numpy as np
import torch
from PIL import Image, ImageDraw
from knee.dicom import PLANES, load_study
from knee.validation import read_csv, read_schema

torch.set_num_threads(2)
root = competition_root()
id_column, targets = read_schema(root / "sample_submission.csv")
_, rows = read_csv(root / "train.csv")
_, series = read_csv(root / "train_series.csv")
selected = sorted([r for r in rows if all(r[t].strip() for t in targets)], key=lambda r: r[id_column])[:2]
summary = []
for study_index, row in enumerate(selected):
    uid = row[id_column]
    volume = load_study(root / "train_series", uid, [s for s in series if s[id_column] == uid],
                        image_size=224, slices_per_plane=8, crop_mm=160)
    tile, gap, left, top = 140, 6, 85, 55
    canvas = Image.new("RGB", (left + 8*(tile+gap), top + 3*(tile+gap) + 45), "#0f172a")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), f"MRI preprocessing QA — example {study_index+1}", fill="white")
    draw.text((12, 30), "Geometric slice order | 160 mm crop | 224 px encoder input | no lesion localization", fill="#cbd5e1")
    for plane_index, plane in enumerate(PLANES):
        draw.text((8, top + plane_index*(tile+gap) + 60), plane, fill="#cbd5e1")
        for slice_index in range(8):
            x, y = left + slice_index*(tile+gap), top + plane_index*(tile+gap)
            pixels = np.clip(volume.images[plane_index, slice_index]*255, 0, 255).astype(np.uint8)
            canvas.paste(Image.fromarray(pixels).resize((tile,tile)), (x,y))
            if not volume.slice_mask[plane_index, slice_index]:
                draw.text((x+10,y+60), "MISSING / PAD", fill="#fb7185")
    draw.text((12, canvas.height-30), "Private competition imagery; patient identifiers and report text omitted.", fill="#cbd5e1")
    canvas.save(f"/kaggle/working/preprocessing-example-{study_index+1}.png")
    summary.append({"example":study_index+1, "counts":volume.audit["counts"],
                    "errors":volume.audit["errors"], "missing_planes":volume.audit["missing_planes"]})
Path("/kaggle/working/gallery-audit.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary), flush=True)
