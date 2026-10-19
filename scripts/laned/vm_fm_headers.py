"""Lane D (VM): read the header of every file under the extracted fastMRI tree (no pixels), group by study/series and write
  /content/fm_meta/fm_series_headers.csv  one row per series with the classifier's header fields + orientation + counts
  /content/fm_meta/fm_files.json          VM-only map series -> sorted file paths (never leaves the VM)
Usage: python -I vm_fm_headers.py <extracted root>"""
import csv, json, os, sys
from collections import defaultdict
from multiprocessing import Pool

FIELDS = ["PatientSex", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "SoftwareVersions", "ReceiveCoilName",
          "TransmitCoilName", "ContrastBolusAgent", "SeriesDescription", "ProtocolName", "Laterality", "ImageLaterality",
          "BodyPartExamined", "SliceThickness", "SpacingBetweenSlices", "PixelSpacing", "RepetitionTime", "EchoTime",
          "InversionTime", "FlipAngle", "EchoTrainLength", "Rows", "Columns", "ScanningSequence", "SequenceVariant", "ScanOptions",
          "MRAcquisitionType", "ImagingFrequency", "PixelBandwidth", "NumberOfAverages", "PercentPhaseFieldOfView",
          "AcquisitionMatrix", "InPlanePhaseEncodingDirection", "PatientPosition", "ImageOrientationPatient", "ImageType",
          "Modality", "SOPClassUID", "NumberOfFrames"]


def val(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)) or type(v).__name__ in ("MultiValue",):
        return "|".join(str(x) for x in v)
    return str(v)


def one(path):
    import pydicom
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        st, se = str(ds.get("StudyInstanceUID", "")), str(ds.get("SeriesInstanceUID", ""))
        if not st or not se:
            return None
        rec = {k: val(ds.get(k)) for k in FIELDS}
        ipp = ds.get("ImagePositionPatient")
        return st, se, path, rec, (val(ipp) if ipp is not None else ""), str(ds.get("InstanceNumber", ""))
    except Exception:
        return None


def main(root):
    files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
    print("files", len(files), flush=True)
    series, recs, bad = defaultdict(list), {}, 0
    with Pool(os.cpu_count()) as p:
        for r in p.imap_unordered(one, files, chunksize=256):
            if r is None:
                bad += 1; continue
            st, se, path, rec, ipp, inst = r
            series[(st, se)].append((path, ipp, inst))
            recs.setdefault((st, se), rec)
    os.makedirs("/content/fm_meta", exist_ok=True)
    cols = ["StudyInstanceUID", "SeriesInstanceUID", "n_files"] + FIELDS
    with open("/content/fm_meta/fm_series_headers.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for (st, se), lst in sorted(series.items()):
            w.writerow([st, se, len(lst)] + [recs[(st, se)][k] for k in FIELDS])
    json.dump({"%s/%s" % k: sorted(p for p, _, _ in v) for k, v in series.items()}, open("/content/fm_meta/fm_files.json", "w"))
    print("series", len(series), "studies", len({k[0] for k in series}), "unreadable", bad, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
