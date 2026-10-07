"""Synthetic DICOM geometry/intensity tests; skip without optional ML packages."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee.dicom import PLANES, StudyLoadError, load_study


_HAS_ML = all(importlib.util.find_spec(name) for name in ("numpy", "pydicom", "torch"))


@unittest.skipUnless(_HAS_ML, "optional numpy/pydicom/torch packages unavailable")
class DicomTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import pydicom
        self.np, self.pydicom = np, pydicom
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = "1.2.3"

    def row(self, series="1.2.4", plane="Sagittal", fluid=True, fat=True):
        return dict(StudyInstanceUID=self.study, SeriesInstanceUID=series,
                    Anatomical_Plane=plane, Fluid_Sensitive=fluid, Fat_Suppression=fat)

    def slice(self, series="1.2.4", index=0, plane=0, pixels=None, **changes):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
        np = self.np
        directory = self.root / self.study / series
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{index}.dcm"
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = self.study, series
        ds.PatientID = "synthetic-person"
        ds.Modality = "MR"
        ds.ImageOrientationPatient = [[0, 1, 0, 0, 0, -1], [1, 0, 0, 0, 0, -1],
                                      [1, 0, 0, 0, 1, 0]][plane]
        position = [0, 0, 0]
        position[plane] = float(index)
        ds.ImagePositionPatient = position
        ds.InstanceNumber = 99 - int(index)
        ds.PixelSpacing = [1, 1]
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SamplesPerPixel = 1
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        if pixels is None:
            pixels = np.arange(16, dtype=np.int16).reshape(4, 4) + int(index) * 16
        pixels = np.asarray(pixels, dtype=np.int16)
        ds.Rows, ds.Columns = pixels.shape
        ds.PixelData = pixels.tobytes()
        for key, value in changes.items():
            if value is None:
                delattr(ds, key)
            else:
                setattr(ds, key, value)
        ds.save_as(path, enforce_file_format=True)
        return path

    def load(self, rows=None, **kwargs):
        return load_study(self.root, self.study, rows or [self.row()],
                          image_size=kwargs.pop("image_size", 4),
                          crop_mm=kwargs.pop("crop_mm", 4),
                          slices_per_plane=kwargs.pop("slices_per_plane", 4), **kwargs)

    def test_physical_order_overrides_filename_and_instance_number(self):
        for index, position in [(0, 2), (1, 0), (2, 1)]:
            self.slice(index=index, pixels=self.np.arange(16).reshape(4, 4) + position * 16,
                       ImagePositionPatient=[position, 0, 0])
        result = self.load()
        means = result.images[0, :3].mean(axis=(1, 2))
        self.assertTrue(self.np.all(self.np.diff(means) > 0))
        self.assertEqual(result.slice_mask.tolist(), [[True, True, True, False], [False] * 4, [False] * 4])
        self.assertEqual(result.plane_mask.tolist(), [True, False, False])
        self.assertEqual(result.images.dtype, self.np.float32)
        self.assertEqual(int(self.np.count_nonzero(result.images[~result.slice_mask])), 0)

    def test_decode_bound_and_unique_sample_positions(self):
        for i in range(10):
            self.slice(index=i)
        with patch.object(self.pydicom, "dcmread", wraps=self.pydicom.dcmread) as read:
            result = self.load(slices_per_plane=3)
        decoded = [c for c in read.call_args_list if not c.kwargs.get("stop_before_pixels", False)]
        self.assertEqual(len(decoded), 3)
        self.assertEqual([Path(c.args[0]).stem for c in decoded], ["0", "4", "9"])
        self.assertEqual(result.audit["counts"]["headers_read"], 10)
        self.assertEqual(result.audit["counts"]["pixel_decode_attempts"], 3)

    def test_duplicate_positions_do_not_create_extra_valid_slices(self):
        for i, pos in enumerate([0, 0, 1]):
            self.slice(index=i, ImagePositionPatient=[pos, 0, 0])
        result = self.load()
        self.assertEqual(result.audit["valid_slices"], 2)
        self.assertEqual(result.audit["counts"]["duplicate_positions_removed"], 1)

    def test_series_preference_deterministic_and_other_studies_ignored(self):
        self.slice(series="1.2.5", pixels=self.np.arange(16).reshape(4, 4).T)
        self.slice()
        rows = [self.row(), self.row("1.2.5", fluid="True", fat="False")]
        rows[0]["Fluid_Sensitive"], rows[0]["Fat_Suppression"] = "true", "true"
        unrelated = {**self.row("1.2.6"), "StudyInstanceUID": "9.9"}
        a, b = self.load(rows + [unrelated]), self.load(list(reversed(rows)))
        self.np.testing.assert_array_equal(a.images, b.images)
        self.np.testing.assert_array_equal(a.images, self.load([self.row()]).images)
        self.assertEqual(a.audit["candidate_series"], 2)
        self.assertEqual(a.audit["counts"]["series_attempted"], 1)

    def test_mixed_orientation_rejected_and_next_series_used(self):
        self.slice(index=0)
        self.slice(index=1, ImageOrientationPatient=[0, -1, 0, 0, 0, -1])
        self.slice(series="1.2.5")
        result = self.load([self.row(), self.row("1.2.5", fat=False)])
        self.assertEqual(result.audit["errors"], {"inconsistent_orientation": 1})
        self.assertEqual(result.audit["counts"]["pixel_decode_attempts"], 1)
        self.assertEqual(result.audit["valid_slices"], 1)

    def test_canonical_transpose_and_flip_keep_anatomical_pixels_aligned(self):
        np = self.np
        pixels = np.arange(16).reshape(4, 4)
        self.slice(pixels=pixels)
        expected = self.load()
        # Swapped width/height, with original height direction reversed.
        self.slice(series="1.2.5", pixels=pixels[::-1].T,
                   ImageOrientationPatient=[0, 0, 1, 0, 1, 0])
        actual = self.load([self.row("1.2.5")])
        np.testing.assert_allclose(actual.images, expected.images)

    def test_anisotropic_physical_crop_and_padding(self):
        from knee.dicom import _physical_crop
        np = self.np
        raw = np.arange(24).reshape(4, 6)
        cropped = _physical_crop(raw, np.asarray([2, 1]), 4, np)
        np.testing.assert_array_equal(cropped, raw[1:3, 1:5])
        padded = _physical_crop(raw, np.asarray([1, 1]), 8, np)
        self.assertEqual(padded.shape, (8, 8))
        np.testing.assert_array_equal(padded[2:6, 1:7], raw)
        self.assertEqual(float(padded[:2].sum()), 0)
        self.slice(PixelSpacing=[2, 1])
        loaded = self.load(image_size=8)
        self.assertEqual(loaded.images.shape, (3, 4, 8, 8))
        self.assertTrue(np.isfinite(loaded.images).all())

    def test_rescale_and_monochrome_inversion(self):
        self.slice()
        original = self.load()
        self.slice(RescaleSlope=2, RescaleIntercept=-100)
        scaled = self.load()
        self.np.testing.assert_allclose(original.images, scaled.images, atol=1e-6)
        self.slice(PhotometricInterpretation="MONOCHROME1")
        inverted = self.load()
        self.np.testing.assert_allclose(original.images[0, 0] + inverted.images[0, 0], 1, atol=1e-6)
        self.slice(RescaleSlope=-2, RescaleIntercept=100)
        negative = self.load()
        self.np.testing.assert_allclose(negative.images[0, 0], inverted.images[0, 0], atol=1e-6)

    def test_dicom_padding_excluded_from_percentiles_and_stays_black(self):
        np = self.np
        pixels = np.arange(16).reshape(4, 4)
        pixels[0] = -1000
        self.slice(pixels=pixels, PixelPaddingValue=-1000, PhotometricInterpretation="MONOCHROME1")
        result = self.load()
        np.testing.assert_array_equal(result.images[0, 0, 0], 0)
        self.assertAlmostEqual(float(result.images[0, 0, 1, 0]), 1)
        self.assertAlmostEqual(float(result.images[0, 0, -1, -1]), 0)

    def test_all_planes_and_patient_hash_without_raw_identifier_in_audit(self):
        rows = []
        for plane, name in enumerate(PLANES):
            series = f"1.2.{4 + plane}"
            self.slice(series=series, plane=plane)
            rows.append(self.row(series, plane=name.title()))
        result = self.load(rows)
        self.assertEqual(result.plane_mask.tolist(), [True, True, True])
        digest = hashlib.sha256(b"knee-patient-v1:synthetic-person").hexdigest()
        self.assertEqual(result.patient_hashes, (digest,))
        audit = json.dumps(result.audit)
        for sensitive in (self.study, "synthetic-person", digest, str(self.root)):
            self.assertNotIn(sensitive, audit)

    def test_invalid_geometry_and_unsupported_images_fail_with_sanitized_codes(self):
        cases = [({"PixelSpacing": [0, 1]}, "missing_or_invalid_spacing"),
                 ({"ImagePositionPatient": None}, "missing_or_invalid_position"),
                 ({"ImageOrientationPatient": [1, 0, 0, 0, 1, 0]}, "metadata_geometry_plane_mismatch"),
                 ({"ImageOrientationPatient": [0, 1, 0, 0, 1, 0]}, "nonorthonormal_orientation"),
                 ({"NumberOfFrames": 2}, "multiframe_not_supported"),
                 ({"PhotometricInterpretation": "RGB"}, "unsupported_photometric_interpretation")]
        for attrs, code in cases:
            with self.subTest(code=code):
                self.slice(**attrs)
                with self.assertRaises(StudyLoadError) as error:
                    self.load()
                self.assertEqual(error.exception.audit["errors"], {code: 1})
                self.assertNotIn(self.study, str(error.exception))

    def test_corrupt_pixels_fail_plane_without_silent_repeated_slices(self):
        path = self.slice()
        ds = self.pydicom.dcmread(path)
        ds.PixelData = b"\0\0"
        ds.save_as(path, enforce_file_format=True)
        with self.assertRaises(StudyLoadError) as error:
            self.load()
        self.assertEqual(error.exception.audit["errors"], {"pixel_decode_failed": 1})
        self.assertEqual(error.exception.audit["valid_slices"], 0)

    def test_header_bound_and_missing_planes_fail_clearly(self):
        self.slice(index=0)
        self.slice(index=1)
        with self.assertRaises(StudyLoadError) as error:
            self.load(max_files_per_series=1)
        self.assertEqual(error.exception.audit["errors"], {"series_exceeds_header_bound": 1})
        with self.assertRaises(StudyLoadError) as error:
            self.load([self.row("9.8.7")])
        self.assertEqual(error.exception.audit["missing_planes"], 3)

    def test_constant_pixels_and_bad_parameters_rejected(self):
        self.slice(pixels=self.np.ones((4, 4)))
        with self.assertRaises(StudyLoadError) as error:
            self.load()
        self.assertEqual(error.exception.audit["errors"], {"constant_intensity_series": 1})
        for kwargs in ({"crop_mm": float("nan")}, {"image_size": 0}, {"slices_per_plane": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.load(**kwargs)


if __name__ == "__main__":
    unittest.main()
