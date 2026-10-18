"""Synthetic-only tests for the label-blind DICOM orientation audit (no real DICOM, labels or reports)."""
import csv
import importlib.util
import glob as glob_module
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest import mock

from knee import orientation_audit as oa

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_orientation_audit', ROOT / 'scripts/build_orientation_audit.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
PUBLIC = Path(os.environ.get('ORIENTATION_AUDIT_PUBLIC_SOURCE', ROOT / 'outputs/research/2026-10-08/raptor-audit/'
              'knee-mri-twelve-findings-from-a-single-model.py'))
HAS_PYDICOM = importlib.util.find_spec('pydicom') is not None
HAS_NUMPY = importlib.util.find_spec('numpy') is not None
TH = oa.thresholds()
STANDARD = {'Sagittal': (0, 1, 0, 0, 0, -1), 'Coronal': (1, 0, 0, 0, 0, -1), 'Axial': (1, 0, 0, 0, 1, 0)}
SLOTS = [('Sagittal', 1, 26), ('Sagittal', 0, 22), ('Coronal', 1, 18), ('Coronal', 0, 12), ('Axial', -1, 18)]


def rotate(v, axis, degrees):
    """Rodrigues rotation of v about a unit axis."""
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    k = axis
    kxv = oa._cross(k, v)
    kdv = oa._dot(k, v)
    return tuple(v[i] * c + kxv[i] * s + k[i] * kdv * (1 - c) for i in range(3))


def header(iop, ipp, spacing=(0.5, 0.5), number=1):
    ns = SimpleNamespace(InstanceNumber=number)
    if iop is not None:
        ns.ImageOrientationPatient = list(iop)
    if ipp is not None:
        ns.ImagePositionPatient = list(ipp)
    if spacing is not None:
        ns.PixelSpacing = list(spacing)
    return ns


def instances(iop, n=6, step=3.0, reverse=False, **drop):
    """Headers of a stack along the series normal; optional tag omission through no_iop / no_ipp / no_spacing"""
    normal = oa._cross(iop[:3], iop[3:])
    out = []
    for k in range(n):
        ipp = tuple(step * k * normal[i] + (10.0, 20.0, 30.0)[i] for i in range(3))
        out.append(header(None if drop.get('no_iop') else iop, None if drop.get('no_ipp') else ipp,
                          None if drop.get('no_spacing') else (0.5, 0.5), number=k + 1))
    return [oa.extract_instance(h, TH) for h in (out[::-1] if reverse else out)]


def analyze(plane, iop, **kw):
    return oa.analyze_series(instances(iop, **kw), plane, TH)


class GeometryTests(unittest.TestCase):
    def test_dominant_axis_and_sign_in_lps(self):
        self.assertEqual(oa.dominant_axis((-1, 0, 0))[:2], ('x', -1))
        self.assertEqual(oa.SIGNED_LABELS[('x', 1)], 'L')
        self.assertEqual(oa.SIGNED_LABELS[('x', -1)], 'R')
        self.assertEqual(oa.SIGNED_LABELS[('y', 1)], 'P')
        self.assertEqual(oa.SIGNED_LABELS[('z', 1)], 'S')
        axis, sign, mag, second = oa.dominant_axis((0.05, 0.1, -0.99))
        self.assertEqual((axis, sign), ('z', -1))
        self.assertAlmostEqual(mag, 0.99)
        self.assertAlmostEqual(second, 0.1)
        self.assertEqual(oa.dominant_axis((0.5, 0.5, 0.0))[0], 'x')  # tie: deterministic x < y < z

    def test_class_thresholds_are_cos20_and_cos45_and_recorded(self):
        self.assertAlmostEqual(TH['standard_min_cosine'], math.cos(math.radians(20)))
        self.assertAlmostEqual(TH['oblique_min_cosine'], math.cos(math.radians(45)))
        c = lambda deg: oa.classify_component(math.cos(math.radians(deg)), TH)
        self.assertEqual([c(0), c(20), c(20.5), c(44.9), c(45), c(45.5), c(80)],
                         ['standard', 'standard', 'oblique', 'oblique', 'oblique', 'nonstandard', 'nonstandard'])
        loose = oa.thresholds(standard_degrees=30.0)
        self.assertEqual(oa.classify_component(math.cos(math.radians(25)), loose), 'standard')
        with self.assertRaises(ValueError):
            oa.thresholds(standard_degrees=50.0)
        with self.assertRaises(ValueError):
            oa.thresholds(unknown=1)

    def test_standard_iop_vectors_for_each_plane(self):
        expected = {'Sagittal': ('P', 'I', 'R'), 'Coronal': ('L', 'I', 'P'), 'Axial': ('L', 'P', 'S')}
        for plane, labels in expected.items():
            r = analyze(plane, STANDARD[plane])
            self.assertEqual(tuple(r['vectors'][v]['label'] for v in oa.VECTORS), labels, plane)
            self.assertEqual(r['series_class'], 'standard')
            self.assertEqual(r['mirror']['status'], 'ok_standard', plane)
            self.assertEqual(r['mirror']['axis'], 'x')
            self.assertEqual(r['order'], 'ascending')
            self.assertFalse(any(r['flags'][f] for f in oa.FLAGS_ROUTING))
        self.assertEqual(analyze('Sagittal', STANDARD['Sagittal'])['mirror']['vector'], 'normal')
        self.assertEqual(analyze('Coronal', STANDARD['Coronal'])['mirror']['vector'], 'row')

    def test_sign_does_not_change_the_mirror_decision_but_is_reported(self):
        pos = analyze('Sagittal', (0, -1, 0, 0, 0, -1))   # normal +x (Left)
        neg = analyze('Sagittal', STANDARD['Sagittal'])   # normal -x (Right)
        self.assertEqual((pos['mirror']['label'], neg['mirror']['label']), ('L', 'R'))
        self.assertEqual({pos['mirror']['status'], neg['mirror']['status']}, {'ok_standard'})
        flipped_row = analyze('Coronal', (-1, 0, 0, 0, 0, -1))
        self.assertEqual((flipped_row['mirror']['label'], flipped_row['mirror']['status']), ('R', 'ok_standard'))
        flipped_col = analyze('Axial', (1, 0, 0, 0, -1, 0))
        self.assertEqual((flipped_col['mirror']['label'], flipped_col['mirror']['status']), ('L', 'ok_standard'))
        self.assertEqual(flipped_col['vectors']['normal']['label'], 'I')

    def test_sorted_ipp_order_runs_along_positive_normal_even_for_reversed_input(self):
        for reverse in (False, True):
            self.assertEqual(analyze('Axial', STANDARD['Axial'], reverse=reverse)['order'], 'ascending')
        self.assertEqual(analyze('Axial', STANDARD['Axial'], n=1)['order'], 'degenerate')
        self.assertEqual(analyze('Axial', STANDARD['Axial'], step=0.0)['order'], 'degenerate')
        self.assertEqual(analyze('Axial', STANDARD['Axial'], no_ipp=True)['order'], 'unavailable')

    def test_degenerate_unavailable_or_ambiguous_order_never_qualifies_for_the_mirror(self):
        for plane in oa.PLANES:
            for name, kw, order in (('identical IPP', {'step': 0.0}, 'degenerate'), ('one slice', {'n': 1}, 'degenerate'),
                                    ('no IPP', {'no_ipp': True}, 'unavailable')):
                r = analyze(plane, STANDARD[plane], **kw)
                self.assertEqual((r['order'], r['mirror']['status'], r['route']), (order, 'geometry_flagged', 'bypass_plain'),
                                 (plane, name))
            self.assertEqual(analyze(plane, STANDARD[plane])['route'], 'mirror_ok')       # control
        tie = instances(STANDARD['Sagittal'])
        tie.append(tie[2])                                      # a second slice at an equal reader position
        r = oa.analyze_series(tie, 'Sagittal', TH)
        self.assertEqual((r['order'], r['mirror']['status'], r['route']), ('ambiguous', 'geometry_flagged', 'bypass_plain'))
        transposed = analyze('Coronal', (0, 0, 1, 1, 0, 0), step=0.0)       # also never routed to a transform
        self.assertEqual(transposed['route'], 'bypass_plain')

    def test_coronal_with_rows_along_y_or_z_fails_the_assumption(self):
        along_z = analyze('Coronal', (0, 0, 1, 1, 0, 0))     # rows +z, columns +x: normal +y (coronal-like stack)
        along_y = analyze('Coronal', (0, 1, 0, 0, 0, -1))    # rows +y: sagittal geometry labelled coronal
        for r, axis in ((along_z, 'z'), (along_y, 'y')):
            self.assertEqual((r['mirror']['axis'], r['mirror']['status']), (axis, 'fail_axis'))
            self.assertEqual(r['vectors']['row']['class'], 'standard')   # aligned, but not L/R
        self.assertEqual(analyze('Axial', (0, 1, 0, 1, 0, 0))['mirror']['status'], 'fail_axis')
        self.assertEqual(analyze('Sagittal', STANDARD['Axial'])['mirror']['status'], 'fail_axis')  # normal is z

    def test_oblique_series_is_counted_apart_from_standard(self):
        row, col = rotate((1, 0, 0), (0, 1, 0), 30), rotate((0, 0, -1), (0, 1, 0), 30)
        r = analyze('Coronal', row + col)
        self.assertEqual((r['mirror']['axis'], r['mirror']['class'], r['mirror']['status']), ('x', 'oblique', 'ok_oblique'))
        self.assertEqual(r['series_class'], 'oblique')
        near = rotate((1, 0, 0), (0, 1, 0), 15), rotate((0, 0, -1), (0, 1, 0), 15)
        self.assertEqual(analyze('Coronal', near[0] + near[1])['mirror']['status'], 'ok_standard')

    def test_nonstandard_with_x_still_dominant_fails_by_class(self):
        row = oa._unit((0.62, 0.55, 0.56))
        col = oa._unit(oa._cross(row, (0, 0, 1)))
        col = oa._unit(oa._cross(oa._cross(row, col), row))
        r = analyze('Axial', row + col)
        self.assertEqual((r['mirror']['axis'], r['mirror']['class'], r['mirror']['status']),
                         ('x', 'nonstandard', 'fail_nonstandard'))

    def test_inconsistent_iop_and_mixed_normal_are_flagged_and_routed(self):
        base = instances(STANDARD['Coronal'])
        tilt = (rotate((1, 0, 0), (0, 1, 0), 2.0), rotate((0, 0, -1), (0, 1, 0), 2.0))
        mixed = base[:3] + instances(tilt[0] + tilt[1])[3:]
        r = oa.analyze_series(mixed, 'Coronal', TH)
        self.assertTrue(r['flags']['inconsistent_iop'])
        self.assertFalse(r['flags']['mixed_normal'])           # 2 degrees: below the 5 degree normal tolerance
        self.assertEqual(r['mirror']['status'], 'geometry_flagged')
        steep = (rotate((1, 0, 0), (0, 0, 1), 12.0), (0, 0, -1))      # rotation about z tilts the normal by 12 degrees
        r = oa.analyze_series(base[:3] + instances(steep[0] + steep[1])[3:], 'Coronal', TH)
        self.assertTrue(r['flags']['mixed_normal'])
        self.assertEqual(r['mirror']['status'], 'geometry_flagged')
        flipped = oa.extract_instance(header((-1, 0, 0, 0, 0, -1), (0, 0, 0)), TH)   # normal reversed
        r = oa.analyze_series(base + [flipped], 'Coronal', TH)
        self.assertTrue(r['flags']['mixed_normal'] and r['flags']['inconsistent_iop'])
        # A tolerance below the actual noise flags; the recorded default does not flag 1e-5 jitter.
        jitter = [oa.extract_instance(header(tuple(v + 1e-5 * (k % 2) for v in STANDARD['Coronal'][:3]) + STANDARD['Coronal'][3:],
                                             (0, 3.0 * k, 0)), TH) for k in range(5)]
        self.assertFalse(oa.analyze_series(jitter, 'Coronal', TH)['flags']['inconsistent_iop'])

    def test_not_ascending_when_normals_disagree_with_the_series_normal(self):
        a = instances(STANDARD['Coronal'], n=4)
        b = [oa.extract_instance(header((-1, 0, 0, 0, 0, -1), (0, 3.0 * k + 100, 0)), TH) for k in range(2)]
        # b's own normal is -y, so the reader sorts by IPP . (-y): its key order reverses against series normal +y.
        r = oa.analyze_series(a + b, 'Coronal', TH)
        self.assertEqual(r['order'], 'not_ascending')
        self.assertEqual(r['mirror']['status'], 'geometry_flagged')

    def test_missing_tags_are_flagged_and_never_silently_standard(self):
        none = oa.analyze_series(instances(STANDARD['Coronal'], no_iop=True), 'Coronal', TH)
        self.assertEqual((none['mirror']['status'], none['vectors'], none['flags']['missing_iop']),
                         ('unverifiable', None, True))
        partial = instances(STANDARD['Coronal'])
        partial[2] = oa.extract_instance(header(None, (0, 6.0, 0)), TH)
        r = oa.analyze_series(partial, 'Coronal', TH)
        self.assertTrue(r['flags']['missing_iop'])
        self.assertEqual(r['mirror']['status'], 'geometry_flagged')
        r = oa.analyze_series(instances(STANDARD['Coronal'], no_ipp=True), 'Coronal', TH)
        self.assertTrue(r['flags']['missing_ipp'])
        self.assertEqual((r['order'], r['mirror']['status']), ('unavailable', 'geometry_flagged'))
        r = oa.analyze_series(instances(STANDARD['Coronal'], no_spacing=True), 'Coronal', TH)
        self.assertTrue(r['flags']['missing_pixel_spacing'])
        self.assertEqual((r['mirror']['status'], r['route']), ('geometry_flagged', 'bypass_plain'))   # complete-geometry contract
        bad = [oa.extract_instance(header((1, 0, 0, 1, 0, 0), (0, 0, 0)), TH)]   # row == col: not a valid IOP
        r = oa.analyze_series(bad, 'Coronal', TH)
        self.assertEqual((r['flags']['invalid_iop'], r['mirror']['status']), (True, 'unverifiable'))
        short = oa.extract_instance(SimpleNamespace(ImageOrientationPatient=[1, 0, 0], ImagePositionPatient=[0, 0, 0]), TH)
        self.assertEqual(short['iop_state'], 'invalid')
        r = oa.analyze_series([], 'Axial', TH)
        self.assertEqual((r['flags']['no_files'], r['mirror']['status']), (True, 'unverifiable'))
        r = oa.analyze_series(instances(STANDARD['Axial']) + [{'error': 'InvalidDicomError'}], 'Axial', TH)
        self.assertEqual((r['flags']['unreadable_header'], r['mirror']['status']), (True, 'geometry_flagged'))

    def test_missing_or_invalid_pixel_spacing_is_bypassed_for_every_plane(self):
        for plane in oa.PLANES:
            self.assertEqual(analyze(plane, STANDARD[plane])['route'], 'mirror_ok')
            self.assertEqual(analyze(plane, STANDARD[plane], no_spacing=True)['route'], 'bypass_plain', plane)
        for bad in ([0.0, 0.5], [-0.5, 0.5], [0.5], [0.5, 0.5, 0.5], 'abc', float('nan')):
            insts = []
            for k in range(6):
                h = header(STANDARD['Axial'], (0, 0, 3.0 * k))
                h.PixelSpacing = bad
                insts.append(oa.extract_instance(h, TH))
            r = oa.analyze_series(insts, 'Axial', TH)
            self.assertTrue(r['flags']['missing_pixel_spacing'], bad)
            self.assertEqual((r['mirror']['status'], r['route']), ('geometry_flagged', 'bypass_plain'), bad)
        one_bad = instances(STANDARD['Axial'])                  # a single bad slice among valid ones is enough
        one_bad[3] = oa.extract_instance(header(STANDARD['Axial'], (0, 0, 9.0), spacing=None), TH)
        r = oa.analyze_series(one_bad, 'Axial', TH)
        self.assertEqual((r['slices']['pixel_spacing_bad'], r['route']), (1, 'bypass_plain'))
        self.assertEqual(analyze('Coronal', (0, 0, 1, 1, 0, 0), no_spacing=True)['route'], 'bypass_plain')  # not a transform

    def test_sampled_or_unobserved_series_can_never_be_mirror_ok(self):
        observed = instances(STANDARD['Sagittal'], n=8)
        self.assertEqual(oa.analyze_series(observed, 'Sagittal', TH, files_total=8)['route'], 'mirror_ok')
        for total in (9, 100):
            r = oa.analyze_series(observed, 'Sagittal', TH, files_total=total)
            self.assertTrue(r['flags']['sampled'])
            self.assertEqual((r['mirror']['status'], r['route']), ('geometry_flagged', 'bypass_plain'))
            r = oa.analyze_series(observed, 'Sagittal', TH, pick=(26, oa.SPAN96), files_total=total)
            self.assertEqual(r['slices']['used'], 0)             # used-slice counts need every instance
        r = oa.analyze_series(instances((0, 0, 1, 1, 0, 0), n=8), 'Coronal', TH, files_total=20)
        self.assertEqual(r['route'], 'bypass_plain')             # a sampled series is not routed to a transform either

    def test_reader_position_matches_the_pinned_sort_key(self):
        h = header((1, 0, 0, 0, 1, 0), (5, 6, 7), number=9)
        self.assertEqual(oa.reader_position(oa.extract_instance(h, TH)), 7.0)       # IPP . (row x col)
        self.assertEqual(oa.reader_position(oa.extract_instance(header(None, (5, 6, 7), number=9), TH)), 9.0)
        self.assertEqual(oa.reader_position({'error': 'x'}), 0.0)

    def test_reader_key_follows_the_pinned_fallbacks(self):
        good = dict(ImageOrientationPatient=[1, 0, 0, 0, 1, 0], ImagePositionPatient=[0, 0, 5.0], InstanceNumber=4)
        cases = [
            ({}, (5.0, 'geometry')),
            ({'PixelSpacing': [0.5, 0.5]}, (5.0, 'geometry')),
            ({'ImagePositionPatient': [0, 5.0]}, (0.0, 'exception')),             # np.dot shape mismatch in the reader
            ({'ImagePositionPatient': [0, 0, 1, 2]}, (0.0, 'exception')),
            ({'ImagePositionPatient': ['a', 'b', 'c']}, (0.0, 'exception')),
            ({'ImageOrientationPatient': [1, 0, 0, 0, 1]}, (4.0, 'instance_number')),   # len(iop) != 6
            ({'ImageOrientationPatient': None}, (4.0, 'instance_number')),
            ({'ImagePositionPatient': None, 'InstanceNumber': None}, (0.0, 'instance_number')),
            ({'ImageOrientationPatient': ['x'] * 6}, (0.0, 'exception')),
            ({'PixelSpacing': 'abc'}, (0.0, 'exception')),                          # ps[0] parse fails after pos was computed
            ({'PixelSpacing': 0.5}, (0.0, 'exception')),
            ({'PixelSpacing': []}, (0.0, 'exception')),
        ]
        for change, expected in cases:
            self.assertEqual(oa.reader_key(SimpleNamespace(**{**good, **change})), expected, change)
        fallback = oa.analyze_series([oa.extract_instance(SimpleNamespace(**{**good, 'PixelSpacing': 'abc'}), TH)] * 3,
                                     'Axial', TH)
        self.assertTrue(fallback['flags']['reader_fallback'])

    def test_any_reader_fallback_alone_routes_to_bypass(self):
        # Defence in depth: even if every other field looks clean, an instance the reader files by fallback
        # (InstanceNumber or the exception position) is not ordered by the geometry we audited.
        for source in ('instance_number', 'exception'):
            series = instances(STANDARD['Sagittal'])
            series[2] = dict(series[2], reader_source=source)
            r = oa.analyze_series(series, 'Sagittal', TH)
            self.assertEqual((r['slices']['reader_fallback'], r['flags']['reader_fallback'], r['route']),
                             (1, True, 'bypass_plain'), source)
            self.assertFalse(r['flags']['missing_iop'] or r['flags']['missing_ipp'] or r['flags']['missing_pixel_spacing'])

    def test_instance_cap_samples_evenly_with_both_ends(self):
        self.assertEqual(oa.sample_indices(5, None), [0, 1, 2, 3, 4])
        self.assertEqual(oa.sample_indices(5, 10), [0, 1, 2, 3, 4])
        picks = oa.sample_indices(100, 5)
        self.assertEqual((picks[0], picks[-1], len(picks)), (0, 99, 5))
        with self.assertRaises(ValueError):
            oa.sample_indices(10, 1)


    def test_per_series_routing_categories(self):
        self.assertEqual(analyze('Coronal', STANDARD['Coronal'])['route'], 'mirror_ok')
        transposed = analyze('Coronal', (0, 0, 1, 1, 0, 0))      # rows +z, columns +x: L/R runs along image height
        self.assertEqual((transposed['route'], transposed['lr_vector']), ('needs_transform', 'col'))
        sagittal_like = analyze('Coronal', STANDARD['Sagittal'])  # coronal label, sagittal geometry: L/R is slice order
        self.assertEqual((sagittal_like['route'], sagittal_like['lr_vector']), ('needs_transform', 'normal'))
        row_x_sagittal = analyze('Sagittal', STANDARD['Axial'])   # sagittal label with L/R along width
        self.assertEqual((row_x_sagittal['route'], row_x_sagittal['lr_vector']), ('needs_transform', 'row'))
        oblique = rotate((1, 0, 0), (0, 1, 0), 30) + rotate((0, 0, -1), (0, 1, 0), 30)
        self.assertEqual(analyze('Coronal', oblique)['route'], 'bypass_plain')
        tilted_transposed = rotate((0, 0, 1), (0, 1, 0), 30) + rotate((1, 0, 0), (0, 1, 0), 30)
        r = analyze('Coronal', tilted_transposed)               # L/R vector exists but is oblique: no exact flip
        self.assertEqual((r['mirror']['status'], r['route']), ('fail_axis', 'bypass_plain'))
        self.assertEqual(analyze('Coronal', STANDARD['Coronal'], no_iop=True)['route'], 'bypass_plain')
        mixed = instances(STANDARD['Coronal'])[:3] + instances((0, 0, 1, 1, 0, 0))[3:]
        r = oa.analyze_series(mixed, 'Coronal', TH)
        self.assertEqual(r['route'], 'bypass_plain')             # inconsistent geometry is never transformed
        self.assertEqual(oa.study_route([]), 'bypass_plain')
        self.assertEqual(oa.study_route(['mirror_ok', 'mirror_ok']), 'mirror_ok')
        self.assertEqual(oa.study_route(['mirror_ok', 'needs_transform']), 'needs_transform')
        self.assertEqual(oa.study_route(['needs_transform', 'bypass_plain', 'mirror_ok']), 'bypass_plain')

    def test_slice_level_counts_do_not_hide_a_minority_behind_a_majority(self):
        base = instances(STANDARD['Coronal'], n=10)
        tilt = instances(rotate((1, 0, 0), (0, 1, 0), 2.0) + rotate((0, 0, -1), (0, 1, 0), 2.0), n=10)
        series = base[:5] + [tilt[5], tilt[6]] + base[7:]       # 2 of 10 slices deviate by 2 degrees
        r = oa.analyze_series(series, 'Coronal', TH)
        self.assertEqual((r['slices']['total'], r['slices']['iop_inconsistent'], r['slices']['routing_flagged']), (10, 2, 2))
        self.assertEqual((r['flags']['inconsistent_iop'], r['mirror']['status'], r['route']),
                         (True, 'geometry_flagged', 'bypass_plain'))
        with_missing = series[:8] + [oa.extract_instance(header(None, (0, 0, 0)), TH), {'error': 'X'}]
        r = oa.analyze_series(with_missing, 'Coronal', TH)
        self.assertEqual((r['slices']['iop_missing'], r['slices']['unreadable'], r['slices']['pixel_spacing_bad']), (1, 1, 0))

    def test_slices_actually_used_by_the_reader_are_counted_separately(self):
        self.assertEqual(oa.used_positions(10, 3, (0.02, 0.98)), [0, 4, 8])
        self.assertEqual(oa.used_positions(1, 4), [0, 0, 0, 0])
        self.assertEqual(oa.used_positions(0, 4), [])
        base = instances(STANDARD['Coronal'], n=10)
        tilt = instances(rotate((1, 0, 0), (0, 1, 0), 2.0) + rotate((0, 0, -1), (0, 1, 0), 2.0), n=10)
        at = lambda k: oa.analyze_series(base[:k] + [tilt[k]] + base[k + 1:], 'Coronal', TH, pick=(3, (0.02, 0.98)))
        self.assertEqual((at(4)['slices']['used'], at(4)['slices']['used_routing_flagged']), (3, 1))   # position 4 is read
        self.assertEqual(at(5)['slices']['used_routing_flagged'], 0)                                  # position 5 is not
        self.assertEqual(at(5)['slices']['iop_inconsistent'], 1)                                      # but it is counted
        self.assertEqual(oa.analyze_series(base, 'Coronal', TH)['slices']['used'], 0)

    @unittest.skipUnless(HAS_NUMPY, 'NumPy unavailable')
    def test_used_positions_match_the_pinned_numpy_expression(self):
        import numpy as np
        for n in list(range(2, 70)) + [127, 300, 512]:
            for count in (1, 12, 18, 22, 26):
                lo, hi = int(n * 0.02), max(int(n * 0.98) - 1, int(n * 0.02))
                picks = np.linspace(lo, hi, count).round().astype(int)
                self.assertEqual(oa.used_positions(n, count), [int(min(p, n - 1)) for p in picks], (n, count))

    def test_declared_parameters_are_frozen(self):
        self.assertEqual({k: v for k, v in TH.items() if not k.endswith('_cosine')},
                         {'standard_degrees': 20.0, 'oblique_degrees': 45.0, 'iop_tolerance': 1e-3,
                          'normal_tolerance_degrees': 5.0, 'ambiguity_margin': 0.10, 'iop_validity_tolerance': 1e-2,
                          'order_tolerance_mm': 1e-3, 'cosine_epsilon': 1e-9})
        self.assertEqual(oa.slot_names(SLOTS), ['sag0', 'sag1', 'cor0', 'cor1', 'ax'])
        self.assertEqual(oa.SPAN96, (0.02, 0.98))
        self.assertEqual(oa.MIRROR_VECTOR, {'Sagittal': 'normal', 'Coronal': 'row', 'Axial': 'row'})


class AggregateTests(unittest.TestCase):
    @staticmethod
    def study(agg, sid, specs):
        """specs: slot -> iop ; slots absent from specs have no series."""
        chosen = [(i, plane, (f'{sid}-s{i}' if i in specs else None)) for i, (plane, _f, _c) in enumerate(SLOTS)]
        agg.add_study(sid, chosen, {i: oa.analyze_series(instances(iop), SLOTS[i][0], TH) for i, iop in specs.items()})

    @staticmethod
    def full():
        return {i: STANDARD[SLOTS[i][0]] for i in range(5)}

    def test_counts_per_plane_per_slot_and_study_routing(self):
        agg = oa.Aggregate(SLOTS)
        transposed = (0, 0, 1, 1, 0, 0)
        oblique = rotate((1, 0, 0), (0, 1, 0), 30) + rotate((0, 0, -1), (0, 1, 0), 30)
        self.study(agg, 'A', self.full())
        self.study(agg, 'B', self.full())
        self.study(agg, 'D', {**self.full(), 2: transposed})                        # needs_transform
        self.study(agg, 'E', {k: v for k, v in self.full().items() if k != 1})     # second sagittal missing
        self.study(agg, 'F', {**self.full(), 3: oblique})                           # bypass_plain
        self.study(agg, 'G', {**self.full(), 2: transposed, 3: oblique})            # bypass dominates transform
        out = agg.to_dict()
        cor, sag = out['planes']['Coronal'], out['planes']['Sagittal']
        self.assertEqual((out['studies_total'], out['studies_with_selected_series']), (6, 6))
        self.assertEqual(out['study_routing'], {'mirror_ok': 3, 'bypass_plain': 2, 'needs_transform': 1})
        self.assertEqual(cor['series'], 12)
        self.assertEqual((cor['studies']['with_plane'], cor['studies']['all_ok_standard']), (6, 3))
        self.assertEqual((cor['studies']['any_assumption_fail'], cor['studies']['any_ok_oblique']), (2, 2))
        self.assertEqual((cor['mirror_status']['fail_axis'], cor['mirror_status']['ok_oblique']), (2, 2))
        self.assertEqual(cor['series_route'], {'mirror_ok': 8, 'bypass_plain': 2, 'needs_transform': 2})
        self.assertEqual(cor['needs_transform_image_axis'], {'width': 0, 'height': 2, 'slice_order': 0})
        self.assertEqual(cor['dominant_axis']['row'], {'x': 10, 'y': 0, 'z': 2})
        self.assertEqual(cor['mirror_vector_lr_sign'], {'L': 10, 'R': 0})
        self.assertEqual((sag['series'], sag['studies']['with_plane'], sag['signed_axis']['normal']['R']), (11, 6, 11))
        self.assertEqual(list(out['slots']), ['sag0', 'sag1', 'cor0', 'cor1', 'ax'])
        self.assertEqual(out['denominators'], {'studies': 6, 'plane_studies': {'Sagittal': 6, 'Coronal': 6, 'Axial': 6},
                                               'slot_series': {'sag0': 6, 'sag1': 5, 'cor0': 6, 'cor1': 6, 'ax': 6}})
        slot = out['slots']
        self.assertEqual((slot['sag1']['studies_with_series'], slot['sag1']['studies_without_series']), (5, 1))
        self.assertEqual(slot['cor0']['mirror_status']['fail_axis'], 2)       # per slot, not a study majority
        self.assertEqual(slot['cor1']['mirror_status']['ok_oblique'], 2)
        self.assertEqual(slot['cor0']['series_route']['needs_transform'], 2)
        self.assertEqual(slot['cor1']['series_route']['bypass_plain'], 2)
        self.assertEqual(slot['sag0']['mirror_status']['ok_standard'], 6)
        self.assertEqual((slot['ax']['plane'], slot['ax']['slices_in_slot']), ('Axial', 18))
        self.assertEqual(slot['cor0']['slice_level']['total'], 36)            # 6 headers per series, 6 series

    def test_routing_arrays_cover_every_study_and_are_private(self):
        agg = oa.Aggregate(SLOTS)
        for k in range(3):
            self.study(agg, f'S{k}', {**self.full(), 2: (0, 0, 1, 1, 0, 0)} if k == 1 else self.full())
        self.study(agg, 'S3', {0: STANDARD['Sagittal']})
        arrays = agg.routing_arrays()
        self.assertEqual(arrays['study_ids'], ['S0', 'S1', 'S2', 'S3'])
        self.assertEqual(arrays['routing'], ['mirror_ok', 'needs_transform', 'mirror_ok', 'mirror_ok'])
        self.assertEqual(arrays['slot_status'][3], ['ok_standard', None, None, None, None])
        self.assertEqual(arrays['slot_route'][1][2], 'needs_transform')
        self.assertEqual(len(arrays['routing']), agg.to_dict()['studies_total'])

    def test_missing_geometry_series_is_unverifiable_and_routed_to_bypass(self):
        agg = oa.Aggregate(SLOTS)
        chosen = [(0, 'Sagittal', 'x'), (1, 'Sagittal', None), (2, 'Coronal', None), (3, 'Coronal', None), (4, 'Axial', None)]
        agg.add_study('only', chosen, {0: oa.analyze_series(instances(STANDARD['Sagittal'], no_iop=True), 'Sagittal', TH)})
        agg.add_study('none', [(i, p, None) for i, (p, _f, _c) in enumerate(SLOTS)], {})
        out = agg.to_dict()
        self.assertEqual(out['planes']['Sagittal']['mirror_status']['unverifiable'], 1)
        self.assertEqual(out['planes']['Sagittal']['studies']['not_all_ok_standard'], 1)
        self.assertEqual(out['planes']['Axial']['series'], 0)
        self.assertIsNone(out['planes']['Axial']['studies']['fraction_all_ok_standard'])
        self.assertEqual(out['study_routing'], {'mirror_ok': 0, 'bypass_plain': 2, 'needs_transform': 0})
        self.assertEqual((out['studies_total'], out['studies_with_selected_series']), (2, 1))

    def test_descriptive_summary_is_not_a_certification(self):
        def build(cor_bad, n=200):
            agg = oa.Aggregate(SLOTS)
            for k in range(n):
                self.study(agg, f'S{k}', {**self.full(), 2: (0, 0, 1, 1, 0, 0)} if k < cor_bad else self.full())
            return agg.to_dict()
        ok = oa.descriptive_summary(build(2))                                       # 198/200 = 99.0%
        self.assertTrue(ok['per_plane']['Coronal']['at_or_above_threshold'])
        self.assertIn('per-study routing governs', ok['role'])
        self.assertEqual(ok['study_routing_fraction']['needs_transform'], 0.01)     # the 1% is still routed per study
        low = oa.descriptive_summary(build(3))                                      # 98.5%
        self.assertFalse(low['per_plane']['Coronal']['at_or_above_threshold'])
        self.assertTrue(low['per_plane']['Sagittal']['at_or_above_threshold'])
        self.assertIsNone(oa.descriptive_summary(build(0, n=3))['per_plane']['Coronal']['at_or_above_threshold'])

    def test_aggregate_contains_no_identifiers(self):
        agg = oa.Aggregate(SLOTS)
        uid = '1.2.826.0.1.3680043.8.498.99999999'
        self.study(agg, uid, {**self.full(), 2: (0, 0, 1, 1, 0, 0)})
        text = json.dumps(agg.to_dict())
        self.assertNotIn('1.2.826', text)
        self.assertNotIn('99999999', text)
        self.assertEqual(len(agg.to_dict()['ordered_study_ids_sha256']), 64)
        self.assertIn(uid, json.dumps(agg.routing_arrays()))     # only the separate private arrays carry IDs


def replica_pick(rows, plane, fluid, used):
    """Verbatim body of the pinned ``_pick_series_for_slot`` (public reader line 295)."""
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def row(sid, uid, plane, fluid):
    return {'StudyInstanceUID': sid, 'SeriesInstanceUID': uid, 'Fluid_Sensitive': str(fluid),
            'Fat_Suppression': str(fluid), 'Anatomical_Plane': plane}


class SelectionTests(unittest.TestCase):
    rows = [row('s', 'S1', 'Sagittal', 1), row('s', 'S2', 'Sagittal', 0), row('s', 'S3', 'Sagittal', 1),
            row('s', 'C1', 'Coronal', 0), row('s', 'A1', 'Axial', 1), row('s', 'A2', 'Axial', 0),
            row('s', 'C2', 'Coronal', 1)]

    def test_slot_assignment_follows_fluid_preference_and_unused_fallback(self):
        chosen = oa.select_series(replica_pick, SLOTS, self.rows)
        self.assertEqual([c[2] for c in chosen], ['S1', 'S2', 'C2', 'C1', 'A1'])
        self.assertEqual([c[1] for c in chosen], ['Sagittal', 'Sagittal', 'Coronal', 'Coronal', 'Axial'])
        both_fluid = [row('t', 'X1', 'Sagittal', 1), row('t', 'X2', 'Sagittal', 1), row('t', 'Y1', 'Coronal', 1)]
        self.assertEqual([c[2] for c in oa.select_series(replica_pick, SLOTS, both_fluid)], ['X1', 'X2', 'Y1', None, None])
        one = [row('u', 'Z1', 'Sagittal', 0)]
        self.assertEqual([c[2] for c in oa.select_series(replica_pick, SLOTS, one)], ['Z1', None, None, None, None])

    @unittest.skipUnless(PUBLIC.is_file() and HAS_NUMPY, 'pinned public reader source or NumPy unavailable')
    def test_pinned_selector_is_the_qualified_wide96_reader(self):
        from knee import source_span_pilot as pilot
        codes = pilot.recipes(PUBLIC.read_text())      # raises unless SHA256 859aae12... matches
        pick, slots = oa.pinned_selector(codes)
        self.assertEqual(slots, [tuple(s) for s in pilot.SLOTS96])
        self.assertEqual(sum(s[2] for s in slots), 96)
        for rows in (self.rows, self.rows[::-1], self.rows[:3], self.rows[3:]):
            self.assertEqual(oa.select_series(pick, slots, rows), oa.select_series(replica_pick, SLOTS, rows))

    def test_series_csv_loader_keeps_file_order_and_rejects_bad_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'series.csv'

            def write(rows, fields=oa.SERIES_FIELDS):
                with path.open('w', newline='') as handle:
                    w = csv.DictWriter(handle, fieldnames=fields)
                    w.writeheader()
                    w.writerows(rows)
            write(self.rows)
            records = oa.load_series_records(path)
            self.assertEqual([r['SeriesInstanceUID'] for r in records['s']], [r['SeriesInstanceUID'] for r in self.rows])
            for bad in (self.rows + [self.rows[0]], [row('s', 'B1', 'Oblique', 1)], [row('s', '../x', 'Axial', 1)],
                        [row('s', 'B2', 'Axial', 2)]):
                write(bad)
                with self.assertRaises(ValueError):
                    oa.load_series_records(path)
            write([], fields=oa.SERIES_FIELDS[:3])
            with self.assertRaises(ValueError):
                oa.load_series_records(path)


def write_dicom(path, iop, ipp, spacing=(0.5, 0.5), number=1):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    if iop is not None:
        ds.ImageOrientationPatient = list(iop)
    if ipp is not None:
        ds.ImagePositionPatient = list(ipp)
    if spacing is not None:
        ds.PixelSpacing = list(spacing)
    if number is not None:
        ds.InstanceNumber = number
    ds.save_as(str(path), enforce_file_format=True)


def make_tree(root, split, layout):
    """layout: study -> list of (series uid, plane, fluid, iop|None). Writes headers-only DICOM-like files."""
    rows = []
    for sid, series in layout.items():
        for uid, plane, fluid, iop in series:
            directory = Path(root) / f'{split}_series' / sid / uid
            directory.mkdir(parents=True)
            for k in range(4):
                n = oa._cross(iop[:3], iop[3:]) if iop else (0, 0, 1)
                write_dicom(directory / f'{k + 1}.dcm', iop, tuple(2.0 * k * c for c in n), number=k + 1)
            rows.append(row(sid, uid, plane, fluid))
    with (Path(root) / f'{split}_series.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=oa.SERIES_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


LAYOUT = {
    'StudyOne': [('a1', 'Sagittal', 1, STANDARD['Sagittal']), ('a2', 'Sagittal', 0, STANDARD['Sagittal']),
                 ('a3', 'Coronal', 1, STANDARD['Coronal']), ('a4', 'Coronal', 0, STANDARD['Coronal']),
                 ('a5', 'Axial', 1, STANDARD['Axial'])],
    'StudyTwo': [('b1', 'Sagittal', 1, STANDARD['Sagittal']), ('b3', 'Coronal', 1, (0, 0, 1, 1, 0, 0)),
                 ('b5', 'Axial', 0, STANDARD['Axial'])],
    'StudyThree': [('c1', 'Sagittal', 1, STANDARD['Sagittal']), ('c5', 'Axial', 1, None)],
}


@unittest.skipUnless(HAS_PYDICOM, 'pydicom unavailable')
class DriverTests(unittest.TestCase):
    def audit(self, root, workers, cap=None, routing=False):
        records = oa.load_series_records(Path(root) / 'train_series.csv')
        out = Path(root) / f'out{workers}.json'
        route_path = Path(root) / f'routing{workers}.json' if routing else None
        state = oa.audit_split(Path(root) / 'train_series', records, replica_pick, SLOTS, TH, out=out,
                               config={'split': 'train'}, routing_out=route_path, workers=workers, cap=cap,
                               chunk_studies=2)
        return state, json.loads(out.read_text()), route_path

    def test_synthetic_tree_end_to_end_inline_and_process_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tree(tmp, 'train', LAYOUT)
            _, saved, _ = self.audit(tmp, 1)
            _, saved2, _ = self.audit(tmp, 2)
            self.assertEqual(saved['aggregate'], saved2['aggregate'])
            a = saved['aggregate']
            self.assertEqual((saved['status'], saved['series_audited'], a['studies_total']), ('COMPLETE', 10, 3))
            self.assertEqual(a['planes']['Sagittal']['mirror_status']['ok_standard'], 4)
            self.assertEqual(a['planes']['Coronal']['mirror_status']['fail_axis'], 1)
            self.assertEqual(a['planes']['Coronal']['mirror_status']['ok_standard'], 2)
            self.assertEqual(a['planes']['Axial']['mirror_status'], {**{s: 0 for s in oa.STATUSES},
                                                                     'ok_standard': 2, 'unverifiable': 1})
            self.assertEqual(a['planes']['Axial']['flags']['missing_iop'], 1)
            self.assertEqual(a['slots']['ax']['slice_level']['iop_missing'], 4)
            self.assertEqual(a['slots']['ax']['slice_level']['used'], 3 * 18)         # 18 reader picks per axial series
            self.assertEqual(a['slots']['ax']['slice_level']['used_routing_flagged'], 18)   # the series without IOP
            self.assertEqual(a['study_routing'], {'mirror_ok': 1, 'bypass_plain': 1, 'needs_transform': 1})
            self.assertEqual(a['planes']['Coronal']['needs_transform_image_axis']['height'], 1)
            self.assertEqual(a['planes']['Sagittal']['header_files_read'], 16)
            self.assertEqual(a['slots']['sag0']['studies_with_series'], 3)
            self.assertEqual(a['slots']['sag1']['studies_without_series'], 2)
            self.assertFalse(saved['pixels_decoded'] or saved['labels_read'] or saved['identifiers_in_aggregate'])
            text = (Path(tmp) / 'out1.json').read_text()
            for secret in ('StudyOne', 'StudyTwo', 'StudyThree', '"a1"', '/train_series/'):
                self.assertNotIn(secret, text)

    def test_private_routing_array_is_positional_complete_and_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tree(tmp, 'train', LAYOUT)
            _, saved, route_path = self.audit(tmp, 1, routing=True)
            private = json.loads(route_path.read_text())
            self.assertEqual(private['study_ids'], ['StudyOne', 'StudyTwo', 'StudyThree'])     # CSV first-seen order
            self.assertEqual(private['routing'], ['mirror_ok', 'needs_transform', 'bypass_plain'])
            self.assertEqual(private['slot_names'], ['sag0', 'sag1', 'cor0', 'cor1', 'ax'])
            self.assertEqual(private['slot_status'][1], ['ok_standard', None, 'fail_axis', None, 'ok_standard'])
            self.assertEqual(private['slot_route'][2], ['mirror_ok', None, None, None, 'bypass_plain'])
            self.assertEqual(len(private['routing']), saved['aggregate']['studies_total'])
            self.assertEqual(saved['private_routing_file'], 'routing1.json')
            self.assertNotIn('StudyTwo', json.dumps(saved))
            self.assertEqual(saved['aggregate']['routing_array_sha256'],
                             __import__('hashlib').sha256(json.dumps(private['routing']).encode()).hexdigest())
            self.assertFalse(Path(tmp, 'routing2.json').exists())

    def test_instance_cap_reads_fewer_headers_and_flags_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tree(tmp, 'train', {'OnlyStudy': LAYOUT['StudyOne']})
            _, saved, _ = self.audit(tmp, 1, cap=2)
            axial = saved['aggregate']['planes']['Axial']
            self.assertEqual((axial['header_files_read'], axial['flags']['sampled']), (2, 1))
            self.assertEqual(axial['slice_level']['used'], 0)       # used-slice counts need every instance
            a = saved['aggregate']
            self.assertEqual(a['study_routing'], {'mirror_ok': 0, 'bypass_plain': 1, 'needs_transform': 0})
            self.assertEqual(a['planes']['Sagittal']['mirror_status']['geometry_flagged'], 2)
            self.assertEqual(sum(g['series_route']['mirror_ok'] for g in a['planes'].values()), 0)   # none certified
            _, full, _ = self.audit(tmp, 1, cap=4)                   # a cap that covers every file observes everything
            self.assertEqual(full['aggregate']['study_routing']['mirror_ok'], 1)

    @unittest.skipUnless(PUBLIC.is_file() and HAS_NUMPY, 'pinned public reader source or NumPy unavailable')
    def test_run_with_the_pinned_reader_on_a_synthetic_competition_tree(self):
        import hashlib
        import time
        from knee import source_span_pilot as pilot
        codes = pilot.recipes(PUBLIC.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            comp = Path(tmp) / 'input/competitions/rsna-knee-abnormality-detection'
            comp.mkdir(parents=True)
            make_tree(comp, 'train', LAYOUT)
            make_tree(comp, 'test', {'TestStudy': LAYOUT['StudyOne']})
            contract = {'thresholds': TH, 'slots': [list(s) for s in SLOTS], 'span': list(oa.SPAN96), 'workers': 1,
                        'instance_cap': None, 'denominators': 'test',
                        'reader_hashes': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in codes.items()},
                        'train_series_csv_sha256': oa._sha256_file(comp / 'train_series.csv'), 'code_revision': 'test'}
            result = oa.run(Path(tmp) / 'input', Path(tmp) / 'out', contract, codes, started=time.monotonic())
            self.assertEqual(result['status'], 'COMPLETE')
            train = json.loads((Path(tmp) / 'out/train-orientation.json').read_text())
            test = json.loads((Path(tmp) / 'out/test-orientation.json').read_text())
            self.assertIn('per-study routing governs', train['descriptive_summary']['role'])
            self.assertIsNone(train['descriptive_summary']['per_plane']['Coronal']['at_or_above_threshold'])
            self.assertEqual(test['aggregate']['studies_total'], 1)
            self.assertEqual(test['aggregate']['study_routing']['mirror_ok'], 1)
            self.assertEqual(sorted(p.name for p in (Path(tmp) / 'out').iterdir()),
                             ['routing-private-test.json', 'routing-private-train.json',
                              'test-orientation.json', 'train-orientation.json'])
            bad = dict(contract, train_series_csv_sha256='0' * 64)
            with self.assertRaises(RuntimeError):
                oa.run(Path(tmp) / 'input', Path(tmp) / 'out2', bad, codes, started=time.monotonic())
            self.assertTrue((Path(tmp) / 'out2/failure.json').exists())


def write_series(directory, specs):
    """specs: list of dicts for write_dicom (iop, ipp, spacing, number); files named 1.dcm, 2.dcm, ..."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    for k, spec in enumerate(specs, 1):
        write_dicom(directory / f'{k}.dcm', spec.get('iop'), spec.get('ipp'), spec.get('spacing', (0.5, 0.5)),
                    spec.get('number', k))


def stack(iop, n=5, step=3.0, **override):
    normal = oa._cross(iop[:3], iop[3:])
    return [{'iop': iop, 'ipp': tuple(step * k * c for c in normal), **override} for k in range(n)]


@unittest.skipUnless(HAS_PYDICOM, 'pydicom unavailable')
class RealHeaderTests(unittest.TestCase):
    """Small synthetic pydicom files through the real read path (stop_before_pixels, glob enumeration)."""

    def audit(self, plane, specs):
        with tempfile.TemporaryDirectory() as tmp:
            write_series(Path(tmp) / 's', specs)
            return oa.audit_series_dir(str(Path(tmp) / 's'), plane, TH)

    def test_control_clean_series_is_mirror_ok_in_every_plane(self):
        for plane in oa.PLANES:
            r = self.audit(plane, stack(STANDARD[plane]))
            self.assertEqual((r['order'], r['mirror']['status'], r['route']), ('ascending', 'ok_standard', 'mirror_ok'), plane)

    def test_identical_sagittal_ipps_do_not_qualify_for_the_lr_mirror(self):
        r = self.audit('Sagittal', [{'iop': STANDARD['Sagittal'], 'ipp': (10.0, 20.0, 30.0)} for _ in range(5)])
        self.assertEqual((r['mirror']['axis'], r['mirror']['class']), ('x', 'standard'))   # orientation alone looks fine
        self.assertEqual((r['order'], r['mirror']['status'], r['route']), ('degenerate', 'geometry_flagged', 'bypass_plain'))
        single = self.audit('Sagittal', stack(STANDARD['Sagittal'], n=1))
        self.assertEqual((single['order'], single['route']), ('degenerate', 'bypass_plain'))

    def test_equal_position_ties_are_ambiguous_and_bypassed(self):
        specs = stack(STANDARD['Axial'], n=5)
        specs.append(dict(specs[2]))
        r = self.audit('Axial', specs)
        self.assertEqual((r['order'], r['route']), ('ambiguous', 'bypass_plain'))

    def test_missing_pixel_spacing_and_malformed_ipp_are_bypassed(self):
        r = self.audit('Axial', stack(STANDARD['Axial'], spacing=None))
        self.assertEqual((r['flags']['missing_pixel_spacing'], r['route']), (True, 'bypass_plain'))
        specs = stack(STANDARD['Axial'], n=6)
        specs[3]['ipp'] = (0.0, 5.0)            # IPP with two values: the reader files this slice at position 0.0
        r = self.audit('Axial', specs)
        self.assertEqual((r['flags']['invalid_ipp'], r['flags']['reader_fallback'], r['route']), (True, True, 'bypass_plain'))
        specs = stack(STANDARD['Axial'], n=6)
        specs[2].update(iop=None)                # IOP absent on one slice: InstanceNumber fallback in the reader
        r = self.audit('Axial', specs)
        self.assertEqual((r['flags']['missing_iop'], r['flags']['reader_fallback'], r['route']), (True, True, 'bypass_plain'))

    def test_sampling_is_not_certifying(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_series(Path(tmp) / 's', stack(STANDARD['Sagittal'], n=8))
            full = oa.audit_series_dir(str(Path(tmp) / 's'), 'Sagittal', TH)
            sampled = oa.audit_series_dir(str(Path(tmp) / 's'), 'Sagittal', TH, cap=3)
        self.assertEqual(full['route'], 'mirror_ok')
        self.assertEqual((sampled['flags']['sampled'], sampled['files'], sampled['files_total'], sampled['route']),
                         (True, 3, 8, 'bypass_plain'))


def reader_files(order_and_meta, directory):
    return [f for f, _ps in order_and_meta(str(directory))[0]]


@unittest.skipUnless(PUBLIC.is_file() and HAS_NUMPY and HAS_PYDICOM, 'pinned public reader source, NumPy or pydicom unavailable')
class ReaderParityTests(unittest.TestCase):
    """The audit must order slices exactly as the pinned reader (built via source_span_pilot.arm_module) does."""

    @staticmethod
    def pinned_order_and_meta():
        from knee import source_span_pilot as pilot
        module = pilot.arm_module(pilot.recipes(PUBLIC.read_text()), 'wide96')
        with mock.patch.dict(sys.modules, {'cv2': types.ModuleType('cv2')}):   # only the unused resize helper needs cv2
            return module._make_reader()[0]

    @staticmethod
    def mine(directory):
        files, instances, _total = oa.read_series(str(directory), TH)
        return [files[k] for k in oa.reader_order(instances)], files, instances

    @staticmethod
    def enumerations():
        """Natural directory order, then a deliberately reversed one: ties must follow the enumeration, never a sort."""
        real = glob_module.glob
        return [mock.patch('glob.glob', real), mock.patch('glob.glob', lambda pattern: sorted(real(pattern), reverse=True))]

    def test_mocked_malformed_headers_sort_like_the_pinned_reader(self):
        axial = [1, 0, 0, 0, 1, 0]
        def ds(**kw):
            base = dict(ImageOrientationPatient=axial, ImagePositionPatient=[0, 0, 10.0], PixelSpacing=[0.5, 0.5], InstanceNumber=1)
            return SimpleNamespace(**{**base, **kw})
        headers = [ds(), ds(ImagePositionPatient=[0, 0, -5.0]), ds(), ds(ImagePositionPatient=[0, 0, -5.0]),
                   ds(ImagePositionPatient=[0, 5.0]), ds(ImagePositionPatient=[0, 0, 1, 2]),
                   ds(ImagePositionPatient=['a', 'b', 'c']), ds(ImageOrientationPatient=axial[:5], InstanceNumber=7),
                   ds(ImageOrientationPatient=None, InstanceNumber=3), ds(ImagePositionPatient=None, InstanceNumber=None),
                   ds(PixelSpacing='abc', ImagePositionPatient=[0, 0, 20.0]), ds(PixelSpacing=0.5, ImagePositionPatient=[0, 0, 30.0]),
                   ds(PixelSpacing=[], ImagePositionPatient=[0, 0, 40.0]), ds(ImageOrientationPatient=['x'] * 6),
                   OSError('unreadable'), ds(ImagePositionPatient=[0, 0, 2.5], InstanceNumber=-1), ds(PixelSpacing=None)]
        with tempfile.TemporaryDirectory() as tmp:
            by_path = {}
            for k, h in enumerate(headers):
                path = Path(tmp) / f'{k:02d}.dcm'
                path.write_bytes(b'')
                by_path[str(path)] = h
            def fake_dcmread(path, **_kw):
                if isinstance(by_path[str(path)], Exception):
                    raise by_path[str(path)]
                return by_path[str(path)]
            orders = []
            for enumeration in self.enumerations():
                with mock.patch('pydicom.dcmread', fake_dcmread), enumeration:
                    pinned = reader_files(self.pinned_order_and_meta(), tmp)
                    mine, files, instances = self.mine(tmp)
                self.assertEqual(mine, pinned)
                orders.append(mine)
            self.assertNotEqual(orders[0], orders[1])              # the reader's order does depend on enumeration
            sources = {i.get('reader_source', 'error') for i in instances}
            self.assertEqual(sources, {'error', 'exception', 'geometry', 'instance_number'})
            self.assertEqual(oa.analyze_series(instances, 'Axial', TH)['route'], 'bypass_plain')

    def test_real_written_files_with_ties_and_malformed_headers_sort_like_the_pinned_reader(self):
        specs = stack(STANDARD['Axial'], n=5, step=4.0)
        specs.append(dict(specs[1]))                                    # tie
        specs.append({**specs[3], 'ipp': (0.0, 5.0)})                   # IPP of two values
        specs.append({**specs[2], 'iop': None, 'number': 2})            # IOP absent
        specs.append({**specs[4], 'iop': STANDARD['Axial'][:5], 'number': 9})   # IOP of five values
        specs.append({**specs[0], 'ipp': None, 'number': 1})            # IPP absent
        with tempfile.TemporaryDirectory() as tmp:
            write_series(Path(tmp) / 's', specs)
            (Path(tmp) / 's' / 'not-dicom.dcm').write_bytes(b'garbage')     # unreadable header
            for enumeration in self.enumerations():
                with enumeration:
                    pinned = reader_files(self.pinned_order_and_meta(), Path(tmp) / 's')
                    mine, _files, instances = self.mine(Path(tmp) / 's')
                self.assertEqual(len(pinned), len(specs) + 1)
                self.assertEqual(mine, pinned)
            r = oa.analyze_series(instances, 'Axial', TH)
            self.assertEqual((r['flags']['unreadable_header'], r['flags']['reader_fallback'], r['route']),
                             (True, True, 'bypass_plain'))

    def test_enumeration_is_unsorted_like_the_pinned_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_series(Path(tmp) / 's', stack(STANDARD['Axial'], n=6))
            real = glob_module.glob
            with mock.patch('glob.glob', lambda pattern: sorted(real(pattern), reverse=True)):
                files, _, _ = oa.read_series(str(Path(tmp) / 's'), TH)
            self.assertEqual(files, sorted(real(f'{tmp}/s/*.dcm'), reverse=True))


class BuilderTests(unittest.TestCase):
    """Cache-contract preconditions of the kernel builder (synthetic contracts only)."""

    def write_contracts(self, root, skip=None, **change):
        from knee import source_span_pilot as pilot
        hashes = {'narrow': 'n' * 64, 'wide': 'w' * 64}
        for name in builder.CACHE_CONTRACTS:
            if name == skip:
                continue
            contract = {'reader_hashes': hashes,
                        'config': {'slots': [list(x) for x in pilot.SLOTS96], 'span': list(oa.SPAN96)},
                        'input_pins': {'competition': {'train_series.csv': 'c' * 64}}}
            for key, value in change.items():
                contract[key] = value
            path = Path(root) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(contract))
        return hashes

    def test_all_four_exact_cache_contracts_are_required(self):
        self.assertEqual(len(builder.CACHE_CONTRACTS), 4)
        with tempfile.TemporaryDirectory() as tmp:
            hashes = self.write_contracts(tmp)
            self.assertEqual(builder.require_cache_contracts(tmp, hashes, 'c' * 64), 4)
        for missing in builder.CACHE_CONTRACTS:
            with tempfile.TemporaryDirectory() as tmp:
                hashes = self.write_contracts(tmp, skip=missing)
                with self.assertRaisesRegex(ValueError, 'missing'):
                    builder.require_cache_contracts(tmp, hashes, 'c' * 64)
        with tempfile.TemporaryDirectory() as tmp:         # no contracts at all is not "nothing to verify"
            with self.assertRaisesRegex(ValueError, 'missing'):
                builder.require_cache_contracts(tmp, {}, 'c' * 64)

    def test_disagreeing_cache_contracts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            hashes = self.write_contracts(tmp)
            with self.assertRaisesRegex(ValueError, 'Reader hashes'):
                builder.require_cache_contracts(tmp, {**hashes, 'wide': 'x' * 64}, 'c' * 64)
            with self.assertRaisesRegex(ValueError, 'train_series.csv'):
                builder.require_cache_contracts(tmp, hashes, 'd' * 64)
        with tempfile.TemporaryDirectory() as tmp:
            hashes = self.write_contracts(tmp, config={'slots': [['Sagittal', 1, 26]], 'span': list(oa.SPAN96)})
            with self.assertRaisesRegex(ValueError, 'Reader hashes'):
                builder.require_cache_contracts(tmp, hashes, 'c' * 64)

    def test_build_never_overwrites_and_v2_names_do_not_collide_with_the_first_build(self):
        self.assertEqual(builder.DEFAULT_DESTINATION, 'outputs/kaggle/orientation-audit-v2-20261012')
        self.assertEqual(builder.DEFAULT_SLUG, 'knee-orientation-audit-v2-20261012')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Fresh immutable'):
                builder.build(tmp, builder.DEFAULT_SLUG, evidence_root=ROOT, public_source=PUBLIC)


if __name__ == '__main__':
    unittest.main()
