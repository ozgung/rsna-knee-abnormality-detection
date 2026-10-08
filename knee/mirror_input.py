"""Audited mirror input policy; preserve study_lut/public math and partial masks."""
import ast
from collections import Counter
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import threading

from knee.raptor_input import fatal_error, propagate_fatal, validate_production_volume

POLICY_VERSION = 'mirror-study-lut-partial-v1'
TRAIN_COUNT = 64
MAX_SECONDS = 9 * 3600  # competition notebook limit, not a spending gate
CHECKPOINT_BYTES = 292840787
STRESS_CASES = frozenset({'healthy', 'missing_iop', 'missing_ipp', 'missing_spacing',
    'bad_header', 'bad_pixel', 'black_frame', 'empty_series', 'missing_series', 'short_series',
    'all_empty_fails', 'fatal_header', 'fatal_geometry', 'fatal_pixel', 'fatal_pydicom',
    'fatal_catch_regions', 'invalid_pixels_fatal'})


def policy_contract():
    return {'version': POLICY_VERSION, 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def protect_decoder(source):
    """Prefix all pinned decoder catches, including OSError (TimeoutError's base).

    No decode, ordering, crop, percentile, normalization or fallback expressions
    change. Prefixing the whole region also covers async errors between callbacks.
    """
    from knee.reference import FASTREAD_SHA256
    if hashlib.sha256(source.encode()).hexdigest() != FASTREAD_SHA256:
        raise ValueError('Unreviewed mirror decoder source')
    tree = ast.parse(source)
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    counts = Counter(ast.unparse(n.type) for n in handlers)
    if counts != {'Exception': 13, 'OSError': 4, 'NeedMore': 2, '(ValueError, OSError)': 1}:
        raise ValueError('Mirror decoder catch inventory changed')
    cleanup_regions = 0
    for handler in handlers:
        if handler.name is not None: raise ValueError('Mirror decoder catch binding changed')
        handler.name = '_mirror_caught'
        # Preserve the existing fd cleanup before rethrowing a fatal header error.
        after_cleanup = isinstance(handler.body[0], ast.If) and ast.unparse(handler.body[0].test) == 'fd is not None'
        cleanup_regions += int(after_cleanup)
        handler.body.insert(int(after_cleanup), ast.parse('_mirror_propagate_fatal(_mirror_caught)').body[0])
    if cleanup_regions != 1: raise ValueError('Mirror header descriptor-cleanup structure changed')
    code = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    return code, {'catch_count': len(handlers), 'cleanup_regions': cleanup_regions, 'sha256': hashlib.sha256(code.encode()).hexdigest()}


@contextmanager
def counted_decoder(decoder, counts):
    """Add observability and unsafe-pixel rejection; public handled errors stay masked."""
    import numpy as np
    lock = threading.Lock()
    originals = decoder._records, decoder._result, decoder._Ctx.px, decoder.fast_hdr
    def records(rows):
        with lock:
            counts['fast_headers'] += len(rows)
            counts['fast_bad_headers'] += sum(not r[3] for r in rows)
        return originals[0](rows)
    def result(selected, got, spacing):
        with lock:
            counts['pixel_failures'] += sum(v[0] is None for v in got.values())
        return originals[1](selected, got, spacing)
    def pixels(ctx, data, info):
        value = originals[2](ctx, data, info)
        if value[0] is not None:
            a = value[0]
            if isinstance(a, decoder.RawSlice):
                bits = 16 - a.shift
                probes = np.array([0, (1 << (bits - 1)) - 1, 1 << (bits - 1), (1 << bits) - 1], np.uint16)
                valid = len(a.shape) == 2 and np.isfinite(a.codes_to_float(probes)).all()
            else:
                valid = a.ndim == 2 and np.isfinite(a).all()
            if not valid:
                with lock: counts['invalid_pixels'] += 1
                raise ValueError('Mirror pixel shape/finiteness invariant failed')
            with lock: counts['decoded_unique_slices'] += 1
        return value
    def header(info):
        value = originals[3](info)
        if value is not None and (not np.isfinite(value).all() or value[1] <= 0):
            with lock: counts['strict_ineligible_headers'] += 1
        return value
    decoder._records, decoder._result, decoder._Ctx.px, decoder.fast_hdr = records, result, pixels, header
    try: yield
    finally:
        decoder._records, decoder._result, decoder._Ctx.px, decoder.fast_hdr = originals


@contextmanager
def observe_headers(counts):
    """Observe pydicom's compatibility path; protected source catches own resources."""
    import numpy as np
    import pydicom
    original = pydicom.dcmread
    lock = threading.Lock()
    def read(*args, **kwargs):
        try: dataset = original(*args, **kwargs)
        except Exception:
            if kwargs.get('stop_before_pixels'):
                with lock: counts['header_parse_failures'] += 1
            raise
        if kwargs.get('stop_before_pixels'):
            iop, ipp, ps = (getattr(dataset,n,None) for n in
                           ('ImageOrientationPatient','ImagePositionPatient','PixelSpacing'))
            with lock:
                counts['headers'] += 1
                counts['geometry_defaults'] += int(iop is None or ipp is None or len(iop) != 6)
                counts['spacing_defaults'] += int(ps is None)
            try:
                arrays = [np.asarray(v,dtype=float) for v in (iop,ipp,ps)]
                valid = ([a.shape for a in arrays] == [(6,),(3,),(2,)] and
                         all(np.isfinite(a).all() for a in arrays) and (arrays[2]>0).all())
            except Exception as error:
                propagate_fatal(error); valid = False
            if not valid:
                with lock: counts['strict_ineligible_headers'] += 1
        return dataset
    pydicom.dcmread = read
    try: yield
    finally: pydicom.dcmread = original


def prepare_study(public, decoder, study, records, root, *, reference=False):
    from knee.reference import slot_plan
    counts = Counter({n: 0 for n in ('headers', 'fast_headers', 'fast_bad_headers', 'header_parse_failures',
        'geometry_defaults', 'spacing_defaults', 'strict_ineligible_headers', 'pixel_failures',
        'decoded_unique_slices', 'resource_errors', 'invalid_pixels')})
    plan, substitutions = slot_plan(public, study, records)
    try:
        with observe_headers(counts):
            if reference:
                volume, mask = public.build_study(study, records, str(root), public._make_reader())
            else:
                with counted_decoder(decoder, counts):
                    fast = decoder.make_builder('study_lut', public.IMG, public.SLOTS, hdr_threads=4, px_threads=4)
                    volume, mask = fast(study, records, str(root))
    except Exception as error:
        if fatal_error(error): counts['resource_errors'] += 1
        error.mirror_input_counts = dict(counts)
        raise
    counts.update(validate_production_volume(volume, mask, plan, image_size=public.IMG))
    counts['sequence_preference_substitutions'] = substitutions
    return volume, mask, dict(counts)


def strict_eligible(counts):
    return not any(counts.get(n, 0) for n in ('fast_bad_headers', 'header_parse_failures',
        'strict_ineligible_headers', 'pixel_failures', 'masked_selected_frames'))


def stress_inputs(public, decoder):
    """Real synthetic DICOM files exercise the native decoder, never competition data."""
    import cv2
    import numpy as np
    import pydicom
    import tempfile
    from unittest.mock import patch
    checks = {}
    def dataset(index):
        meta = pydicom.Dataset(); meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        meta.MediaStorageSOPInstanceUID = '1.2.826.0.1.3680043.8.498.' + str(index + 1)
        ds = pydicom.FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Rows = 32; ds.Columns = 40; ds.SamplesPerPixel = 1; ds.BitsAllocated = 16
        ds.BitsStored = 12; ds.HighBit = 11; ds.PixelRepresentation = 0
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]; ds.ImagePositionPatient = [0, 0, index]
        ds.PixelSpacing = [.5, .5]; ds.InstanceNumber = index
        ds.PixelData = ((np.arange(1280, dtype=np.uint16).reshape(32,40) + index * 11) % 2000).tobytes()
        return ds
    with tempfile.TemporaryDirectory(prefix='mirror-synthetic-') as directory:
        root = Path(directory); sag = root/'synthetic'/'sag'; cor = root/'synthetic'/'cor'
        sag.mkdir(parents=True); cor.mkdir()
        for i in range(8): pydicom.dcmwrite(cor/f'{i}.dcm', dataset(i), enforce_file_format=True)
        rows = [dict(SeriesInstanceUID='sag', Anatomical_Plane='Sagittal', Fluid_Sensitive=1),
                dict(SeriesInstanceUID='cor', Anatomical_Plane='Coronal', Fluid_Sensitive=1)]
        normal = STRESS_CASES - {'all_empty_fails', 'fatal_header', 'fatal_geometry', 'fatal_pixel',
                                 'fatal_pydicom', 'fatal_catch_regions', 'invalid_pixels_fatal'}
        for case in sorted(normal):
            for p in sag.glob('*.dcm'): p.unlink()
            for i in range(1 if case == 'short_series' else 8):
                ds = dataset(i)
                field = {'missing_iop':'ImageOrientationPatient', 'missing_ipp':'ImagePositionPatient',
                         'missing_spacing':'PixelSpacing'}.get(case)
                if field: delattr(ds, field)
                if i == 0 and case == 'black_frame': ds.PixelData = bytes(2560)
                if i == 0 and case == 'bad_pixel': ds.PixelData = bytes(2)
                pydicom.dcmwrite(sag/f'{i}.dcm', ds, enforce_file_format=True)
            if case == 'bad_header': (sag/'0.dcm').write_bytes(b'not a DICOM')
            if case == 'empty_series':
                for p in sag.glob('*.dcm'): p.unlink()
            chosen = [dict(r) for r in rows]
            if case == 'missing_series': chosen[0]['SeriesInstanceUID'] = 'missing'
            records = {'synthetic': chosen}
            expected = prepare_study(public, decoder, 'synthetic', records, root, reference=True)
            actual = prepare_study(public, decoder, 'synthetic', records, root)
            if not all(np.array_equal(a,b) for a,b in zip(expected[:2],actual[:2])):
                raise ValueError('Mirror native/public synthetic volume or mask differs')
            if case == 'healthy' and not strict_eligible(actual[2]): raise ValueError('Healthy fixture audit ineligible')
            checks[case] = 'PASS'
        try: prepare_study(public, decoder, 'synthetic', {}, root)
        except ValueError: checks['all_empty_fails'] = 'PASS'
        else: raise AssertionError('Empty mirror study accepted')
        records = {'synthetic': [rows[1]]}
        opencv = cv2.error('synthetic memory'); opencv.code = -4
        for case, function in [('fatal_header','_pread'), ('fatal_geometry','fast_hdr'), ('fatal_pixel','fast_raw')]:
            for error in (MemoryError('synthetic'), TimeoutError('synthetic'), opencv):
                with patch.object(decoder, function, side_effect=error):
                    try: prepare_study(public, decoder, 'synthetic', records, root)
                    except type(error): pass
                    else: raise AssertionError('Mirror decoder swallowed a resource error')
            checks[case] = 'PASS'
        for error in (MemoryError('synthetic'), TimeoutError('synthetic'), opencv):
            with patch.object(decoder, 'fast_hdr', return_value=None), patch.object(pydicom, 'dcmread', side_effect=error):
                try: prepare_study(public, decoder, 'synthetic', records, root)
                except type(error): pass
                else: raise AssertionError('Mirror pydicom path swallowed a resource error')
        checks['fatal_pydicom'] = 'PASS'
        # Every broad protected catch must rethrow a TimeoutError, including OSError
        # cleanup regions; inject directly after callback success without signal races.
        handlers = [n for n in ast.walk(ast.parse(decoder._mirror_protected_source)) if isinstance(n,ast.ExceptHandler)]
        for handler in handlers:
            if ast.unparse(handler.type) == 'NeedMore': continue
            attempt = ast.Try(body=[ast.Raise(exc=ast.Name(id='_injected',ctx=ast.Load()),cause=None)],
                              handlers=[handler],orelse=[],finalbody=[])
            function = ast.FunctionDef(name='_catch_probe', args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]), body=[attempt], decorator_list=[])
            code = compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])), '<mirror-catch-test>', 'exec')
            namespace = dict(decoder.__dict__, fd=None, _injected=TimeoutError('synthetic after callback'))
            exec(code, namespace)
            try: namespace['_catch_probe']()
            except TimeoutError: pass
            else: raise AssertionError('Mirror catch consumed deadline')
        checks['fatal_catch_regions'] = 'PASS'
        with patch.object(decoder, 'fast_raw', return_value=np.full((2,2),np.nan)):
            try: prepare_study(public, decoder, 'synthetic', records, root)
            except ValueError: checks['invalid_pixels_fatal'] = 'PASS'
            else: raise AssertionError('Unsafe pixels accepted')
    return {'status':'PASS','cases':checks,'uses_raw_data':False,'uses_labels':False}


def require_rehearsal(contract, report, *, bundle_sha256):
    from knee.reference import canonical_bytes, sha256, RECIPE, WEIGHT_SHA256
    policy = policy_contract(); studies = report.get('studies', [])
    env = report.get('environment', {}); packages = env.get('packages', {})
    stress = report.get('stress_checks', {})
    source = contract.get('source', {})
    sampling = report.get('sampling', {})
    def digest(value):
        return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    if (contract.get('input_policy') != policy or report.get('input_policy') != policy or
        source.get('recipe') != RECIPE or source.get('checkpoint_sha256') != WEIGHT_SHA256 or
        report.get('manifest') != source or report.get('manifest_sha256') != sha256(canonical_bytes(source)) or
        report.get('source_bundle_sha256') != bundle_sha256 or not digest(bundle_sha256) or
        report.get('checkpoint_verified_sha256') != WEIGHT_SHA256 or report.get('checkpoint_bytes') != CHECKPOINT_BYTES or
        report.get('status') != 'passed' or report.get('prediction_fallbacks') != 0 or
        contract.get('private') is not True or contract.get('internet') is not False or
        contract.get('train_count') != TRAIN_COUNT or
        report.get('sampling', {}).get('train_count') != TRAIN_COUNT or
        sampling.get('test_count') != 3 or sampling.get('uses_labels') is not False or
        sampling.get('seed') != contract.get('seed') or not digest(sampling.get('ordered_ids_sha256')) or
        sampling.get('selection') != 'SHA256(seed:StudyInstanceUID) order' or
        env.get('python_major_minor') != [3,12] or packages.get('torch') != '2.10.0+cu128' or packages.get('timm') != '1.0.26' or
        stress.get('status') != 'PASS' or stress.get('cases') != dict.fromkeys(STRESS_CASES,'PASS') or
        stress.get('uses_raw_data') is not False or stress.get('uses_labels') is not False or
        not math.isfinite(report.get('runtime_seconds',float('inf'))) or not 0 < report['runtime_seconds'] <= MAX_SECONDS):
        raise ValueError('Fresh exact mirror67 source/weights/policy qualification required')
    if (len(studies) != 67 or Counter(s.get('split') for s in studies) != {'train':64,'test':3} or
        [s.get('ordinal') for s in studies] != list(range(67)) or
        not any(s.get('strict_comparison') == 'PASS' for s in studies) or
        report.get('strict_comparison_counts') != dict(Counter(s.get('strict_comparison') for s in studies))):
        raise ValueError('Mirror expanded cohort/strict audit coverage missing')
    for s in studies:
        if (s.get('public_pixel_max_error') != 0 or s.get('window_max_error') != 0 or
            not 0 <= s.get('repeat_prediction_max_error',float('inf')) <= 1e-6 or s.get('resource_errors') != 0 or
            s.get('valid_slices',0) <= 0 or s.get('eval_windows') != 94 or
            s.get('strict_comparison') not in ('PASS','ineligible_partial_input') or
            s.get('decoder_pixel_max_error') != (0 if s['strict_comparison']=='PASS' else None)):
            raise ValueError('Mirror input/window/repeat evidence incomplete')
