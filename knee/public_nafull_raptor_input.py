from collections import Counter

from contextlib import contextmanager

import ast

import hashlib

from pathlib import Path

STRESS_CASES = frozenset({"healthy", "missing_iop", "missing_ipp", "missing_spacing", "bad_header",
    "bad_pixel", "black_frame", "bad_series_other_valid", "empty_series_other_valid", "short_series",
    "all_empty_fails", "fatal_header", "fatal_pixel", "fatal_geometry", "fatal_pixel_catch",
    "invalid_pixels_fatal"})

def fatal_error(error):
    import cv2
    return isinstance(error, (MemoryError, TimeoutError)) or (isinstance(error, cv2.error) and error.code == -4)

def propagate_fatal(error):
    if fatal_error(error): raise error

def transform_exception_handlers(source):
    """Change only the two pinned catch policies; keep scientific AST identical."""
    tree = ast.parse(source)
    names = {'_make_reader', 'build_study'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names: raise ValueError('Raptor protected definition inventory changed')
    count = 0
    class Protect(ast.NodeTransformer):
        def visit_ExceptHandler(self, node):
            nonlocal count
            if not isinstance(node.type, ast.Name) or node.type.id != 'Exception' or node.name is not None:
                raise ValueError('Raptor public catch structure changed')
            count += 1
            node.name = '_raptor_caught'
            node.body.insert(0, ast.Expr(value=ast.Call(func=ast.Name(id='_raptor_propagate_fatal', ctx=ast.Load()),
                args=[ast.Name(id=node.name, ctx=ast.Load())], keywords=[])))
            return node
    for node in nodes:
        before = count
        Protect().visit(node)
        if count != before + 1: raise ValueError('Expected exactly one protected catch per Raptor function')
    if count != 2: raise ValueError('Expected exactly two protected Raptor catches')
    code = ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))) + '\n'
    return code, {'catch_count': count, 'sha256': hashlib.sha256(code.encode()).hexdigest()}

def exercise_pixel_catch(public, error):
    """Stress the actual protected append-region handler without timing a signal."""
    tree = ast.parse(public._raptor_protected_source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_study')
    handlers = [node for node in ast.walk(function) if isinstance(node, ast.ExceptHandler)]
    if len(handlers) != 1: raise ValueError('Raptor pixel catch inventory changed')
    attempt = ast.Try(body=[ast.Raise(exc=ast.Name(id='_injected', ctx=ast.Load()), cause=None)],
                      handlers=handlers, orelse=[], finalbody=[])
    compiled = compile(ast.fix_missing_locations(ast.Module(body=[attempt], type_ignores=[])), '<pixel-catch-stress>', 'exec')
    exec(compiled, public.__dict__, {'_injected': error, 'arrs': [], 'pss': [], 'med_ps': .5})

@contextmanager
def guarded_reader(public, counters):
    """Sequential only: prevent the two public Exception catches hiding resources.

    Restore both objects even on failure. The public reader/preprocessing source
    remains unmodified; normal missing-header defaults and pixel exceptions retain
    their original control flow. Unsafe nonfinite/non-2D pixels remain fatal.
    """
    import cv2
    import numpy as np
    import pydicom
    original_read, original_factory = pydicom.dcmread, public._make_reader
    class AbortRead(BaseException):
        def __init__(self, error): self.error = error
    def guarded_call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            if fatal_error(error):
                counters['resource_errors'] += 1
                raise AbortRead(error) from None
            raise
    def read(*args, **kwargs):
        try:
            dataset = guarded_call(original_read, *args, **kwargs)
        except Exception:
            if kwargs.get('stop_before_pixels'):
                counters['header_parse_failures'] += 1
            raise
        if kwargs.get('stop_before_pixels'):
            counters['headers'] += 1
            try:
                iop, ipp, ps = (getattr(dataset, name, None) for name in
                               ('ImageOrientationPatient', 'ImagePositionPatient', 'PixelSpacing'))
                if iop is None or ipp is None or len(iop) != 6:
                    counters['geometry_defaults'] += 1
                if ps is None:
                    counters['spacing_defaults'] += 1
                # Eligibility for the separate strict audit; never changes the dataset.
                arrays = [np.asarray(value, dtype=float) for value in (iop, ipp, ps)]
                if ([a.shape for a in arrays] != [(6,), (3,), (2,)] or
                        not all(np.isfinite(a).all() for a in arrays) or (arrays[2] <= 0).any()):
                    counters['strict_ineligible_headers'] += 1
            except (MemoryError, TimeoutError) as error:
                counters['resource_errors'] += 1
                raise AbortRead(error) from None
            except Exception:
                counters['strict_ineligible_headers'] += 1
        return dataset
    def factory():
        order, pixel, resize = original_factory()
        def checked_pixel(path):
            value = pixel(path)
            if value.ndim != 2 or not np.isfinite(value).all():
                counters['invalid_pixels'] += 1
                raise AbortRead(ValueError('Raptor pixel shape/finiteness invariant failed'))
            return value
        def guarded_pixel(path):
            return guarded_call(checked_pixel, path)
        return order, guarded_pixel, resize
    pydicom.dcmread, public._make_reader = read, factory
    try:
        yield
    except AbortRead as stopped:
        raise stopped.error from None
    except Exception as error:
        if fatal_error(error): counters['resource_errors'] += 1
        raise
    finally:
        pydicom.dcmread, public._make_reader = original_read, original_factory

def validate_production_volume(volume, mask, plan, *, image_size):
    import numpy as np
    size = sum(count for _, count in plan)
    if volume.shape != (size, image_size, image_size) or volume.dtype != np.uint8:
        raise ValueError('Raptor volume shape/type mismatch')
    mask = np.asarray(mask)
    nonzero = volume.reshape(size, -1).sum(1) > 0
    if mask.shape != (size,) or not np.isin(mask, [0, 1]).all() or not np.array_equal(mask.astype(bool), nonzero):
        raise ValueError('Raptor pixel/mask inconsistency')
    counts = Counter(valid_slices=int(nonzero.sum()), missing_slots=0, selected_slots=0,
                     masked_selected_frames=0, partial_slots=0, empty_selected_slots=0, zero_studies=0)
    offset = 0
    for row, count in plan:
        valid = int(nonzero[offset:offset+count].sum())
        if row is None:
            counts['missing_slots'] += 1
            if valid: raise ValueError('Absent Raptor series unexpectedly contains pixels')
        else:
            counts['selected_slots'] += 1
            counts['masked_selected_frames'] += count - valid
            counts['partial_slots'] += int(0 < valid < count)
            counts['empty_selected_slots'] += int(valid == 0)
        offset += count
    if not nonzero.any():
        raise ValueError(f'Raptor study has no usable image content: missing_slots={counts["missing_slots"]}, empty_selected_slots={counts["empty_selected_slots"]}; no fallback')
    return dict(counts)

def prepare_study(public, study, records, root, *, cache_pixels=True, reference=False):
    """Call exact public build_study; cache only successful reads within one study."""
    from knee.reference import slot_plan
    counters = Counter({name: 0 for name in ('headers', 'header_parse_failures', 'geometry_defaults',
        'spacing_defaults', 'strict_ineligible_headers', 'pixel_failures', 'decoded_unique_slices',
        'empty_series', 'resource_errors', 'invalid_pixels')})
    plan, substitutions = slot_plan(public, study, records)
    with guarded_reader(public, counters):
        order, pixel, resize = public._make_reader()
        cache, decoded = {}, set()
        def counted_order(path):
            result = order(path)
            counters['empty_series'] += int(not result[0])
            return result
        def counted_pixel(path):
            if path in cache:
                return cache[path]
            try:
                value = pixel(path)
            except Exception:
                counters['pixel_failures'] += 1
                raise
            if path not in decoded:
                counters['decoded_unique_slices'] += 1; decoded.add(path)
            if cache_pixels: cache[path] = value
            return value
        # Reference mode deliberately bypasses memoization/instrumented wrappers.
        reader = (order, pixel, resize) if reference else (counted_order, counted_pixel, resize)
        volume, mask = public.build_study(study, records, str(root), reader)
    counters.update(validate_production_volume(volume, mask, plan, image_size=public.IMG))
    counters['sequence_preference_substitutions'] += substitutions
    return volume, mask, dict(counters)

def strict_eligible(counts):
    return not any(counts.get(key, 0) for key in ('header_parse_failures', 'strict_ineligible_headers',
        'pixel_failures', 'empty_series', 'masked_selected_frames'))

def strict_compare(public, study, records, root, volume, mask):
    """Only called on explicitly eligible healthy inputs; every mismatch is fatal."""
    import numpy as np
    from knee.reference import reference_reader, slot_plan, validate_volume
    with guarded_reader(public, Counter()):
        reader, failures = reference_reader(public)
        other, other_mask = public.build_study(study, records, str(root), reader)
    validate_volume(other, other_mask, slot_plan(public, study, records)[0], image_size=public.IMG)
    if failures or not np.array_equal(volume, other) or not np.array_equal(mask, other_mask):
        raise ValueError('Strict healthy Raptor decoder comparison differs')

def stress_inputs(public):
    """Production-path parity using synthetic datasets/files only, then restore state."""
    import cv2
    import numpy as np
    import pydicom
    import tempfile
    original_read, original_factory = pydicom.dcmread, public._make_reader
    checks, mode, error = {}, 'healthy', None
    base = (np.arange(public.IMG ** 2, dtype=np.float32).reshape(public.IMG, public.IMG) % 1000) + 50
    class SyntheticDataset(pydicom.Dataset):
        @property
        def pixel_array(self):
            if mode == 'fatal_pixel': raise error
            return self._array
    def read(path, stop_before_pixels=False):
        index, series = int(Path(path).stem), Path(path).parent.name
        if mode == 'fatal_header' and stop_before_pixels: raise error
        if mode == 'bad_header' and stop_before_pixels and index == 0: raise ValueError('synthetic header')
        ds = SyntheticDataset()
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, index]
        ds.PixelSpacing = [.5, .5]; ds.InstanceNumber = index
        for case, attribute in (('missing_iop', 'ImageOrientationPatient'), ('missing_ipp', 'ImagePositionPatient'),
                                ('missing_spacing', 'PixelSpacing')):
            if mode == case: delattr(ds, attribute)
        if not stop_before_pixels:
            if (mode == 'bad_pixel' and index == 0) or (mode == 'bad_series_other_valid' and series == 'sag'):
                raise ValueError('synthetic pixel')
            ds._array = np.zeros_like(base) if mode == 'black_frame' and index == 0 else base + index * 5
            if mode == 'invalid_pixels_fatal': ds._array = np.full_like(base, np.nan)
        return ds
    try:
        pydicom.dcmread = read
        with tempfile.TemporaryDirectory(prefix='raptor-synthetic-') as directory:
            root = Path(directory)
            for series, n in (('sag', 8), ('cor', 8), ('empty', 0), ('short', 1)):
                path = root / 'synthetic' / series; path.mkdir(parents=True)
                for i in range(n): (path / f'{i}.dcm').touch()
            rows = [dict(SeriesInstanceUID='sag', Anatomical_Plane='Sagittal', Fluid_Sensitive=1)]
            for mode in sorted(STRESS_CASES - {'all_empty_fails', 'fatal_header', 'fatal_pixel', 'fatal_geometry',
                                              'fatal_pixel_catch', 'invalid_pixels_fatal'}):
                selected = [dict(rows[0])]
                if mode == 'empty_series_other_valid': selected[0]['SeriesInstanceUID'] = 'empty'
                if mode == 'short_series': selected[0]['SeriesInstanceUID'] = 'short'
                if mode in ('empty_series_other_valid', 'bad_series_other_valid'):
                    selected.append(dict(SeriesInstanceUID='cor', Anatomical_Plane='Coronal', Fluid_Sensitive=1))
                records = {'synthetic': selected}
                expected = public.build_study('synthetic', records, str(root), public._make_reader())
                actual = prepare_study(public, 'synthetic', records, root)
                if not all(np.array_equal(a, b) for a, b in zip(expected, actual[:2])):
                    raise ValueError('Synthetic Raptor public-input parity failed')
                checks[mode] = 'PASS'
            mode = 'healthy'
            try:
                prepare_study(public, 'synthetic', {}, root)
            except ValueError:
                checks['all_empty_fails'] = 'PASS'
            else: raise AssertionError('Empty Raptor study accepted')
            opencv = cv2.error('synthetic allocation'); opencv.code = -4
            for mode in ('fatal_header', 'fatal_pixel'):
                for error in (MemoryError('synthetic'), TimeoutError('synthetic'), opencv):
                    try:
                        prepare_study(public, 'synthetic', {'synthetic': rows}, root)
                    except type(error): pass
                    else: raise AssertionError('Fatal Raptor read was swallowed')
                    if pydicom.dcmread is not read or public._make_reader is not original_factory:
                        raise AssertionError('Raptor resource guard did not restore reader')
                checks[mode] = 'PASS'
            mode = 'healthy'
            original_cross = public.np.cross
            try:
                for error in (MemoryError('synthetic geometry'), TimeoutError('synthetic geometry'), opencv):
                    def failing_cross(*args, **kwargs): raise error
                    public.np.cross = failing_cross
                    try:
                        prepare_study(public, 'synthetic', {'synthetic': rows}, root)
                    except type(error): pass
                    else: raise AssertionError('Fatal geometry arithmetic was swallowed')
                checks['fatal_geometry'] = 'PASS'
            finally:
                public.np.cross = original_cross
            # Execute the actual transformed pixel catch body with a failure after
            # the reader returned (the append/timer region), without racing SIGALRM.
            for error in (MemoryError('synthetic append'), TimeoutError('synthetic append'), opencv):
                try:
                    exercise_pixel_catch(public, error)
                except type(error): pass
                else: raise AssertionError('Fatal pixel catch policy returned')
            checks['fatal_pixel_catch'] = 'PASS'
            mode = 'invalid_pixels_fatal'
            try:
                prepare_study(public, 'synthetic', {'synthetic': rows}, root)
            except ValueError:
                checks[mode] = 'PASS'
            else: raise AssertionError('Nonfinite pixels accepted')
        return {'cases': checks, 'uses_raw_data': False, 'uses_labels': False}
    finally:
        pydicom.dcmread, public._make_reader = original_read, original_factory
