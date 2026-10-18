"""Audited recovery boundaries, exact per-study cleanup and genuine-only ledgers."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

POLICY = 'classified-data-local-v1'
REGISTERED_DATA_CLASSES=set()
REGISTERED_DICOM_CLASS=None
REGISTERED_OLD_PARSER_CLASSES=set()
REGISTERED_DECODE_CLASS=None
DICOM_READ_SITES=frozenset({'training.native_cache:prepare:87','training.native_cache:prepare:105','training.dicom_geometry:order_series:435','training.dicom_geometry:read_pixels:648','old_public:_hdr','old_public:_px','old_decoder:_hdr','old_decoder:_px','old_decoder:hdr_from_bytes','old_decoder:px'})


class DataLocalFailure(Exception):
    """Only explicit unusable-image terminal outcomes are arm-recoverable."""
    def __init__(self,code,*,site=None,detail=None):
        self.code=code;self.site=site;self.detail=detail or code
        super().__init__(code)


class AuditedPixelBufferError(ValueError):
    """Exact pydicom3.0.2 byte-count/photometric corruption raise sites."""

PYDICOM_BUFFER_FILE_SHA='4f2b345d28bffc75d9b4866b0dac92217c260b1de2476be4c3ea215ef5455d6a'
PYDICOM_BUFFER_FUNCTION_SHA='d2303443992fee21189cf32d5fcc3c01262dff18cb7c5d6910bed304614a6bba'
PIXEL_BUFFER_CATCH_SITES=frozenset({'old_public:_px','old_decoder:px','training.native_cache:prepare:105','training.dicom_geometry:read_pixels:648'})


def install_pixel_buffer_guard():
    import inspect,textwrap
    import pydicom.pixels.decoders.base as base
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()!=PYDICOM_BUFFER_FILE_SHA:raise ValueError('EXACT_PYDICOM_BUFFER_SOURCE')
    function=base.DecodeRunner._validate_buffer
    if getattr(function,'_s6_buffer_guarded',False):
        if function is not getattr(base,'_s6_guarded_buffer_function',None) or function.__globals__.get('_s6_pixel_buffer_error') is not AuditedPixelBufferError:raise ValueError('EXACT_INSTALLED_PIXEL_GUARD')
        return function._s6_buffer_guard_sha256
    source=textwrap.dedent(inspect.getsource(function))
    if hashlib.sha256(source.encode()).hexdigest()!=PYDICOM_BUFFER_FUNCTION_SHA:raise ValueError('EXACT_PYDICOM_BUFFER_FUNCTION')
    tree=ast.parse(source);raises=[x for x in ast.walk(tree) if isinstance(x,ast.Raise)]
    if {x.lineno for x in raises}!={22,33} or any(not isinstance(x.exc,ast.Call) or not isinstance(x.exc.func,ast.Name) or x.exc.func.id!='ValueError' for x in raises):raise ValueError('EXACT_PYDICOM_BUFFER_RAISES')
    for node in raises:node.exc.func=ast.Name('_s6_pixel_buffer_error',ast.Load())
    transformed=ast.unparse(ast.fix_missing_locations(tree))+'\n';base.__dict__['_s6_pixel_buffer_error']=AuditedPixelBufferError
    exec(compile(transformed,'<s6-pydicom-buffer-guard>','exec'),base.__dict__)
    function=base.__dict__['_validate_buffer'];function._s6_buffer_guarded=True;function._s6_buffer_guard_sha256=hashlib.sha256(transformed.encode()).hexdigest()
    base.DecodeRunner._validate_buffer=function;base._s6_guarded_buffer_function=function
    return function._s6_buffer_guard_sha256


class FatalSourceError(BaseException):
    """Bypass upstream Exception catches without permitting a partial prediction."""
    def __init__(self, error):
        self.error = error
        super().__init__(type(error).__name__ + ': ' + str(error))


def guard_source_error(error, scope):
    """Unknown exceptions escape every upstream catch as BaseException.

    Accepted data errors carry an audited AST raise-site identity. Only primitive
    malformed-tag conversions are accepted at their exact original catch sites.
    """
    chain=[];current=error
    while current is not None and id(current) not in {id(x) for x in chain}:
        chain.append(current);current=current.__cause__ or current.__context__
    for cause in chain:
        if type(cause) in REGISTERED_DATA_CLASSES:continue
        if isinstance(cause,(MemoryError,TimeoutError,OSError,RuntimeError,KeyboardInterrupt,SystemExit,KeyError,IndexError,AssertionError,ImportError)):
            raise FatalSourceError(cause) from cause
    if type(error) is DataLocalFailure:raise error
    cls=type(error)
    if cls in REGISTERED_DATA_CLASSES:return
    if cls is AuditedPixelBufferError and scope in PIXEL_BUFFER_CATCH_SITES:return
    if cls in REGISTERED_OLD_PARSER_CLASSES and scope in ('old_decoder:_scan_full','old_decoder:_hdr_prefix'):return
    if cls is REGISTERED_DICOM_CLASS and scope in DICOM_READ_SITES:return
    if scope in ('training.dicom_geometry:_finite_float:132', 'training.dicom_geometry:_float_vector:142') and cls in (ValueError,TypeError):
        return
    if scope == 'training.dicom_geometry:header_from_dataset:322' and (cls in (ValueError,TypeError,OverflowError) or (cls is AttributeError and getattr(error,'name',None)=='InstanceNumber')):
        return
    raise FatalSourceError(error) from error


# Absolute line identities in the verified pinned geometry file. Unknown plane,
# algorithm residual, corrupt artifacts and optimizer errors are intentionally absent.
GEOMETRY_DATA_RAISES = frozenset({124,135,152,165,187,323,336,341,438,442,444,447,451,454,464,486,499,503,511,520,523,659,662,665,669,674,677})


def transformed_function(source, *, name, scope, unwrap=False, data_raise_lines=(), source_start=1):
    tree=ast.parse(source)
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    handlers=[x for x in ast.walk(node) if isinstance(x,ast.ExceptHandler) and isinstance(x.type,ast.Name) and x.type.id=='Exception']
    for handler in handlers:
        if unwrap:handler.body=[ast.Raise(exc=None,cause=None)]
        else:
            handler.name=handler.name or '_s6_caught'
            site=scope+':'+str(source_start+handler.lineno-1)
            handler.body.insert(0,ast.Expr(ast.Call(ast.Name('_s6_guard_source_error',ast.Load()),[ast.Name(handler.name,ast.Load()),ast.Constant(site)],[])))
    tagged=[]
    for raise_node in ast.walk(node):
        if data_raise_lines and isinstance(raise_node,ast.Raise) and source_start+raise_node.lineno-1 in data_raise_lines:
            line=source_start+raise_node.lineno-1
            if not isinstance(raise_node.exc,ast.Call) or not isinstance(raise_node.exc.func,ast.Name) or raise_node.exc.func.id!='GeometryError':
                raise ValueError('EXACT_GEOMETRY_RAISE_SITE')
            tagged.append(line)
            raise_node.exc.func=ast.Name('_s6_geometry_error_'+str(line),ast.Load())
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    return ast.unparse(module)+'\n',len(handlers),tagged


def install_source_guards(CI, ds, cfg):
    """Numerical AST unchanged; only exact raise/catch policies are rewritten."""
    import textwrap
    import training.native_cache as native
    import training.dicom_geometry as geometry
    import knee_anchor_bank_v2 as anchors
    import px_cache
    global REGISTERED_DICOM_CLASS,REGISTERED_DECODE_CLASS
    from pydicom.errors import InvalidDicomError
    REGISTERED_DICOM_CLASS=InvalidDicomError
    REGISTERED_DECODE_CLASS=type('_S6DecodedPixelData',(geometry.GeometryError,),{'_s6_data_site':'649_audited_cause_only'})
    REGISTERED_DATA_CLASSES.add(REGISTERED_DECODE_CLASS)
    code=Path(cfg['code']).resolve();receipts={'pydicom:DecodeRunner._validate_buffer':{'catches':0,'data_raise_sites':[22,33],'guarded_sha256':install_pixel_buffer_guard()}}
    def source_file(relative):
        path=code/relative
        if not path.resolve().is_relative_to(code) or hashlib.sha256(path.read_bytes()).hexdigest()!=cfg['guard_source_sha256'][relative]:
            raise ValueError('GUARD_EXACT_SOURCE_BYTES_'+relative)
        return path.read_text()
    def definition(relative,name,class_name=None):
        full=source_file(relative);tree=ast.parse(full)
        owner=tree if class_name is None else next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==class_name)
        node=next(x for x in owner.body if isinstance(x,ast.FunctionDef) and x.name==name)
        start=min([node.lineno]+[x.lineno for x in node.decorator_list])
        return textwrap.dedent('\n'.join(full.splitlines()[start-1:node.end_lineno])),start
    def apply(module,relative,name,*,class_name=None,unwrap=False,lean=False):
        source,start=definition(relative,name,class_name)
        if lean:
            changes=[('pydicom.dcmread(io.BytesIO(archive.read(i)))','pydicom.dcmread(io.BytesIO(archive.read(i)), stop_before_pixels=True)'),
                     ('geom.pixels_from_dataset(objects[sid][name],path=name)','geom.pixels_from_dataset(pydicom.dcmread(io.BytesIO(archive.read(name))),path=name)')]
            for before,after in changes:
                if source.count(before)!=1:raise ValueError('EXACT_NATIVE_LEAN_PATCH')
                source=source.replace(before,after)
        scope=module.__name__+':'+name
        lines=GEOMETRY_DATA_RAISES if relative=='training/dicom_geometry.py' else ()
        transformed,count,tagged=transformed_function(source,name=name,scope=scope,unwrap=unwrap,data_raise_lines=lines,source_start=start)
        tree=ast.parse(transformed)
        if relative=='compact_infer.py':
            terminal=[x for x in ast.walk(tree) if isinstance(x,ast.Raise) and isinstance(x.exc,ast.Call) and isinstance(x.exc.func,ast.Name) and x.exc.func.id=='ValueError' and ast.unparse(x.exc.args[0])=="'no usable window'"]
            if len(terminal)!=1:raise ValueError('EXACT_NO_WINDOW_SITE')
            terminal[0].exc=ast.Call(ast.Name('_s6_data_local_failure',ast.Load()),[ast.Constant('s6_no_usable_window')],[ast.keyword('site',ast.Constant('compact_infer.py:CStudies.__getitem__:210')),ast.keyword('detail',ast.Constant('no usable window'))])
        if relative=='px_cache.py':
            terminal=[x for x in ast.walk(tree) if isinstance(x,ast.Raise) and isinstance(x.exc,ast.Call) and isinstance(x.exc.func,ast.Name) and x.exc.func.id=='ValueError' and 'no series file present' in ast.unparse(x.exc)]
            if len(terminal)!=1:raise ValueError('EXACT_NO_SERIES_SITE')
            terminal[0].exc=ast.Call(ast.Name('_s6_empty_native_failure',ast.Load()),[ast.Name('study',ast.Load())],[])
        if relative=='training/dicom_geometry.py' and name=='read_pixels':
            wrapper=[x for x in ast.walk(tree) if isinstance(x,ast.Raise) and isinstance(x.exc,ast.Call) and isinstance(x.exc.func,ast.Name) and x.exc.func.id=='GeometryError']
            if len(wrapper)!=1:raise ValueError('EXACT_PIXEL_DECODE_WRAPPER')
            wrapper[0].exc=ast.Call(ast.Name('_s6_audited_decode_failure',ast.Load()),[ast.Name('exc',ast.Load())],[])
        transformed=ast.unparse(ast.fix_missing_locations(tree))+'\n'
        namespace=module.__dict__
        namespace.update(_s6_guard_source_error=guard_source_error,_s6_data_local_failure=DataLocalFailure,_s6_empty_native_failure=empty_native_failure,_s6_audited_decode_failure=audited_decode_failure)
        for line in tagged:
            error_class=type('_S6GeometryData'+str(line),(geometry.GeometryError,),{'_s6_data_site':line})
            REGISTERED_DATA_CLASSES.add(error_class)
            namespace['_s6_geometry_error_'+str(line)]=error_class
        exec(compile(transformed,'<s6-guard-'+scope+'>','exec'),namespace)
        function=namespace[name]
        if class_name:setattr(getattr(module,class_name),name,function)
        else:setattr(module,name,function)
        receipts[relative+':'+name]={'catches':count,'data_raise_sites':tagged,'guarded_sha256':hashlib.sha256(transformed.encode()).hexdigest()}
    # Guard geometry first so native/anchor namespaces resolve guarded functions.
    tree=ast.parse(source_file('training/dicom_geometry.py'));found=set()
    for node in tree.body:
        if isinstance(node,ast.FunctionDef):
            tagged=[x.lineno for x in ast.walk(node) if isinstance(x,ast.Raise) and x.lineno in GEOMETRY_DATA_RAISES]
            broad=any(isinstance(x,ast.ExceptHandler) and isinstance(x.type,ast.Name) and x.type.id=='Exception' for x in ast.walk(node))
            if broad or tagged:
                apply(geometry,'training/dicom_geometry.py',node.name);found.update(tagged)
    if found!=GEOMETRY_DATA_RAISES:raise ValueError('PINNED_GEOMETRY_RAISE_INVENTORY')
    apply(CI,'compact_infer.py','__getitem__',class_name='CStudies',unwrap=True)
    apply(native,'training/native_cache.py','prepare',lean=True)
    apply(anchors,'knee_anchor_bank_v2.py','record',class_name='AnchorBank')
    apply(px_cache,'px_cache.py','build',class_name='StudySlices')
    return receipts


def audited_decode_failure(error):
    if type(error) in (AuditedPixelBufferError,REGISTERED_DICOM_CLASS) and REGISTERED_DECODE_CLASS is not None:return REGISTERED_DECODE_CLASS(str(error))
    return FatalSourceError(error)


def empty_native_failure(study):
    if not study['series']:return DataLocalFailure('s6_no_usable_native_series',site='px_cache.py:StudySlices.build:161',detail='Native study has no successfully prepared series')
    return FatalSourceError(FileNotFoundError('Nonempty native series artifact vanished'))


def install_old_guards(public,decoder):
    """Guard old-source catches and tag only the actual all-empty invariant."""
    import inspect
    import textwrap
    import knee.mirror_input as MI
    global REGISTERED_DICOM_CLASS
    from pydicom.errors import InvalidDicomError
    REGISTERED_DICOM_CLASS=InvalidDicomError
    pixel_guard_sha=install_pixel_buffer_guard()
    if decoder.NeedMore.__module__!=decoder.__name__ or decoder.NeedMore.__name__!='NeedMore' or decoder.NeedMore.__bases__!=(Exception,):raise ValueError('EXACT_PINNED_PARSER_CONTROL_CLASS')
    REGISTERED_OLD_PARSER_CLASSES.add(decoder.NeedMore)
    source=textwrap.dedent(inspect.getsource(MI.validate_production_volume))
    tree=ast.parse(source)
    raises=[x for x in ast.walk(tree) if isinstance(x,ast.Raise) and isinstance(x.exc,ast.Call) and isinstance(x.exc.func,ast.Name) and x.exc.func.id=='ValueError' and 'Raptor study has no usable image content:' in ast.unparse(x.exc)]
    if len(raises)!=1:raise ValueError('EXACT_OLD_ALL_EMPTY_SITE')
    original_detail=raises[0].exc.args[0]
    raises[0].exc=ast.Call(ast.Name('_s6_data_local_failure',ast.Load()),[ast.Constant('old_no_usable_image')],[ast.keyword('site',ast.Constant('raptor_input.py:validate_production_volume:164')),ast.keyword('detail',original_detail)])
    transformed=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    namespace=dict(MI.validate_production_volume.__globals__);namespace['_s6_data_local_failure']=DataLocalFailure
    exec(compile(transformed,'<s6-old-all-empty>','exec'),namespace)
    MI.validate_production_volume=namespace['validate_production_volume']
    import sys
    def public_guard(error):guard_source_error(error,'old_public:'+sys._getframe(1).f_code.co_name)
    def decoder_guard(error):guard_source_error(error,'old_decoder:'+sys._getframe(1).f_code.co_name)
    # Exact pinned parser callbacks resolve the actual original catch function.
    public._raptor_propagate_fatal=public_guard
    decoder._mirror_propagate_fatal=decoder_guard
    return {'all_empty_guard_sha256':hashlib.sha256(transformed.encode()).hexdigest(),'pixel_buffer_guard_sha256':pixel_guard_sha,'unknown_source_exceptions':'fatal'}


def cleanup_study(scratch, sid, forced_side):
    root = Path(scratch)
    exact = root / ('s_' + hashlib.sha256(sid.encode()).hexdigest()[:16])
    if root.is_symlink() or exact.is_symlink(): raise ValueError('SCRATCH_PATH_ALIAS')
    if exact.exists(): shutil.rmtree(exact)
    forced_side.pop(sid, None)
    if exact.exists() or sid in forced_side: raise ValueError('PER_STUDY_CLEANUP_FAILED')
    return {'scratch_removed': True, 'side_override_removed': True}


def save_ledger(path, ids, probabilities, statuses, *, logits=None):
    import numpy as np
    from knee.public_adaptation_fit import atomic_save
    p = np.asarray(probabilities, dtype=np.float64).reshape((-1, 12))
    good = np.asarray(statuses) == 'genuine'
    if len(ids) != len(set(ids)) or p.shape != (len(ids), 12) or len(statuses) != len(ids) or any(s not in ('genuine', 'data_local_failure') for s in statuses):
        raise ValueError('RECOVERY_LEDGER_SCHEMA')
    if not np.isfinite(p[good]).all() or not ((p[good] >= 0) & (p[good] <= 1)).all() or not np.isnan(p[~good]).all():
        raise ValueError('GENUINE_ONLY_LEDGER_VALUES')
    arrays = dict(study_ids=np.asarray(ids), probabilities=p, statuses=np.asarray(statuses))
    if logits is not None:
        z = np.asarray(logits, dtype=np.float32).reshape((-1, 4, 12))
        if z.shape != (len(ids), 4, 12) or not np.isfinite(z[good]).all() or not np.isnan(z[~good]).all():
            raise ValueError('GENUINE_ONLY_LOGITS')
        arrays['logits'] = z
    atomic_save(path, None, lambda _, h: np.savez(h, **arrays))
