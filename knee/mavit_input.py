"""Active MaViT reader only: unchanged healthy math, declared malformed-view masking."""
import ast
from collections import Counter
import errno
from pathlib import Path
import sys
import threading
import types

from knee.mavit_checkpoint import SOURCE_HASHES, canonical, digest

VERSION='mavit-active-reader-v2'
DEFINITIONS={'IOStats','_fopen','_pread','_read_whole','_make_pyd','NeedMore','scan_header','_ds','_us','fast_hdr',
 '_pydicom_major','_transform','_lerp_percentile','RawSlice','fast_raw','_pct_selftest','Cfg','_Ctx','_scan_full',
 '_knee_cs','_knee_header_values','_knee_get_laterality','_knee_materialize','_knee_percentile_context','_knee_to_u8',
 '_knee_centers','_knee_crop_resize_u8','KneeFastReader','make_knee_dataset_reader'}
CONSTANTS={'STATS','ACCOUNT','_O_FLAGS','_EXT_VR','_STD_VR','T_IN','T_IPP','T_IOP','T_SPP','T_PI','T_NF','T_ROWS','T_COLS',
 'T_PS','T_BA','T_BS','T_PR','T_RI','T_RS','T_MLUT','_WANT_VR','TS_EXPLICIT_LE','TS_IMPLICIT_LE','_TS_BAD_HDR','_PIX_TAGS',
 '_uH','_uI','_uHH','_NUM','_CORRECT_UNUSED','_ALL_CODES','_PCT_OK','T_LAT','T_ILAT','KNEE_VIEW_KEYS','KNEE_VIEW','KNEE_LABELS'}
PREFIX='import glob, io, os, re, struct, threading\nfrom concurrent.futures import ThreadPoolExecutor\nimport numpy as np\n'
RECIPE={'out_size':288,'num_slice':8,'percentile':[1.,99.],'crop_mm':130.,'io_threads':32,'crop_only_if_nonsquare':True,
 'mode':'valid','selection':'first complete plane/fluid/fat key in metadata order; no alternative retry',
 'intensity':'full unique-modal-shape series1/99; truncate uint8 before scipy order1 zoom',
 'laterality':'public first explicit tag, then median-X; right sagittal depth/channels, other planes width',
 'positions':'public center/D; rounded clipped20-80% centers','malformed':'mask selected unusable view; no frame salvage',
 'empty':'whole-empty/whole-black fatal; preserve black-view public mask if another view usable'}
COUNTERS=('selected_views','missing_views','masked_views','decoded_files','shape_discarded_frames','mixed_shape_views',
 'ambiguous_shape_views','short_views','unsafe_pixel_views','invalid_geometry_views','invalid_spacing_views',
 'header_default_frames','header_geometry_default_frames','header_fallbacks','pixel_fallbacks','pixel_decode_failures',
 'laterality_unknown_views','laterality_conflict_views','black_views','resource_errors','whole_empty_studies')
CATCHES={'_make_pyd.hdr_from_bytes':1,'_pydicom_major':1,'_Ctx.px':1,'_knee_get_laterality':2}


class MalformedView(ValueError):
    def __init__(self,reason):self.reason=reason;super().__init__('Malformed selected MaViT view')


def fatal(error):
    torch=sys.modules.get('torch');oom=getattr(torch,'OutOfMemoryError',()) if torch else ()
    return (isinstance(error,(MemoryError,TimeoutError)) or (oom and isinstance(error,oom)) or
            (isinstance(error,OSError) and error.errno in {errno.ENOMEM,errno.EMFILE,errno.ENFILE,errno.ENOSPC,errno.EIO,errno.EAGAIN,getattr(errno,'EDQUOT',-1)}) or
            (type(error).__module__=='cv2' and getattr(error,'code',None)==-4))


def declared_catch(path,error):
    """Recognized malformed DICOM/geometry exceptions, never arbitrary program errors."""
    if isinstance(error,(ValueError,EOFError,OverflowError)):return True
    if path in ('_make_pyd.hdr_from_bytes','_knee_get_laterality') and isinstance(error,(TypeError,AttributeError,IndexError)):
        return True  # Original missing/malformed geometry defaults remain valid.
    if type(error).__module__=='pydicom.errors' and type(error).__name__=='InvalidDicomError':return True
    # These pixel exceptions must actually originate inside the pinned decoder.
    if path=='_Ctx.px' and isinstance(error,(AttributeError,RuntimeError,NotImplementedError)):
        tb=error.__traceback__
        while tb:
            if tb.tb_frame.f_globals.get('__name__','').startswith('pydicom.'):return True
            tb=tb.tb_next
    return False


def maskable_view_error(error):
    return (isinstance(error,(MalformedView,FileNotFoundError)) or
            (isinstance(error,RuntimeError) and str(error).startswith('Failed to decode DICOM:')))


def collect_workers(executor,one,jobs):
    """Observe every submitted result before allowing malformed-view masking.

    `Executor.map` stops yielding at its first exception, which can hide a later
    resource/programming error. Preserve successful job order and the original
    executor lifetime, then prioritize fatal, unknown, and declared errors.
    """
    futures=[executor.submit(one,job) for job in jobs];results=[None]*len(futures);errors=[]
    for index,future in enumerate(futures):
        try:results[index]=future.result()
        except BaseException as error:
            if not future.done():raise  # Main-thread alarm/interrupt: never consume it as a worker result.
            errors.append(error)
    if errors:
        fatal_errors=[e for e in errors if fatal(e) or not isinstance(e,Exception)]
        unknown_errors=[e for e in errors if not maskable_view_error(e)]
        raise (fatal_errors or unknown_errors or errors)[0]
    return results


def extract(source):
    if digest(source.encode())!=SOURCE_HASHES['my_fastread_knee.py']:raise ValueError('Pinned MaViT reader changed')
    nodes=[];found=set();mutations=0
    for n in ast.parse(source).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in DEFINITIONS:nodes.append(n);found.add(n.name)
        elif isinstance(n,ast.Assign):
            names={x.id for t in n.targets for x in ast.walk(t) if isinstance(x,ast.Name)}
            if names and names<=CONSTANTS:nodes.append(n);found.update(names)
        elif isinstance(n,ast.Expr) and ast.unparse(n) in ("_WANT_VR.setdefault(T_LAT, b'CS')","_WANT_VR.setdefault(T_ILAT, b'CS')"):
            nodes.append(n);mutations+=1
    if found!=DEFINITIONS|CONSTANTS or mutations!=2:raise ValueError('Active MaViT reader inventory differs')
    return PREFIX+ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n'


def transform(source,*,production):
    """Five catch prefixes plus one drained worker collector; science stays unchanged."""
    tree=ast.parse(source);counts=Counter();inserted=Counter();stack=[]
    class Protect(ast.NodeTransformer):
        def visit_ClassDef(self,node):
            stack.append(node.name);self.generic_visit(node);stack.pop();return node
        def visit_FunctionDef(self,node):
            stack.append(node.name);self.generic_visit(node);stack.pop();return node
        def visit_ExceptHandler(self,node):
            path='.'.join(stack)
            if path in CATCHES:
                if not isinstance(node.type,ast.Name) or node.type.id!='Exception' or node.name is not None:raise ValueError('Active catch changed')
                node.name='_mavit_error';node.body.insert(0,ast.parse(f'_mavit_caught({path!r}, _mavit_error)').body[0]);counts[path]+=1
            return node
        def visit_Assign(self,node):
            path='.'.join(stack);target=ast.unparse(node.targets[0])
            if path=='KneeFastReader._read_series' and target=='records' and ast.unparse(node.value)=='list(ex.map(_one, jobs))':
                inserted['workers']+=1;node.value=ast.parse('_mavit_collect(ex, _one, jobs)',mode='eval').body;return node
            if not production:return node
            if path=='KneeFastReader._read_series':
                if target=='shapes':
                    inserted['records']+=1;return [ast.parse('_mavit_records(records)').body[0],node]
                if target=='pixel_spacing0':
                    inserted['spacing']+=1;return [node,ast.parse('_mavit_spacing(pixel_spacing0)').body[0]]
                if target=='(lo, hi, pct_ctx)':
                    inserted['percentiles']+=1;return [node,ast.parse('_mavit_percentiles(lo, hi)').body[0]]
            if path=='KneeFastReader.__call__' and target=='(x, s, m)':
                if ast.unparse(node.value)!='self._read_series(series_dir, key[0], mode)':raise ValueError('Selected-view call changed')
                inserted['view']+=1
                handler=ast.ExceptHandler(type=ast.Name(id='Exception',ctx=ast.Load()),name='_mavit_view_error',
                    body=[*ast.parse('_mavit_mask(_mavit_view_error)').body,ast.Continue()])
                return ast.Try(body=[node],handlers=[handler],orelse=[],finalbody=[])
            return node
    tree=Protect().visit(tree)
    if dict(counts)!=CATCHES or dict(inserted)!=dict.fromkeys(('workers','records','spacing','percentiles','view') if production else ('workers',),1):
        raise ValueError('MaViT active catch/check inventory changed')
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def definitions(source,*,production):
    code=transform(extract(source),production=production);module=types.ModuleType('mavit_active_reader')
    def caught(path,error):
        if fatal(error):raise error
    module._mavit_caught=caught;module._mavit_collect=collect_workers
    def forbidden(*args,**kwargs):raise RuntimeError('Inactive reader branch must not run')
    module.fast_px=forbidden  # fast=True/lut=True always uses fast_raw instead.
    exec(compile(code,'<audited-mavit-active-reader>','exec'),module.__dict__)
    return module


def policy_contract(source):
    original=extract(source)
    return {'version':VERSION,'sha256':digest(Path(__file__).read_bytes()),'recipe':RECIPE,
            'source_sha256':SOURCE_HASHES['my_fastread_knee.py'],'active_source_sha256':digest(original.encode()),
            'public_exception_only_sha256':digest(transform(original,production=False).encode()),
            'production_sha256':digest(transform(original,production=True).encode()),'fatal_catches':CATCHES,'workers':'drain all submitted futures; preserve result order; fatal then unknown errors take precedence'}


def prepare(module,reader,study,rows,root,*,production):
    """Install only observation/acceptance hooks; no input-array mutation."""
    import numpy as np
    counts=Counter(dict.fromkeys(COUNTERS,0));lock=threading.Lock();lateral=set()
    def add(key,n=1):
        with lock:counts[key]+=n
    def propagate(error):
        if fatal(error):
            if not getattr(error,'_mavit_counted',False):add('resource_errors');error._mavit_counted=True
            raise error
    def caught(path,error):
        propagate(error)
        if production and not declared_catch(path,error):raise error
        if path=='_make_pyd.hdr_from_bytes':add('header_default_frames')
        if path=='_Ctx.px':add('pixel_decode_failures')
    def records_check(records):
        add('decoded_files',len(records));shapes=Counter()
        for r in records:
            a=r['a'];shape=a.shape
            if len(shape)!=2 or min(shape)<=0:raise MalformedView('unsafe_pixel_views')
            if isinstance(a,module.RawSlice):
                values=a.codes_to_float(np.flatnonzero(a.cnt).astype(np.uint16))
                if a.mono1:values=values.max()-values
            else:values=np.asarray(a)
            if not np.isfinite(values).all():raise MalformedView('unsafe_pixel_views')
            if not np.isfinite(r['rec'][0]):raise MalformedView('invalid_geometry_views')
            shapes[tuple(shape)]+=1
        if not shapes:raise MalformedView('short_views')
        maximum=max(shapes.values())
        if sum(n==maximum for n in shapes.values())!=1:raise MalformedView('ambiguous_shape_views')
        if maximum<3:raise MalformedView('short_views')
        add('shape_discarded_frames',len(records)-maximum);add('mixed_shape_views',int(len(shapes)>1))
    def spacing(value):
        if not np.isfinite(value) or value<=0:raise MalformedView('invalid_spacing_views')
    def percentiles(lo,hi):
        if not np.isfinite([lo,hi]).all() or hi<lo:raise MalformedView('unsafe_pixel_views')
    def mask(error):
        propagate(error)
        if isinstance(error,MalformedView):add(error.reason)
        elif maskable_view_error(error):pass
        else:raise error  # Unknown programming/data exceptions are not guessed away.
        add('masked_views')
    originals={n:getattr(module,n,None) for n in ('_mavit_caught','_mavit_records','_mavit_spacing','_mavit_percentiles','_mavit_mask')}
    original_header=module._knee_header_values;original_laterality=module._knee_get_laterality
    original_hdr=reader.ctx.hdr_b;original_px=reader.ctx.px_b
    def header(*args):
        h=original_header(*args)
        if any(h[k] is None for k in ('iop','ipp','ps')):add('header_geometry_default_frames')
        for k in ('image_laterality','laterality'):
            v=h[k]
            if v and v[0] in ('L','R'):lateral.add(v[0])
        return h
    def laterality(records):
        lateral.clear();value=original_laterality(records)
        add('laterality_unknown_views',int(value is None));add('laterality_conflict_views',int(len(lateral)>1));return value
    def hdr(*args):add('header_fallbacks');return original_hdr(*args)
    def px(*args):add('pixel_fallbacks');return original_px(*args)
    try:
        module._mavit_caught=caught
        if production:
            module._mavit_records=records_check;module._mavit_spacing=spacing;module._mavit_percentiles=percentiles;module._mavit_mask=mask
            module._knee_header_values=header;module._knee_get_laterality=laterality;reader.ctx.hdr_b=hdr;reader.ctx.px_b=px
        selected={(r['Anatomical_Plane'],int(r['Fluid_Sensitive']),int(r['Fat_Suppression'])) for r in rows if str(r['StudyInstanceUID'])==str(study)}
        if not selected<=set(module.KNEE_VIEW_KEYS):raise ValueError('Unsupported MaViT metadata view key')
        counts['selected_views']=len(selected)
        result=reader(study,rows,str(root),'valid')
        v,m,p=(result[k] for k in ('volume','view_mask','slice'))
        if (v.shape!=(6,8,3,reader.out_size,reader.out_size) or v.dtype!=np.uint8 or m.shape!=(6,) or m.dtype!=bool or
                p.shape!=(6,8) or p.dtype!=np.float32 or not np.isfinite(p).all() or np.any((p<0)|(p>1))):
            raise ValueError('MaViT tensor schema differs')
        counts['missing_views']=6-int(m.sum());counts['black_views']=sum(bool(m[i]) and not np.any(v[i]) for i in range(6))
        if not m.any() or not v.any():add('whole_empty_studies');raise ValueError('Whole-empty/black MaViT study')
        return (v,m,p),dict(counts)
    except Exception as error:
        if fatal(error) and not getattr(error,'_mavit_counted',False):add('resource_errors')
        error.mavit_counts=dict(counts);raise
    finally:
        for name,value in originals.items():
            if value is None:module.__dict__.pop(name,None)
            else:setattr(module,name,value)
        module._knee_header_values=original_header;module._knee_get_laterality=original_laterality
        reader.ctx.hdr_b=original_hdr;reader.ctx.px_b=original_px
