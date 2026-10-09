"""Image-only exact-pixel diagnostic; historical44 recipe remains a hypothesis."""
import ast
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
import types
import warnings

from knee import public_training_smoke as smoke
from knee.fresh_reader_setup import dataset_root, notebook_root, exact_root, time_limit
from knee.raptor_input import guarded_reader

SOURCE_SHA = '859aae12ecea593b45ab0a5f6463fa573cd3083f340c161e920f10fbba29e81b'
SLOTS = [('Sagittal', 1, 12), ('Sagittal', 0, 10), ('Coronal', 1, 8), ('Coronal', 0, 6), ('Axial', -1, 8)]
SUBSTITUTIONS = {'SLOTS': SLOTS, 'MAXS': 44, 'build_study.span': [[0.06, 0.15], [0.94, 0.85]]}


def recipe(source):
    """Extract only three pinned functions; never run public imports/main/models."""
    if smoke.digest(source.encode()) != SOURCE_SHA:
        raise ValueError('Archived reader source differs')
    tree = ast.parse(source)
    names = ['_make_reader', '_pick_series_for_slot', 'build_study']
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if [n.name for n in nodes] != names:
        raise ValueError('Reader definition inventory differs')
    changed = Counter()
    class Span(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is float and node.value in (.06, .94):
                changed[node.value] += 1
                return ast.copy_location(ast.Constant({.06:.15,.94:.85}[node.value]),node)
            return node
    Span().visit(nodes[-1])
    if changed != {.06:1,.94:1}:
        raise ValueError('Span substitution inventory differs')
    code = ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])))+'\n'
    return code


def reader_module(code):
    import glob
    import numpy as np
    module = types.ModuleType('corpus44_hypothesis')
    module.__dict__.update(np=np,glob=glob,IMG=336,CROP_MM=140.,SLOTS=SLOTS,MAXS=44)
    exec(compile(code,'<pinned44-hypothesis>','exec'),module.__dict__)
    return module


def compare(volume, mask, expected, expected_mask):
    import numpy as np
    if volume.shape != (44,336,336) or expected.shape != volume.shape or volume.dtype != np.uint8 or expected.dtype != np.uint8:
        raise ValueError('Volume schema differs')
    for m in (mask,expected_mask):
        if m.shape != (44,) or not np.isin(m,[0,1]).all():
            raise ValueError('Mask schema differs')
    slots=[];offset=0
    for plane,fluid,count in SLOTS:
        a=volume[offset:offset+count];b=expected[offset:offset+count]
        d=np.abs(a.astype(np.int16)-b.astype(np.int16))
        slots.append({'slot':len(slots),'plane':plane,'fluid_preference':fluid,'shape':list(a.shape),
            'exact_hash_match':smoke.digest(a.tobytes())==smoke.digest(b.tobytes()),
            'mask_match':bool(np.array_equal(mask[offset:offset+count],expected_mask[offset:offset+count])),
            'mask_disagreements':int(np.count_nonzero(mask[offset:offset+count]!=expected_mask[offset:offset+count])),
            'different_pixels':int(np.count_nonzero(d)),'mean_absolute_error_uint8':float(d.mean()),
            'max_absolute_error_uint8':int(d.max()),
            'reconstructed_valid_frames':int(mask[offset:offset+count].sum()),
            'corpus_valid_frames':int(expected_mask[offset:offset+count].sum())})
        offset+=count
    return slots


def run(inputs,out,contract,code,*,started):
    """Never persist IDs, images, labels, exception text or raw DICOM metadata."""
    import numpy as np
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','phase':'setup','version':1,'recipe_status':'hypothesis',
        'contract_sha256':smoke.digest(smoke.canonical(contract)),'studies':[]}
    def save():
        report['elapsed_seconds']=time.monotonic()-started
        report['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        p=out/'parity-report.json.tmp';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(out/'parity-report.json')
    save()
    try:
        with time_limit(43200-(time.monotonic()-started),'actual platform12h limit'):
            if smoke.digest(code.encode())!=contract['reader_sha256'] or contract['substitutions']!=json.loads(json.dumps(SUBSTITUTIONS)):
                raise ValueError('Reader contract differs')
            import subprocess
            warnings.filterwarnings('ignore',module=r'pydicom(?:\..*)?')
            codecroot=dataset_root(inputs,'goodpjw2008/rsna-knee-2-5d-convnext-reader',list(contract['codecs']))
            wheels=[]
            for name,sha in contract['codecs'].items():
                smoke.require_file(codecroot/name,sha);wheels.append(str(codecroot/name))
            subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*wheels],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            report['codec_hashes']=contract['codecs']
            identity=notebook_root(inputs,smoke.IDENTITY_KERNEL,'sealed_identity',['sealed-manifest-private.json'])
            selected,cohort=smoke.membership(identity/'sealed-manifest-private.json')
            if cohort!=contract['smoke_cohort']:
                raise ValueError('Smoke membership differs')
            selected=selected[:5]
            if smoke.digest(smoke.canonical(selected))!=contract['selected5_sha256']:
                raise ValueError('Five-study membership differs')
            inputs=Path(inputs)
            competition=exact_root([inputs/'competitions/rsna-knee-abnormality-detection',inputs/'rsna-knee-abnormality-detection'],['train_series.csv'])
            roots=[dataset_root(inputs,ref,[f'{prefix}_{suffix}.npy' for suffix in ('ids','masks','vols')]) for ref,prefix in zip(smoke.CORPORA,['all','extra'])]
            ids=[];vols=[];masks=[]
            for root,prefix,size in zip(roots,['all','extra'],[3200,1207]):
                for suffix in ('ids','masks'):
                    name=f'{prefix}_{suffix}.npy';smoke.require_file(root/name,smoke.PINS[name])
                ids.extend(np.load(root/f'{prefix}_ids.npy',allow_pickle=False).astype(str))
                v=np.load(root/f'{prefix}_vols.npy',mmap_mode='r',allow_pickle=False)
                if v.shape!=(size,44,336,336) or v.dtype!=np.uint8:raise ValueError('Mounted volume schema differs')
                vols.append(v);masks.append(np.load(root/f'{prefix}_masks.npy',allow_pickle=False))
            if len(set(ids))!=4407 or not set(selected)<=set(ids):raise ValueError('Corpus coverage differs')
            records={u:[] for u in selected}
            with (competition/'train_series.csv').open(newline='') as f:
                for row in csv.DictReader(f):
                    if row['StudyInstanceUID'] in records:
                        records[row['StudyInstanceUID']].append(row)
            report['environment']={'python':sys.version.split()[0],'device':'cpu','packages':{p:importlib.metadata.version(p) for p in ('numpy','pydicom','opencv-python','pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg')}}
            import cv2
            cv2.setNumThreads(4)
            public=reader_module(code);report['phase']='compare';save()
            for ordinal,uid in enumerate(selected):
                counters=Counter();begin=time.monotonic()
                report['active_ordinal']=ordinal
                report['active_reader_counts']=counters
                original_factory=public._make_reader
                def tracked_factory():
                    order,pixel,resize=original_factory()
                    def tracked_pixel(path):
                        try:
                            value=pixel(path);counters['decoded_frames']+=1;return value
                        except BaseException:
                            counters['decode_failures']+=1;raise
                    return order,tracked_pixel,resize
                public._make_reader=tracked_factory
                with guarded_reader(public,counters):
                    volume,mask=public.build_study(uid,records,str(competition/'train_images'),public._make_reader())
                public._make_reader=original_factory
                report.pop('active_ordinal');report.pop('active_reader_counts')
                row=ids.index(uid);part=0 if row<3200 else 1;offset=row if part==0 else row-3200
                slots=compare(volume,mask,vols[part][offset],masks[part][offset])
                report['studies'].append({'ordinal':ordinal,'slots':slots,'reader_counts':dict(counters),'elapsed_seconds':time.monotonic()-begin})
                save()
            report['status']='EXACT_MATCH' if all(s['exact_hash_match'] and s['mask_match'] for r in report['studies'] for s in r['slots']) else 'MISMATCH'
            report['phase']='complete';save()
    except BaseException as error:
        report['status']='ERROR';report['failure']={'phase':report['phase'],'type':type(error).__name__};save()
        raise RuntimeError('Image-only parity diagnostic failed; sanitized aggregate report saved') from None
    return report
