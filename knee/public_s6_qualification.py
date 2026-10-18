"""Fail-closed, image-only frozen32 S6 qualification. ML imports are worker-local.

The released source modules are executed unchanged. Instrumentation is an external
observer; its input fingerprints must equal an uninstrumented fresh process.
Anatomy always remains UNKNOWN until the private complete inventory is reviewed.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

UID = 'StudyInstanceUID'
SAMPLE_SHA = '1e4cf23b048d1ced8c960ee8035abbfe1f43aba4992feb6c4d42d4949748c414'
TRAIN_SHA = '8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33'
FOLD_SHA = '12bebc897a549d660cf5e6ab597c9a8d8f84dc887b4abd4c99b67fb4a25cb3ae'
EXCLUDED_SHA = '44834d7638eb60142a668a7597bd3093fe569e13b6f77e60a9c185a4ac3206f1'
LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
SERIES_COLUMNS = [UID,'SeriesInstanceUID','Fluid_Sensitive','Fat_Suppression','Anatomical_Plane']
RECIPE = dict(backbone='convnext_small',size=384,crop=352,views='v4',slices=3,phys_mm=3.5,eval_stride=1,chunk=16,workers=0,folds=4,amp='fp16')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(json.dumps(value, indent=2, allow_nan=False).encode()+b'\n')
    temp.replace(path)


def checked_json(path, sha):
    require(file_sha(path)==sha, 'Pinned file identity differs')
    return json.loads(Path(path).read_bytes())


def validate_membership(ids, folds, excluded, sample_sha=SAMPLE_SHA):
    require(isinstance(ids,list) and len(ids)==32 and len(set(ids))==32, 'Exactly32 unique studies required')
    require(all(isinstance(u,str) and u and '/' not in u and '\\' not in u and u not in ('.','..') for u in ids), 'Invalid study ID')
    require(digest(ids)==sample_sha, 'Original frozen32 order differs')
    require(all(u in folds and isinstance(folds[u],int) and folds[u]>0 for u in ids), 'Training-only non-gold/nonfold0 membership required')
    require(not set(ids).intersection(excluded['quarantined_study_ids']), 'Quarantined membership')


def schema_header(path):
    with Path(path).open(newline='') as f:
        row=next(csv.reader(f))
    require(row==[UID]+LABELS, 'Official target schema/order differs')
    return row


def stage_inputs(raw, staged, ids, series_sha):
    """Read only selected metadata columns; train.csv never inspected for values."""
    raw, staged = Path(raw), Path(staged)
    require(not staged.exists(), 'Fresh isolated input root required')
    require(file_sha(raw/'train.csv')==TRAIN_SHA, 'Official train bytes differ')
    require(file_sha(raw/'train_series.csv')==series_sha, 'Official series bytes differ')
    rows=[]
    with (raw/'train_series.csv').open(newline='') as f:
        reader=csv.DictReader(f)
        require(set(SERIES_COLUMNS).issubset(reader.fieldnames), 'Missing series schema')
        for row in reader:
            if row[UID] in set(ids): rows.append({k:row[k] for k in SERIES_COLUMNS})
    require({r[UID] for r in rows}==set(ids), 'Selected metadata coverage differs')
    pairs=[(r[UID],r['SeriesInstanceUID']) for r in rows]
    require(len(pairs)==len(set(pairs)), 'Duplicate series metadata')
    require(all(s and '/' not in s and '\\' not in s and s not in ('.','..') for _,s in pairs), 'Invalid series ID')
    staged.mkdir(parents=True)
    with (staged/'train.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow([UID]);w.writerows([[u] for u in ids])
    with (staged/'train_series.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=SERIES_COLUMNS);w.writeheader();w.writerows(rows)
    images=staged/'train_series'; images.mkdir()
    inventory=[]
    for u,s in pairs:
        src=raw/'train_series'/u/s
        require(src.is_dir() and src.resolve().is_relative_to((raw/'train_series'/u).resolve()), 'Image path escapes selected study')
        require(any(src.glob('*.dcm')), 'Selected series has no DICOM')
        dst=images/u/s;dst.parent.mkdir(exist_ok=True);dst.symlink_to(src.resolve(),target_is_directory=True)
        inventory.append([u,s,str(src.resolve())])
    receipt={'train_sha256':file_sha(staged/'train.csv'),'series_sha256':file_sha(staged/'train_series.csv'),
             'membership_sha256':digest(ids),'series_membership_sha256':digest(pairs),'image_paths_sha256':digest(inventory),'studies':32,'series':len(rows)}
    verify_staged(staged,ids,receipt)
    return receipt


def verify_staged(staged, ids, receipt):
    staged=Path(staged)
    require(set(p.name for p in staged.iterdir())=={'train.csv','train_series.csv','train_series'}, 'Unexpected exposed table')
    with (staged/'train.csv').open(newline='') as f:
        r=csv.DictReader(f);require(r.fieldnames==[UID], 'Label-bearing train schema prohibited');got=[x[UID] for x in r]
    require(got==ids, 'Staged study order differs')
    with (staged/'train_series.csv').open(newline='') as f:
        r=csv.DictReader(f);require(r.fieldnames==SERIES_COLUMNS, 'Unexpected series columns');rows=list(r)
    pairs=[(r[UID],r['SeriesInstanceUID']) for r in rows]
    require({u for u,_ in pairs}==set(ids) and len(pairs)==len(set(pairs)), 'Series membership differs')
    require(file_sha(staged/'train.csv')==receipt['train_sha256'] and file_sha(staged/'train_series.csv')==receipt['series_sha256'], 'Staged bytes changed')
    actual=[(p.parent.name,p.name) for p in (staged/'train_series').glob('*/*')]
    require(set(actual)==set(pairs), 'Exposed image membership differs')
    inventory=[]
    for u,s in pairs:
        p=staged/'train_series'/u/s
        require(p.is_symlink() and p.is_dir(), 'Images must be immutable source symlinks')
        inventory.append([u,s,str(p.resolve())])
    require(digest(inventory)==receipt['image_paths_sha256'], 'Resolved image paths changed')


def verify_sources(code, pins):
    code=Path(code)
    require(len([n for n in pins if n.startswith('training/vendor_runtime/monai060/')])==194, 'Incomplete194-file vendor closure')
    actual={p.relative_to(code).as_posix() for p in code.rglob('*.py')}
    require(actual==set(pins), 'Source closure membership differs')
    for name,pin in pins.items():
        require(file_sha(code/name)==pin, 'Source module bytes changed')


def verify_assets(compact, anatomy, manifest, binding):
    require(file_sha(Path(compact)/'manifest.json')==binding['manifest_sha256'], 'Mounted manifest differs')
    require(manifest['models']==['E_lrprior_s0_f%d.pt'%i for i in range(4)], 'Four exact fold files required')
    for name,pin in manifest['sha256'].items():
        require(file_sha(Path(compact)/name)==pin, 'Mounted reader/side/spec identity differs')
    require(file_sha(anatomy)==manifest['anchor_weights_sha256'], 'Mounted anatomy identity differs')
    for key in ('backbone','size','crop','views','slices','phys_mm','eval_stride'):
        require(manifest[key]==RECIPE[key], 'Released S6 recipe differs')


def sentinels(window_counts):
    require(len(window_counts)==32 and all(n>0 for n in window_counts), 'Complete32 window counts required')
    result=[0,1]
    for i in sorted(range(32),key=lambda i:(-window_counts[i],i)):
        if i not in result:result.append(i)
        if len(result)==4:break
    return result


def retained_rectangle(geometry, crop=352):
    """Actual retained native rectangle after384->352 and letterbox clipping."""
    size=geometry.size;off=(size-crop)//2
    x0,y0,x1,y1=geometry.crop_xyxy;px,py,px1,py1=geometry.valid_xyxy
    a,b,c,d=max(off,px),max(off,py),min(off+crop,px1),min(off+crop,py1)
    if c<=a or d<=b:return None
    return [x0+(a-px)*(x1-x0)/(px1-px),y0+(b-py)*(y1-y0)/(py1-py),
            x0+(c-px)*(x1-x0)/(px1-px),y0+(d-py)*(y1-y0)/(py1-py)]


def slab_positions(series):
    """Audit the exact source stable physical ordering and thin-stack grouping."""
    import statistics
    pos=list(map(float,series['positions']))
    if series['plane']=='Sagittal' and series.get('side')=='R':pos=[-v for v in pos]
    order=sorted(range(len(pos)),key=lambda i:pos[i]);ordered=[pos[i] for i in order]
    spacing=statistics.median([b-a for a,b in zip(ordered,ordered[1:])]) if len(ordered)>1 else 0.
    groups=[[i] for i in order]
    if 0<spacing<1.75:
        g=max(1,round(3.5/spacing));n=len(order)//g+(1 if len(order)%g>=max(1,g//2) else 0)
        if n>=2:groups=[order[j*g:min(len(order),(j+1)*g)] for j in range(n)]
    return [sum(pos[i] for i in group)/len(group) for group in groups],groups


def synthetic_source_checks():
    """Exercise actual pinned source functions before any real pixels."""
    import numpy as np
    import compact_train as CT
    import px_cache as PX
    import knee_anchor_bank_v2 as KB
    import training.input_provenance as IP
    import training.dicom_geometry as G
    import training.geometry_bank as GB
    projection_errors=[]
    for angle in (0.,.4):
        rot=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
        aff=np.stack([rot@np.array([[1.2,0.,10.],[0.,.7,20.],[0.,0.,3.5*z]]) for z in range(5)])
        for stack in (aff,aff[::-1]):
            for z in (0,2,4):
                point=stack[z]@np.array([12.25,7.75,1.]);j,c,r,d=KB.project(stack,point)
                error=float(np.linalg.norm(stack[j]@np.array([c,r,1.])-point));projection_errors.append(error)
                require(j==z and error<=1e-4 and d<=1e-4,'Synthetic patient projection failed')
    aligned=np.repeat(np.eye(3)[None],3,axis=0);aligned[:,2,2]=[0.,3.5,7.]
    require(GB.alignment_quality(aligned)['aligned'],'Synthetic aligned geometry failed')
    shifted=aligned.copy();shifted[2,0,2]+=2.
    require(not GB.alignment_quality(shifted)['aligned'],'Synthetic unaligned geometry accepted')
    for plane in ('Sagittal','Coronal','Axial'):
        require(G.flip_for_plane(plane,'R')==(plane!='Sagittal') and not G.flip_for_plane(plane,'L'),'Synthetic side canonicalization differs')
        views=PX.series_views(plane,(3,80,160),[[.7,1.2]]*3,384,'v4',None,(.7,1.2),None)
        require(views[0]==(0,None,(.5,.5)) and len(views)==(3 if plane=='Coronal' else 4),'Synthetic missing-anchor view semantics differ')
    geo=IP.view_geometry((80,160),(.7,1.2),384)
    require(geo.resized_shape[0]<384 and geo.resized_shape[1]==384,'Synthetic anisotropic letterbox failed')
    require(retained_rectangle(geo) is not None,'Synthetic retained crop missing')
    for n in (1,2,5):
        meta=np.zeros((n,12),np.float32);meta[:,5]=np.log1p(3.5)
        blocks=[{'n_slices':n,'views':[(0,0,0)]}]
        rows=CT.window_rows(blocks,lambda f,r:meta[r],'eval',3,1,phys_mm=3.5)
        require(len(rows)==n and all(0<=v<n for row in rows for v in row[2:]),'Synthetic short-stack padding failed')
        require(rows[0][2]==0 and rows[-1][-1]==n-1,'Synthetic repeated boundary index failed')
    require(CT.window_rows([],lambda f,r:None,'eval',3,1,phys_mm=3.5)==[],'Missing-plane semantics differ')
    p,g=slab_positions({'positions':[7.,3.5,3.5,0.],'plane':'Sagittal','side':'L'})
    require(g==[[3],[1],[2],[0]],'Duplicate stable ordering failed')
    return {'status':'PASS','projection_max_error_mm':max(projection_errors),'cases':['anisotropic','reversed','oblique','unaligned','left_right','short','duplicate','missing_planes','missing_anchors','boundary_padding']}


def worker(config_path, mode, ordinals):
    """One fresh process; no source monkeypatch is installed twice."""
    import numpy as np
    import torch
    import contextlib
    import resource
    import setuptools
    import distutils.util  # exact MONAI0.6 Python3.12 compatibility shim
    cfg=json.loads(Path(config_path).read_bytes());out=Path(cfg['out'])/mode;out.mkdir(parents=True)
    receipt={'status':'RUNNING','completed':0,'mode':mode,'pid':os.getpid(),'phase':'setup','recipe':RECIPE,'unknowns':['actual anatomical acceptance','source default chunk192 equivalence'],'records':[]}
    save(out/'receipt.json',receipt)
    started=time.perf_counter()
    try:
        verify_sources(cfg['code'],cfg['source_pins'])
        manifest=cfg['manifest'];verify_assets(cfg['compact'],cfg['anatomy'],manifest,cfg['binding'])
        verify_staged(cfg['staged'],cfg['ids'],cfg['stage_binding'])
        require(torch.cuda.is_available(), 'Explicit CUDA unavailable')
        require(torch.cuda.get_device_name(0)=='Tesla T4', 'Pinned T4 device differs')
        require('monai' not in sys.modules, 'Fresh MONAI namespace required')
        sys.path.insert(0,cfg['code']);os.environ['RSNA_CODE_ROOT']=cfg['code']
        import compact_infer as CI
        import compact_train as CT
        receipt['synthetic_source_checks']=synthetic_source_checks()
        require(CT.LABELS==LABELS, 'Explicit source-to-official mapping differs')
        state_spec=json.loads((Path(cfg['compact'])/'state_spec.json').read_bytes())
        def load_models():
            result=[]
            for name in manifest['models']:
                sd=torch.load(Path(cfg['compact'])/name,map_location='cpu',weights_only=True)
                kw={'spatial':any(k=='sp_q' or k.endswith('.sp_q') for k in sd),
                    'extent':any(k.split('.')[-1] in ('ext_u','ext_view','ext_gamma') for k in sd)}
                m=CT.Net('convnext_small',None,freeze_blocks=16,crop=352,state_spec=state_spec,in_chans=3,**kw)
                m.load_state_dict(sd,strict=True)
                result.append(m.eval().requires_grad_(False).to('cuda').to(memory_format=torch.channels_last))
                del sd
            return result
        torch.manual_seed(42);np.random.seed(42)
        torch.backends.cudnn.benchmark=False
        models=load_models()
        ds=CI.CStudies(cfg['staged'],'train',[cfg['ids'][i] for i in ordinals],out/'scratch',cfg['code'],384,1,'v4',3,cfg['anatomy'],3.5,
                       str(Path(cfg['compact'])/'side_cls.pt'),2.,0.,20.,extent_meta=any(m.extent for m in models))
        ds._setup()
        import monai
        require(Path(monai.__file__).resolve().is_relative_to(Path(cfg['code'])/'training/vendor_runtime/monai060'), 'MONAI import shadowed')
        require(monai.__version__=='0.6.0', 'MONAI version differs')
        states=models+[ds.bank.model,ds.side_net]
        def finite_auxiliary(module,inputs,output):
            require(torch.isfinite(output).all().item(),'Nonfinite localizer/side output')
        ds.bank.model.register_forward_hook(finite_auxiliary);ds.side_net.register_forward_hook(finite_auxiliary)
        def state_digest():
            h=hashlib.sha256()
            for index,m in enumerate(states):
                all_tensors={'parameter/'+n:t for n,t in m.named_parameters()}
                all_tensors.update({'buffer/'+n:t for n,t in m.named_buffers()})
                for name,tensor in sorted(all_tensors.items()):
                    a=tensor.detach().cpu().contiguous().numpy()
                    h.update(canonical([index,name,str(a.dtype),list(a.shape)]));h.update(a.tobytes())
            return h.hexdigest()
        initial=state_digest();receipt['state_before']=initial;receipt['setup_hash_load_seconds']=time.perf_counter()-started
        def array_sha(a):
            a=np.ascontiguousarray(a);h=hashlib.sha256(canonical([str(a.dtype),list(a.shape)]));h.update(a.tobytes());return h.hexdigest()
        capture={'ordinal':None,'item':None,'geometry':[],'anchors':None,'candidates':[],'series_views':[],'block_positions':[]}
        # Instrument external observers only. Each observer returns the source result unchanged.
        # Minimal byte observers run on the reference as well; no preprocessing changes.
        orig_window_rows=ds.window_rows
        def observed_windows(*args,**kwargs):
            rows=orig_window_rows(*args,**kwargs);capture['window_rows']=rows;return rows
        ds.window_rows=observed_windows
        if mode!='wrapper':
            original_build=ds.builder.build
            def byte_observer(uid):
                result=original_build(uid);capture['item']=result;return result
            ds.builder.build=byte_observer
        if mode=='wrapper':
            import training.input_provenance as IP
            import px_cache as PX
            orig_geo=IP.view_geometry;orig_build=ds.builder.build;orig_record=ds.bank.record;orig_candidate=ds.bank.candidate;orig_views=PX.series_views
            def observed_geo(*args,**kwargs):
                g=orig_geo(*args,**kwargs);capture['geometry'].append({'shape':list(g.native_shape),'spacing':list(g.spacing),'crop':list(g.crop_xyxy),'valid':list(g.valid_xyxy),'retained':retained_rectangle(g)});return g
            def observed_candidate(s,g):
                try:
                    c,why=orig_candidate(s,g)
                    capture['candidates'].append({'series':s['SeriesInstanceUID'],'why':why,'candidate':None if c is None else {k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in c.items()}})
                    return c,why
                except BaseException as e:
                    capture['candidates'].append({'series':s['SeriesInstanceUID'],'exception_type':type(e).__name__});raise
            def observed_record(*args,**kwargs):
                r=orig_record(*args,**kwargs);capture['anchors']=r;return r
            def observed_views(*args,**kwargs):
                v=orig_views(*args,**kwargs);capture['series_views'].append(v);return v
            def observed_build(uid):
                item=orig_build(uid);capture['item']=item
                key=hashlib.sha256(uid.encode()).hexdigest()[:24]
                study=json.loads((Path(ds.builder.cache)/'studies'/(key+'.json')).read_bytes())
                require(not study.get('errors',[]) and all(not s.get('errors') for s in study['series']), 'Source swallowed decoder/side errors')
                require(not item['skipped_series'], 'Source skipped series')
                anchors=capture['anchors']
                require(not any(str(w).startswith('error:') for w in anchors['why']), 'Source swallowed localizer error')
                require(not any('exception_type' in c for c in capture['candidates']), 'Source swallowed candidate exception')
                capture['block_positions']=[slab_positions(next(s for s in study['series'] if s['SeriesInstanceUID']==b['series']))[0] for b in item['blocks']]
                require(all(len(pos)==b['n_slices'] for pos,b in zip(capture['block_positions'],item['blocks'])),'Physical slab mapping differs')
                geometry_dir=Path(ds.builder.cache)/'geometry'
                geo_private={p.name:json.loads(p.read_bytes()) for p in geometry_dir.glob('*.json')}
                errors=[]
                for geo in geo_private.values():
                    if 'affines_col_row_origin_lps_mm' not in geo:continue
                    aff=np.array(geo['affines_col_row_origin_lps_mm']);require(np.isfinite(aff).all(),'Nonfinite source affine')
                    for point in (anchors['joint_lps'],anchors['patella_lps']):
                        if point is None:continue
                        point=np.array(point);z,c,r,d=ds.KB.project(aff,point)
                        normal=np.cross(aff[0,:,0],aff[0,:,1]);normal/=np.linalg.norm(normal)
                        distances=(aff[:,:,2]-point)@normal;require(z==int(np.argmin(np.abs(distances))),'Source nearest-slice projection differs')
                        expected=point+normal*distances[z];err=float(np.linalg.norm(aff[z]@np.array([c,r,1.])-expected))
                        require(err<=1e-4,'Actual patient-coordinate projection differs');errors.append(err)
                capture['projection_max_error_mm']=max(errors,default=0.)
                capture['scratch_bytes']=sum(p.stat().st_size for p in Path(ds.builder.cache).rglob('*') if p.is_file())
                _contact_sheets(out,capture['ordinal'],study,item,anchors,capture['geometry'],geo_private)
                save(out/('private-%03d.json'%capture['ordinal']),{'anchors':anchors,'candidates':capture['candidates'],'geometry':geo_private,
                     'render_geometry':capture['geometry'],'view_centers':capture['series_views'],'blocks':item['blocks']})
                return item
            IP.view_geometry=observed_geo;ds.builder.build=observed_build;ds.bank.record=observed_record;ds.bank.candidate=observed_candidate;PX.series_views=observed_views
        np_logits=[];np_probs=[];np_ord=[]
        for j,ordinal in enumerate(ordinals):
            receipt['phase']='study';receipt['current_ordinal']=ordinal
            capture.update(ordinal=ordinal,item=None,geometry=[],anchors=None,candidates=[],series_views=[])
            item=ds[j]
            require(item['error'] is None and len(item['images'])>0, 'Source returned failed/empty study')
            images=item['images'].numpy();meta=item['meta'].numpy()
            require(images.dtype==np.uint8 and images.shape[1:]==(3,384,384), 'Input image contract differs')
            fingerprint={'render_windows':array_sha(images),'metadata':array_sha(meta),'retained_crop':array_sha(images[:,:,16:368,16:368]),'mask':digest([True]*len(images))}
            if capture['item'] is not None:
                fingerprint.update(pixel_rows=array_sha(capture['item']['px'].numpy()),row_metadata=array_sha(capture['item']['meta'].numpy()),window_blocks=digest(capture['item']['blocks']),window_rows=digest(capture['window_rows']))
                rawmeta=capture['item']['meta'].numpy();wr=np.array(capture['window_rows'],dtype=np.int64)
                for triplet in wr[:,2:]:
                    require(len(set(rawmeta[triplet,0]))==1 and len(set(rawmeta[triplet,3]))==1,'Triplet crosses series/view')
                if mode=='wrapper':
                    save(out/('windows-%03d-private.json'%ordinal),{'rows':capture['window_rows'],'channel_positions_mm':[[capture['block_positions'][int(rawmeta[r,0])][int(rawmeta[r,9])] for r in t] for t in wr[:,2:]],'physical_gaps_mm':[[capture['block_positions'][int(rawmeta[t[0],0])][int(rawmeta[t[j+1],9])]-capture['block_positions'][int(rawmeta[t[0],0])][int(rawmeta[t[j],9])] for j in range(2)] for t in wr[:,2:]],'repeated_triplets':int(sum(len(set(t))<3 for t in wr[:,2:]))})
            if mode=='reference':
                receipt['records'].append({'ordinal':ordinal,'windows':len(images),'input':fingerprint,'preprocess_seconds':item['seconds']})
                receipt['completed']=j+1;save(out/'receipt.json',receipt)
                continue
            t=time.perf_counter()
            img=item['images'].to('cuda')[None,:,: ,16:368,16:368];mt=item['meta'].to('cuda')[None]
            mask=torch.ones((1,img.shape[1]),dtype=torch.bool,device='cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                zs=torch.stack([m(img,mt,mask,chunk=16).float()[0] for m in models])
            require(torch.isfinite(zs).all().item(), 'Nonfinite FP16 logits: FP32 recovery forbidden')
            zmean=zs.mean(0);p=torch.sigmoid(zmean).double().cpu().numpy();z=zs.cpu().numpy()
            require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(), 'Invalid probabilities')
            # Recompute exactly the released float32 mean/sigmoid path outside model execution.
            oracle=torch.sigmoid(torch.from_numpy(z).mean(0)).double().numpy()
            reconstruction_error=float(np.max(np.abs(p-oracle)))
            require(reconstruction_error<=1e-6, 'Mean-logit/sigmoid reconstruction differs')
            torch.cuda.synchronize();dt=time.perf_counter()-t
            np_logits.append(z);np_probs.append(p);np_ord.append(ordinal)
            with (out/'predictions-private.tmp').open('wb') as checkpoint:
                np.savez(checkpoint,ordinals=np_ord,logits=np_logits,probabilities=np_probs);checkpoint.flush();os.fsync(checkpoint.fileno())
            (out/'predictions-private.tmp').replace(out/'predictions-private.npz')
            anchor=item.get('anchor') or {}
            row={'ordinal':ordinal,'windows':len(images),'input':fingerprint,'preprocess_seconds':item['seconds'],'four_model_seconds':dt,'mean_sigmoid_reconstruction_error':reconstruction_error,
                 'anchor_method':anchor.get('method'),'anchor_reasons':anchor.get('why',[]),'side':item.get('side'),
                 'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'working_disk_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
            if mode=='wrapper':
                row['field_center_fallbacks']=sum(not e.get('in_field',True) for e in capture['anchors']['series'].values())+sum(b['series'] not in capture['anchors']['series'] for b in capture['item']['blocks'])
                row['used_series']=len(capture['item']['blocks']);row['geometry_sha256']=digest(capture['geometry'])
                row['projection_max_error_mm']=capture['projection_max_error_mm'];row['native_scratch_bytes']=capture['scratch_bytes']
                row['missing_planes']=sorted({'Sagittal','Coronal','Axial'}-{b['plane'] for b in capture['item']['blocks']})
            receipt['records'].append(row);receipt['completed']=j+1;receipt['elapsed_seconds']=time.perf_counter()-started
            save(out/'receipt.json',receipt)
            del img,mt,mask,zs,item,images,meta
        receipt['state_after']=state_digest();require(receipt['state_after']==initial, 'Model parameters/buffers mutated')
        with (out/'predictions-six-decimal-private.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow([UID]+LABELS)
            for i,p in zip(np_ord,np_probs):w.writerow([cfg['ids'][i]]+['%.6f'%x for x in p])
        receipt.update(status='PASS',zero_imputed=True,zero_failed=True,fp32_recoveries=0,environment={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'monai':monai.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'timm':__import__('timm').__version__,'pydicom':__import__('pydicom').__version__,'pandas':__import__('pandas').__version__,'scipy':__import__('scipy').__version__})
        save(out/'receipt.json',receipt)
    except BaseException as e:
        save(out/'failure-private.json',{'type':type(e).__name__,'detail':str(e)})
        receipt.update(status='FAIL',failure_type=type(e).__name__,failure_stage='worker',elapsed_seconds=time.perf_counter()-started)
        save(out/'receipt.json',receipt)
        raise RuntimeError('S6 qualification failed; preserve private outputs and inspect sanitized receipt') from None


def _contact_sheets(out, ordinal, study, item, anchors, geometries, geo_private):
    """Private ordinal inventory: every used series/view at10/50/90% + anchor depth."""
    import numpy as np
    from PIL import Image, ImageDraw
    series={s['SeriesInstanceUID']:s for s in study['series']};inventory=[]
    geo_cursor=0
    for si,b in enumerate(item['blocks']):
        s=series[b['series']];physical,groups=slab_positions(s);n=b['n_slices'];entry=anchors['series'].get(b['series']);zs={int(round((n-1)*p)) for p in (.1,.5,.9)}
        gp=hashlib.sha256((study[UID]+'/'+b['series']).encode()).hexdigest()[:24]+'.json'
        anchor_native_z=None
        if entry and anchors['joint_lps'] is not None and gp in geo_private:
            import knee_anchor_bank_v2 as KB
            anchor_native_z=KB.project(np.array(geo_private[gp]['affines_col_row_origin_lps_mm']),np.array(anchors['joint_lps']))[0]
            # Canonical native position sign matches px_cache's right-sagittal ordering.
            pos=np.array(s['positions']);pos=-pos if s['plane']=='Sagittal' and s.get('side')=='R' else pos
            anchor_group=int(np.argmin(np.abs(np.array(physical)-pos[anchor_native_z])));zs.add(anchor_group)
        zs=sorted(zs);tiles=[];rects=[]
        for vid,start in b['views']:
            for z in zs:
                image=item['px'][start+z].numpy()
                g=geometries[geo_cursor+z];rects.append({'view':vid,'depth_index':z,'retained_native_xyxy':g['retained'],'padding_valid_xyxy':g['valid']})
                tile=Image.new('RGB',(384,412),'black');tile.paste(Image.fromarray(image).convert('RGB'),(0,28));draw=ImageDraw.Draw(tile)
                draw.text((5,5),'%03d series%02d %s %s view%d z%d/%d'%(ordinal,si,s['plane'],s.get('side'),vid,z,n),fill='white')
                draw.rectangle((16,44,367,395),outline='yellow',width=2);tiles.append(tile)
            geo_cursor+=n
        # Native references with union of actual post-crop rectangles and projected anchors.
        native=np.load(s['path'],mmap_mode='r');positions=np.array(s['positions']);positions=-positions if s['plane']=='Sagittal' and s.get('side')=='R' else positions
        order=np.argsort(positions,kind='stable')
        for z in zs:
            nz=anchor_native_z if anchor_native_z is not None and z==anchor_group else groups[z][0]
            a=np.array(native[nz],np.float32);lo,hi=np.percentile(a,[1,99]);a=np.clip((a-lo)/max(hi-lo,1),0,1)
            im=Image.fromarray(np.rint(a*255).astype('uint8')).convert('RGB');draw=ImageDraw.Draw(im)
            for r in rects:
                if r['depth_index']==z and r['retained_native_xyxy'] is not None:draw.rectangle(r['retained_native_xyxy'],outline=['yellow','red','cyan','lime','orange','magenta'][r['view']%6],width=2)
            if entry:
                h,w=a.shape;rf,cf=entry['joint'];draw.ellipse((cf*(w-1)-5,rf*(h-1)-5,cf*(w-1)+5,rf*(h-1)+5),outline='white',width=2)
                if entry.get('patella'):
                    rf,cf=entry['patella'];draw.ellipse((cf*(w-1)-5,rf*(h-1)-5,cf*(w-1)+5,rf*(h-1)+5),outline='orange',width=2)
            im.thumbnail((384,384));tile=Image.new('RGB',(384,412),'black');tile.paste(im,(0,28));ImageDraw.Draw(tile).text((5,5),'Native %s %s depth%d anchorZ%s'%(s['plane'],s.get('side'),nz,anchor_native_z),fill='white');tiles.append(tile)
        sheet=Image.new('RGB',(384*len(zs),412*((len(tiles)+len(zs)-1)//len(zs))),'black')
        for k,tile in enumerate(tiles):sheet.paste(tile,((k%len(zs))*384,(k//len(zs))*412))
        filename='%03d-series-%02d.png'%(ordinal,si);sheet.save(out/filename)
        inventory.append({'ordinal':ordinal,'series_ordinal':si,'sheet':filename,'sheet_sha256':file_sha(out/filename),'plane':s['plane'],'side':s.get('side'),'views':[v for v,_ in b['views']],
                          'depth_indices':zs,'joint_anchor_native_z':anchor_native_z,'field_center_fallback':entry is None or not entry.get('in_field',True),'retained_coverage':rects})
    require(geo_cursor==len(geometries), 'Render geometry inventory differs')
    save(out/('%03d-inventory-private.json'%ordinal),inventory)


def run(config):
    """Orchestrate isolated workers; never scores labels or advances to870."""
    import numpy as np
    config=Path(config);cfg=json.loads(config.read_bytes());out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    receipt={'status':'RUNNING','source_state':'UNKNOWN','mechanics':'UNKNOWN','anatomy':'UNKNOWN','prediction':'UNKNOWN','optimizer_updates':0,'completed_studies':0,'binding':cfg['binding'],'recipe':RECIPE,'unknowns':['private complete contact-sheet review required','source default chunk192 equivalence untested']}
    require(not (out/'receipt.json').exists(), 'Existing qualification cannot be restarted automatically')
    save(out/'receipt.json',receipt);started=time.perf_counter()
    try:
        cfg['stage_binding']=stage_inputs(cfg['raw'],cfg['staged'],cfg['ids'],cfg['binding']['series_sha256']);save(config,cfg)
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
        for mode,ords in [('reference',list(range(32))),('wrapper',list(range(32)))]:
            subprocess.run([sys.executable,'-m','knee.public_s6_qualification','worker',str(config),mode,json.dumps(ords)],check=True,env=env)
        ref=json.loads((out/'reference/receipt.json').read_bytes());wrapped=json.loads((out/'wrapper/receipt.json').read_bytes())
        require(ref['status']==wrapped['status']=='PASS','Initial workers failed')
        require(ref['state_before']==wrapped['state_before'],'Strict fresh reconstruction differs')
        for a,b in zip(ref['records'],wrapped['records']):
            require(all(a['input'][k]==b['input'][k] for k in a['input']),'All32 reference/wrapper inputs differ')
        sentinel=sentinels([r['windows'] for r in wrapped['records']]);receipt['sentinel_ordinals']=sentinel
        for mode in ('reference_prediction','repeat','reload'):
            subprocess.run([sys.executable,'-m','knee.public_s6_qualification','worker',str(config),mode,json.dumps(sentinel)],check=True,env=env)
        baseline=np.load(out/'wrapper/predictions-private.npz',allow_pickle=False);errors={}
        for mode in ('reference_prediction','repeat','reload'):
            result=np.load(out/mode/'predictions-private.npz',allow_pickle=False);rs=json.loads((out/mode/'receipt.json').read_bytes())
            require(rs['status']=='PASS' and rs['state_before']==wrapped['state_before'],'Worker state/reconstruction differs')
            inds=result['ordinals'].tolist();error=float(np.max(np.abs(result['probabilities']-baseline['probabilities'][inds])))
            require(error<=1e-6,'Frozen probability tolerance exceeded');errors[mode]=error
            for row in rs['records']:
                require(all(row['input'][k]==wrapped['records'][row['ordinal']]['input'][k] for k in row['input']),'Fresh decode/window fingerprint differs')
        require(any(str(r['anchor_method']).startswith('skm_tea_joint') for r in wrapped['records']),'No learned-anchor path exercised')
        inventories=[json.loads((out/'wrapper'/('%03d-inventory-private.json'%i)).read_bytes()) for i in range(32)]
        require(all(len(inv)==wrapped['records'][i]['used_series'] for i,inv in enumerate(inventories)),'Contact inventory coverage differs')
        counts=[r['windows'] for r in wrapped['records']]
        preprocess=sum(sum(r['preprocess_seconds']) for r in wrapped['records']);inference=sum(r['four_model_seconds'] for r in wrapped['records'])
        methods=[r['anchor_method'] for r in wrapped['records']]
        strata={'learned_anchor':sum(str(m).startswith('skm_tea_joint') for m in methods),'slab_prior':methods.count('slab_prior'),
                'field_center':sum(r['field_center_fallbacks'] for r in wrapped['records']),'side_flipped':sum(bool((r.get('side') or {}).get('flipped')) for r in wrapped['records']),
                'missing_planes':sum(bool(r['missing_planes']) for r in wrapped['records'])}
        receipt.update(status='MECHANICS_PASS_ANATOMY_UNREVIEWED',source_state='PASS',mechanics='PASS',prediction='PASS',completed_studies=32,parity_max_abs_errors=errors,
                       inventory_series=sum(map(len,inventories)),windows_total=sum(counts),windows_max=max(counts),windows_median=float(np.median(counts)),
                       zero_imputed=True,zero_failed=True,fp32_recoveries=0,elapsed_seconds=time.perf_counter()-started,
                       advancement_allowed=False,environment=wrapped['environment'],strata=strata,untested_strata=[k for k,v in strata.items() if v==0],
                       resource={'startup_seconds':wrapped['setup_hash_load_seconds'],'preprocess_seconds':preprocess,'four_model_seconds':inference,'seconds_per_window':inference/sum(counts),
                                 'estimated870_seconds':wrapped['setup_hash_load_seconds']+870*(preprocess+inference)/32,'estimate_limit':'frozen32 observed-window extrapolation; hidden-test tails and actual platform limit unverified',
                                 'peak_cuda_bytes':max(r['peak_cuda_bytes'] for r in wrapped['records']),'peak_rss_kib_linux':max(r['peak_rss_raw'] for r in wrapped['records']),
                                 'peak_native_scratch_bytes':max(r['native_scratch_bytes'] for r in wrapped['records'])})
        save(out/'receipt.json',receipt);return receipt
    except BaseException as e:
        receipt.update(status='FAIL',failure_type=type(e).__name__,elapsed_seconds=time.perf_counter()-started,advancement_allowed=False)
        save(out/'receipt.json',receipt);raise RuntimeError('S6 qualification failed; preserve outputs; no automatic retry') from None


if __name__=='__main__':
    import signal
    def interrupted(signum,frame):raise KeyboardInterrupt('platform interruption')
    signal.signal(signal.SIGTERM,interrupted)
    if sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3],json.loads(sys.argv[4]))
    elif sys.argv[1]=='run':run(sys.argv[2])
    else:raise SystemExit('Expected run or worker')
