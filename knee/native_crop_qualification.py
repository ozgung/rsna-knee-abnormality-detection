"""CPU-only native crop mechanics; a geometric center prior, not a detector."""
from collections import Counter
import csv
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import sys
import time
from knee.reference import bounded_ids, canonical_bytes, sha256, slot_plan

SEED, COUNT, SIDE_MM = 20261011, 32, 90.0
FOLD_SHA = '12bebc897a549d660cf5e6ab597c9a8d8f84dc887b4abd4c99b67fb4a25cb3ae'
EXCLUDED_SHA = '44834d7638eb60142a668a7597bd3093fe569e13b6f77e60a9c185a4ac3206f1'


class CropInvalid(ValueError):
    """Explicit validation failure permits unchanged global fallback."""


def geometry(iop, ipp, spacing, shape):
    import numpy as np
    try: a,p,s = [np.asarray(x,dtype=float) for x in (iop,ipp,spacing)]
    except (TypeError,ValueError): raise CropInvalid('missing_geometry') from None
    if [x.shape for x in (a,p,s)] != [(6,),(3,),(2,)]: raise CropInvalid('missing_geometry')
    if not all(np.isfinite(x).all() for x in (a,p,s)) or (s<=0).any(): raise CropInvalid('invalid_geometry')
    axes = a.reshape(2,3)
    if (np.abs(np.linalg.norm(axes,axis=1)-1)>1e-4).any() or abs(np.dot(axes[0],axes[1]))>1e-4: raise CropInvalid('nonorthonormal_geometry')
    if len(shape)!=2 or min(shape)<=0: raise CropInvalid('unsupported_pixels')
    # First IOP vector follows columns; second follows rows.
    center = p+axes[0]*s[1]*(shape[1]-1)/2+axes[1]*s[0]*(shape[0]-1)/2
    return axes,p,s,center


def physical_box(g,center,reference_axes,shape,side=SIDE_MM):
    """Fractional cell-boundary extent is exactly side mm on both axes.

    Pixel centers are integral; never shrink, clip or pad an out-of-bounds box.
    """
    import numpy as np
    axes,origin,spacing,_ = g
    if not np.allclose(axes,reference_axes,atol=1e-4,rtol=0): raise CropInvalid('inconsistent_orientation')
    normal = np.cross(axes[0],axes[1]); normal=normal/np.linalg.norm(normal)
    delta = center-origin; delta = delta-normal*np.dot(delta,normal)
    rc = np.array([np.dot(delta,axes[1])/spacing[0],np.dot(delta,axes[0])/spacing[1]])
    extent = side/spacing
    start = rc-extent/2; end=rc+extent/2
    if (start < -.5).any() or (end > np.asarray(shape)-.5).any(): raise CropInvalid('out_of_bounds')
    # Full bilinear interpolation support must also be in the actual array.
    first=start+extent/(2*320); last=end-extent/(2*320)
    if (first<0).any() or (last>np.asarray(shape)-1).any(): raise CropInvalid('out_of_bounds')
    return tuple(map(float,(start[0],end[0],start[1],end[1])))


def resample_box(array,box):
    import cv2
    import numpy as np
    y0,y1,x0,x1=box
    yy,xx=np.mgrid[0:320,0:320].astype(np.float32)
    mx=(x0+(xx+.5)*(x1-x0)/320).astype(np.float32)
    my=(y0+(yy+.5)*(y1-y0)/320).astype(np.float32)
    return cv2.remap(array,mx,my,cv2.INTER_LINEAR)



def native_series(arrays,metadata):
    """Normalize on full public-selected arrays including duplicated frames."""
    import cv2
    import numpy as np
    if len(arrays)!=len(metadata) or not arrays:
        raise ValueError("Native arrays and metadata must have equal nonzero length")
    valid=[a for a in arrays if a is not None]
    if any(a.ndim!=2 or not np.isfinite(a).all() for a in valid): raise CropInvalid('unsupported_pixels')
    lo,hi=np.percentile(np.concatenate([a.ravel() for a in valid]),[2.,98.]) if valid else (0.,1.)
    geometries,reasons=[],[]
    for a,meta in zip(arrays,metadata):
        try:
            if a is None: raise CropInvalid('missing_pixels')
            geometries.append(geometry(*meta,a.shape)); reasons.append(None)
        except CropInvalid as error: geometries.append(None); reasons.append(str(error))
    anchor=geometries[(len(geometries)-1)//2]
    center=anchor[3] if anchor is not None else None
    axes=anchor[0] if anchor is not None else None
    crops,boxes=[],[]
    for index,(a,g,reason) in enumerate(zip(arrays,geometries,reasons)):
        try:
            if reason: raise CropInvalid(reason)
            if anchor is None: raise CropInvalid("missing_anchor_geometry")
            box=physical_box(g,center,axes,a.shape)
            aw=np.clip((a-lo)/(hi-lo+1e-6),0,1)
            crop=(resample_box(aw,box)*255).astype(np.uint8)
            crops.append(crop); boxes.append(box)
        except CropInvalid as error:
            crops.append(None); boxes.append(None); reasons[index]=str(error)
    return crops,boxes,reasons


def triplets(volume,native,reasons,slots,centers=None):
    import numpy as np
    global_frames=volume[:,32:352,32:352]
    windows,counts=[],Counter()
    for center in (centers if centers is not None else range(1,95)):
        ix=[center-1,center,center+1]
        reason='cross_slot_triplet' if len({slots[i] for i in ix})!=1 else next((reasons[i] for i in ix if reasons[i]),None)
        if reason: windows.append(global_frames[ix].copy()); counts[reason]+=1
        else: windows.append(np.stack([native[i] for i in ix]))
    return np.stack(windows),dict(counts)


def select_inputs(root,targets,exclusions,folds,count=COUNT):
    from knee.public_rich_cache import select_inputs as official_inputs
    _,records,membership=official_inputs(root,targets,exclusions,count=1)
    gold=set()
    with (Path(root)/'train.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if all(row[t] is not None and row[t].strip() for t in targets): gold.add(row['StudyInstanceUID'])
    eligible=set(records)-gold-set(exclusions['quarantined_study_ids'])
    if not eligible<=folds.keys() or any(type(folds[s]) is not int or folds[s] not in range(5) for s in eligible):
        raise ValueError('Fixed fold mapping coverage invalid')
    train=sorted(s for s in eligible if folds[s]!=0)
    return bounded_ids(train,count=count,seed=SEED),records,{**membership,'fold0_training_eligible':len(train)}


def safe_geometry_metadata(meta):
    import numpy as np
    result=[]
    for value in meta:
        try:
            array=np.asarray(value,dtype=float)
            result.append(array.tolist() if np.isfinite(array).all() and array.ndim==1 else None)
        except (TypeError,ValueError): result.append(None)
    return result


def capture_study(public,decoder,study,records,root):
    import numpy as np
    import pydicom
    from threading import Lock
    from knee.mirror_input import prepare_study
    from knee.raptor_input import propagate_fatal,fatal_error
    fatal_latch=[]
    order,read,resize=public._make_reader(); lock=Lock()
    plan,_=slot_plan(public,study,records)
    frames,boxes,reasons,slots,contexts=[],[],[],[],[]
    for slot,(_,k) in enumerate(plan):
        frames.extend([None]*k); boxes.extend([None]*k); reasons.extend(['missing_pixels']*k)
        slots.extend([slot]*k); contexts.extend([(None,None)]*k)
    current={}; source_stats=Counter({f'slot{i}_present':int(row is not None) for i,(row,_) in enumerate(plan)}); private_sources=[]
    def ordering(directory):
        # Release all raw selected arrays from previous source series here.
        current.clear()
        files,med=order(directory)
        slot=next(i for i,(row,_) in enumerate(plan) if row and directory.endswith('/'+row['SeriesInstanceUID']))
        k=plan[slot][1]; n=len(files)
        picks=np.linspace(int(n*.02),max(int(n*.98)-1,int(n*.02)),k).round().astype(int) if n>1 else [0]*k
        selected=[files[min(int(i),n-1)] for i in picks] if n else []
        source_stats[f'slot{slot}_selected']+=len(selected)
        source_stats[f'slot{slot}_unique_selected']+=len(set(p for p,_ in selected))
        fingerprints={}
        for p,_ in selected:
            if p not in fingerprints:
                import hashlib
                digest=hashlib.sha256()
                with Path(p).open('rb') as handle:
                    for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
                fingerprints[p]=digest.hexdigest()
            private_sources.append({'slot':slot,'source_path':str(Path(p).relative_to(root)),
                                    'size_bytes':Path(p).stat().st_size,'sha256':fingerprints[p]})
        current.update(slot=slot,selected=selected,pixels={},metadata={},cursor=0,initialized=False)
        return files,med
    def capture(path):
        with lock:
            try:
                if path not in current['pixels']:
                    current['pixels'][path]=read(path)
                    try:
                        h=pydicom.dcmread(path,stop_before_pixels=True)
                        meta=tuple(getattr(h,n,None) for n in ('ImageOrientationPatient','ImagePositionPatient','PixelSpacing'))
                        if str(getattr(h,'AnatomicalOrientationType','BIPED'))!='BIPED': meta=(None,None,None)
                        current['metadata'][path]=meta
                    except Exception as error:
                        current['metadata'][path]=(None,None,None)
                        propagate_fatal(error)
                return current['pixels'][path]
            except Exception as error:
                # Public build_study._px can consume this exception. Keep the
                # first fatal object outside that catch and rethrow after join.
                if fatal_error(error) and not fatal_latch: fatal_latch.append(error)
                raise
    def cropping(aw,ps):
        if not current['initialized']:
            arrays=[current['pixels'].get(p) for p,_ in current['selected']]
            metas=[current['metadata'].get(p,(None,None,None)) for p,_ in current['selected']]
            crops,bounds,why=native_series(arrays,metas)
            offset=sum(k for _,k in plan[:current['slot']])
            for j,(crop,box,reason) in enumerate(zip(crops,bounds,why)):
                frames[offset+j]=crop; boxes[offset+j]=box; reasons[offset+j]=reason
                path,_=current['selected'][j]
                for item in private_sources:
                    if item['slot']==current['slot'] and item['source_path']==str(Path(path).relative_to(root)):
                        item['native_box_pixel_boundaries']=box
                        item['geometry']=safe_geometry_metadata(metas[j])
                        item['matrix']=list(arrays[j].shape) if arrays[j] is not None else None
                a=arrays[j]
                if a is not None:
                    source_stats[f'slot{current["slot"]}_matrix_{a.shape[0]}x{a.shape[1]}']+=1
                    try:
                        spacing=geometry(*metas[j],a.shape)[2]
                        source_stats[f'slot{current["slot"]}_spacing_{spacing[0]:.8g}x{spacing[1]:.8g}']+=1
                        gy,gy1,gx,gx1=inherited_box(a.shape,current['selected'][j][1])
                        source_stats[f'slot{current["slot"]}_public_field_mm_{(gy1-gy)*spacing[0]:.8g}x{(gx1-gx)*spacing[1]:.8g}']+=1
                        if box:
                            y0,y1,x0,x1=box
                            source_stats[f'slot{current["slot"]}_native_field_mm_{(y1-y0)*spacing[0]:.8g}x{(x1-x0)*spacing[1]:.8g}']+=1
                    except CropInvalid: source_stats[f'slot{current["slot"]}_invalid_geometry']+=1
            current.update(initialized=True,offset=offset)
        # Public calls resize only for successfully read pixels; track skipped frames.
        selected=current['selected']; j=current['cursor']
        while j<len(selected) and selected[j][0] not in current['pixels']: j+=1
        path,_=selected[j]; current['cursor']=j+1; index=current['offset']+j
        result=resize(aw,ps)
        a=current['pixels'][path]; meta=current['metadata'][path]
        contexts[index]=frame_panels(a,meta,ps,boxes[index],(result*255).astype(np.uint8))
        return result
    # Execute the exact protected public function per slot in an isolated globals
    # dictionary; prevents its previous-slot arrs/valid locals retaining raw pixels
    # during the next slot. Full original fast-builder bytes independently verify it.
    import types
    reference=np.zeros((96,384,384),np.uint8); offset=0
    for slot,(row,k) in enumerate(plan):
        if row is not None:
            namespace=dict(public.__dict__,SLOTS=[public.SLOTS[slot]],MAXS=k)
            single=types.FunctionType(public.build_study.__code__,namespace)
            try:
                part,_=single(study,{study:[row]},str(root),(ordering,capture,cropping))
            finally:
                if fatal_latch: raise fatal_latch[0]
            reference[offset:offset+k]=part
            del part,single,namespace
        offset+=k; current.clear()
    mask=(reference.reshape(96,-1).sum(1)>0).astype(np.uint8)
    volume,fast_mask,counts=prepare_study(public,decoder,study,records,root)
    if not np.array_equal(reference,volume) or not np.array_equal(mask,fast_mask): raise ValueError('Global reference/fast parity failed')
    window_hashes={}
    for mirror in (False,True):
        x=public.eval_windows(reference,mask,94,320,mirror=mirror)
        y=public.eval_windows(volume,fast_mask,94,320,mirror=mirror)
        if not __import__('torch').equal(x,y): raise ValueError('Global window parity failed')
        window_hashes["mirrored" if mirror else "normal"]=sha256(x.numpy().tobytes())
        del x,y
    del reference
    centers=public._eval_centers(fast_mask,96,94)
    crops,fallback=triplets(volume,frames,reasons,slots,centers)
    counts["window_centers"]=list(map(int,centers))
    counts["global_window_hashes"]=window_hashes
    counts["source_strata"]=dict(source_stats)
    counts["private_sources"]=private_sources
    return volume,fast_mask,crops,boxes,reasons,slots,contexts,counts,fallback


def inherited_box(shape,ps):
    h,w=shape; size=min(int(round(140/max(ps,1e-3))),h,w)
    return (h-size)//2,(h-size)//2+size,(w-size)//2,(w-size)//2+size


def comparison_crop(frame,shape,ps,box):
    """Identical native pixel boundaries mapped through original square→384 resize.

    Return None when ROI lies outside inherited field. No comparison padding.
    """
    import cv2
    import numpy as np
    gy,gy1,gx,gx1=inherited_box(shape,ps); y0,y1,x0,x1=box
    if not (gy-.5<=y0<y1<=gy1-.5 and gx-.5<=x0<x1<=gx1-.5): return None
    yy,xx=np.mgrid[0:320,0:320].astype(np.float32); size=gy1-gy
    mx=((x0+(xx+.5)*(x1-x0)/320+.5-gx)*384/size-.5).astype(np.float32)
    my=((y0+(yy+.5)*(y1-y0)/320+.5-gy)*384/size-.5).astype(np.float32)
    if mx.min()<0 or my.min()<0 or mx.max()>383 or my.max()>383: return None
    return cv2.remap(frame,mx,my,cv2.INTER_LINEAR)


def frame_panels(a,meta,ps,box,global_frame):
    import cv2
    import numpy as np
    lo,hi=np.percentile(a,[2,98]) # Display only, no crop renormalization.
    display=cv2.cvtColor((np.clip((a-lo)/(hi-lo+1e-6),0,1)*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    gy,gy1,gx,gx1=inherited_box(a.shape,ps)
    cv2.rectangle(display,(gx,gy),(gx1-1,gy1-1),(0,200,255),2)
    caption='public pixel box; geometry invalid'; control=None
    try:
        spacing=geometry(*meta,a.shape)[2]
        caption=f'public px {gy1-gy}x{gx1-gx} / mm {(gy1-gy)*spacing[0]:.1f}x{(gx1-gx)*spacing[1]:.1f}'
    except CropInvalid: pass
    if box:
        y0,y1,x0,x1=box
        cv2.rectangle(display,(round(x0),round(y0)),(round(x1),round(y1)),(0,255,0),2)
        control=comparison_crop(global_frame,a.shape,ps,box)
    preview=cv2.resize(display,(320,320),interpolation=cv2.INTER_AREA)
    cv2.putText(preview,caption,(4,12),cv2.FONT_HERSHEY_SIMPLEX,.27,(255,120,120),1)
    return preview,control


def save_overlays(out,ordinal,volume,crops,boxes,reasons,slots,contexts,centers):
    import cv2
    import numpy as np
    selected={j for j,c in enumerate(centers) if len(set(slots[c-1:c+2]))!=1 or any(reasons[c-1:c+2])}
    for slot in set(slots):
        starts=[j for j,c in enumerate(centers) if slots[c-1:c+2]==[slot]*3]
        if starts: selected.update(starts[int(round((len(starts)-1)*q))] for q in (.25,.5,.75))
    covered=set()
    for window in sorted(selected):
        start=centers[window]-1
        panels=[]
        for i in range(start,start+3):
            covered.add(i)
            preview,control=contexts[i]
            full=preview if preview is not None else np.zeros((320,320,3),np.uint8)
            roi=cv2.cvtColor(crops[window,i-start],cv2.COLOR_GRAY2BGR)
            reason='cross_slot_triplet' if len(set(slots[start:start+3]))!=1 else next((r for r in reasons[start:start+3] if r),'native90mm')
            cv2.putText(roi,reason,(4,12),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,100,255),1)
            comparison=cv2.cvtColor(control if control is not None else np.zeros((320,320),np.uint8),cv2.COLOR_GRAY2BGR)
            cv2.putText(comparison,'public384 same box' if control is not None else 'comparison unavailable',(4,12),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,100,255),1)
            panels.append(np.concatenate([full,roi,comparison],axis=1))
        if not cv2.imwrite(str(Path(out)/f'{ordinal:03d}-window-{window:02d}.png'),np.concatenate(panels,axis=0)):
            raise OSError('Overlay write failed')
    # Geometry anomalies outside mask-selected K94 still get a private contact sheet.
    for i,(preview,control) in enumerate(contexts):
        if reasons[i] and preview is not None and i not in covered:
            fallback=cv2.cvtColor(volume[i,32:352,32:352],cv2.COLOR_GRAY2BGR)
            cv2.putText(fallback,reasons[i],(4,12),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,100,255),1)
            image=np.concatenate([preview,fallback,np.zeros_like(preview)],axis=1)
            if not cv2.imwrite(str(Path(out)/f'{ordinal:03d}-anomaly-frame-{i:02d}.png'),image): raise OSError('Anomaly overlay write failed')


def mirrored_triplets(public,crops,centers):
    import torch
    return public._mirror_tri(torch.from_numpy(crops),centers,96).numpy()


def run(public_source,decoder_source,manifest,root,out,exclusions,folds,binding):
    from knee.reference import load_definitions
    import torch
    out=Path(out); out.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    report={'status':'RUNNING','purpose':'geometric center prior mechanics; no anatomical containment or prediction claims',
            'models_loaded':0,'device':'cpu','sampling_seed':SEED,'requested_studies':COUNT,'completed_studies':0,
            'config':{'side_mm':SIDE_MM,'grid':'fractional cell boundaries exactly90mm; cv2 linear; no clipping/padding',
                      'comparison':'linear mapped identical native pixel box in public384; unavailable outside inherited field',
                      'global':'unchanged square PS[0] crop and96x384 masks; central320 fallback',
                      'selection':'public slots/order/2–98%; same selected-series frames'},
            'binding':binding,'source_manifest_sha256':sha256(canonical_bytes(manifest)),'source_series_residency_max':1,
            'free_disk_before':shutil.disk_usage(out).free,'python':sys.version,'packages':{},'studies':[],'fallback_triplets':{}}
    for p in ('torch','numpy','pydicom','opencv-python-headless','opencv-python'):
        try: report['packages'][p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: pass
    totals=Counter()
    def save():
        report.update(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        (out/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        torch.set_num_threads(4); public,decoder=load_definitions(public_source,decoder_source,manifest)
        ids,records,membership=select_inputs(root,public.LAB,exclusions,folds)
        selected_sha=sha256(canonical_bytes(ids))
        if selected_sha!=binding['selected_order_sha256']: raise ValueError('Frozen sampling mismatch')
        if sha256((Path(root)/'train.csv').read_bytes())!=binding['train_csv_sha256']: raise ValueError('Official table binding changed')
        (out/'sample-private.json').write_text(json.dumps({'study_ids':ids,'selected_order_sha256':selected_sha},indent=2)+'\n')
        report.update(membership=membership,selected_order_sha256=selected_sha,series_sha256=sha256((Path(root)/'train_series.csv').read_bytes()),anatomical_acceptance='UNVERIFIED')
        for ordinal,sid in enumerate(ids):
            v,m,c,b,r,s,ctx,counts,fallback=capture_study(public,decoder,sid,records,Path(root)/'train_series')
            repeat=capture_study(public,decoder,sid,records,Path(root)/'train_series')
            import numpy as np
            if any(not np.array_equal(a,b) for a,b in zip((v,m,c),repeat[:3])) or counts['global_window_hashes']!=repeat[7]['global_window_hashes']: raise ValueError('Repeat pixels/windows differ')
            mirror=mirrored_triplets(public,c,counts["window_centers"])
            if not np.array_equal(mirror,mirrored_triplets(public,repeat[2],repeat[7]["window_centers"])): raise ValueError('Repeat mirror differs')
            mirror_sha=sha256(mirror.tobytes()); del mirror,repeat
            before=sha256(v.tobytes()); save_overlays(out,ordinal,v,c,b,r,s,ctx,counts["window_centers"])
            if before!=sha256(v.tobytes()): raise ValueError('Global mutated')
            private=counts.pop('private_sources')
            (out/f'{ordinal:03d}-sources-private.json').write_text(json.dumps(private,indent=2)+'\n')
            totals.update(fallback)
            report['studies'].append({'ordinal':ordinal,'global_sha256':before,'mask_sha256':sha256(m.tobytes()),
                                      'native_triplets_sha256':sha256(c.tobytes()),'native_mirrored_sha256':mirror_sha,'repeat_normal_mirror_parity':True,'global_normal_mirror_window_parity':True,'global_reference_fast_parity':True,
                                      'native_eligible_frames':sum(x is None for x in r),'decode_counts':counts,'fallback_triplets':fallback})
            report['completed_studies']+=1; report['fallback_triplets']=dict(totals); save()
        report['storage_bytes']=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        report['status']='PASS'
    except Exception as error:
        report.update(status='FAIL',failure_type=type(error).__name__)
        raise RuntimeError('Native qualification failed; see aggregate receipt') from None
    finally:
        report["free_disk_after"]=shutil.disk_usage(out).free
        save()
    return report
