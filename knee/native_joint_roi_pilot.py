"""Build-only qualification helpers: frozen S6 pixels, one joint view, no fit.

Imports of the pinned external reader and ML packages are deliberately local.
The same reader function is the prospective training and inference entry point.
"""
from collections import Counter
import json
from pathlib import Path
import resource
import sys
import time
from knee.public_s6_qualification import digest, file_sha, require, save, stage_inputs, verify_sources

WEIGHT_SHA = '5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e'
WEIGHT_BYTES = 112496037
SIDE_MM, SIZE = 100., 384
WEIGHT_URL = 'https://huggingface.co/arjundd/skm-tea-models/blob/main/neurips2021/segmentation-models/V-Net_E1/model-raw.ckpt'


def joint_view(series, geo, anchor, view_geometry, project):
    """Fail closed on unresolved source mapping; explicit global crop fallbacks.

    Source clipping/shrinking is forbidden: require the requested rounded100mm
    rectangle to fit. Source letterboxing is retained and counted, not hidden.
    """
    import numpy as np
    if any(str(x).startswith('error:') for x in anchor['why']):
        raise ValueError('Source swallowed localizer error')
    base={'reason': None, 'side_mm_requested': SIDE_MM, 'padding_pixels': 0}
    if anchor['joint_lps'] is None:
        return None, {**base, 'reason': 'missing_anchor'}
    if geo is None:
        return None, {**base, 'reason': 'missing_geometry'}
    if not geo['quality']['aligned']:
        return None, {**base, 'reason': 'unaligned_geometry'}
    aff=np.asarray(geo['affines_col_row_origin_lps_mm'], float)
    shape=tuple(series['native_shape'][-2:]);sp=np.asarray(series['spacing_mm'],float)
    require(aff.shape==(series['native_shape'][0],3,3) and np.isfinite(aff).all(), 'Affine/native slice contract differs')
    require(sp.shape==(len(aff),2) and np.isfinite(sp).all() and (sp>0).all(), 'Spacing/native slice contract differs')
    require(np.allclose(sp,sp[0],rtol=1e-4,atol=0), 'Changing in-plane spacing unsupported')
    require(np.allclose(np.linalg.norm(aff[:,:,:2],axis=1)[:,::-1],sp,rtol=1e-4,atol=1e-4), 'Canonical affine/spacing contract differs')
    z,c,r,d=project(aff,np.asarray(anchor['joint_lps'],float))
    normal=np.cross(aff[z,:,0],aff[z,:,1]);normal/=np.linalg.norm(normal)
    point=np.asarray(anchor['joint_lps'],float)
    expected=point+normal*np.dot(aff[z,:,2]-point,normal)
    err=float(np.linalg.norm(aff[z]@np.array([c,r,1.])-expected))
    require(err<=1e-4,'Patient projection differs')
    # Aligned source permits less than1mm drift. One fixed in-plane rectangle
    # follows the pinned source's per-series projection across every triplet.
    center=(r/max(1,shape[0]-1),c/max(1,shape[1]-1))
    sy,sx=sp[0];ch,cw=round(SIDE_MM/sy),round(SIDE_MM/sx)
    y,x=round(r-(ch-1)/2),round(c-(cw-1)/2)
    base.update(nearest_native_slice=int(z),projection_error_mm=err,plane_distance_mm=float(d),
                center_row_col=list(center),native_shape=list(shape),spacing_mm=sp[0].tolist(),
                requested_box_xyxy=[x,y,x+cw,y+ch], boundary_distance_mm=float(min(r*sy,c*sx,(shape[0]-1-r)*sy,(shape[1]-1-c)*sx)))
    outside=max(0,ch*cw-max(0,min(x+cw,shape[1])-max(x,0))*max(0,min(y+ch,shape[0])-max(y,0)))
    base['out_of_field_fraction']=outside/max(1,ch*cw)
    if ch<1 or cw<1 or outside:
        return None,{**base,'reason':'out_of_bounds_100mm'}
    g=view_geometry(shape,sp[0],SIZE,crop_mm=SIDE_MM,center=center)
    require(tuple(g.crop_xyxy)==(x,y,x+cw,y+ch), 'Source clipped/shrank joint crop')
    oh,ow=g.resized_shape
    base.update(actual_field_mm=[ch*sy,cw*sx],native_support_pixels=ch*cw,
                resampling_ratio=[oh/ch,ow/cw],padding_pixels=SIZE*SIZE-oh*ow,
                retained_box_xyxy=list(g.crop_xyxy),valid_xyxy=list(g.valid_xyxy))
    return center,base


def qualify_frame_references(geometries, anchor):
    """No cross-plane registration assumption when projecting a shared anchor."""
    if anchor['joint_lps'] is None:return
    refs={g.get('frame_of_reference_uid') for g in geometries.values() if g is not None}
    require(len(refs)==1 and None not in refs and '' not in refs,
            'BLOCKER: shared anchor requires one explicit FrameOfReferenceUID')


def read_pair(cache,uid,geometries,anchor):
    """Shared train/inference adapter. The pinned builder owns all pixel math.

    There are two source builds with identical series/order/slab paths. The
    external view selector changes only geometry; source modules stay intact.
    """
    import numpy as np
    import torch
    import px_cache as PX
    import knee_anchor_bank_v2 as KB
    from training.input_provenance import view_geometry
    qualify_frame_references(geometries,anchor)
    study=json.loads(Path(PX.study_json(str(cache),uid)).read_bytes())
    centers={};receipts={}
    for s in study['series']:
        sid=s['SeriesInstanceUID']
        centers[sid],receipts[sid]=joint_view(s,geometries.get(sid),anchor,view_geometry,KB.project)
        gg=view_geometry(s['native_shape'][-2:],s['spacing_mm'][0],SIZE)
        gh,gw=gg.resized_shape;nh,nw=gg.native_shape
        receipts[sid].update(native_shape=list(gg.native_shape),spacing_mm=list(gg.spacing),
                            global_padding_pixels=SIZE*SIZE-gh*gw,global_resampling_ratio=[gh/nh,gw/nw],
                            global_field_mm=[nh*gg.spacing[0],nw*gg.spacing[1]])
    original=PX.series_views
    # The source calls series_views in representative block order. Recover IDs
    # with the same exact-duplicate resolver, rather than assuming all rows used.
    from training.exact_duplicates import ExactDuplicateResolver
    plan=ExactDuplicateResolver(None,None,False).resolve(study)
    ids=[study['series'][i]['SeriesInstanceUID'] for i in plan['representative_indices']]
    def render(local):
        cursor=iter(ids)
        def views(*args,**kwargs):
            sid=next(cursor)
            center=centers[sid] if local else None
            return [(0,None,(.5,.5))] if center is None else [(0,SIDE_MM,center)]
        try:
            PX.series_views=views
            item=PX.StudySlices(str(cache),[],SIZE,'v1').build(uid)
            require(not item['skipped_series'],'Source skipped native series')
            require(next(cursor,None) is None,'Source representative order differs')
            return item
        finally:PX.series_views=original
    global_item,local_item=render(False),render(True)
    require(global_item['blocks']==local_item['blocks'],'Global/local slice blocks differ')
    # Geometric metadata column6 changes with FOV; all ordering/meta columns
    # otherwise coincide. Never conceal an intensity/slice-path discrepancy.
    columns=[i for i in range(12) if i!=6]
    require(torch.equal(global_item['meta'][:,columns],local_item['meta'][:,columns]),'Global/local source ordering differs')
    for b in global_item['blocks']:
        if receipts[b['series']]['reason']:
            off=b['views'][0][1];n=b['n_slices']
            require(torch.equal(global_item['px'][off:off+n],local_item['px'][off:off+n]),'Fallback differs from corresponding global')
    return global_item,local_item,receipts


def paired_triplets(global_item,local_item):
    """12 identical series/depth triplets duplicated to a matched24 view plan."""
    import numpy as np
    blocks=global_item['blocks'];require(bool(blocks),'No source blocks')
    plan=[]
    for j in range(12):
        b=blocks[j%len(blocks)];n=b['n_slices'];off=b['views'][0][1]
        center=int(round((n-1)*((j//len(blocks)+.5)/max(1,int(np.ceil(12/len(blocks)))))))
        center=min(n-1,center)
        ix=[off+min(n-1,max(0,center+k)) for k in (-1,0,1)]
        plan.append({'series':b['series'],'indices':ix})
    indices=[x['indices'] for x in plan]
    g=global_item['px'].numpy()[indices];l=local_item['px'].numpy()[indices]
    return np.concatenate([g,g]),np.concatenate([g,l]),plan


def containment_native(out,ordinal,study,receipts):
    """Every source series: uncropped canonical native pixels with ROI overlay."""
    import cv2
    import numpy as np
    for index,s in enumerate(study['series']):
        r=receipts[s['SeriesInstanceUID']]
        volume=np.load(s['path'],mmap_mode='r',allow_pickle=False)
        z=r.get('nearest_native_slice',len(volume)//2)
        a=np.asarray(volume[z],float);lo,hi=s['intensity']['display_window_uint16']
        preview=cv2.cvtColor(np.rint(np.clip((a-lo)/max(hi-lo,1e-6),0,1)*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        if 'requested_box_xyxy' in r:
            x,y,x1,y1=r['requested_box_xyxy']
            cv2.rectangle(preview,(x,y),(x1-1,y1-1),(0,255,0) if not r['reason'] else (0,0,255),2)
            cy,cx=r['center_row_col'];h,w=a.shape
            cv2.drawMarker(preview,(round(cx*(w-1)),round(cy*(h-1))),(0,255,255),cv2.MARKER_CROSS,12,2)
        # Native sheet is for display only, never an input normalization.
        scale=min(768/preview.shape[1],768/preview.shape[0])
        preview=cv2.resize(preview,(max(1,round(preview.shape[1]*scale)),max(1,round(preview.shape[0]*scale))),interpolation=cv2.INTER_AREA)
        caption=f"{s['plane']} native {a.shape} z={z} / {r['reason'] or 'joint100mm'}"
        cv2.putText(preview,caption,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,100,100),1)
        require(cv2.imwrite(str(Path(out)/f'{ordinal:03d}-native-series-{index:02d}.png'),preview),'Native containment artifact write failed')


def run(config):
    """One private bounded reader qualification; no predictor or outcome labels."""
    import numpy as np
    import torch
    from knee.public_s6_qualification import validate_membership
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','completed_studies':0,'advancement_allowed':False,
            'anatomical_containment':'UNVERIFIED','train_inference_parity':'UNVERIFIED',
            'binding':config['binding'],'studies':[],'fallback_series':{}}
    started=time.perf_counter();totals=Counter()
    try:
        verify_sources(config['code'],config['source_pins'])
        require(file_sha(config['anatomy'])==WEIGHT_SHA and Path(config['anatomy']).stat().st_size==WEIGHT_BYTES,'SKM-TEA weight identity differs')
        validate_membership(config['ids'],config['folds'],config['excluded'])
        require(digest(config['folds'])==config['binding']['selected_folds_sha256'],'Selected membership changed')
        sys.path.insert(0,config['code'])
        torch.set_num_threads(1)
        import knee_anchor_bank_v2 as KB
        from training import native_cache as NC,geometry_bank as GB
        stage_inputs(config['raw'],config['staged'],config['ids'],config['binding']['series_sha256'])
        device=config.get('device','cpu');require(device in ('cpu','cuda'),'Unsupported explicit localizer device')
        require(device!='cuda' or torch.cuda.is_available(),'Explicit CUDA unavailable')
        t=time.perf_counter();model=KB.load_model(config['anatomy'],device);bank=KB.AnchorBank(model,device)
        report['localizer_load_seconds']=time.perf_counter()-t
        import monai
        require(Path(monai.__file__).resolve().is_relative_to(Path(config['code'])/'training/vendor_runtime/monai060') and monai.__version__=='0.6.0','Pinned MONAI runtime shadowed')
        import importlib.metadata
        report.update(device=device,python=sys.version,packages={p:importlib.metadata.version(p) for p in ('torch','numpy','pydicom','scipy','pandas')},
                      wrapper_exposure=config['binding']['wrapper_exposure'],models_loaded=1)
        def model_sha():
            h=__import__('hashlib').sha256()
            for name,value in sorted(model.state_dict().items()):
                h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        report['localizer_state_before']=model_sha()
        candidates=[];original_candidate=bank.candidate
        def observed_candidate(s,g):
            c,why=original_candidate(s,g)
            candidates.append({'series':s['SeriesInstanceUID'],'why':why,
                'candidate':None if c is None else {k:v.tolist() if hasattr(v,'tolist') else v for k,v in c.items()}})
            return c,why
        bank.candidate=observed_candidate
        for ordinal,uid in enumerate(config['ids']):
            candidates.clear()
            work=out/f'{ordinal:03d}-native';t=time.perf_counter()
            manifest=NC.prepare(config['staged'],work,split='train',study_ids=[uid],source_directory=True)
            require(len(manifest['studies'])==1,'Native study count differs');study=manifest['studies'][0]
            require(not study['errors'] and all(not s['errors'] for s in study['series']),'Source swallowed native decoder errors')
            GB.build(config['staged'],work/'manifest.json',work/'geometry',split='train',source_directory=True)
            decode_seconds=time.perf_counter()-t
            geos={s['SeriesInstanceUID']:json.loads((work/'geometry'/(KB.key_for(uid,s['SeriesInstanceUID'])+'.json')).read_bytes()) for s in study['series']}
            t=time.perf_counter();anchor=bank.record(uid,study,str(work/'geometry'));anchor_seconds=time.perf_counter()-t
            g,l,receipts=read_pair(work,uid,geos,anchor)
            control,treatment,plan=paired_triplets(g,l)
            # A fresh second call through exactly the same adapter qualifies
            # the prospective train/inference deterministic reader contract.
            gg,ll,rr=read_pair(work,uid,geos,anchor);cc,tt,pp=paired_triplets(gg,ll)
            require(np.array_equal(control,cc) and np.array_equal(treatment,tt) and plan==pp and receipts==rr,'Shared train/inference reader bytes differ')
            import cv2
            containment_native(out,ordinal,study,receipts)
            for j,p in enumerate(plan):
                image=np.concatenate([control[j,1],treatment[j+12,1]],axis=1)
                require(cv2.imwrite(str(out/f'{ordinal:03d}-containment-{j:02d}.png'),image),'Containment artifact write failed')
            source_inventory=[]
            for s in study['series']:
                for name in s['files']:
                    path=Path(config['staged'])/'train_series'/uid/s['SeriesInstanceUID']/name
                    import pydicom
                    header=pydicom.dcmread(path,stop_before_pixels=True,specific_tags=['Rows','Columns','PixelSpacing'])
                    source_inventory.append({'series':s['SeriesInstanceUID'],'file':name,'sha256':file_sha(path),'bytes':path.stat().st_size,
                        'dicom_matrix':[int(header.Rows),int(header.Columns)],'dicom_pixel_spacing':list(map(float,header.PixelSpacing)) if 'PixelSpacing' in header else None})
            save(out/f'{ordinal:03d}-source-private.json',{'study':study,'geometry':geos,'anchor':anchor,'candidates':list(candidates),'series_crops':receipts,'triplet_plan':plan,'dicom_inventory':source_inventory})
            totals.update(r['reason'] for r in receipts.values() if r['reason'])
            report['studies'].append({'ordinal':ordinal,'decode_geometry_seconds':decode_seconds,'anchor_seconds':anchor_seconds,
                'anchor_method':anchor['method'],'anchor_reasons':anchor['why'],'control_sha256':__import__('hashlib').sha256(control.tobytes()).hexdigest(),
                'treatment_sha256':__import__('hashlib').sha256(treatment.tobytes()).hexdigest(),'series':list(receipts.values()),
                'dicom_inventory_sha256':digest(source_inventory),
                'fallback_treatment_triplets':sum(receipts[p['series']]['reason'] is not None for p in plan),
                'out_of_bounds_series':sum(r['reason']=='out_of_bounds_100mm' for r in receipts.values()),
                'local_padding_pixels_per_frame':sum(r['padding_pixels'] for r in receipts.values()),
                'global_padding_pixels_per_frame':sum(r['global_padding_pixels'] for r in receipts.values()),
                'missing_planes':sorted({'Sagittal','Coronal','Axial'}-{s['plane'] for s in study['series']}),'shared_reader_repeat_parity':True})
            report.update(completed_studies=ordinal+1,fallback_series=dict(totals));save(out/'receipt.json',report)
            # Keep source receipts/overlays, release one-study native cache.
            import shutil
            shutil.rmtree(work)
        report['localizer_state_after']=model_sha()
        require(report['localizer_state_before']==report['localizer_state_after'],'Localizer state changed')
        report.update(status='PASS_READER_MECHANICS',train_inference_parity='PASS_SHARED_ADAPTER_REPEAT')
    except BaseException as error:
        report.update(status='BLOCKED_OR_FAILED',failure_type=type(error).__name__,failure_message=str(error));raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        save(out/'receipt.json',report)
    return report
