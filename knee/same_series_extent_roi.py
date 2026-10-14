"""Image-only, same-series sagittal ROI reader. No study anchor or label access.

The pinned source segment path runs only on the qualification host.
Public pixel preprocessing is unchanged; this module owns binding and allocation.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

from knee.native_joint_roi_pilot import SIZE, WEIGHT_SHA, WEIGHT_BYTES
from knee.same_series_roi import triplet_plan, retained_support
from knee.public_s6_qualification import digest, file_sha, require, save, stage_inputs, verify_sources, validate_membership

PROTOCOL = 'same-series-sagittal-extent-matched24-v2'
MARGIN_MM = 10.
MIN_SIDE_MM = 100
MAX_SIDE_MM = 140


def series_extent(uid, series, geometry, bank, view_geometry):
    """Own-series support envelope; return crop selector, JSON receipt, masks.

    Class names are deliberately limited to source p0 patellar tissue and
    p1:4 joint tissues. No centroid heuristic or cross-series point is used.
    """
    import math
    import numpy as np
    sid = series['SeriesInstanceUID']
    base = {'origin_series': sid, 'reason': None, 'padding_pixels': 0,
            'margin_mm': MARGIN_MM, 'support_channel_semantics':
            ['source_p0_patellar_tissue', 'source_p1_joint_tissue',
             'source_p2_joint_tissue', 'source_p3_joint_tissue']}
    if series['plane'] != 'Sagittal':
        return None, {**base, 'reason': 'unsupported_plane_no_independent_segmenter'}, None
    if geometry is None:
        return None, {**base, 'reason': 'missing_geometry'}, None
    require(geometry['StudyInstanceUID'] == uid and geometry['SeriesInstanceUID'] == sid,
            'Geometry origin binding differs')
    shape = np.asarray(series['native_shape'])
    require(shape.shape == (3,) and np.issubdtype(shape.dtype, np.integer) and (shape > 0).all(),
            'Native shape contract differs')
    aff = np.asarray(geometry['affines_col_row_origin_lps_mm'], float)
    sp = np.asarray(series['spacing_mm'], float)
    require(aff.shape == (shape[0], 3, 3) and np.isfinite(aff).all(), 'Affine/native contract differs')
    require(sp.shape == (shape[0], 2) and np.isfinite(sp).all() and (sp > 0).all(), 'Spacing/native contract differs')
    require(np.allclose(sp, sp[0], rtol=1e-4, atol=0), 'Changing in-plane spacing unsupported')
    require(np.allclose(np.linalg.norm(aff[:, :, :2], axis=1)[:, ::-1], sp, rtol=1e-4, atol=1e-4), 'Affine/spacing contract differs')
    require(np.all(np.linalg.norm(np.cross(aff[:, :, 0], aff[:, :, 1]), axis=1) > 0), 'Degenerate affine')
    retained_support(series)  # Validate native depth positions even before segment execution.
    require(isinstance(geometry['quality']['aligned'], bool), 'Alignment contract differs')
    if not geometry['quality']['aligned']:
        return None, {**base, 'reason': 'unaligned_geometry'}, None
    segs, why = bank.segment(series, geometry, n_slices=7)
    if segs is None:
        require(why == 'oblique', 'Unknown localizer status: ' + str(why))
        return None, {**base, 'reason': 'oblique'}, None
    require(why == 'ok' and isinstance(segs, (list, tuple)) and bool(segs), 'Malformed segment result')
    dz = float(np.linalg.norm(aff[1, :, 2] - aff[0, :, 2])) if len(aff) > 1 else 4.
    step = max(1, int(round(4. / max(dz, 1e-3))))
    middle = int(shape[0]) // 2
    expected_depths = sorted({min(int(shape[0])-1, max(0, middle+k*step)) for k in range(-3,4)})
    depths, masks = [], []
    for entry in segs:
        require(isinstance(entry, (tuple, list)) and len(entry) == 2, 'Malformed segment entry')
        z, probability = entry
        require(isinstance(z, (int, np.integer)) and not isinstance(z, (bool, np.bool_)) and 0 <= z < shape[0], 'Segment depth contract differs')
        p = np.asarray(probability)
        require(p.shape == (4, int(shape[1]), int(shape[2])) and np.issubdtype(p.dtype, np.floating), 'Segment probability shape/type differs')
        require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'Segment probability contract differs')
        depths.append(int(z)); masks.append(p > .5)
    require(depths == expected_depths, 'Segment sampled depth mapping differs')
    binary = np.stack(masks)
    counts = binary.sum(axis=(2,3))
    accepted = counts >= 40
    supported = binary & accepted[:, :, None, None]
    joint = supported[:,1:4].any(axis=1)
    mass = int(joint.sum())
    support = {'depth_indices': np.asarray(depths, np.int64), 'binary_masks': binary,
               'accepted_class_slice': accepted, 'pixel_counts': counts.astype(np.int64)}
    base.update(geometry_sha256=digest(geometry), segment_reason=why,
                sampled_depth_indices=depths, per_slice_class_pixels=counts.tolist(),
                accepted_class_slice=accepted.tolist(), accepted_class_pixels=supported.sum(axis=(0,2,3)).tolist(),
                accepted_joint_union_pixels=mass,
                support_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in support.items()})
    def fallback(reason):
        return None, {**base, 'reason': reason}, support
    def border_counts(array):
        edge = array.copy()
        edge[:,:,1:-1,1:-1] = False
        return edge.sum(axis=(2,3)).tolist()
    base.update(raw_border_class_slice_pixels=border_counts(binary),
                accepted_border_class_slice_pixels=border_counts(supported))
    if mass < 250:
        return fallback('insufficient_joint_support')
    if int(accepted[:,1:4].any(axis=0).sum()) < 2:
        return fallback('partial_joint_channel_support')
    if supported[:,:,0,:].any() or supported[:,:,-1,:].any() or supported[:,:,:,0].any() or supported[:,:,:,-1].any():
        return fallback('foreground_touches_source_border')
    yy, xx = np.nonzero(supported.any(axis=(0,1)))
    x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
    h,w = map(int,shape[1:]); sy,sx = map(float,sp[0])
    row,col = (y0+y1-1)/2., (x0+x1-1)/2.
    center = (row/max(1,h-1), col/max(1,w-1))
    required = max((y1-y0)*sy, (x1-x0)*sx)+2*MARGIN_MM
    base.update(support_box_xyxy=[x0,y0,x1,y1], center_row_col=list(center),
                required_side_mm=required, native_shape=[h,w], spacing_mm=[sy,sx])
    # Whole-mm ceiling plus source rounding can require an extra millimetre.
    # Select the first size whose actual rounded pixel rectangle supplies the
    # full margin on all four sides. Never shift, clip or shrink the rectangle.
    # The pinned source consumes normalized centers. Reconstituting that float
    # can move a half-even tie by one pixel; use its arithmetic before testing
    # margins/bounds, while retaining exact equality with the source rectangle.
    source_row, source_col = center[0]*(h-1), center[1]*(w-1)
    for side in range(max(MIN_SIDE_MM, math.ceil(required)), MAX_SIDE_MM+1):
        ch,cw = round(side/sy),round(side/sx)
        y,x = round(source_row-(ch-1)/2),round(source_col-(cw-1)/2)
        margins = [(y0-y)*sy, (y+ch-y1)*sy, (x0-x)*sx, (x+cw-x1)*sx]
        if min(margins) >= MARGIN_MM-1e-9:
            break
    else:
        return fallback('required_side_exceeds_140mm')
    base.update(side_mm_requested=side, requested_box_xyxy=[x,y,x+cw,y+ch],
                actual_margin_mm=margins, actual_margin_order=['top','bottom','left','right'],
                actual_field_mm=[ch*sy,cw*sx],
                out_of_field_fraction=1-(max(0,min(x+cw,w)-max(x,0))*
                                          max(0,min(y+ch,h)-max(y,0)))/max(1,ch*cw))
    if x < 0 or y < 0 or x+cw > w or y+ch > h or side > min(h*sy,w*sx):
        return fallback('support_crop_margin_does_not_fit_source')
    g = view_geometry((h,w), sp[0], SIZE, crop_mm=side, center=center)
    require(tuple(g.crop_xyxy) == (x,y,x+cw,y+ch),
            f'Source clipped/shrank extent crop: requested={(x,y,x+cw,y+ch)}, source={tuple(g.crop_xyxy)}')
    oh,ow = g.resized_shape
    base.update(retained_box_xyxy=list(g.crop_xyxy), valid_xyxy=list(g.valid_xyxy),
                native_support_pixels=ch*cw, resampling_ratio=[oh/ch,ow/cw],
                padding_pixels=SIZE*SIZE-oh*ow)
    return (side,center), base, support


def source_render(cache, uid, study, centers):
    """Pinned global/slab/intensity path, with one geometry selector per series."""
    import px_cache as PX
    from training.exact_duplicates import ExactDuplicateResolver
    plan = ExactDuplicateResolver(None, None, False).resolve(study)
    ids = [study['series'][i]['SeriesInstanceUID'] for i in plan['representative_indices']]
    cursor = iter(ids)
    original = PX.series_views
    def views(*args, **kwargs):
        selector = centers[next(cursor)]
        return [(0, None, (.5, .5))] if selector is None else [(0, selector[0], selector[1])]
    try:
        PX.series_views = views
        item = PX.StudySlices(str(cache), [], SIZE, 'v1').build(uid)
        require(not item['skipped_series'] and item['error'] is None, 'Source skipped/error native series')
        require(next(cursor, None) is None, 'Source representative order differs')
        by_id = {s['SeriesInstanceUID']: s for s in study['series']}
        require(all(b['n_slices'] == len(retained_support(by_id[b['series']])) for b in item['blocks']), 'Native/retained slab mapping differs')
        return item
    finally:
        PX.series_views = original


def read_pair(cache, uid, study, geometries, bank, *, view_geometry=None, renderer=None, support_store=None):
    """Single shared prospective training/inference adapter; no outcome inputs."""
    import numpy as np
    if view_geometry is None:
        from training.input_provenance import view_geometry
    renderer = source_render if renderer is None else renderer
    require(study['StudyInstanceUID'] == uid, 'Study origin binding differs')
    ids = [s['SeriesInstanceUID'] for s in study['series']]
    require(len(set(ids)) == len(ids), 'Duplicate source series')
    centers, receipts = {}, {}
    for s in study['series']:
        sid = s['SeriesInstanceUID']
        centers[sid], receipts[sid], support = series_extent(uid, s, geometries.get(sid), bank, view_geometry)
        if support_store is not None:
            support_store[sid] = support
        gg = view_geometry(s['native_shape'][-2:], s['spacing_mm'][0], SIZE)
        gh, gw = gg.resized_shape; nh, nw = gg.native_shape
        receipts[sid].update(native_shape=list(gg.native_shape), spacing_mm=list(gg.spacing),
                             global_padding_pixels=SIZE * SIZE - gh * gw, global_resampling_ratio=[gh / nh, gw / nw],
                             global_field_mm=[nh * gg.spacing[0], nw * gg.spacing[1]])
    global_item = renderer(cache, uid, study, dict.fromkeys(ids))
    local_item = renderer(cache, uid, study, centers)
    require(global_item['blocks'] == local_item['blocks'], 'Global/local slice blocks differ')
    # Pinned items are CPU tensors; synthetic tests may supply numpy arrays.
    def array(value):
        return value.numpy() if hasattr(value, 'numpy') else np.asarray(value)
    gm, lm = array(global_item['meta']), array(local_item['meta'])
    columns = [i for i in range(12) if i != 6]
    require(np.array_equal(gm[:, columns], lm[:, columns]), 'Global/local source ordering differs')
    gp, lp = array(global_item['px']), array(local_item['px'])
    require(gp.shape == lp.shape and np.isfinite(gp).all() and np.isfinite(lp).all(), 'Reader pixel contract differs')
    for b in global_item['blocks']:
        if receipts[b['series']]['reason']:
            off = b['views'][0][1]; n = b['n_slices']
            require(np.array_equal(gp[off:off+n], lp[off:off+n]), 'Fallback differs from corresponding global')
    plan = triplet_plan(global_item['blocks'], {sid: centers[sid] is not None for sid in ids})
    ix = [p['indices'] for p in plan]
    control = gp[ix].copy(); treatment = control.copy()
    treatment[12:] = lp[ix[12:]]
    require(control.shape == (24, 3, SIZE, SIZE), 'Matched24 shape differs')
    # Every slot is valid; missing planes do not inject zero views. The plan
    # always fills with available genuine global/eligible series triplets.
    masks = np.ones(24, dtype=np.bool_)
    return control, treatment, plan, receipts, masks


def native_overview(out,ordinal,study,receipts):
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
        caption=f"{s['plane']} native {a.shape} z={z} / {r['reason'] or str(r['side_mm_requested'])+'mm extent'}"
        cv2.putText(preview,caption,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,100,100),1)
        require(cv2.imwrite(str(Path(out)/f'{ordinal:03d}-native-series-{index:02d}.png'),preview),'Native containment artifact write failed')


def render_triplets(out, ordinal, study, plan, receipts, control, treatment):
    """All eligible series and every selected triplet, including slab support."""
    import cv2
    import numpy as np
    native_overview(out, ordinal, study, receipts)
    by_id = {s['SeriesInstanceUID']: s for s in study['series']}
    supports = {sid: retained_support(s) for sid, s in by_id.items()}
    native_plans = []
    for j, p in enumerate(plan):
        s = by_id[p['series']]; r = receipts[p['series']]
        volume = np.load(s['path'], mmap_mode='r', allow_pickle=False)
        rows = []
        for channel, retained in enumerate(p['retained_indices']):
            group = supports[p['series']][retained]
            lo, hi = s['intensity']['display_window_uint16']
            native = []
            for z in group:
                a = np.rint(np.clip((np.asarray(volume[z], float)-lo)/max(hi-lo,1e-6),0,1)*255).astype(np.uint8)
                a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
                if 'requested_box_xyxy' in r:
                    x,y,x1,y1 = r['requested_box_xyxy']; cv2.rectangle(a,(x,y),(x1-1,y1-1),(0,255,0) if not r['reason'] else (0,0,255),2)
                h,w = a.shape[:2]; scale=min(384/h,384/w)
                a=cv2.resize(a,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
                pane=np.zeros((384,384,3),np.uint8);pane[:a.shape[0],:a.shape[1]]=a
                cv2.putText(pane,f'native z={z} retained={retained}',(3,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,100,100),1)
                native.append(pane)
            pair=[cv2.cvtColor(v,cv2.COLOR_GRAY2BGR) for v in (control[j,channel],treatment[j,channel])]
            rows.append(np.concatenate(pair+native,axis=1))
        width=max(row.shape[1] for row in rows)
        rows=[cv2.copyMakeBorder(row,0,0,0,width-row.shape[1],cv2.BORDER_CONSTANT) for row in rows]
        require(cv2.imwrite(str(Path(out)/f'{ordinal:03d}-paired-triplet-{j:02d}.png'),np.concatenate(rows,axis=0)), 'Triplet evidence write failed')
        native_plans.append({'slot': j, 'series': p['series'], 'native_support_indices': [supports[p['series']][k] for k in p['retained_indices']]})
    return native_plans


def render_evidence(out, ordinal, study, plan, receipts, control, treatment, support_store):
    """Preserve every triplet's native support and add all segmentation masks.

    Saved masks are raw >.5 binary channels; accepted flags distinguish small
    fragments from support used by the envelope. Contours show all channels.
    """
    import cv2
    import numpy as np
    native_plan = render_triplets(out, ordinal, study, plan, receipts, control, treatment)
    for j, series in enumerate(study['series']):
        support = support_store[series['SeriesInstanceUID']]
        if support is None:
            continue
        prefix = f'{ordinal:03d}-segmentation-series-{j:02d}'
        mask_path = Path(out)/(prefix+'.npz')
        np.savez_compressed(mask_path, **support)
        volume = np.load(series['path'], mmap_mode='r', allow_pickle=False)
        require(volume.shape == tuple(series['native_shape']), 'Evidence native shape differs')
        lo,hi = series['intensity']['display_window_uint16']
        colors = [(255,0,255),(255,255,0),(0,255,255),(0,128,255)]
        for i,z in enumerate(support['depth_indices']):
            native = np.rint(np.clip((np.asarray(volume[z],float)-lo)/max(hi-lo,1e-6),0,1)*255).astype(np.uint8)
            pane = cv2.cvtColor(native,cv2.COLOR_GRAY2BGR)
            for channel,color in enumerate(colors):
                mask = support['binary_masks'][i,channel].astype(np.uint8)
                contours,_ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(pane,contours,-1,color,1)
            r = receipts[series['SeriesInstanceUID']]
            if 'requested_box_xyxy' in r:
                x,y,x1,y1 = r['requested_box_xyxy']
                cv2.rectangle(pane,(x,y),(x1-1,y1-1),(0,255,0) if not r['reason'] else (0,0,255),2)
            require(cv2.imwrite(str(Path(out)/f'{prefix}-native-z{z:04d}.png'),pane), 'Support overlay write failed')
        # Verify serialization preserves every native mask/depth/count exactly.
        with np.load(mask_path,allow_pickle=False) as saved:
            require(set(saved.files) == set(support) and all(np.array_equal(saved[k],v) for k,v in support.items()), 'Saved segmentation support differs')
    return native_plan


def run(config):
    """CPU/offline frozen32 mechanics; anatomy review and fitting stay pending."""
    import numpy as np
    import torch
    require(config.get('device') == 'cpu', 'Same-series qualification is CPU-only')
    require(config['binding']['protocol_id'] == PROTOCOL, 'Qualification protocol binding differs')
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','protocol':PROTOCOL,'completed_studies':0,'advancement_allowed':False,
            'anatomical_containment':'UNVERIFIED','train_inference_parity':'UNVERIFIED',
            'binding':config['binding'],'studies':[]}
    started=time.perf_counter();counts=Counter();fallbacks=Counter();ratios=[];global_ratios=[]
    import shutil
    report['minimum_free_disk_bytes']=shutil.disk_usage(out).free
    report['peak_native_cache_bytes']=0
    try:
        verify_sources(config['code'],config['source_pins'])
        require(file_sha(config['anatomy'])==WEIGHT_SHA and Path(config['anatomy']).stat().st_size==WEIGHT_BYTES,'SKM-TEA weight identity differs')
        validate_membership(config['ids'],config['folds'],config['excluded'])
        require(digest(config['folds'])==config['binding']['selected_folds_sha256'],'Selected membership changed')
        sys.path.insert(0,config['code']);torch.set_num_threads(1)
        import knee_anchor_bank_v2 as KB
        from training import native_cache as NC, geometry_bank as GB
        require((KB.N_SLICES,KB.MASS_MIN,KB.LO,KB.HI,KB.OBLIQUE_COS,KB.SPAN_MM)==(7,250,.15,.85,.85,4.), 'Pinned candidate thresholds differ')
        import px_cache as PX
        require((PX.THIN_MM,PX.SLAB_MM)==(1.75,3.5), 'Pinned slab support mapping differs')
        stage_inputs(config['raw'],config['staged'],config['ids'],config['binding']['series_sha256'])
        t=time.perf_counter();model=KB.load_model(config['anatomy'],'cpu');bank=KB.AnchorBank(model,'cpu')
        report['localizer_load_seconds']=time.perf_counter()-t
        import monai, importlib.metadata
        require(Path(monai.__file__).resolve().is_relative_to(Path(config['code'])/'training/vendor_runtime/monai060') and monai.__version__=='0.6.0','Pinned MONAI runtime shadowed')
        report.update(device='cpu',python=sys.version,packages={p:importlib.metadata.version(p) for p in ('torch','numpy','pydicom','scipy','pandas')})
        def state_sha():
            h=hashlib.sha256()
            for name,value in sorted(model.state_dict().items()):
                h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        report['localizer_state_before']=state_sha()
        for ordinal,uid in enumerate(config['ids']):
            work=out/f'{ordinal:03d}-native';t=time.perf_counter()
            manifest=NC.prepare(config['staged'],work,split='train',study_ids=[uid],source_directory=True)
            require(len(manifest['studies'])==1,'Native study count differs');study=manifest['studies'][0]
            require(not study['errors'] and all(not s['errors'] for s in study['series']),'Source swallowed native decoder errors')
            GB.build(config['staged'],work/'manifest.json',work/'geometry',split='train',source_directory=True)
            geos={s['SeriesInstanceUID']:json.loads((work/'geometry'/(KB.key_for(uid,s['SeriesInstanceUID'])+'.json')).read_bytes()) for s in study['series']}
            decode_seconds=time.perf_counter()-t;t=time.perf_counter()
            native_hashes={s['SeriesInstanceUID']:file_sha(s['path']) for s in study['series']}
            report['peak_native_cache_bytes']=max(report['peak_native_cache_bytes'],sum(p.stat().st_size for p in work.rglob('*') if p.is_file()))
            supports, repeated_supports = {}, {}
            c,l,plan,receipts,masks=read_pair(work,uid,study,geos,bank,support_store=supports)
            cc,ll,pp,rr,mm=read_pair(work,uid,study,geos,bank,support_store=repeated_supports)
            require(set(supports)==set(repeated_supports) and all(
                (v is None and repeated_supports[sid] is None) or
                (v is not None and repeated_supports[sid] is not None and set(v)==set(repeated_supports[sid]) and
                 all(np.array_equal(a,repeated_supports[sid][k]) for k,a in v.items()))
                for sid,v in supports.items()), 'Repeated segmentation support differs')
            require(np.array_equal(c,cc) and np.array_equal(l,ll) and plan==pp and receipts==rr and np.array_equal(masks,mm),'Shared reader repeat differs')
            require(state_sha()==report['localizer_state_before'],'Localizer state changed')
            reader_seconds=time.perf_counter()-t
            require(native_hashes=={s['SeriesInstanceUID']:file_sha(s['path']) for s in study['series']},'Native input bytes changed')
            # Freeze/check the plan first; rendering adds only native-support evidence.
            plan_sha=digest(plan);native_plan=render_evidence(out,ordinal,study,plan,receipts,c,l,supports)
            require(digest(plan)==plan_sha,'Frozen selection plan changed during rendering')
            save(out/f'{ordinal:03d}-source-private.json',{'study':study,'geometry':geos,'series_crops':receipts,'triplet_plan':plan,'selection_plan_sha256':plan_sha,'native_triplet_support':native_plan,'native_volume_sha256':native_hashes})
            native=sum(receipts[p['series']]['reason'] is None for p in plan[12:])
            counts.update(studies=1,series=len(receipts),eligible_series=sum(r['reason'] is None for r in receipts.values()),
                          context_slots=12,supplemental_slots=12,native_slots=native,fallback_slots=12-native)
            fallbacks.update(r['reason'] for r in receipts.values() if r['reason'])
            ratios.extend(v for r in receipts.values() if r['reason'] is None for v in r['resampling_ratio'])
            global_ratios.extend(v for r in receipts.values() for v in r['global_resampling_ratio'])
            def ratio_summary(values):
                return {'count':len(values),'min':min(values),'max':max(values),'mean':float(np.mean(values))} if values else {'count':0}
            report['resampling']={'local':ratio_summary(ratios),'global':ratio_summary(global_ratios)}
            report['minimum_free_disk_bytes']=min(report['minimum_free_disk_bytes'],shutil.disk_usage(out).free)
            report['studies'].append({'ordinal':ordinal,'decode_geometry_seconds':decode_seconds,'reader_twice_seconds':reader_seconds,
                'control_sha256':hashlib.sha256(c.tobytes()).hexdigest(),'treatment_sha256':hashlib.sha256(l.tobytes()).hexdigest(),
                'selection_plan_sha256':plan_sha,'native_triplets':native,'fallback_triplets':12-native,'series':list(receipts.values()),
                'missing_planes':sorted({'Sagittal','Coronal','Axial'}-{s['plane'] for s in study['series']}),
                'view_mask_valid_count':int(masks.sum()),'shared_reader_repeat_parity':True,
                'segmentation_support_repeat_parity':True})
            report.update(completed_studies=ordinal+1,counts=dict(counts),fallback_series=dict(fallbacks));save(out/'receipt.json',report)
            import shutil
            shutil.rmtree(work)
        report['localizer_state_after']=state_sha()
        require(report['localizer_state_before']==report['localizer_state_after'],'Localizer state changed')
        report.update(status='PASS_READER_MECHANICS',train_inference_parity='PASS_SHARED_ADAPTER_REPEAT')
    except BaseException as error:
        report.update(status='FAILED',failure_type=type(error).__name__,failure_message=str(error));raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        save(out/'receipt.json',report)
    return report
