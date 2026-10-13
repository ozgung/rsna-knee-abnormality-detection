"""Image-only, same-series sagittal ROI reader. No study anchor or label access.

The pinned source candidate/segment path runs only on the qualification host.
Public pixel preprocessing is unchanged; this module owns binding and allocation.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

from knee.native_joint_roi_pilot import SIDE_MM, SIZE, WEIGHT_SHA, WEIGHT_BYTES, joint_view
from knee.public_s6_qualification import digest, file_sha, require, save, stage_inputs, verify_sources, validate_membership

PROTOCOL = 'same-series-sagittal-100mm-matched24-v1'


def series_candidate(uid, series, geometry, bank, view_geometry, project):
    """Invoke candidate directly, bound to exactly this series and its affines.

    candidate() retains seven-slice segment preprocessing, .5 probability,
    40-pixel centroid, 250-pixel mass, .15-.85 plausibility and .85 obliquity.
    record(), slab_prior(), cross-series filtering/averaging are never called.
    """
    import numpy as np
    sid = series['SeriesInstanceUID']
    base = {'origin_series': sid, 'reason': None, 'padding_pixels': 0}
    if series['plane'] != 'Sagittal':
        return None, {**base, 'reason': 'unsupported_plane_no_independent_segmenter'}
    if geometry is None:
        return None, {**base, 'reason': 'missing_geometry'}
    require(geometry['StudyInstanceUID'] == uid and geometry['SeriesInstanceUID'] == sid,
            'Geometry origin binding differs')
    if not geometry['quality']['aligned']:
        return None, {**base, 'reason': 'unaligned_geometry'}
    aff = np.asarray(geometry['affines_col_row_origin_lps_mm'], float)
    sp = np.asarray(series['spacing_mm'], float)
    require(aff.shape == (series['native_shape'][0], 3, 3) and np.isfinite(aff).all(), 'Affine/native contract differs')
    require(sp.shape == (len(aff), 2) and np.isfinite(sp).all() and (sp > 0).all(), 'Spacing/native contract differs')
    require(np.allclose(sp, sp[0], rtol=1e-4, atol=0), 'Changing in-plane spacing unsupported')
    require(np.allclose(np.linalg.norm(aff[:, :, :2], axis=1)[:, ::-1], sp, rtol=1e-4, atol=1e-4), 'Affine/spacing contract differs')
    require(np.all(np.linalg.norm(np.cross(aff[:, :, 0], aff[:, :, 1]), axis=1) > 0), 'Degenerate affine')
    candidate, why = bank.candidate(series, geometry)
    # Recognize only the pinned candidate's explicit data conditions. Opaque
    # statuses/errors from an altered source cannot become successful fallback.
    if candidate is None:
        import re
        require(why == 'oblique' or bool(re.fullmatch(r'mass<250\(\d+\)|implausible\(rf=-?\d+\.\d{2},cf=-?\d+\.\d{2}\)', why)),
                'Unknown localizer status: ' + str(why))
        return None, {**base, 'reason': 'no_same_series_candidate', 'candidate_reason': why}
    require(why == 'ok' and candidate['sid'] == sid, 'Candidate origin binding differs')
    point = np.asarray(candidate['X'], float)
    require(point.shape == (3,) and np.isfinite(point).all(), 'Candidate point contract differs')
    center, receipt = joint_view(series, geometry, {'joint_lps': point, 'why': []}, view_geometry, project)
    return center, {**receipt, 'origin_series': sid, 'geometry_sha256': digest(geometry),
                    'candidate_reason': why, 'candidate_mass': candidate['mass'], 'candidate_lps': point.tolist()}


def source_render(cache, uid, study, centers):
    """Pinned global/slab/intensity path, with one geometry selector per series."""
    import px_cache as PX
    from training.exact_duplicates import ExactDuplicateResolver
    plan = ExactDuplicateResolver(None, None, False).resolve(study)
    ids = [study['series'][i]['SeriesInstanceUID'] for i in plan['representative_indices']]
    cursor = iter(ids)
    original = PX.series_views
    def views(*args, **kwargs):
        center = centers[next(cursor)]
        return [(0, None, (.5, .5))] if center is None else [(0, SIDE_MM, center)]
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


def triplet_plan(blocks, eligible):
    """Frozen 12 all-series context + 12 eligible sagittal round-robin slots."""
    import math
    require(bool(blocks), 'No source blocks')
    require(len({b['series'] for b in blocks}) == len(blocks), 'Duplicate source blocks')
    def allocate(selected, arm):
        result = []
        for j in range(12):
            b = selected[j % len(selected)]; n = b['n_slices']; off = b['views'][0][1]
            require(n > 0 and len(b['views']) == 1, 'Source slice/view contract differs')
            # Preserve the original all-series plan exactly. Supplemental uses
            # the same evenly spaced rule over its own eligible series list.
            center = min(n - 1, int(round((n - 1) * ((j // len(selected) + .5) / max(1, math.ceil(12 / len(selected)) if arm == 'global_context' else (12 - (j % len(selected)) + len(selected) - 1) // len(selected))))))
            relative = [min(n - 1, max(0, center + k)) for k in (-1, 0, 1)]
            result.append({'series': b['series'], 'indices': [off + i for i in relative],
                           'retained_indices': relative, 'role': arm})
        return result
    context = allocate(blocks, 'global_context')
    selected = [b for b in blocks if eligible[b['series']] and b['plane'] == 'Sagittal']
    supplement = allocate(selected, 'same_series_supplement') if selected else [dict(p, role='global_supplement_fallback') for p in context]
    return context + supplement


def read_pair(cache, uid, study, geometries, bank, *, view_geometry=None, project=None, renderer=None):
    """Single shared prospective training/inference adapter; no outcome inputs."""
    import numpy as np
    if view_geometry is None:
        from training.input_provenance import view_geometry
    if project is None:
        from knee_anchor_bank_v2 import project
    renderer = source_render if renderer is None else renderer
    require(study['StudyInstanceUID'] == uid, 'Study origin binding differs')
    ids = [s['SeriesInstanceUID'] for s in study['series']]
    require(len(set(ids)) == len(ids), 'Duplicate source series')
    centers, receipts = {}, {}
    for s in study['series']:
        sid = s['SeriesInstanceUID']
        centers[sid], receipts[sid] = series_candidate(uid, s, geometries.get(sid), bank, view_geometry, project)
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


def retained_support(series):
    """Source slice groups behind retained frames; must match pinned blocks."""
    import numpy as np
    pos = np.asarray(series['positions'], float)
    require(pos.shape == (series['native_shape'][0],) and len(pos) and np.isfinite(pos).all(), 'Native depth mapping differs')
    if series['plane'] == 'Sagittal' and series.get('side') == 'R': pos = -pos
    order = np.argsort(pos, kind='stable'); pos = pos[order]
    spacing = float(np.median(np.diff(pos))) if len(pos) > 1 else 0.
    if 0 < spacing < 1.75:
        group = max(1, int(round(3.5 / spacing)))
        count = len(pos) // group + (1 if len(pos) % group >= max(1, group // 2) else 0)
        if count >= 2:
            return [order[j*group:min(len(pos), (j+1)*group)].tolist() for j in range(count)]
    return [[int(i)] for i in order]


def render_evidence(out, ordinal, study, plan, receipts, control, treatment):
    """All eligible series and every selected triplet, including slab support."""
    import cv2
    import numpy as np
    from knee.native_joint_roi_pilot import containment_native
    containment_native(out, ordinal, study, receipts)
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
            c,l,plan,receipts,masks=read_pair(work,uid,study,geos,bank)
            cc,ll,pp,rr,mm=read_pair(work,uid,study,geos,bank)
            require(np.array_equal(c,cc) and np.array_equal(l,ll) and plan==pp and receipts==rr and np.array_equal(masks,mm),'Shared reader repeat differs')
            require(state_sha()==report['localizer_state_before'],'Localizer state changed')
            reader_seconds=time.perf_counter()-t
            require(native_hashes=={s['SeriesInstanceUID']:file_sha(s['path']) for s in study['series']},'Native input bytes changed')
            # Freeze/check the plan first; rendering adds only native-support evidence.
            plan_sha=digest(plan);native_plan=render_evidence(out,ordinal,study,plan,receipts,c,l)
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
                'view_mask_valid_count':int(masks.sum()),'shared_reader_repeat_parity':True})
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
