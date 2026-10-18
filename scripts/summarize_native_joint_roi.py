"""Summarize private reader receipts without emitting study/series identifiers."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ratio(fallback, total):
    return dict(fallback=fallback, total=total,
                fraction=fallback / total if total else None)


def summarize(directory, build_path):
    root = Path(directory)
    report = json.loads((root / 'receipt.json').read_text())
    build = json.loads(Path(build_path).read_text())
    require(report['binding'] == build['binding'], 'Immutable build binding differs')
    require(report['status'] == 'PASS_READER_MECHANICS', 'Reader is not successfully terminal')
    studies = report['studies']
    require(len(studies) == report['completed_studies'] ==
            report['binding']['protocol']['study_count'], 'Incomplete study coverage')
    require(report['train_inference_parity'] == 'PASS_SHARED_ADAPTER_REPEAT', 'Reader parity failed')
    require(report['localizer_state_before'] == report['localizer_state_after'], 'Localizer state changed')
    require(report['advancement_allowed'] is False, 'Unexpected advancement authorization')
    require(math.isfinite(report['elapsed_seconds']) and report['elapsed_seconds'] > 0,
            'Invalid runtime')
    planes = {p: Counter() for p in ('Sagittal', 'Coronal', 'Axial')}
    totals = Counter()
    reasons = Counter()
    seen = set()
    for row in studies:
        ordinal = row['ordinal']
        require(ordinal not in seen, 'Duplicate study ordinal')
        seen.add(ordinal)
        private = json.loads((root / f'{ordinal:03d}-source-private.json').read_text())
        source_series = private['study']['series']
        crops = private['series_crops']
        require(len(crops) == len(source_series) == len(row['series']), 'Series coverage differs')
        crop_counts = lambda rows: Counter(json.dumps(r, sort_keys=True, allow_nan=False) for r in rows)
        require(crop_counts(crops.values()) == crop_counts(row['series']), 'Series crop receipt differs')
        series_planes = {}
        for source in source_series:
            sid = source['SeriesInstanceUID']
            plane = source['plane']
            require(sid not in series_planes and plane in planes, 'Invalid series or plane')
            series_planes[sid] = plane
            crop = crops[sid]
            fallback = crop['reason'] is not None
            planes[plane]['series'] += 1
            planes[plane]['fallback_series'] += fallback
            totals['series'] += 1
            totals['fallback_series'] += fallback
            if fallback:
                reasons[crop['reason']] += 1
        plan = private['triplet_plan']
        require(len(plan) == report['binding']['protocol']['treatment_joint_views'], 'Slot count differs')
        fallback_slots = 0
        for slot in plan:
            sid = slot['series']
            require(sid in series_planes, 'Unknown treatment series')
            plane = series_planes[sid]
            fallback = crops[sid]['reason'] is not None
            planes[plane]['slots'] += 1
            planes[plane]['fallback_slots'] += fallback
            fallback_slots += fallback
        require(fallback_slots == row['fallback_treatment_triplets'], 'Fallback slot total differs')
        require(row['shared_reader_repeat_parity'] is True, 'Study parity failed')
        totals['slots'] += len(plan)
        totals['fallback_slots'] += fallback_slots
        totals['studies_with_any_fallback'] += fallback_slots > 0
        totals['studies_all_fallback'] += fallback_slots == len(plan)
    require(seen == set(range(len(studies))), 'Study ordinals differ')
    require(dict(reasons) == report['fallback_series'], 'Fallback reason totals differ')
    return dict(status='PASS_AGGREGATE_REVIEW', binding=report['binding'],
                studies_with_any_slot_fallback=ratio(totals['studies_with_any_fallback'], len(studies)),
                studies_with_all_slots_fallback=ratio(totals['studies_all_fallback'], len(studies)),
                series=ratio(totals['fallback_series'], totals['series']),
                treatment_joint_slots=ratio(totals['fallback_slots'], totals['slots']),
                per_plane={p: dict(series=ratio(c['fallback_series'], c['series']),
                                   treatment_joint_slots=ratio(c['fallback_slots'], c['slots']))
                           for p, c in planes.items()},
                fallback_reasons=dict(reasons), elapsed_seconds=report['elapsed_seconds'],
                peak_rss_bytes=report['peak_rss_bytes'],
                anatomical_containment='UNVERIFIED_REQUIRES_VISUAL_REVIEW',
                advancement_allowed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('build_receipt')
    parser.add_argument('output')
    args = parser.parse_args()
    result = summarize(args.directory, args.build_receipt)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'advancement_allowed': False}))
