"""Build-only immutable Raptor64 fold-0 package for one arm (init x head); historical AST checked.

Arms: in-orig, in-plane (ImageNet-384 init) and dr-orig, dr-plane (Dread CC0 v4_full backbone
init), each with the original public RaptorClassifier head or the plane-aware head. The
historical own224 Track-A trainer/entry are adapted with anchored replacements only; the
sampler/model/optimizer/schedule/loss/training loop/top-3/SWA AST must stay identical.
"""
import argparse
import ast
import difflib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee import raptor64_training as guard, public_training_smoke as smoke

_spec = importlib.util.spec_from_file_location('build_source96_training_helpers', ROOT / 'scripts/build_source96_training.py')
s96 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s96)
replace_once, function_source, scientific_ast = s96.replace_once, s96.function_source, s96.scientific_ast
TRAIN_SHA, ENTRY_SHA = s96.TRAIN_SHA, s96.ENTRY_SHA
FILES = ('knee/__init__.py', 'knee/raptor64_training.py', 'knee/raptor64_loader.py', 'knee/raptor64_cache.py',
         'knee/source96_training.py', 'knee/source96_loader.py', 'knee/source96_cache.py', 'knee/source_span_pilot.py',
         'knee/corpus44_parity.py', 'knee/public_training_smoke.py', 'knee/public_rich_cache.py', 'knee/reference.py',
         'knee/fresh_reader_setup.py', 'knee/raptor_input.py')
HEAD_HOOK = 'StudyWindows, RaptorClassifier, collate = raptor64.head_variant(StudyWindows, RaptorClassifier, collate, __file__)\n'
VARIANT_HOOK = 'StudyWindows = raptor64.recipe_variant(StudyWindows, __file__)\n'


def adapt_trainer(original, full=False, variant=None):
    text = original
    doc = next(n for n in ast.parse(original).body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))
    text = replace_once(text, ast.get_source_segment(original, doc),
        '"""Raptor64 fold-0 experiment: Dread widedense v4 input contract (64 slots, 6-94%, 336->384).\n'
        'Historical sampler/model/optimizer/selection/SWA AST retained; per-arm initializer and head\n'
        'variant come from raptor64-recipe.json. Verified private cache; no gold scoring."""')
    for name, code in [('_open_corpus', 'def _open_corpus():\n    return raptor64.staged_corpus()[:2]'),
                       ('_open_ids', 'def _open_ids():\n    return raptor64.staged_corpus()[2]')]:
        text = replace_once(text, function_source(text, name), code)
    text = replace_once(text, '    a = ap.parse_args()', '    a = ap.parse_args()\n    raptor64.require_recipe(a, __file__)')
    text = replace_once(text, '        dev = "cuda" if torch.cuda.is_available() else "cpu"', '        dev = raptor64.strict_cuda(torch)')
    text = replace_once(text, '    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)',
        '    raptor64.verify_membership(train_ids, oof_ids, gold_ids, a.fold, a.fold_file, a.excluded_ids, __file__)\n'
        '    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)')
    old = ('    use_timm = a.ckpt in ("timm", "pretrained")\n    bb = build_backbone(a.arch, pretrained=use_timm)\n'
           '    src = load_raptor(bb, a.ckpt)\n    if use_timm: print(f"[raptor] timm-pretrained backbone: {a.arch}", flush=True)')
    text = replace_once(text, old, '    bb = build_backbone(a.arch, pretrained=False)\n'
        '    INITIALIZER = raptor64.initialize_backbone(bb, __file__)\n'
        '    src = INITIALIZER["init"] + ":" + INITIALIZER["file_sha256"]\n'
        '    print(f"[raptor64] init {INITIALIZER[\'init\']} loaded {INITIALIZER[\'loaded_tensors\']} tensors, missing 0, unexpected 0", flush=True)')
    text = replace_once(text, '            print(f"[raptor] grad_ckpt unsupported for {a.arch}: {e}", flush=True)',
        '            raise RuntimeError("Required gradient checkpointing unavailable") from None')
    text = replace_once(text, '    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]',
        '    INITIAL = raptor64.initial_record(model, INITIALIZER, __file__)\n    IDENT = raptor64.run_identity(a, __file__, _lab, INITIAL)\n'
        '    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]')
    text = replace_once(text, '        R_ = torch.load(a.resume, map_location="cpu", weights_only=False)',
        '        if smoke_file_sha(a.resume) != os.environ["RAPTOR64_RESUME_SHA256"]: raise ValueError("Pinned private resume bytes differ")\n'
        '        R_ = torch.load(a.resume, map_location="cpu", weights_only=False)\n        raptor64.validate_resume(R_, IDENT)')
    text = replace_once(text, '        del R_', '        raptor64.restore_rng(R_["rng"])\n        del R_')
    start = text.index('        if a.time_budget_h > 0 and (time.time() - t0) > a.time_budget_h * 3600 and ep < a.epochs - 1:')
    end = text.index('    if best_state is None: raise SystemExit("no epoch was selected")', start)
    text = text[:start] + '''        rp = os.path.join(HERE, f"raptor_resume_{a.tag}.pt")
        raptor64.save_resume(rp, {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None, "best": best,
            "best_state": best_state, "best_P": best_P, "hist": hist, "topk": topk,
            "epoch": ep, "elapsed": time.time() - t0}, IDENT)
        print(f"[resume] recoverable completed epoch {ep}; worker RNG continuation limitation recorded", flush=True)
''' + text[end:]
    if full:
        start = text.index('    # TRACKA PATCH: checkpoint selection = masked macro AUC of the FULL held-out fold')
        end = text.index('\n', text.index('    print(f"[oof-sub] FULL held-out fold', start)) + 1
        text = text[:start] + ('    # RAPTOR64 FULL-DATA: no held-out fold; per-epoch score is the pre-registered SWA-epoch indicator.\n'
                               '    if a.folds != 0 or oof_ids: raise SystemExit("full-data mode requires --folds 0 and no held-out studies")\n'
                               '    sl = None\n') + text[end:]
        start = text.index('        model.eval(); Ps = []\n        with torch.no_grad():\n            for x, y in sl:')
        end = text.index('\n', text.index('        print(f"ep{ep} OOF{len(P)} masked macro-AUC {au:.4f}", flush=True)', start)) + 1
        text = text[:start] + ('        au = raptor64.full_selection_score(ep, __file__); aucs = {}; P = None\n'
                               '        hist.append({"ep": ep, "loss": tot / len(tl), "oof_auc": au, "selection": "pre-registered fixed SWA epochs; no held-out"})\n') + text[end:]
        text = replace_once(text, 'print(f"ep{ep} loss {tot/len(tl):.3f} | OOF masked macro-AUC {au:.4f} (best {best:.4f}) | topk by OOF-full-fold: {[int(t[\'epoch\']) for t in topk]} | {time.time()-t0:.0f}s",',
                            'print(f"ep{ep} loss {tot/len(tl):.3f} | FULL-DATA fixed-epoch score {au:.1f} | topk fixed: {[int(t[\'epoch\']) for t in topk]} | {time.time()-t0:.0f}s",')
    # Head variant: subclasses from the reviewed package module; the public class bodies stay verbatim.
    text = replace_once(text, '\ndef main():\n', '\n' + HEAD_HOOK + (VARIANT_HOOK if variant else '') + '\n\ndef main():\n')
    if variant:
        # Explicit single-factor loss variant; the historical lossf assignment node is kept verbatim (AST-checked).
        text = replace_once(text, '    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)\n',
                            '    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)\n    lossf = raptor64.loss_variant(lossf, __file__)\n')
    text = 'from knee import raptor64_training as raptor64\nfrom knee.public_training_smoke import file_sha as smoke_file_sha\n' + text
    if scientific_ast(original) != scientific_ast(text):
        raise ValueError('Historical scientific AST changed')
    compile(text, 'raptor64-trainer', 'exec')
    return text


def adapt_entry(original, arm, workers, full=False, variant=None):
    text = original
    frozen = guard.frozen_for(arm); axial = variant == 'axial'
    defaults = {'FOLD_A': '0', 'FOLD_B': '', 'TIME_BUDGET_H': '0', 'RES': '384', 'ARCH': guard.ARCH, 'BS': '4', 'WORKERS': str(workers),
                'EPOCHS': str(guard.EPOCHS), 'CKPT': 'pinned-descriptor', 'K': str(frozen['k']), 'K_EVAL': str(frozen['k_eval']),
                'WORKDIR': f'/content/raptor64-{arm}-fold0', 'ASSET_DIR': f'/content/raptor64-{arm}-pkg', 'RAPTOR_INPUT': '/content/input',
                'GRAD_CKPT': '1', 'WIN_CHUNK': '8', 'ACCUM': '2'}
    tree = ast.parse(text)
    assign = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PARAM_DEFAULTS' for t in n.targets))
    old = ast.get_source_segment(text, assign); values = {kw.arg: ast.literal_eval(kw.value) for kw in assign.value.keywords}
    values.update(defaults); values['RESUME'] = '0'
    text = replace_once(text, old, 'PARAM_DEFAULTS = ' + repr(values))
    text = 'from knee import raptor64_training as raptor64\n' + text
    text = replace_once(text, '    return c\n', '    raptor64.entry_guard(c, __file__)\n    return c\n')
    text = replace_once(text, '    shutil.copy(assets / "train_knee.patch", work / "train_knee.patch")',
        '    shutil.copy(assets / "train_knee.patch", work / "train_knee.patch")\n'
        '    shutil.copy(assets / "raptor64-recipe.json", work / "raptor64-recipe.json")\n'
        '    shutil.copy(assets / "manifest.json", work / "manifest.json")\n'
        '    shutil.copytree(assets / "knee", work / "knee", dirs_exist_ok=True)')
    text = replace_once(text, '    return cmd',
        '    if c["RESUME"].lower() in ("1", "true", "yes"):\n        rp = p["work"] / f"raptor_resume_fold{fold}.pt"\n'
        '        if not rp.is_file(): raise ValueError("Resume requested but state missing")\n        cmd += ["--resume", str(rp)]\n    return cmd')
    start = text.index('    if c["CKPT"] == "timm":'); end = text.index('    def env_for(i):', start)
    text = text[:start] + '''    raptor64.strict_cuda(torch)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for k in folds:
        if c["RESUME"].lower() not in ("1", "true", "yes") and any(work.glob(f"raptor_*fold{k}*")):
            raise ValueError("Fresh training requires unused work directory; never delete recovery artifacts")
''' + text[end:]
    start = text.index('    for k in folds:\n        rp = work / f"raptor_resume_fold{k}.pt"'); end = text.index('    tot = sum(', start)
    text = text[:start] + '    # Keep every completed-epoch resume until external persistence has verified it.\n' + text[end:]
    text = replace_once(text, '"time_budget_note": "10.0 h = 12 h Kaggle session minus checkpoint/export margin"',
        '"time_budget_note": "No assistant cap; checkpoint against caller-supplied actual platform deadline only"')
    text = replace_once(text, '    k_eval = int(c["SMOKE_K_EVAL"] if smoke else c["K_EVAL"])',
        '    raptor64.export_guard(work, k, ckpt_path, kind, sel_epochs, partial)\n    k_eval = raptor64.' + ('AXIAL_WINDOWS' if axial else 'EXPORT_K'))
    text = replace_once(text, '    dev = "cuda" if torch.cuda.is_available() else "cpu"', '    dev = raptor64.strict_cuda(torch)')
    text = replace_once(text, '    truth = pd.read_parquet(tk._find("train_labels_v1.parquet"))',
        '    raptor64.verify_export_inputs(tk._find(c["LABELS_FILE"]), tk._find("train_labels_v1.parquet"), work / "folds_v0.json", work / "excluded_ids.json", val_ids, k, __file__)\n'
        '    truth = pd.read_parquet(tk._find("train_labels_v1.parquet"))')
    text = replace_once(text, '    pv, tv = predict(val_ids)',
        '    if gold_ids or len(val_ids) != raptor64.SPLITS[k]["development"]: raise ValueError("Export must use pinned nongold development only")\n'
        '    checkpoint_before = sha256(ckpt_path)\n    pv, tv = predict(val_ids)\n'
        '    if sha256(ckpt_path) != checkpoint_before: raise ValueError("Frozen checkpoint mutated during export")')
    text = replace_once(text, '"caveat": "val AUC is selection-biased (the same fold chose the epochs); gold-58 is the only clean number and is noisy (+-0.03).",',
        '"caveat": "Exposed report-selected development (top-3 by ' + ('K10' if axial else 'K16') + ' report-masked AUC); no gold scored; dread_v4 init additionally exposed (Dread trained on all non-gold studies).",\n'
        '        "raptor64_identity": json.load(open(work / "initial-state-record.json")),\n'
        '        "raptor64_policy": raptor64.recipe(work / "train_knee_patched.py"),\n'
        '        "raptor64_experiment_split": raptor64.runtime_split(work / "folds_v0.json", work / "excluded_ids.json", work / "train_knee_patched.py")[0],\n'
        '        "raptor64_cache_content_sha256": os.environ["RAPTOR64_CONTENT_SHA256"],\n' +
        ('        "comparison_endpoint": "K10 primary (selection K10; axial windows)",' if axial else
         '        "comparison_endpoint": "K42 primary (selection K16)",'))
    if variant == 'mirror':
        text = replace_once(text, '    df.to_csv(work / f"oof_fold{k}.csv", index=False)\n',
                            '    df.to_csv(work / f"oof_fold{k}.csv", index=False)\n'
                            '    raptor64.mirror_tta_csv(tk, model, val_ids, id2row, res, k_eval, dev, int(c["WORKERS"]), pv, work / f"oof_fold{k}_mirror_tta.csv")\n')
    if full:
        text = replace_once(text, '"--folds", "5", "--fold", str(fold),', '"--folds", "0", "--fold", "-1",')
        text = replace_once(text, 'def export_main(k):\n    c = cfg()\n',
                            'def export_main(k):\n    c = cfg()\n    if raptor64.full_mode(raptor64.recipe(Path(c["WORKDIR"]) / "train_knee_patched.py")):\n'
                            '        return raptor64.full_record(Path(c["WORKDIR"]), k, __file__)\n')
    compile(text, 'raptor64-entry', 'exec')
    return text


def build(destination, *, evidence_root, arm, workers=2, worker_choice_name=None, swa_epochs=None, teacher_sha256=None):
    out = Path(destination); evidence = Path(evidence_root)
    arm = guard.arm_slug(arm); init, head = guard.ARMS[arm]
    if out.exists():
        raise ValueError('Fresh immutable package directory required')
    if type(workers) is not int or not 0 <= workers <= 64 or (workers != 2 and not worker_choice_name):
        raise ValueError('Named frozen worker choice required')
    if worker_choice_name and not re.fullmatch('[a-z0-9][a-z0-9-]{2,63}', worker_choice_name):
        raise ValueError('Invalid worker choice name')
    old = evidence / 'outputs/kaggle/trackA-coatnet224-fold01-20261009'
    train = (old / 'train_knee_patched.py').read_text(); entry = (old / 'tracka_entry.py').read_text()
    if smoke.digest(train.encode()) != TRAIN_SHA or smoke.digest(entry.encode()) != ENTRY_SHA:
        raise ValueError('Historical source identity differs')
    full = arm in guard.FULL_ARMS
    if full != (swa_epochs is not None):
        raise ValueError('Pre-registered SWA epochs are required for, and only for, the full-data arm')
    variant = guard.VARIANTS.get(arm)
    if (variant in ('distil', 'distil_flip', 'axial')) != (teacher_sha256 is not None):
        raise ValueError('A pinned teacher SHA is required for, and only for, the distil arm')
    trainer = adapt_trainer(train, full, variant); runner = adapt_entry(entry, arm, workers, full, variant)
    sources = {name: (ROOT / name).read_text() for name in FILES}
    for name, value in sources.items():
        compile(value, name, 'exec')
    split = guard.FULL_SPLIT if full else guard.SPLITS[0]
    policy = {'version': 1, 'recipe': f'raptor64-v4-384-{arm}-{"full" if full else "fold0"}-v1', 'arm': arm, 'init': init, 'head': head,
              **({'mode': 'full', 'swa_epochs': list(swa_epochs)} if full else {}),
              **({'variant': variant} if variant else {}),
              **({'mirror': {'p': guard.MIRROR_P, 'perm': guard.SIDE_PERM, 'masked': [guard.MCL_INDEX], 'tta_export': True}} if variant == 'mirror' else {}),
              **({'flip': guard.FLIP_SPEC} if variant == 'distil_flip' else {}),
              **({'axial': guard.AXIAL_SPECS[arm]} if variant == 'axial' else {}),
              **({'teacher': {'source': 'dread_v4_full_k42', 'sha256': teacher_sha256, 'weight': guard.DISTIL_WEIGHTS[arm],
                              'loss': f'report BCE(pos_weight) + {guard.DISTIL_WEIGHTS[arm]} x plain BCE toward teacher probabilities on all 12 targets',
                              'exposure': 'second-order: the teacher saw fold-0 development studies'}} if variant in ('distil', 'distil_flip', 'axial') else {}),
              'workers': workers, 'worker_choice_name': worker_choice_name, 'frozen': guard.frozen_for(arm), 'export_k': guard.export_k_for(arm),
              'experiment_split': split, 'experiment_split_sha256': smoke.digest(smoke.canonical(split)),
              'input_contract': {'cache': 'raptor64-v4-336-sharded-v1', 'slots': [18, 14, 12, 8, 12], 'span': [0.06, 0.94],
                                 'storage': 336, 'model_input': 384, 'resize': 'StudyWindows bilinear (align_corners=False), as public eval_windows'},
              'initializer': {'imagenet': guard.IMAGENET_PIN, 'dread_v4': guard.DREAD_PIN, 'nartaa_oai': guard.NARTAA_PIN}[init],
              'head_definition': ('public RaptorClassifier' if head == 'original' else
                                  f'plane-aware: Embedding(6,{guard.CODE_DIM}) of plane*2+FS and Linear(1,{guard.DEPTH_DIM}) of normalized series depth, '
                                  'concatenated to LayerNorm(h) before att and clsW'),
              'selection': ('pre-registered fixed SWA epochs (no held-out set); historical top-3 rule applied to an epoch indicator' if full else
                            ('per-epoch K10 (all axial windows) report-masked macro AUC on the 870 fold-0 development; top-3 SWA' if variant == 'axial' else
                             'per-epoch K16 report-masked macro AUC on the 870 fold-0 development (own224/source96 protocol); top-3 SWA')),
              'worker_rng_resume': 'Main RNG only; persistent worker RNG/prefetch not restored; no bit-identical continuation claim',
              'runtime_required': ['RAPTOR64_STAGE_ROOT', 'RAPTOR64_MANIFEST_SHA256', 'RAPTOR64_CONTENT_SHA256', 'RAPTOR64_STAGE_RECEIPT_SHA256',
                                   'RAPTOR64_INITIALIZER_DESCRIPTOR', 'RAPTOR64_INITIALIZER_DESCRIPTOR_SHA256'],
              'optional_actual_platform_deadline': 'RAPTOR64_ACTUAL_PLATFORM_DEADLINE_UTC_SECONDS; no assistant budget'}
    guard.validate_policy(policy)
    out.mkdir(parents=True)
    for name, value in sources.items():
        p = out / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(value)
    for name in ('train_knee_original.py', 'folds_v0.json', 'excluded_ids.json'):
        shutil.copyfile(old / name, out / name)
    (out / 'train_knee_patched.py').write_text(trainer); (out / 'tracka_entry.py').write_text(runner)
    patch = ''.join(difflib.unified_diff((old / 'train_knee_original.py').read_text().splitlines(True), trainer.splitlines(True),
                                         fromfile='train_knee.py', tofile='train_knee.py'))
    (out / 'train_knee.patch').write_text(patch); (out / 'raptor64-recipe.json').write_text(json.dumps(policy, indent=2) + '\n')
    manifest = {'original_sha256': smoke.SOURCE_SHA, 'historical_patched_sha256': TRAIN_SHA, 'historical_entry_sha256': ENTRY_SHA,
                'patched_sha256': smoke.file_sha(out / 'train_knee_patched.py'), 'entry_sha256': smoke.file_sha(out / 'tracka_entry.py'),
                'patch_sha256': smoke.file_sha(out / 'train_knee.patch'), 'folds_sha256': smoke.file_sha(out / 'folds_v0.json'),
                'excluded_ids_sha256': smoke.file_sha(out / 'excluded_ids.json'), 'recipe_sha256': smoke.file_sha(out / 'raptor64-recipe.json'),
                'sources': {name: smoke.digest(value.encode()) for name, value in sources.items()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    status = subprocess.run(['git', 'status', '--short'], cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
    receipt = {'build_only': True, 'arm': arm, 'builder_revision': revision, 'working_source_status': status,
               'manifest_sha256': smoke.file_sha(out / 'manifest.json'), 'trainer_sha256': manifest['patched_sha256'],
               'entry_sha256': manifest['entry_sha256'], 'scientific_ast_equal': scientific_ast(train) == scientific_ast(trainer), 'policy': policy,
               'limits': ['No GPU/model training executed', 'Cache staging/initializer/persistence provisioning are separate VM steps']}
    (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('destination', type=Path)
    parser.add_argument('--evidence-root', type=Path, required=True); parser.add_argument('--arm', choices=sorted(guard.ARMS), required=True)
    parser.add_argument('--workers', type=int, default=2); parser.add_argument('--worker-choice-name')
    parser.add_argument('--teacher-sha256', help='distil arm only: pinned teacher npz SHA256')
    parser.add_argument('--swa-epochs', type=int, nargs=3, help='full-data arm only: pre-registered SWA epochs (e.g. dr-orig fold-0 selected epochs)')
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
