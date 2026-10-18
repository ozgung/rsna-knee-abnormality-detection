"""Build-only immutable own224-d96 package; historical scientific AST is checked."""
import argparse
import ast
import difflib
import json
from pathlib import Path
import re
import shutil
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from knee import source96_training as guard,public_training_smoke as smoke

TRAIN_SHA='eb3d34ffc6aa6f7c500f8ef2039dba0f313b42e371e075e58d4b7a47dde0bd3c'
ENTRY_SHA='c1abaf134622f4e4ea9ffd9330991fd34c0d1185d1ccabae28e3ec55095aa519'
FILES=('knee/__init__.py','knee/source96_training.py','knee/source96_loader.py','knee/source96_cache.py','knee/source_span_pilot.py',
       'knee/corpus44_parity.py','knee/public_training_smoke.py','knee/public_rich_cache.py','knee/reference.py','knee/fresh_reader_setup.py','knee/raptor_input.py')


def replace_once(text,before,after):
    if text.count(before)!=1:raise ValueError('Historical patch anchor differs')
    return text.replace(before,after,1)


def function_source(text,name):
    node=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return ast.get_source_segment(text,node)


def scientific_ast(text):
    tree=ast.parse(text);result={}
    for name in ('StudyWindows','RaptorClassifier'):
        result[name]=ast.dump(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    for name in ('head_params','bb_params','opt','steps','sched','lossf'):
        node=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        result[name]=ast.dump(node)
    result['training_updates']=ast.dump(next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.iter,ast.Name) and n.iter.id=='tl'))
    result['top3_selection']=ast.dump(next(n for n in ast.walk(main) if isinstance(n,ast.If) and ast.unparse(n.test)=='len(topk) < TOPK or sel > min((t[\'sel\'] for t in topk))'))
    result['swa']=ast.dump(next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=='len(topk) > 1'))
    return result


def adapt_trainer(original):
    text=original
    doc=next(n for n in ast.parse(original).body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))
    text=replace_once(text,ast.get_source_segment(original,doc),'"""Own224-d96wide recipe-level experiment. Verified private96 cache; no gold scoring.\nHistorical sampler/model/optimizer/selection/SWA AST retained. Generic initializer identity\nfor the historical run remains unresolved. See source96-recipe.json for frozen policy."""')
    for name,code in [('_open_corpus','def _open_corpus():\n    return source96.staged_corpus()[:2]'),('_open_ids','def _open_ids():\n    return source96.staged_corpus()[2]')]:
        text=replace_once(text,function_source(text,name),code)
    text=replace_once(text,'    a = ap.parse_args()','    a = ap.parse_args()\n    from knee import source96_training as source96\n    source96.require_recipe(a, __file__)')
    # Hooks above use module globals, also needed by export-only imports.
    text='from knee import source96_training as source96\n'+text
    text=replace_once(text,'        dev = "cuda" if torch.cuda.is_available() else "cpu"','        dev = source96.strict_cuda(torch)')
    text=replace_once(text,'    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)',
        '    source96.verify_membership(train_ids, oof_ids, gold_ids, a.fold, a.fold_file, a.excluded_ids, __file__)\n    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)')
    old='    use_timm = a.ckpt in ("timm", "pretrained")\n    bb = build_backbone(a.arch, pretrained=use_timm)\n    src = load_raptor(bb, a.ckpt)\n    if use_timm: print(f"[raptor] timm-pretrained backbone: {a.arch}", flush=True)'
    text=replace_once(text,old,'    bb = build_backbone(a.arch, pretrained=False)\n    INITIALIZER = source96.generic_backbone(bb)\n    src = "pinned-generic-imagenet:" + INITIALIZER["file_sha256"]')
    text=replace_once(text,'            print(f"[raptor] grad_ckpt unsupported for {a.arch}: {e}", flush=True)',
        '            raise RuntimeError("Required gradient checkpointing unavailable") from None')
    text=replace_once(text,'    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]',
        '    INITIAL = source96.initial_record(model, INITIALIZER, __file__)\n    IDENT = source96.run_identity(a, __file__, _lab, INITIAL)\n    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]')
    text=replace_once(text,'        R_ = torch.load(a.resume, map_location="cpu", weights_only=False)',
        '        if smoke_file_sha(a.resume) != os.environ["SOURCE96_RESUME_SHA256"]: raise ValueError("Pinned private resume bytes differ")\n        R_ = torch.load(a.resume, map_location="cpu", weights_only=False)\n        source96.validate_resume(R_, IDENT)')
    text=replace_once(text,'        del R_','        source96.restore_rng(R_["rng"])\n        del R_')
    start=text.index('        if a.time_budget_h > 0 and (time.time() - t0) > a.time_budget_h * 3600 and ep < a.epochs - 1:')
    end=text.index('    if best_state is None: raise SystemExit("no epoch was selected")',start)
    text=text[:start]+'''        rp = os.path.join(HERE, f"raptor_resume_{a.tag}.pt")
        source96.save_resume(rp, {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None, "best": best,
            "best_state": best_state, "best_P": best_P, "hist": hist, "topk": topk,
            "epoch": ep, "elapsed": time.time() - t0}, IDENT)
        print(f"[resume] recoverable completed epoch {ep}; worker RNG continuation limitation recorded", flush=True)
'''+text[end:]
    text='from knee.public_training_smoke import file_sha as smoke_file_sha\n'+text
    if scientific_ast(original)!=scientific_ast(text):raise ValueError('Historical scientific AST changed')
    compile(text,'source96-trainer','exec');return text


def adapt_entry(original,workers,fold=0):
    text=original
    defaults={'FOLD_A':str(fold),'FOLD_B':'','TIME_BUDGET_H':'0','RES':'224','ARCH':guard.ARCH,'BS':'4','WORKERS':str(workers),
              'WORKDIR':'/content/trackA96','ASSET_DIR':'/content/trackA96_pkg','RAPTOR_INPUT':'/content/input','GRAD_CKPT':'1','WIN_CHUNK':'8','ACCUM':'2'}
    tree=ast.parse(text);assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARAM_DEFAULTS' for t in n.targets))
    old=ast.get_source_segment(text,assign);values={kw.arg:ast.literal_eval(kw.value) for kw in assign.value.keywords};values.update(defaults);values['RESUME']='0'
    text=replace_once(text,old,'PARAM_DEFAULTS = '+repr(values))
    text='from knee import source96_training as source96\n'+text
    text=replace_once(text,'    return c\n','    source96.entry_guard(c, __file__)\n    return c\n')
    text=replace_once(text,'    shutil.copy(assets / "train_knee.patch", work / "train_knee.patch")',
        '    shutil.copy(assets / "train_knee.patch", work / "train_knee.patch")\n    shutil.copy(assets / "source96-recipe.json", work / "source96-recipe.json")\n    shutil.copy(assets / "manifest.json", work / "manifest.json")\n    shutil.copytree(assets / "knee", work / "knee", dirs_exist_ok=True)')
    text=replace_once(text,'    return cmd',
        '    if c["RESUME"].lower() in ("1", "true", "yes"):\n        rp = p["work"] / f"raptor_resume_fold{fold}.pt"\n        if not rp.is_file(): raise ValueError("Resume requested but state missing")\n        cmd += ["--resume", str(rp)]\n    return cmd')
    start=text.index('    if c["CKPT"] == "timm":');end=text.index('    def env_for(i):',start)
    text=text[:start]+'''    source96.strict_cuda(torch)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for k in folds:
        if c["RESUME"].lower() not in ("1", "true", "yes") and any(work.glob(f"raptor_*fold{k}*")):
            raise ValueError("Fresh training requires unused work directory; never delete recovery artifacts")
'''+text[end:]
    start=text.index('    for k in folds:\n        rp = work / f"raptor_resume_fold{k}.pt"');end=text.index('    tot = sum(',start)
    text=text[:start]+'    # Keep every completed-epoch resume until external persistence has verified it.\n'+text[end:]
    text=replace_once(text,'"time_budget_note": "10.0 h = 12 h Kaggle session minus checkpoint/export margin"',
        '"time_budget_note": "No assistant cap; checkpoint against caller-supplied actual platform deadline only"')
    text=replace_once(text,'def export_main(k):','def export_main(k, secondary=False):')
    text=replace_once(text,'    k_eval = int(c["SMOKE_K_EVAL"] if smoke else c["K_EVAL"])',
        '    source96.export_guard(work, k, ckpt_path, kind, sel_epochs, partial, secondary)\n    suffix = "_k94" if secondary else ""\n    k_eval = 94 if secondary else 16')
    text=replace_once(text,'    dev = "cuda" if torch.cuda.is_available() else "cpu"','    dev = source96.strict_cuda(torch)')
    text=replace_once(text,'    truth = pd.read_parquet(tk._find("train_labels_v1.parquet"))',
        '    source96.verify_export_inputs(tk._find(c["LABELS_FILE"]), tk._find("train_labels_v1.parquet"), work / "folds_v0.json", work / "excluded_ids.json", val_ids, k, __file__)\n    truth = pd.read_parquet(tk._find("train_labels_v1.parquet"))')
    text=replace_once(text,'    pv, tv = predict(val_ids)',
        '    if gold_ids or len(val_ids) != source96.SPLITS[k]["development"]: raise ValueError("Export must use pinned nongold development only")\n    checkpoint_before = sha256(ckpt_path)\n    pv, tv = predict(val_ids)\n    if sha256(ckpt_path) != checkpoint_before: raise ValueError("Frozen checkpoint mutated during export")')
    text=text.replace('f"oof_fold{k}.csv"','f"oof_fold{k}{suffix}.csv"').replace('f"run_fold{k}.json"','f"run_fold{k}{suffix}.json"')
    text=replace_once(text,'"caveat": "val AUC is selection-biased (the same fold chose the epochs); gold-58 is the only clean number and is noisy (+-0.03).",',
        '"caveat": "Exposed report-selected development; no gold scored; historical initializer unresolved; worker RNG resume not bit-identical.",\n        "source96_identity": json.load(open(work / "initial-state-record.json")),\n        "source96_experiment_split": source96.runtime_split(k, work / "folds_v0.json", work / "excluded_ids.json", __file__)[0],\n        "cache_reference_partition_fold": 0,\n        "source96_cache_content_sha256": os.environ["SOURCE96_CONTENT_SHA256"],\n        "comparison_endpoint": "fixed checkpoint K94 secondary" if secondary else "matched K16 primary",')
    text=replace_once(text,'    else:\n        main()','    elif len(sys.argv) >= 3 and sys.argv[1] == "secondary":\n        export_main(int(sys.argv[2]), secondary=True)\n    else:\n        main()')
    compile(text,'source96-entry','exec');return text


def derive_split(evidence,fold):
    import numpy as np
    from knee import source96_cache as cache
    ids=[]
    for sub,prefix in [('corpus','all'),('corpus-ext','extra')]:
        p=evidence/f'outputs/analysis/public-train-recipe-20261009/{sub}/{prefix}_ids.npy'
        smoke.require_file(p,smoke.PINS[prefix+'_ids.npy']);ids.extend(np.load(p,allow_pickle=False).astype(str).tolist())
    ip=evidence/'outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json';smoke.require_file(ip,smoke.MANIFEST_SHA)
    fp=evidence/'outputs/kaggle/trackA-coatnet224-fold01-20261009/folds_v0.json';smoke.require_file(fp,guard.FOLDS_SHA);folds=json.loads(fp.read_bytes())['folds']
    official=evidence/'data/raw/train.csv';smoke.require_file(official,'8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33')
    eligible,_=cache.membership(ids,official,folds,json.loads(ip.read_bytes()))
    ep=evidence/'outputs/kaggle/trackA-coatnet224-fold01-20261009/excluded_ids.json';smoke.require_file(ep,guard.EXCLUSIONS_SHA)
    return guard.experiment_split(eligible,folds,json.loads(ep.read_bytes())['quarantined_study_ids'],fold)


def build(destination,*,evidence_root,workers=2,worker_choice_name=None,fold=0):
    out=Path(destination);evidence=Path(evidence_root)
    if type(fold) is not int or fold not in (0,1):raise ValueError('Only reviewed folds0/1 supported')
    if out.exists():raise ValueError('Fresh immutable package directory required')
    if type(workers) is not int or not 0<=workers<=64 or (workers!=2 and not worker_choice_name):raise ValueError('Named frozen worker choice required')
    if worker_choice_name and not re.fullmatch('[a-z0-9][a-z0-9-]{2,63}',worker_choice_name):raise ValueError('Invalid worker choice name')
    old=evidence/'outputs/kaggle/trackA-coatnet224-fold01-20261009'
    train=(old/'train_knee_patched.py').read_text();entry=(old/'tracka_entry.py').read_text()
    if smoke.digest(train.encode())!=TRAIN_SHA or smoke.digest(entry.encode())!=ENTRY_SHA:raise ValueError('Historical source identity differs')
    trainer=adapt_trainer(train);runner=adapt_entry(entry,workers,fold)
    sources={name:(ROOT/name).read_text() for name in FILES}
    for name,value in sources.items():compile(value,name,'exec')
    policy={'version':1,'recipe':'own224-d96wide-tenepoch-report-top3-v1','workers':workers,'worker_choice_name':worker_choice_name,
            'worker_rng_resume':'Main RNG only; persistent worker RNG/prefetch not restored; no bit-identical continuation claim',
            'historical_comparator':'recipe-level; exact original generic initializer bytes unresolved','frozen':guard.FROZEN,
            'runtime_required':['SOURCE96_STAGE_ROOT','SOURCE96_MANIFEST_SHA256','SOURCE96_CONTENT_SHA256','SOURCE96_STAGE_RECEIPT_SHA256',
                                'SOURCE96_INITIALIZER_DESCRIPTOR','SOURCE96_INITIALIZER_DESCRIPTOR_SHA256'],
            'optional_actual_platform_deadline':'SOURCE96_ACTUAL_PLATFORM_DEADLINE_UTC_SECONDS; no assistant budget; last completed epoch recoverable',
            'secondary':'K94 export of unchanged primaryK16 SWA only; no epoch selection'}
    if fold==1:
        split,_,_=derive_split(evidence,fold)
        policy.update(version=2,recipe='own224-d96wide-tenepoch-report-top3-fold1-v1',frozen={**guard.FROZEN,'fold':1},experiment_split=split,experiment_split_sha256=smoke.digest(smoke.canonical(split)),cache_reference_partition_fold=0)
    out.mkdir(parents=True)
    for name,value in sources.items():
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    for name in ('train_knee_original.py','folds_v0.json','excluded_ids.json'):shutil.copyfile(old/name,out/name)
    (out/'train_knee_patched.py').write_text(trainer);(out/'tracka_entry.py').write_text(runner)
    # Full original-to-generated patch remains reproducible and reviewable.
    patch=''.join(difflib.unified_diff((old/'train_knee_original.py').read_text().splitlines(True),trainer.splitlines(True),fromfile='train_knee.py',tofile='train_knee.py'))
    (out/'train_knee.patch').write_text(patch);cache_policy=(json.dumps(policy,indent=2)+'\n');(out/'source96-recipe.json').write_text(cache_policy)
    manifest={'original_sha256':smoke.SOURCE_SHA,'historical_patched_sha256':TRAIN_SHA,'historical_entry_sha256':ENTRY_SHA,
              'patched_sha256':smoke.file_sha(out/'train_knee_patched.py'),'entry_sha256':smoke.file_sha(out/'tracka_entry.py'),
              'patch_sha256':smoke.file_sha(out/'train_knee.patch'),'folds_sha256':smoke.file_sha(out/'folds_v0.json'),
              'excluded_ids_sha256':smoke.file_sha(out/'excluded_ids.json'),'recipe_sha256':smoke.file_sha(out/'source96-recipe.json'),
              'sources':{name:smoke.digest(value.encode()) for name,value in sources.items()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    status=subprocess.run(['git','status','--short'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.splitlines()
    receipt={'build_only':True,'builder_revision':revision,'working_source_status':status,'manifest_sha256':smoke.file_sha(out/'manifest.json'),'trainer_sha256':manifest['patched_sha256'],'entry_sha256':manifest['entry_sha256'],
             'scientific_ast_equal':scientific_ast(train)==scientific_ast(trainer),'policy':policy,
             'limits':['No GPU/model training executed','Staging/orchestrator/persistence/deadline provisioning remain separate',
                       'Generic initializer descriptor/normalized artifact not supplied; actual initialization pending','Existing Colab scripts must not launch unchanged']}
    (out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path);parser.add_argument('--evidence-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--worker-choice-name');parser.add_argument('--fold',type=int,choices=(0,1),default=0)
    print(json.dumps(build(**vars(parser.parse_args())),indent=2))
