"""Build-only frozen Versia common-visible qualification or qualified full entry."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from build_mirror_submission import archived_build, SCIENTIFIC
from build_versia_rehearsal import build_versia, BUNDLE
from knee import mirror_input, versia
from knee.reference import canonical_bytes, sha256, WEIGHT_DATASET
from knee.reference_blend import CONVNEXT_DATASET, DOCKER_IMAGE
from knee.versia_fusion import VERSION, MAX_SECONDS, POLICY, TARGETS, WEIGHTS, compare_component, require_visible

NEW=('knee/mirror_submission.py','knee/versia_fusion.py','scripts/versia_fusion_entry.py')


def archived_versia(directory,spec_dir):
    """Literal extraction plus exact regeneration by the unchanged pinned builder."""
    directory=Path(directory)
    raw={n:(directory/n).read_bytes() for n in ('notebook.ipynb','versia-contract.json','build-receipt.json','kernel-metadata.json')}
    nb,c,b,meta=(json.loads(raw[n]) for n in raw)
    cells=[x for x in nb['cells'] if x['cell_type']=='code']
    if len(cells)!=1:raise ValueError('One archived Versia code cell required')
    code=''.join(cells[0]['source']);values={}
    for node in ast.parse(code).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('sources','VERSIA_SOURCES','VERSIA_CONTRACT'):
                if name in values:raise ValueError('Duplicate archived Versia assignment')
                values[name]=ast.literal_eval(node.value)
    own=values['sources'];digest=sha256(canonical_bytes(own));revision=b.get('source_revision','')
    if (not re.fullmatch('[0-9a-f]{40}',revision) or set(own)!=set(BUNDLE) or
            any(own[n]!=(ROOT/n).read_text() for n in BUNDLE) or c!=values['VERSIA_CONTRACT'] or
            b.get('source_bundle_sha256')!=digest or b.get('contract_sha256')!=sha256(canonical_bytes(c)) or
            b.get('notebook_sha256')!=sha256(raw['notebook.ipynb']) or b.get('notebook_bytes')!=len(raw['notebook.ipynb'])):
        raise ValueError('Archived Versia source/build binding differs')
    # Regeneration executes only our builder, never the public code/notebook. The
    # sole allowed revision difference is the provenance literal, not source bytes.
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'verify';made=build_versia(out,meta['id'].split('/')[-1],spec_dir=spec_dir)
        expected=json.loads((out/'notebook.ipynb').read_text())
        expected['cells'][-1]['source']=''.join(expected['cells'][-1]['source']).replace(
            f"os.environ['KNEE_CODE_REVISION']={made['source_revision']!r}",
            f"os.environ['KNEE_CODE_REVISION']={revision!r}").splitlines(True)
        if (nb!=expected or c!=json.loads((out/'versia-contract.json').read_text()) or
                meta!=json.loads((out/'kernel-metadata.json').read_text())):
            raise ValueError('Archived Versia executable/metadata differs from pinned reconstruction')
    return own,values['VERSIA_SOURCES'],c,digest,revision,{n:sha256(v) for n,v in raw.items()}


def build_fusion(destination,slug,*,mirror_build,mirror_report,mirror_csv,public_source,
                 versia_build,versia_report,versia_csv,spec_dir,submission=False,visible_results=None):
    destination=Path(destination)
    if destination.exists():raise ValueError('Fresh immutable build destination required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug):raise ValueError('Invalid notebook slug')
    mf,public,decoder,mc,md,mr,ma=archived_build(mirror_build,public_source,ROOT)
    vf,vs,vc,vd,vr,va=archived_versia(versia_build,spec_dir)
    _,spec=versia.audit_sources(spec_dir)
    if (spec['target_order']!=TARGETS or [spec['convnext_weights'][t] for t in TARGETS]!=WEIGHTS or
            spec['mirror_checkpoint_sha256']!=mc['source']['checkpoint_sha256']):
        raise ValueError('Frozen pre-score fusion specification differs')
    reports={arm:json.loads(Path(p).read_text()) for arm,p in [('mirror',mirror_report),('convnext',versia_report)]}
    mirror_input.require_rehearsal(mc,reports['mirror'],bundle_sha256=md)
    versia.require_rehearsal(vc,reports['convnext'],bundle_sha256=vd)
    for arm,csv,revision in [('mirror',mirror_csv,mr),('convnext',versia_csv,vr)]:
        if reports[arm].get('source_revision')!=revision or reports[arm].get('visible_csv_sha256')!=sha256(Path(csv).read_bytes()):
            raise ValueError('Authentic component receipt/CSV/revision binding differs')
        compare_component(csv,csv,arm=arm)
    # Shared scientific sources must agree with each other and the existing module
    # pins. The historical mirror all-knee bundle is never recomputed from new files.
    own={n:mf[n] for n in SCIENTIFIC}
    for n,v in vf.items():
        if n.startswith('scripts/'):continue
        if n in own and own[n]!=v:raise ValueError('Qualified components disagree on shared source')
        own[n]=v
    own.update({n:(ROOT/n).read_text() for n in NEW})
    packages=reports['convnext']['environment']['packages']
    if any(reports['mirror']['environment']['packages'].get(p)!=packages.get(p) for p in ('torch','timm','numpy','pydicom','opencv-python')):
        raise ValueError('Qualified components require different common environments')
    if mc['codec_hashes']!=vc['codecs']:raise ValueError('Component codecs differ')
    q={'version':VERSION,'policy':POLICY,'spec_sha256':versia.SPEC_SHA256,
       'source_url':spec['source_url'],'frozen_at_utc':spec['frozen_at_utc'],'notebook_sha256':spec['notebook_sha256'],
       'fusion_cell_sha256':spec['source_cell_sha256']['published-fusion.py'],
       'mirror':{'manifest':mc['source'],'input_policy':mc['input_policy'],'archive_sha256':ma,
          'bundle_sha256':md,'source_revision':mr,'report_sha256':sha256(Path(mirror_report).read_bytes()),
          'csv_sha256':sha256(Path(mirror_csv).read_bytes()),'sampling':reports['mirror']['sampling'],
          'strict_comparison_counts':reports['mirror']['strict_comparison_counts']},
       'convnext':{'contract':vc,'archive_sha256':va,'bundle_sha256':vd,'source_revision':vr,
          'report_sha256':sha256(Path(versia_report).read_bytes()),'csv_sha256':sha256(Path(versia_csv).read_bytes()),
          'sampling':reports['convnext']['sampling'],'strict_coverage':reports['convnext']['strict_coverage']},
       'runtime_sources':{n:sha256(v.encode()) for n,v in own.items()},'builder_sha256':sha256(Path(__file__).read_bytes()),
       'qualification_helpers':{n:sha256((ROOT/n).read_bytes()) for n in ('scripts/build_mirror_submission.py','scripts/build_versia_rehearsal.py')},
       'codecs':vc['codecs'],'environment':{'python':[3,12],'packages':packages,'docker_image':DOCKER_IMAGE}}
    fingerprint=sha256(canonical_bytes(q));evidence=None
    if submission:
        if visible_results is None:raise ValueError('Full fusion entry requires new common visible results')
        evidence,paths=require_visible(q,visible_results)
        evidence['component_cross_run_max_abs_error']={arm:compare_component(ref,paths[arm],arm=arm)
            for arm,ref in [('mirror',mirror_csv),('convnext',versia_csv)]}
        evidence['visible_rank_and_tie_parity']=True
    elif visible_results is not None:raise ValueError('Common visible evidence is for full qualification only')
    c={'version':VERSION,'qualification':q,'qualification_sha256':fingerprint,'visible':not submission,
       'visible_evidence':evidence,'max_seconds':MAX_SECONDS,'private':True,'internet':False}
    bundle=sha256(canonical_bytes(own))
    code=("import time,signal\nFUSION_STARTED=time.monotonic()\n"
        "def _fusion_expired(signum,frame):\n    raise TimeoutError('Actual nine-hour fusion runtime exhausted')\n"
        "signal.signal(signal.SIGALRM,_fusion_expired)\nsignal.setitimer(signal.ITIMER_REAL,32400)\n"
        "import os,sys\nfrom pathlib import Path\nproject=Path('/kaggle/working/project')\n"
        f'sources={own!r}\n'
        "for name,content in sources.items():\n    p=project/name\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_text(content)\n"
        "sys.path.insert(0,str(project))\n"
        f'FUSION_CONTRACT={c!r}\nMIRROR_PUBLIC={public!r}\nMIRROR_DECODER={decoder!r}\nVERSIA_SOURCES={vs!r}\n'
        "exec(compile(sources['scripts/versia_fusion_entry.py'],'<versia-fusion-entry>','exec'))\n")
    compile(code,'<fusion>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
         'language_info':{'name':'python','version':'3.12'}},'cells':[
       {'cell_type':'markdown','id':'attribution','metadata':{},'source':[
        '# Frozen published Versia fusion\n',f"[Versia source]({spec['source_url']}) by evgendvorkin; specification frozen {spec['frozen_at_utc']}.\n",
        'Mirror model/source by nartaa and attributed original authors; [weights](https://www.kaggle.com/datasets/'+WEIGHT_DATASET+') have publisher competition-use permission (`other` license), not explicit host/prize clearance. ConvNeXt source/weights by goodpjw2008 (Apache-2.0).\n',
        'Unchanged native320/K94/two-view mirror and ConvNeXt16/slot-copy/three-fold/batch4/workers4. Separate sequential processes; both must succeed. Image-only, private, offline; no labels or reference prediction fixtures attached.\n',
        'Exact double argsort and average-tie rank fusion with frozen target weights; no score-based tuning. Actual9-hour ceiling covers setup and both components.\n',
        f'Qualification `{fingerprint}`.\n']},
       {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    raw=(json.dumps(nb,indent=2)+'\n').encode()
    if len(raw)>900000 or len(code.encode())>900000:raise ValueError('Fusion source exceeds notebook-size limit')
    meta={'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
          'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
          'dataset_sources':[WEIGHT_DATASET,CONVNEXT_DATASET],'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE}
    receipt={'source_bundle_sha256':bundle,'qualification_sha256':fingerprint,'contract_sha256':sha256(canonical_bytes(c)),
       'notebook_sha256':sha256(raw),'notebook_bytes':len(raw),'build_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    destination.mkdir(parents=True,exist_ok=False);(destination/'notebook.ipynb').write_bytes(raw)
    for n,v in [('kernel-metadata.json',meta),('fusion-contract.json',c),('build-receipt.json',receipt)]:
        (destination/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('slug')
    for n in ('mirror-build','mirror-report','mirror-csv','public-source','versia-build','versia-report','versia-csv','spec-dir'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--submission',action='store_true');p.add_argument('--visible-results',type=Path)
    print(json.dumps(build_fusion(**vars(p.parse_args())),indent=2))
