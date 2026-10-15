"""Fresh synthetic K62 CPU check using the exact existing dr-full qualifier."""
import argparse
import json
from pathlib import Path
import sys
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import qualify_b3_drfull_cpu as Q
from knee import public_s6_recovery_b0prime as P


def drfull_manifest(source_text):
    from knee.reference import audit_source
    module,decoder,manifest=audit_source(source_text,profile='raptor',checkpoint_sha256=P.DREAD_CHECKPOINT_SHA256,raptor_windows=P.RAPTOR_WINDOWS)
    manifest.update(checkpoint_file=P.CHECKPOINT_FILE,weight_dataset=P.DATASET,
                    attribution='Own dr-full SWA9-11, Dread v4 init; B0 K62 single-checkpoint swap')
    P.raptor_profile_gate(None,manifest)
    return module,decoder,manifest


def qualify(source_path,checkpoint_path,dread_checkpoint=None):
    namespace={**vars(Q),'DRFULL_WINDOWS':P.RAPTOR_WINDOWS,'drfull_manifest':drfull_manifest}
    function=types.FunctionType(Q.qualify.__code__,namespace,Q.qualify.__name__,Q.qualify.__defaults__)
    return function(source_path,checkpoint_path,dread_checkpoint,forward_windows=P.RAPTOR_WINDOWS)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','dread-checkpoint','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise ValueError('FRESH_RECEIPT')
    r=qualify(a.source,a.checkpoint,a.dread_checkpoint)
    out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:r[k] for k in ('state_sha256','constructor_grid_sha256','windows_shape','forward_windows','cpu_repeat_max_abs','forward_seconds','dread_pin_method_check')}))
