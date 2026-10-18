"""B0 rehearsal-only faults and guarded Raptor partial inputs. Never in a pure entry."""
import json
import os
from pathlib import Path
from unittest.mock import patch

# Each declared case must exercise its intended source diagnostic.
CASE_COUNTERS={'bad_header':'header_parse_failures','bad_pixel':'pixel_failures','black_frame':'masked_selected_frames','empty_series':'empty_series',
               'missing_series':'empty_series','missing_iop':'geometry_defaults','missing_ipp':'geometry_defaults','missing_spacing':'spacing_defaults'}


def _dataset(index):
    import numpy as np
    import pydicom
    meta=pydicom.Dataset();meta.TransferSyntaxUID=pydicom.uid.ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=pydicom.uid.MRImageStorage;meta.MediaStorageSOPInstanceUID='1.2.826.0.1.3680043.8.498.'+str(index+1)
    ds=pydicom.FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
    ds.SOPClassUID=meta.MediaStorageSOPClassUID;ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    ds.Rows=32;ds.Columns=40;ds.SamplesPerPixel=1;ds.BitsAllocated=16;ds.BitsStored=12;ds.HighBit=11;ds.PixelRepresentation=0
    ds.PhotometricInterpretation='MONOCHROME2'
    ds.ImageOrientationPatient=[1,0,0,0,1,0];ds.ImagePositionPatient=[0,0,index];ds.PixelSpacing=[.5,.5];ds.InstanceNumber=index
    ds.PixelData=((np.arange(1280,dtype=np.uint16).reshape(32,40)+index*11)%2000).tobytes()
    return ds


def guarded_raptor_partial(public):
    """Ten declared partial-input cases through the installed guarded Raptor reader.

    Real synthetic DICOM files exercise actual pydicom errors. Guarded production,
    pydicom reference and the unguarded public catch policy must give identical
    volumes and masks; typed failure and fatal cases have separate regression tests.
    """
    import tempfile
    import numpy as np
    import pydicom
    from knee import raptor_input as RI
    from knee.public_s6_qualification import require
    from knee.public_s6_recovery_b0 import GUARDED_PARTIAL_CASES
    guard=public._raptor_propagate_fatal;require(guard is not RI.propagate_fatal,'GUARDED_RAPTOR_READER_REQUIRED')
    checks={};diagnostics={}
    with tempfile.TemporaryDirectory(prefix='b0-raptor-guarded-') as directory:
        root=Path(directory);sag=root/'synthetic'/'sag';cor=root/'synthetic'/'cor';sag.mkdir(parents=True);cor.mkdir()
        for i in range(8):pydicom.dcmwrite(cor/f'{i}.dcm',_dataset(i),enforce_file_format=True)
        rows=[dict(SeriesInstanceUID='sag',Anatomical_Plane='Sagittal',Fluid_Sensitive=1),dict(SeriesInstanceUID='cor',Anatomical_Plane='Coronal',Fluid_Sensitive=1)]
        for case in sorted(GUARDED_PARTIAL_CASES):
            for p in sag.glob('*.dcm'):p.unlink()
            for i in range(1 if case=='short_series' else 8):
                ds=_dataset(i)
                field={'missing_iop':'ImageOrientationPatient','missing_ipp':'ImagePositionPatient','missing_spacing':'PixelSpacing'}.get(case)
                if field:delattr(ds,field)
                if i==0 and case=='black_frame':ds.PixelData=bytes(2560)
                if i==0 and case=='bad_pixel':ds.PixelData=bytes(2)
                pydicom.dcmwrite(sag/f'{i}.dcm',ds,enforce_file_format=True)
            if case=='bad_header':(sag/'0.dcm').write_bytes(b'not a DICOM')
            if case=='empty_series':
                for p in sag.glob('*.dcm'):p.unlink()
            chosen=[dict(x) for x in rows]
            if case=='missing_series':chosen[0]['SeriesInstanceUID']='missing'
            records={'synthetic':chosen}
            guarded=RI.prepare_study(public,'synthetic',records,root)
            reference=RI.prepare_study(public,'synthetic',records,root,cache_pixels=False,reference=True)
            public._raptor_propagate_fatal=RI.propagate_fatal
            try:source=RI.prepare_study(public,'synthetic',records,root)
            finally:public._raptor_propagate_fatal=guard
            require(all(np.array_equal(a,b) and np.array_equal(a,c) for a,b,c in zip(guarded[:2],reference[:2],source[:2])) and guarded[2]==source[2],'GUARDED_RAPTOR_SOURCE_PARITY')
            if case in CASE_COUNTERS:require(guarded[2][CASE_COUNTERS[case]]>0,'GUARDED_CASE_NOT_EXERCISED_'+case)
            require(guarded[2]['resource_errors']==guarded[2]['invalid_pixels']==0,'GUARDED_CASE_FATAL_COUNTER')
            checks[case]='PASS';diagnostics[case]={k:guarded[2][k] for k in ('header_parse_failures','pixel_failures','masked_selected_frames','empty_series','valid_slices')}
    require(checks==dict.fromkeys(GUARDED_PARTIAL_CASES,'PASS'),'GUARDED_RAPTOR_TEN')
    return {'status':'PASS','cases':checks,'case_count':len(checks),'diagnostics':diagnostics,'guarded':True,'policy':'classified-data-local-v1','uses_raw_data':False,'uses_labels':False}


def injection_hook(arm,cfg,context):
    """Raptor all-empty injection for the old arm; the unchanged default S6 hook."""
    import numpy as np
    if arm!='old':
        from knee.public_s6_recovery_rehearsal import injection_hook as default_hook
        return default_hook(arm,cfg,context)
    import knee.raptor_input as RI
    from knee.public_s6_recovery_runtime import DataLocalFailure
    infer=context['infer'];sid=cfg['old_sentinels'][0];genuine=context['sentinels'][0]['probabilities']
    original=RI.validate_production_volume
    def all_empty(volume,mask,plan,**kw):
        return original(np.zeros_like(volume),np.zeros_like(mask),plan,**kw)
    caught=False
    with patch.object(RI,'validate_production_volume',side_effect=all_empty):
        try:infer(sid)
        except DataLocalFailure as e:caught=str(e)=='old_no_usable_image'
    if not caught:raise ValueError('ACTUAL_OLD_ALL_EMPTY_INJECTION')
    context['state_check']()
    p,fp,row=infer(sid)
    if np.max(np.abs(p-np.asarray(genuine)))>1e-6:raise ValueError('OLD_NEXT_HEALTHY_PARITY')
    context['state_check']()
    return {'status':'PASS','cases':['actual_raptor_all_empty','next_healthy_same_study'],'failure_status':'data_local_failure','injected_rows_genuine':False,
            'next_healthy_max_abs':float(np.max(np.abs(p-np.asarray(genuine)))),'cleanup':{'scratch_removed':True,'side_override_removed':True},'state_healthy':True}


if __name__=='__main__':
    import argparse
    from knee import public_s6_recovery_b0 as entry
    if Path(__file__).resolve()!=(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_s6_recovery_b0_rehearsal.py').resolve():
        raise ValueError('EXACT_REHEARSAL_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.worker).read_text())
    if cfg['mode']!='rehearsal':raise ValueError('INJECTIONS_REHEARSAL_ONLY')
    entry.rehearsal_hook=injection_hook
    (entry.old_worker if cfg['arm']=='old' else entry.s6_worker)(a.worker)
