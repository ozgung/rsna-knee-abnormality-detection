import copy
import unittest
from unittest.mock import patch

from knee import coverage_cache as cc
from knee.fresh_reader import fingerprint


def stamp(a):
    return a | {'fingerprint':fingerprint(a)}


def fixtures():
    rows=[{'uid':str(i),'split':'train' if i<2000 else 'weak_validation','group':str(i),'labels':[.5,.9]}
          for i in range(2200)]
    original=stamp({'records':rows[:512]+rows[2000:],'targets':['a','b'],'files':{'schema':'fixed'}})
    identity=stamp({'records':[{'uid':r['uid'],'component':r['group'],'original_split':r['split'],
        'future_role':'training_candidate' if r['split']=='train' else 'development',
        'report_group':r['group'],'quarantine_reasons':[]} for r in rows]})
    candidate={'records':rows,'targets':['a','b'],'files':{'schema':'fixed'},'weak_labels_sha256':'fixed'}
    return candidate,original,identity


class CohortTests(unittest.TestCase):
    def bind(self,c,o,i):
        with patch.object(cc,'ORIGINAL_PLAN',o['fingerprint']),patch.object(cc,'IDENTITY_FP',i['fingerprint']):
            return cc.bind_cohort(c,o,i)

    def test_fixed_membership_and_unchanged_dev(self):
        c,o,i=fixtures();p=self.bind(c,o,i)
        self.assertEqual(len(p['records']),2200)
        c['records'][-1]['labels']=[.1,.2]
        # fixtures shared row objects; isolate original before corruption
        c,o,i=fixtures();c=copy.deepcopy(c);c['records'][-1]['labels']=[.1,.2]
        with self.assertRaisesRegex(ValueError,'Development'):self.bind(c,o,i)

    def test_sealed_and_component_leakage_rejected(self):
        for mutation in ('sealed','component'):
            c,o,i=fixtures()
            if mutation=='sealed':i['records'][0]['future_role']='sealed'
            else:i['records'][0]['component']=i['records'][-1]['component']
            i=stamp({k:v for k,v in i.items() if k!='fingerprint'})
            with self.assertRaises(ValueError):self.bind(c,o,i)

    def test_pilot_cost_gates(self):
        self.assertTrue(cc.projection([{'seconds':2,'bytes':1000000}]*64,2200,100)['passed'])
        self.assertFalse(cc.projection([{'seconds':20,'bytes':1000000}]*64,2200,100)['passed'])
        self.assertFalse(cc.projection([{'seconds':2,'bytes':100000000}]*64,2200,100)['passed'])
        with self.assertRaises(ValueError):cc.projection([{'seconds':float('nan'),'bytes':1}],2200,0)
        measured=[{'seconds':4,'bytes':6300000}]*64
        self.assertFalse(cc.projection(measured,2200,25)['passed'])
        self.assertTrue(cc.projection(measured,2200,25,bytes_limit=17*1024**3)['passed'])
        with self.assertRaises(ValueError):cc.projection(measured,2200,25,bytes_limit=18*1024**3)


try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipIf(np is None,'NumPy unavailable')
class WindowTests(unittest.TestCase):
    def volumes(self):
        return [np.stack([np.full((4,4),s*35+z+1,np.uint8) for z in range(30)])
                if s!=2 else np.empty((0,4,4),np.uint8) for s in range(6)]

    def test_series_boundaries_masks_positions_and_repeat(self):
        v=self.volumes();rng=np.random.default_rng(42)
        x,m,p=cc.sample_windows(v,12,train=False,rng=rng,image_size=4)
        self.assertEqual(x.shape,(6,12,3,4,4));self.assertFalse(m[2]);self.assertFalse(x[2].any())
        for s in (0,1,3,4,5):
            self.assertTrue(np.all(np.diff(x[s,:,:,0,0].astype(int),axis=1)==1))
            self.assertTrue(np.all((p[s]>0)&(p[s]<1)))
        y,_,_=cc.sample_windows(v,12,train=False,rng=rng,image_size=4)
        self.assertTrue(np.array_equal(x,y))

    def test_random_coverage_changes_and_is_seeded(self):
        v=self.volumes();a=np.random.default_rng(42)
        x=cc.sample_windows(v,8,train=True,rng=a,image_size=4)[0]
        y=cc.sample_windows(v,8,train=True,rng=a,image_size=4)[0]
        z=cc.sample_windows(v,8,train=True,rng=np.random.default_rng(42),image_size=4)[0]
        self.assertFalse(np.array_equal(x,y));self.assertTrue(np.array_equal(x,z))

    def test_short_and_all_empty_rejected(self):
        for count in (0,1,2):
            with self.assertRaises(ValueError):
                cc.validate_volumes([np.ones((count,4,4),np.uint8)]*6,image_size=4)
        cc.validate_volumes([np.ones((3,4,4),np.uint8)]*6,image_size=4)


if __name__=='__main__':unittest.main()
