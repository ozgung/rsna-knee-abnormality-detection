import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee import coverage_training as ct

try:
    import numpy as np
    import torch
except ImportError:
    np=torch=None


class BudgetTests(unittest.TestCase):
    def test_all_twelve_epochs_must_fit(self):
        self.assertTrue(ct.timing_projection(.2,.1,100)['passed'])
        self.assertFalse(ct.timing_projection(1.,.1,100)['passed'])
        self.assertFalse(ct.timing_projection(.2,.1,10000)['passed'])
        with self.assertRaises(ValueError):ct.timing_projection(float('nan'),.1,100)

    def test_pair_rejects_sampler_or_environment_drift(self):
        keys=['config','plan_fingerprint','runner_sha256','source_hashes','runtime_sources',
              'index_sha256','complete_sha256','weights_fingerprint','runtime_environment']
        reference={k:'same' for k in keys}|{'variant':'v1'}
        current=reference|{'variant':'consensus','labels_fingerprint':'different'}
        ct.require_pair_invariants(current,reference)
        for key in keys:
            with self.assertRaises(ValueError):ct.require_pair_invariants(current|{key:'changed'},reference)


@unittest.skipIf(torch is None,'ML environment unavailable')
class SupervisionTests(unittest.TestCase):
    def test_unknown_mask_and_soft_half_are_independent(self):
        logits=torch.zeros((1,2),requires_grad=True)
        targets=torch.tensor([[.9,.5]]);weights=torch.tensor([[0.,.8]])
        n,d=ct.weighted_terms(logits,targets,weights)
        self.assertEqual(d.tolist(),[0.,float(weights[0,1])])
        self.assertGreater(float(n[1].detach()),0)
        n.sum().backward();self.assertEqual(float(logits.grad[0,0]),0.)
        # Soft .5 remains supervised and pulls a nonzero logit toward zero.
        z=torch.ones((1,2),requires_grad=True);ct.weighted_terms(z,targets,weights)[0].sum().backward()
        self.assertGreater(float(z.grad[0,1]),0)

    def test_consensus_does_not_read_dev_labels_or_fill_unknowns(self):
        plan={'targets':['a','b'], 'records':[{'uid':'train','split':'train','labels':[.5,.9]},
                                             {'uid':'dev','split':'weak_validation','labels':[.8,.5]}]}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.csv'
            with path.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['StudyInstanceUID','a','b'])
                w.writerow(['train',.01,.5]);w.writerow(['dev','never_parse','never_parse'])
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(ct,'LABEL_SOURCES',{'synthetic':('source.csv',digest)}):
                y,w=ct.supervision(plan,source_paths={'synthetic':path})
            self.assertEqual(float(y[0,0]),.5);self.assertEqual(float(w[0,0]),0)
            self.assertEqual(float(y[0,1]),.5);self.assertGreater(float(w[0,1]),0)
            self.assertTrue(np.array_equal(y[1],np.asarray([.8,.5],np.float32)))

    def test_label_hash_and_row_status_fail_closed(self):
        plan={'targets':['a'],'records':[{'uid':'train','split':'train','labels':[.9]}]}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.csv';path.write_text('StudyInstanceUID,a,status\ntrain,.9,error\n')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            for wanted in (digest,'0'*64):
                with patch.object(ct,'LABEL_SOURCES',{'synthetic':('source.csv',wanted)}):
                    with self.assertRaises(ValueError):ct.supervision(plan,source_paths={'synthetic':path})

    def test_host_cache_evicts_and_obeys_byte_bound(self):
        value=[np.ones((3,4,4),np.uint8)]*6
        size=sum(v.nbytes for v in value)
        cache=ct.VolumeCache('.',{'a':{'file':'a'},'b':{'file':'b'}},limit=size)
        with patch.object(ct,'read_volumes',return_value=value) as read:
            cache.get('a');cache.get('a');cache.get('b')
            self.assertEqual(read.call_count,2);self.assertEqual(list(cache.items),['b'])
            self.assertLessEqual(cache.bytes,cache.limit)


if __name__=='__main__':unittest.main()
