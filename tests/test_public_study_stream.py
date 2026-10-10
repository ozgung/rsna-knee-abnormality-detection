import random
import threading
import time
import unittest
from knee.public_study_stream import StudyLRU, StudyStream


class StreamTests(unittest.TestCase):
    def test_order_identity_cursor_resume_and_rng(self):
        values = {i: (bytearray([i]), bytearray([1])) for i in range(9)}
        rng = random.getstate()
        with StudyStream(range(9), values.__getitem__, contract='recipe-sha') as stream:
            self.assertIs(next(stream), values[0])
            self.assertEqual(stream.consumed_cursor, 0)
            with self.assertRaises(RuntimeError): next(stream)
            stream.acknowledge()
            self.assertIs(next(stream), values[1])
            cursor = stream.consumed_cursor
        self.assertEqual(cursor, 1)
        with StudyStream(range(9), values.__getitem__, contract='recipe-sha', consumed_cursor=cursor) as resumed:
            for i in range(1, 9):
                self.assertIs(next(resumed), values[i])
                resumed.acknowledge()
            with self.assertRaises(StopIteration): next(resumed)
            self.assertEqual(resumed.consumed_cursor, 9)
        self.assertEqual(random.getstate(), rng)

    def test_optional_numpy_torch_rng_unchanged(self):
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest('Optional NumPy/Torch unavailable')
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        with StudyStream(range(6), lambda i: (bytes([i]), bytes([1])), contract='a') as stream:
            for _ in stream:
                stream.acknowledge()
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        self.assertTrue(np.array_equal(numpy_before[1], numpy_after[1]))
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_backpressure_and_early_close(self):
        reached = threading.Event()
        calls = []
        def read(i):
            calls.append(i)
            if len(calls) == 4: reached.set()
            return i
        stream = StudyStream(range(100), read, contract='a')
        self.assertTrue(reached.wait(2))
        time.sleep(.03)
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(stream.fetched_cursor, 4)
        self.assertEqual(stream.consumed_cursor, 0)
        stream.close()
        self.assertFalse(stream._thread.is_alive())
        stream.close()
        with self.assertRaises(RuntimeError): next(stream)

    def test_failure_no_replacement(self):
        calls = []
        def read(i):
            calls.append(i)
            if i == 2: raise LookupError('synthetic decoder failure')
            return i
        with StudyStream(range(8), read, contract='a') as stream:
            for i in range(2):
                self.assertEqual(next(stream), i)
                stream.acknowledge()
            with self.assertRaisesRegex(LookupError, 'synthetic'): next(stream)
            self.assertEqual(stream.consumed_cursor, 2)
        self.assertEqual(calls, [0, 1, 2])

    def test_cache_eviction_hit_and_contract_binding(self):
        cache = StudyLRU(2)
        calls = []
        def read(i):
            calls.append(i)
            return object()
        a = cache.read('a', 1, read)
        cache.read('a', 2, read)
        self.assertIs(cache.read('a', 1, read), a)
        cache.read('a', 3, read)
        cache.read('a', 2, read)
        self.assertEqual(calls, [1, 2, 3, 2])
        self.assertIsNot(cache.read('b', 1, read), a)
        empty = StudyLRU(0)
        self.assertIsNot(empty.read('a', 1, read), empty.read('a', 1, read))

    def test_callback_close_timeout_is_visible_and_recoverable(self):
        entered, release = threading.Event(), threading.Event()
        def read(i):
            entered.set()
            release.wait(2)
            return i
        stream = StudyStream([0], read, contract='a', close_timeout=.01)
        self.assertTrue(entered.wait(1))
        with self.assertRaisesRegex(RuntimeError, 'timeout'): stream.close()
        release.set()
        stream._thread.join(1)
        stream.close()
        self.assertEqual(stream.consumed_cursor, 0)

    def test_invalid_configuration(self):
        for kwargs in ({'prefetch': 5}, {'consumed_cursor': 2}, {'contract': ''}):
            with self.assertRaises(ValueError):
                StudyStream([0], lambda i: i, **({'contract':'a'} | kwargs))
        with self.assertRaises(ValueError): StudyLRU(65)


if __name__ == '__main__': unittest.main()
