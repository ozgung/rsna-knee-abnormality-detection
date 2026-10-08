import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('reports', Path(__file__).resolve().parents[1] / 'scripts/download_kaggle_reports.py')
reports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reports)


class Limited(Exception):
    def __init__(self, retry='1'):
        self.response = SimpleNamespace(status_code=429, headers={'Retry-After': retry})


class ReportTests(unittest.TestCase):
    def test_arrays_are_not_downloaded_and_stops_when_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            reads, pages = [], []
            def listing(token, size):
                pages.append((token, size))
                return [('array.npz', 'large'), ('cache/report.json', 'small')], 'unneeded-next'
            def read(url):
                reads.append(url)
                return b'{"status":"PASS"}'
            state = reports.download_reports('owner/job', tmp, ['cache/report.json'], listing, read)
            self.assertTrue(state['complete'])
            self.assertEqual(reads, ['small'])
            self.assertEqual(pages, [('', 100)])

    def test_rate_limit_resume_preserves_page_and_validates_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, waits = [], []
            def listing(token, size):
                calls.append(token)
                if not token:
                    return [('a.json', 'a')], 'second'
                raise Limited()
            with self.assertRaises(RuntimeError):
                reports.download_reports('owner/job', tmp, ['a.json', 'b.json'], listing, lambda _: b'{}', waits.append)
            self.assertEqual(calls, ['', 'second', 'second', 'second'])
            self.assertEqual(waits, [2, 30, 60])
            resumed = []
            def listing2(token, size):
                resumed.append(token)
                return [('b.json', 'b')], ''
            state = reports.download_reports('owner/job', tmp, ['a.json', 'b.json'], listing2, lambda _: b'{}', waits.append)
            self.assertTrue(state['complete'])
            self.assertEqual(resumed, ['second'])
            Path(tmp, 'a.json').write_text('{"tampered":true}')
            with self.assertRaises(ValueError):
                reports.download_reports('owner/job', tmp, ['a.json', 'b.json'], listing2, lambda _: b'{}')

    def test_missing_invalid_and_escaping_reports_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                reports.download_reports('owner/job', tmp, ['../report.json'], lambda *_: ([], ''), lambda _: b'{}')
            with self.assertRaises(ValueError):
                reports.download_reports('owner/job', tmp, ['a.json'], lambda *_: ([], ''), lambda _: b'{}')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(json.JSONDecodeError):
                reports.download_reports('owner/job', tmp, ['a.json'], lambda *_: ([('a.json', 'a')], ''), lambda _: b'<error>')
            self.assertFalse(Path(tmp, 'a.json').exists())

    def test_long_retry_after_is_not_ignored(self):
        waits = []
        def blocked():
            raise Limited('3600')
        with self.assertRaises(RuntimeError):
            reports.rate_limited_read(blocked, waits.append)
        self.assertEqual(waits, [])


if __name__ == '__main__':
    unittest.main()
