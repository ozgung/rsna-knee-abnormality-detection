"""Retrieve explicit JSON reports without recursively listing outputs at SDK defaults.

Uses 100-file pages, two-second pacing, bounded 429 retries, and saved page
progress. Image arrays and weights are never downloaded. Kaggle is CLI-only.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import time


PAGE_SIZE = 100
MAX_PAGES = 100
MAX_REPORT_BYTES = 20 * 1024**2


def report_path(root, name):
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or '\\' in name or
            not name.endswith('.json') or str(path) != name):
        raise ValueError('Expected a safe relative JSON report path')
    target = root / name
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Report path escapes output directory')
    return target


def write_state(path, state):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(path)


def rate_limited_read(operation, sleep):
    for attempt in range(3):
        try:
            return operation()
        except Exception as exc:
            response = getattr(exc, 'response', None)
            if response is None or getattr(response, 'status_code', None) != 429:
                raise
            retry = response.headers.get('Retry-After')
            delay = 30 * (attempt + 1)
            if retry is not None:
                try:
                    delay = max(delay, float(retry))
                except ValueError:
                    raise RuntimeError('Rate limited with nonnumeric Retry-After; progress saved') from None
            if attempt == 2 or not 0 <= delay <= 60:
                raise RuntimeError('Rate limit exceeds bounded retry allowance; progress saved') from None
            print(json.dumps({'event': 'report_rate_limit', 'retry_seconds': delay}), flush=True)
            sleep(delay)


def download_reports(kernel, destination, required, list_page, read_file, sleep=time.sleep):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    required = sorted(set(required))
    if not required:
        raise ValueError('Explicit required reports are mandatory')
    for name in required:
        report_path(root, name)
    state_path = root / '.report-download-state.json'
    identity = {'kernel': kernel, 'required': required, 'page_size': PAGE_SIZE}
    state = {'identity': identity, 'token': '', 'pages': 0, 'files': {}, 'complete': False}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['identity'] != identity:
            raise ValueError('Download resume identity changed')
        for name, digest in state['files'].items():
            if name not in required or hashlib.sha256(report_path(root, name).read_bytes()).hexdigest() != digest:
                raise ValueError('Previously downloaded report changed')
    while not state['complete']:
        if state['pages'] >= MAX_PAGES:
            raise RuntimeError('Output listing page bound exceeded')
        if state['pages']:
            sleep(2)
        files, next_token = rate_limited_read(lambda: list_page(state['token'], PAGE_SIZE), sleep)
        if next_token and next_token == state['token']:
            raise ValueError('Output pagination did not advance')
        for name, url in files:
            if name not in required or name in state['files']:
                continue
            data = rate_limited_read(lambda: read_file(url), sleep)
            if len(data) > MAX_REPORT_BYTES:
                raise ValueError('JSON report exceeds size bound')
            json.loads(data)
            target = report_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix('.download')
            tmp.write_bytes(data)
            tmp.replace(target)
            state['files'][name] = hashlib.sha256(data).hexdigest()
            write_state(state_path, state)
        state['pages'] += 1
        state['token'] = next_token or ''
        state['complete'] = set(state['files']) == set(required)
        write_state(state_path, state)
        print(json.dumps({'event': 'report_page', 'pages': state['pages'],
                          'reports': len(state['files']), 'complete': state['complete']}), flush=True)
        if not next_token and not state['complete']:
            raise ValueError('Required reports missing from completed output listing')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernel')
    parser.add_argument('destination')
    parser.add_argument('--required', nargs='+', required=True)
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
    import requests
    api = KaggleApi()
    api.authenticate()
    owner, slug, version = api.parse_kernel_string(args.kernel)
    if version is not None:
        raise ValueError('This SDK output endpoint cannot select a version; use an immutable job slug')

    def list_page(token, page_size):
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        request.page_size = page_size
        request.page_token = token
        with api.build_kaggle_client() as client:
            result = client.kernels.kernels_api_client.list_kernel_session_output(request)
        return [(f.file_name, f.url) for f in result.files or []], result.next_page_token

    def read_file(url):
        with requests.get(url, stream=True, timeout=(15, 45)) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > MAX_REPORT_BYTES:
                    raise ValueError('JSON report exceeds size bound')
            return bytes(data)

    download_reports(args.kernel, args.destination, args.required, list_page, read_file)


if __name__ == '__main__':
    main()
