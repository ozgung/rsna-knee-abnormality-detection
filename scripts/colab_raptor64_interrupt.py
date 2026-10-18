"""Local: interrupt the busy foreground cell of the owned raptor64 runtime (detached controllers are unaffected).

Run with the colab-cli tool interpreter. Never allocates, restarts or stops a runtime.
"""
import json
import sys
import requests
from colab_cli.common import state
ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else 'gpu-a100-s-kkb-usc1c2-3eu5ppbx4wl4k'
hits = [a for a in state.client.list_assignments() if a.endpoint == ENDPOINT]
assert len(hits) == 1, 'Exact runtime not present'
proxy = hits[0].runtime_proxy_info
headers = {'X-Colab-Client-Agent': 'colab-cli', 'X-Colab-Runtime-Proxy-Token': proxy.token, 'Authorization': 'token ' + proxy.token}
kernels = requests.get(proxy.url.rstrip('/') + '/api/kernels', headers=headers, timeout=30).json()
assert len(kernels) == 1, 'Ambiguous kernel'
kid = kernels[0]['id']; before = kernels[0].get('execution_state')
r = requests.post(proxy.url.rstrip('/') + f'/api/kernels/{kid}/interrupt', headers=headers, timeout=30)
print(json.dumps({'kernel_id': kid, 'state_before': before, 'interrupt_http': r.status_code}))
