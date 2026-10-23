"""Refresh the runtime proxy token of an EXISTING, still-assigned Lane B runtime through the Colab CLI's own client
(list_assignments), re-register the local session alias and keep-alive.  Never allocates or stops a VM; prints no token.
Usage: <colab-cli python> reconnect.py <session name> <endpoint>"""
import json, os, sys
import requests
from colab_cli.common import state
from colab_cli.state import SessionState
from colab_cli.commands.session import spawn_keep_alive
NAME, ENDPOINT = sys.argv[1], sys.argv[2]
state.config_path = os.path.expanduser(os.environ.get("LB_COLAB_CONFIG", "~/.config/colab-cli/sessions-laneb-20261019.json"))
hits = [a for a in state.client.list_assignments() if a.endpoint == ENDPOINT]
assert len(hits) == 1, "runtime not assigned; no allocation allowed here"
proxy = hits[0].runtime_proxy_info
hdr = {"X-Colab-Client-Agent": "colab-cli", "X-Colab-Runtime-Proxy-Token": proxy.token, "Authorization": "token " + proxy.token}
r = requests.get(proxy.url.rstrip("/") + "/api/kernels", headers=hdr, timeout=30)
assert r.status_code == 200, "kernel listing HTTP %d" % r.status_code
ks = r.json(); assert len(ks) >= 1
kid = ks[0]["id"]
prev = state.store.get(NAME)
assert prev is None or prev.endpoint == ENDPOINT
s = prev or SessionState(name=NAME, token=proxy.token, url=proxy.url, endpoint=ENDPOINT, variant="GPU", accelerator="A100", kernel_id=kid)
s.token, s.url, s.kernel_id = proxy.token, proxy.url, kid
state.client.keep_alive_assignment(ENDPOINT)
state.store.add(s)
alive = False
if s.keep_alive_pid:
    try:
        os.kill(s.keep_alive_pid, 0); alive = True
    except ProcessLookupError:
        pass
if not alive:
    s.keep_alive_pid = spawn_keep_alive(ENDPOINT, NAME, auth_provider=state.auth_provider, config_path=state.config_path)
    state.store.add(s)
print(json.dumps({"status": "RECONNECTED_EXISTING", "session": NAME, "kernels": len(ks), "token_lifetime_s": proxy.token_expires_in_seconds,
                  "keep_alive_pid": s.keep_alive_pid, "new_runtime_created": False}))
