"""Read-only: the account's Colab compute-unit balance and consumption rate (tun/m/ccu-info, as the Colab VS Code client)."""
import json, os
from urllib.parse import urljoin
from colab_cli.common import state
from colab_cli import client as C
state.config_path = os.path.expanduser(os.environ.get("LB_COLAB_CONFIG", "~/.config/colab-cli/sessions-laneb-20261019.json"))
cl = state.client
r = cl.session.request("GET", urljoin(cl.colab_domain, C.TUN_ENDPOINT + "/ccu-info"), params={"authuser": "0"},
                       headers={"Accept": "application/json", "X-Colab-Client-Agent": "colab-cli"})
body = cl._strip_xssi_prefix(r.text)
d = json.loads(body) if r.ok else {"http": r.status_code}
print(json.dumps({k: d.get(k) for k in ("currentBalance", "consumptionRateHourly", "assignmentsCount", "eligibleGpus", "ineligibleGpus") if k in d} or d))
