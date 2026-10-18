"""Writes the competition archive's signed download URL to /content/.signed_url (mode 600); prints nothing secret.
Runs under the VM's system Python (kagglesdk 0.1.37, as the R224 stager), whose download_data_files does not read the
archive body (the kaggle 1.7.4.5 client in the training venv buffered the whole archive in RAM and killed the first VM)."""
import os, sys, time
os.environ.setdefault("KAGGLE_CONFIG_DIR", "/content")
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest as D
api = KaggleApi(); api.authenticate()
for i in range(10):
    try:
        with api.build_kaggle_client() as c:
            d = D(); d.competition_name = "rsna-knee-abnormality-detection"
            r = c.competitions.competition_api_client.download_data_files(d)
            url = r.url; r.close()
        fd = os.open("/content/.signed_url.tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(fd, url.encode()); os.close(fd); os.replace("/content/.signed_url.tmp", "/content/.signed_url")
        print("signed url written"); sys.exit(0)
    except requests.HTTPError as e:
        wait = int(e.response.headers.get("retry-after", "60")) + 5
        print("signed url HTTP", e.response.status_code, "waiting", wait, flush=True); time.sleep(wait)
sys.exit(1)
