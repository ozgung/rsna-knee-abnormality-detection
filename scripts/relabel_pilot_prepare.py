#!/usr/bin/env python3
"""Prepare a blinded LLM relabeling pilot (reports only, no labels).

Outputs (all under the git-ignored ``--out-dir``):

* ``pilot_reports.jsonl``  one ``{"hid": ..., "report": ...}`` object per line, shuffled.
* ``batch_XX.jsonl``       the same lines split into batches of about ``--batch-size``.
* ``pilot_key.json``       salt, seed and hid -> {uid, lang, batch, gold}. Never give to labelers.
* ``pilot_manifest.json``  aggregate counts only (no IDs, no text).

Blinding rules enforced in code:

* ``read_reports`` reads only the ``StudyInstanceUID`` and ``Report`` columns.
* ``read_gold_ids`` is a separate pass that reads the target columns only to decide
  gold membership and returns a set of IDs; no label value leaves that function.
* ``write_jsonl`` refuses any record whose keys are not exactly ``{"hid", "report"}``.
* No label table is opened anywhere in this script.

Selection: all gold rows plus ``--n-nongold`` non-gold rows. For every language in
``REQUIRED_LANGS`` the sample takes ``min(--min-per-lang, available)`` non-gold rows first,
then fills the remaining quota uniformly at random from the rest of the non-gold pool.

Language detection (``detect_language``), a deliberately simple heuristic:

1. Count alphabetic characters by Unicode script. If Greek letters are at least 30% of all
   letters the report is ``el``; if Cyrillic letters are, it is ``cyr`` (Russian, Ukrainian,
   Bulgarian, Serbian and so on are not separated).
2. Otherwise tokenise the lower-cased text and score each Latin-script language by the number
   of tokens found in its cue-word list (function words plus common knee-MRI terms), plus
   3 points per language-specific character (Turkish ``ı ğ ş``; German ``ä ß``; Spanish
   ``ñ ¿ ¡``; French ``è ê ç à œ``; Croatian/Serbian/Bosnian Latin ``č ć đ ž``, reported as ``hr``). Dutch (``nl``) is scored by cue words only.
3. The highest score wins if it is at least 3 and strictly above the runner-up; otherwise
   the report is ``other``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import secrets
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "StudyInstanceUID"
REPORT = "Report"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
REQUIRED_LANGS = ["en", "es", "tr", "el", "de", "cyr"]
DEFAULT_OUT = ROOT / "outputs" / "labels" / "relabel-pilot-20261013"
SEED = 20261013
JSONL_KEYS = {"hid", "report"}

CUE_WORDS = {
    "en": set("""the and of with is are there no not without which this that seen noted
        knee tear torn joint effusion ligament intact normal signal evidence fluid bone
        marrow edema cyst cartilage anterior posterior cruciate collateral meniscus impression
        findings small moderate large mild grade horn body""".split()),
    "es": set("""el los las con sin en y del se que por una un sus rodilla rotura desgarro
        derrame ligamento ligamentos menisco meniscos cruzado articular hueso óseo edema
        quiste líquido cartílago señal hallazgos conclusión leve moderado posterior interno
        externo cuerno técnica impresión resultados almohadilla grasa síndrome
        pinzamiento""".split()),
    "tr": set("""ve ile bir olarak izlendi izlenmektedir izlenmiştir mevcut diz yırtık yırtığı
        sıvı bağ bağı menisküs menisküste eklem dejenerasyon dejeneratif ödem ön arka çapraz
        saptanmadı saptandı kemik iliği kıkırdak sinyal artışı hafif sonuç kist medial lateral
        değişiklikler""".split()),
    "de": set("""der die das und mit kein keine keinen ist im des dem den ein eine nicht
        knie riss ruptur erguss kreuzband vorderes hinteres meniskus innenmeniskus
        außenmeniskus gelenk ohne knochenmarködem knorpel befund beurteilung zyste
        signal hinterhorn""".split()),
    "fr": set("""le les du des et avec pas au aux une un est sans sur genou épanchement
        ménisque méniscale croisé croisés fissure rupture lésion lésions articulaire
        osseux cartilage kyste conclusion interne externe postérieure antérieur signal
        corne""".split()),
    "hr": set("""i se u je su na uz bez te od za koji koja nalaz zaključak koljeno koljena
        zgloba zglob ligament ligamenta križni križnog prednjeg stražnjeg menisk meniska
        medijalni medijalnog lateralni lateralnog ruptura rupture izljev edem koštane
        hrskavice hondromalacija ciste signala intenziteta uredna uredne promjene roga
        tijela znakova stupnja""".split()),
    "nl": set("""het van een met op geen niet zonder knie bevindingen besluit conclusie
        kruisband kruisbanden voorste achterste banden vocht gewrichtsvocht kraakbeen
        kraakbeenlijden scheur normaal graad aanvraag vraagstelling inlichtingen links
        rechts""".split()),
}
CUE_CHARS = {"tr": "ığş", "de": "äß", "es": "ñ¿¡", "fr": "èêçàœ", "hr": "čćđž"}
TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def script_counts(text: str) -> Counter:
    counts: Counter = Counter()
    for ch in text:
        if ch.isalpha():
            try:
                counts[unicodedata.name(ch).split()[0]] += 1
            except ValueError:
                counts["UNKNOWN"] += 1
    return counts


def detect_language(text: str) -> str:
    """Return one of en, es, tr, el, de, fr, hr, nl, cyr, other (see module docstring)."""
    if not isinstance(text, str) or not text.strip():
        return "other"
    counts = script_counts(text)
    letters = sum(counts.values())
    if letters == 0:
        return "other"
    if counts["GREEK"] / letters >= 0.3:
        return "el"
    if counts["CYRILLIC"] / letters >= 0.3:
        return "cyr"
    lower = text.lower()
    tokens = TOKEN_RE.findall(lower)
    scores = {lang: sum(tok in words for tok in tokens) for lang, words in CUE_WORDS.items()}
    for lang, chars in CUE_CHARS.items():
        scores[lang] += 3 * sum(lower.count(c) for c in chars)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    return best if top >= 3 and top > second else "other"


def read_reports(train_csv: Path):
    """Read ONLY the ID and report columns (the blinded path never sees targets)."""
    import pandas as pd

    df = pd.read_csv(train_csv, usecols=[ID, REPORT], dtype=str, keep_default_na=False)
    if df[ID].duplicated().any():
        raise ValueError("duplicate StudyInstanceUID in reports")
    return df


def read_gold_ids(train_csv: Path) -> set[str]:
    """Separate pass: return IDs whose 12 targets are all present. Returns IDs only."""
    import pandas as pd

    df = pd.read_csv(train_csv, usecols=[ID] + TARGETS, dtype={ID: str})
    gold = df.loc[df[TARGETS].notna().all(axis=1), ID]
    return set(gold.tolist())


def make_hid(salt: str, uid: str) -> str:
    return hashlib.sha256(f"{salt}:{uid}".encode("utf-8")).hexdigest()[:16]


def select_pilot(langs: dict[str, str], gold_ids: set[str], n_nongold: int, min_per_lang: int,
                 seed: int, required=tuple(REQUIRED_LANGS)) -> tuple[list[str], dict]:
    """Return (sorted selected UIDs, aggregate selection info)."""
    rng = random.Random(seed)
    gold = sorted(u for u in langs if u in gold_ids)
    pool = sorted(u for u in langs if u not in gold_ids)
    by_lang: dict[str, list[str]] = {}
    for u in pool:
        by_lang.setdefault(langs[u], []).append(u)
    chosen: list[str] = []
    shortfall = {}
    for lang in required:
        cands = by_lang.get(lang, [])
        k = min(min_per_lang, len(cands))
        if k < min_per_lang:
            shortfall[lang] = {"wanted": min_per_lang, "available": len(cands)}
        chosen.extend(rng.sample(cands, k))
    if len(chosen) > n_nongold:
        raise ValueError(f"language minimums need {len(chosen)} rows, more than n_nongold={n_nongold}")
    taken = set(chosen)
    rest = [u for u in pool if u not in taken]
    chosen.extend(rng.sample(rest, min(n_nongold - len(chosen), len(rest))))
    return sorted(gold + chosen), {"n_gold": len(gold), "n_nongold": len(chosen), "shortfall": shortfall}


def write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            if set(rec) != JSONL_KEYS:
                raise ValueError(f"jsonl record keys must be exactly {sorted(JSONL_KEYS)}")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split_batches(items: list, batch_size: int) -> list[list]:
    n = max(1, math.ceil(len(items) / batch_size))
    base, extra = divmod(len(items), n)
    out, start = [], 0
    for b in range(n):
        size = base + (1 if b < extra else 0)
        out.append(items[start:start + size])
        start += size
    return out


def prepare(train_csv: Path, out_dir: Path, *, n_nongold: int = 300, min_per_lang: int = 40,
            batch_size: int = 60, seed: int = SEED, salt: str | None = None, force: bool = False) -> dict:
    out_dir = Path(out_dir)
    key_path = out_dir / "pilot_key.json"
    if key_path.exists() and not force:
        raise FileExistsError(f"{key_path} exists; refusing to re-key an existing pilot (use --force)")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("batch_*.jsonl"):
        old.unlink()

    reports = read_reports(train_csv)                       # ID + Report only
    text = dict(zip(reports[ID], reports[REPORT]))
    langs = {u: detect_language(t) for u, t in text.items()}
    gold_ids = read_gold_ids(train_csv)                     # IDs only
    selected, info = select_pilot(langs, gold_ids, n_nongold, min_per_lang, seed)

    salt = salt or secrets.token_hex(16)
    hids = {u: make_hid(salt, u) for u in selected}
    if len(set(hids.values())) != len(hids):
        raise RuntimeError("hid collision; rerun with another salt")
    order = list(selected)
    random.Random(seed + 1).shuffle(order)

    write_jsonl(out_dir / "pilot_reports.jsonl", ({"hid": hids[u], "report": text[u]} for u in order))
    batches = split_batches(order, batch_size)
    batch_of = {}
    for b, members in enumerate(batches):
        write_jsonl(out_dir / f"batch_{b:02d}.jsonl", ({"hid": hids[u], "report": text[u]} for u in members))
        batch_of.update({u: b for u in members})

    key = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed, "salt": salt, "hash": "sha256(salt + ':' + uid)[:16]",
        "source": str(train_csv),
        "entries": {hids[u]: {"uid": u, "lang": langs[u], "batch": batch_of[u], "gold": u in gold_ids}
                    for u in order},
    }
    key_path.write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")

    sel_set = set(selected)
    manifest = {
        "created_utc": key["created_utc"], "seed": seed, "n_reports": len(order),
        "n_gold": info["n_gold"], "n_nongold": info["n_nongold"], "batch_size_target": batch_size,
        "n_batches": len(batches), "batch_sizes": [len(b) for b in batches],
        "pilot_lang_counts": dict(Counter(langs[u] for u in selected).most_common()),
        "pilot_lang_counts_gold": dict(Counter(langs[u] for u in selected if u in gold_ids).most_common()),
        "pilot_lang_counts_nongold": dict(Counter(langs[u] for u in selected if u not in gold_ids).most_common()),
        "corpus_lang_counts": dict(Counter(langs.values()).most_common()),
        "language_shortfall": info["shortfall"],
        "required_langs": REQUIRED_LANGS, "min_per_lang": min_per_lang,
        "jsonl_keys": sorted(JSONL_KEYS),
        "n_selected_check": len(sel_set),
    }
    (out_dir / "pilot_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-csv", type=Path, default=ROOT / "data" / "raw" / "train.csv")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-nongold", type=int, default=300)
    ap.add_argument("--min-per-lang", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=60)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite an existing pilot key")
    args = ap.parse_args(argv)
    m = prepare(args.train_csv, args.out_dir, n_nongold=args.n_nongold, min_per_lang=args.min_per_lang,
                batch_size=args.batch_size, seed=args.seed, force=args.force)
    print(json.dumps({k: m[k] for k in ("n_reports", "n_gold", "n_nongold", "n_batches", "batch_sizes",
                                         "pilot_lang_counts", "corpus_lang_counts", "language_shortfall")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
