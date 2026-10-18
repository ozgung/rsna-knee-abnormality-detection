"""Synthetic tests for the blinded relabeling pilot (prepare + score)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = pd = None

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SAMPLES = {
    "en": "The anterior cruciate ligament is torn and there is a joint effusion with no fracture.",
    "es": "Rotura del ligamento cruzado anterior con derrame articular en la rodilla, sin fractura.",
    "tr": "Ön çapraz bağda yırtık ve eklem içi sıvı artışı izlendi, kırık saptanmadı.",
    "el": "Ρήξη του πρόσθιου χιαστού συνδέσμου με υγρό στην άρθρωση χωρίς κάταγμα.",
    "de": "Riss des vorderen Kreuzbands mit Gelenkerguss, keine Fraktur im Knie.",
    "cyr": "Разрыв передней крестообразной связки, выпот в суставе, перелома нет.",
    "fr": "Rupture du ligament croisé antérieur avec épanchement articulaire, pas de fracture du genou.",
    "hr": "Ruptura prednjeg križnog ligamenta uz izljev u zglobu, bez znakova prijeloma koljena.",
    "nl": "Scheur van de voorste kruisband met vocht in de knie, geen fractuur.",
}


def make_train(path: Path, n_per_lang: int = 30, n_gold: int = 12, flip_gold: bool = False):
    """Synthetic train.csv: gold rows fully labelled, others NaN; UIDs never appear in text."""
    prep = _load("relabel_pilot_prepare")
    rows, i = [], 0
    langs = list(SAMPLES)
    for lang in langs:
        for j in range(n_per_lang):
            rows.append({"StudyInstanceUID": f"9.9.{i}", "Report": f"{SAMPLES[lang]} (case {j})"})
            i += 1
    df = pd.DataFrame(rows)
    for k, t in enumerate(prep.TARGETS):
        df[t] = np.nan
    rng = np.random.default_rng(0)
    gold_idx = rng.choice(len(df), n_gold, replace=False)
    vals = (rng.random((n_gold, len(prep.TARGETS))) < 0.5).astype(float)
    vals[0], vals[1] = 1.0, 0.0  # both classes for every target
    if flip_gold:
        vals = 1.0 - vals
    df.loc[gold_idx, prep.TARGETS] = vals
    df.to_csv(path, index=False)
    return df, set(df.loc[gold_idx, "StudyInstanceUID"])


@unittest.skipIf(np is None, "numpy/pandas not installed")
class PrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load("relabel_pilot_prepare")

    def test_language_heuristic(self):
        for lang, text in SAMPLES.items():
            self.assertEqual(self.m.detect_language(text), lang, lang)
        self.assertEqual(self.m.detect_language(""), "other")
        self.assertEqual(self.m.detect_language("12345 ---"), "other")

    def run_prepare(self, tmp, **kw):
        train = Path(tmp) / "train.csv"
        df, gold = make_train(train, flip_gold=kw.pop("flip_gold", False))
        out = Path(tmp) / "out"
        man = self.m.prepare(train, out, n_nongold=100, min_per_lang=10, batch_size=25, seed=7,
                             salt="fixedsalt", **kw)
        return df, gold, out, man

    def test_outputs_blinded_complete_and_batched(self):
        with tempfile.TemporaryDirectory() as tmp:
            df, gold, out, man = self.run_prepare(tmp)
            lines = (out / "pilot_reports.jsonl").read_text(encoding="utf-8").splitlines()
            recs = [json.loads(x) for x in lines]
            self.assertEqual(len(recs), len(gold) + 100)
            self.assertTrue(all(set(r) == {"hid", "report"} for r in recs))
            raw = "\n".join(lines)
            self.assertFalse(any(u in raw for u in df["StudyInstanceUID"]))
            key = json.loads((out / "pilot_key.json").read_text())
            ent = key["entries"]
            self.assertEqual(set(ent), {r["hid"] for r in recs})
            self.assertEqual({e["uid"] for e in ent.values() if e["gold"]}, gold)
            for e in ent.values():  # key holds membership only, never target values
                self.assertEqual(set(e), {"uid", "lang", "batch", "gold"})
            nongold_langs = [e["lang"] for e in ent.values() if not e["gold"]]
            for lang in self.m.REQUIRED_LANGS:
                self.assertGreaterEqual(nongold_langs.count(lang), 10)
            batches = sorted(out.glob("batch_*.jsonl"))
            self.assertEqual(len(batches), man["n_batches"])
            sizes = [len(b.read_text().splitlines()) for b in batches]
            self.assertEqual(sum(sizes), len(recs))
            self.assertLessEqual(max(sizes) - min(sizes), 1)
            joined = [json.loads(x) for b in batches for x in b.read_text().splitlines()]
            self.assertEqual(joined, recs)
            for h, e in ent.items():
                self.assertIn(h, (out / f"batch_{e['batch']:02d}.jsonl").read_text())
            pos = [i for i, r in enumerate(recs) if ent[r["hid"]]["gold"]]
            self.assertNotEqual(pos, list(range(len(gold))))  # gold not grouped at the front

    def test_gold_values_do_not_change_reports_output(self):
        outs = []
        for flip in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                _, _, out, _ = self.run_prepare(tmp, flip_gold=flip)
                outs.append([(out / f).read_bytes() for f in ["pilot_reports.jsonl"] +
                             sorted(p.name for p in out.glob("batch_*.jsonl"))])
        self.assertEqual(outs[0], outs[1])

    def test_report_reader_requests_only_id_and_report_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "train.csv"
            make_train(train)
            with mock.patch("pandas.read_csv", wraps=pd.read_csv) as rc:
                df = self.m.read_reports(train)
            self.assertEqual(list(rc.call_args.kwargs["usecols"]), ["StudyInstanceUID", "Report"])
            self.assertEqual(list(df.columns), ["StudyInstanceUID", "Report"])

    def test_refuses_to_rekey_and_rejects_extra_jsonl_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_prepare(tmp)
            with self.assertRaises(FileExistsError):
                self.m.prepare(Path(tmp) / "train.csv", Path(tmp) / "out")
            with self.assertRaises(ValueError):
                self.m.write_jsonl(Path(tmp) / "x.jsonl", [{"hid": "a", "report": "b", "gold": 1}])

    def test_split_batches(self):
        self.assertEqual([len(b) for b in self.m.split_batches(list(range(358)), 60)], [60, 60, 60, 60, 59, 59])
        self.assertEqual([len(b) for b in self.m.split_batches(list(range(5)), 60)], [5])


@unittest.skipIf(np is None, "numpy/pandas not installed")
class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep = _load("relabel_pilot_prepare")
        cls.m = _load("relabel_pilot_score")

    def test_auc_hand_values(self):
        auc = self.m.auc
        self.assertEqual(auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertEqual(auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)
        self.assertIsNone(auc([1, 1], [0.2, 0.3]))
        self.assertEqual(auc([0, 1, 1], [0.1, float("nan"), 0.9]), 1.0)

    def build(self, tmp, label_fn, labelers=("a", "b"), steven_noise=True):
        """Prepare a synthetic pilot, public tables and labeler files built from label_fn."""
        tmp = Path(tmp)
        train = tmp / "train.csv"
        df, _ = make_train(train)
        out = tmp / "pilot"
        self.prep.prepare(train, out, n_nongold=100, min_per_lang=10, batch_size=25, seed=7, salt="s")
        ent = json.loads((out / "pilot_key.json").read_text())["entries"]
        T = self.m.TARGETS
        truth = df.set_index("StudyInstanceUID")[T]
        rng = np.random.default_rng(1)
        # Non-gold "consensus" direction per study: deterministic pseudo-random.
        cons = {u: (rng.random(len(T)) < 0.3) for u in df["StudyInstanceUID"]}
        tables = {}
        for name in ("steven_v4", "gemlow", "pilkwang", "nartaa_v1_blend"):
            rows = []
            for u in df["StudyInstanceUID"]:
                if u in truth.index and truth.loc[u].notna().all():
                    if name in ("gemlow", "nartaa_v1_blend"):
                        continue  # like nartaa: no gold rows
                    y = truth.loc[u].to_numpy()
                    p = np.where(y == 1, 0.7, 0.3) + (rng.normal(0, 0.25, len(T)) if steven_noise else 0)
                else:
                    p = np.where(cons[u], 0.8, 0.2)
                rows.append({"StudyInstanceUID": u, **dict(zip(T, np.clip(p, 0, 1)))})
            path = tmp / f"{name}.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            tables[name] = path
        for lab in labelers:
            per_batch = {}
            for h, e in ent.items():
                u = e["uid"]
                y = truth.loc[u].to_numpy() if e["gold"] else cons[u].astype(float)
                per_batch.setdefault(e["batch"], {})[h] = label_fn(lab, e, y)
            for b, d in per_batch.items():
                (out / f"labels_{lab}_batch_{b:02d}.json").write_text(json.dumps(d))
        return out, train, tables, df

    @staticmethod
    def perfect(lab, e, y):
        return {t: ({"p": 0.95, "state": "pos"} if v == 1 else {"p": 0.05, "state": "neg"})
                for t, v in zip(ScoreTests.m.TARGETS, y)}

    def test_perfect_labelers_go_and_receipt_has_no_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, train, tables, df = self.build(tmp, self.perfect)
            r = self.m.score(out, train, tables, gates={"language_min_reports": 5})
            self.assertEqual(r["gold"]["labeler_mean"]["macro"], 1.0)
            self.assertLess(r["gold"]["steven_v4"]["macro"], 1.0)
            self.assertIsNone(r["gold"]["nartaa_v1_blend"]["macro"])
            self.assertEqual(r["gold"]["nartaa_v1_blend"]["gold_rows_present"], 0)
            self.assertEqual(r["directional"]["labeler_mean"]["agreement"], 1.0)
            self.assertEqual(r["inter_rater"]["a|b"]["state"], 1.0)
            self.assertTrue(r["coverage_complete"])
            self.assertEqual(r["decision"], "GO", r["checks"])
            blob = json.dumps(r) + "\n".join(self.m.summary_lines(r))
            ent = json.loads((out / "pilot_key.json").read_text())["entries"]
            for h, e in ent.items():
                self.assertNotIn(h, blob)
                self.assertNotIn(e["uid"], blob)
            self.assertNotIn("case", blob)  # no report text

    def test_state_disagreement_and_bad_language_fail(self):
        def noisy(lab, e, y):
            d = self.perfect(lab, e, y)
            if lab == "b" and e["lang"] == "tr":
                for t in d:
                    d[t] = {"p": d[t]["p"], "state": "uncertain"}
            return d
        with tempfile.TemporaryDirectory() as tmp:
            out, train, tables, _ = self.build(tmp, noisy)
            r = self.m.score(out, train, tables, gates={"language_min_reports": 5})
            self.assertLess(r["inter_rater"]["a|b"]["state"], 0.95)
            self.assertEqual(r["inter_rater"]["a|b"]["by_language"]["tr"]["state"], 0.0)
            self.assertEqual(r["inter_rater"]["a|b"]["by_language"]["tr"]["p_within"], 1.0)
            self.assertFalse(r["checks"]["inter_rater_state"]["pass"])
            self.assertIn("tr", r["checks"]["minority_language_gap"]["detail"]["failing_languages"])
            self.assertEqual(r["decision"], "NO-GO")

    def test_worse_than_steven_fails_gold_gates(self):
        def inverted(lab, e, y):
            d = self.perfect(lab, e, y)
            if e["gold"]:
                d = {t: {"p": 1 - v["p"], "state": v["state"]} for t, v in d.items()}
            return d
        with tempfile.TemporaryDirectory() as tmp:
            out, train, tables, _ = self.build(tmp, inverted)
            r = self.m.score(out, train, tables, gates={"language_min_reports": 5})
            self.assertFalse(r["checks"]["gold_macro_vs_steven"]["pass"])
            self.assertEqual(r["decision"], "NO-GO")

    def test_single_labeler_and_invalid_cells_incomplete(self):
        def bad(lab, e, y):
            d = self.perfect(lab, e, y)
            d["ACL"] = {"p": 1.5, "state": "pos"}
            d["MCL"] = {"p": 0.2, "state": "maybe"}
            return d
        with tempfile.TemporaryDirectory() as tmp:
            out, train, tables, _ = self.build(tmp, bad, labelers=("solo",))
            r = self.m.score(out, train, tables, gates={"language_min_reports": 5})
            c = r["integrity"]["solo"]
            self.assertEqual(c["invalid_cells"], 2 * r["n_reports"])
            self.assertFalse(c["complete"])
            self.assertIsNone(r["checks"]["inter_rater_state"]["pass"])
            self.assertIn("ACL", r["gold"]["solo"]["undefined"])
            self.assertEqual(r["decision"], "INCOMPLETE")

    def test_cli_writes_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, train, tables, _ = self.build(tmp, self.perfect)
            argv = ["--pilot-dir", str(out), "--train-csv", str(train)]
            for n, p in tables.items():
                argv += [f"--{n.replace('_', '-')}", str(p)]
            with mock.patch("builtins.print"):
                self.assertEqual(self.m.main(argv), 0)
            r = json.loads((out / "score_receipt.json").read_text())
            self.assertIn(r["decision"], {"GO", "INCOMPLETE"})  # default language floor (20) exceeds synthetic counts
            self.assertTrue(r["checks"]["inter_rater_state"]["pass"])


if __name__ == "__main__":
    unittest.main()
