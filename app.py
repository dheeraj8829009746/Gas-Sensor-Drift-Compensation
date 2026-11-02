"""
Gas Sensor Drift Dashboard  (Flask)
-----------------------------------
Pipeline:  Input -> Preprocessing -> Drift Compensation -> Selected Model -> Gas Class

Folder layout
    app.py
    raw/batch1.dat ... batch10.dat      (UCI Gas Sensor Array Drift dataset)
    templates/index.html
    static/style.css, static/script.js

Run:  python app.py      then open http://127.0.0.1:5000
The first start runs the full drift analysis (about 1-2 minutes) and caches it in
artifacts/analysis_cache.json; later starts are fast.
"""
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# ======================================================================
# CORE: drift compensation, evaluation and statistics (same code as the notebook)
# ======================================================================
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 129)]
GAS_NAMES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
             4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}
GAS_LABELS = sorted(GAS_NAMES)
MODEL_NAMES = ["Random Forest", "SVM", "XGBoost"]
BATCHES = list(range(1, 11))
REF_BATCH = 1
CORAL_LAMBDA = 0.5
RANDOM_STATE = 42

COMPENSATION_METHODS = {
    "none": "No compensation (baseline)",
    "batch_standardization": "Batch-wise Standardization",
    "reference_mean": "Reference Mean Alignment",
    "reference_coral": "Reference CORAL Alignment",
}


# ----------------------------------------------------------------------
# Classifier wrapper (handles label encoding for every model type)
# ----------------------------------------------------------------------
class GasClassifier:
    """Random Forest / SVM / XGBoost behind one fit-predict interface."""

    def __init__(self, name, seed=RANDOM_STATE, probability=False):
        self.name, self.seed, self.probability = name, seed, probability
        if name == "Random Forest":
            self.model = RandomForestClassifier(
                n_estimators=300, random_state=seed, n_jobs=-1)
        elif name == "SVM":
            self.model = SVC(kernel="rbf", C=10, gamma="scale",
                             probability=probability, random_state=seed)
        elif name == "XGBoost":
            self.model = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=seed)
        else:
            raise ValueError(f"Unknown model: {name}")

    def fit(self, X, y):
        self.le = LabelEncoder().fit(y)
        self.model.fit(X, self.le.transform(y))
        return self

    def predict(self, X):
        pred = np.asarray(self.model.predict(X)).astype(int)
        return self.le.inverse_transform(pred)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba") and (self.name != "SVM" or self.probability):
            return self.model.predict_proba(X)
        return None


# ----------------------------------------------------------------------
# Drift compensation
# ----------------------------------------------------------------------
def _sym_power(M, power):
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 1e-8, None) ** power
    return (vecs * vals) @ vecs.T


class BatchCompensator:
    """Maps every batch towards the reference batch (Batch 1).

    method = none                  -> identity
             batch_standardization -> z-score using the batch's own mean/std
             reference_mean        -> x - mean_batch + mean_ref
             reference_coral       -> mean + covariance (CORAL) alignment to ref

    Only UNLABELLED samples of each batch are used to estimate statistics.
    """

    def __init__(self, method="none", ref_batch=REF_BATCH, coral_lambda=CORAL_LAMBDA):
        if method not in COMPENSATION_METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.method, self.ref_batch, self.coral_lambda = method, ref_batch, coral_lambda

    def fit(self, X, batch):
        X, batch = np.asarray(X, float), np.asarray(batch)
        self.mu, self.sd, self.T = {}, {}, {}
        for b in np.unique(batch):
            Xb = X[batch == b]
            sd = Xb.std(axis=0, ddof=1)
            self.mu[int(b)] = Xb.mean(axis=0)
            self.sd[int(b)] = np.where(sd < 1e-12, 1.0, sd)
        self.mu_ref, self.sd_ref = self.mu[self.ref_batch], self.sd[self.ref_batch]
        if self.method == "reference_coral":
            lam = self.coral_lambda
            eye = np.eye(X.shape[1])
            Xr = (X[batch == self.ref_batch] - self.mu_ref) / self.sd_ref
            Cr_half = _sym_power(np.cov(Xr, rowvar=False) + lam * eye, 0.5)
            for b in self.mu:
                Xb = (X[batch == b] - self.mu[b]) / self.sd_ref
                Cb_inv_half = _sym_power(np.cov(Xb, rowvar=False) + lam * eye, -0.5)
                self.T[b] = Cb_inv_half @ Cr_half
        return self

    def transform(self, X, batch):
        X = np.atleast_2d(np.asarray(X, float))
        b = int(batch)
        if self.method == "none":
            return X.copy()
        if b not in self.mu:
            raise KeyError(f"No statistics available for batch {b}")
        if self.method == "batch_standardization":
            return (X - self.mu[b]) / self.sd[b]
        if self.method == "reference_mean":
            return X - self.mu[b] + self.mu_ref
        # reference_coral
        Z = ((X - self.mu[b]) / self.sd_ref) @ self.T[b]
        return Z * self.sd_ref + self.mu_ref


def compensate_all(X, batch, comp):
    """Apply a fitted compensator batch by batch."""
    X, batch = np.asarray(X, float), np.asarray(batch)
    out = np.empty_like(X)
    for b in np.unique(batch):
        m = batch == b
        out[m] = comp.transform(X[m], b)
    return out


# ----------------------------------------------------------------------
# Experiment 1: train on Batch 1 (80 %), test on Batch 1 hold-out (20 %)
#               and on every later batch (cross-batch testing)
# ----------------------------------------------------------------------
def run_setup_a(X, y, batch, method="none", model_names=MODEL_NAMES,
                seed=RANDOM_STATE, probability=False):
    X, y, batch = np.asarray(X, float), np.asarray(y), np.asarray(batch)
    comp = BatchCompensator(method).fit(X, batch)
    Xt = compensate_all(X, batch, comp)

    ref_idx = np.where(batch == REF_BATCH)[0]
    train_idx, test_idx = train_test_split(
        ref_idx, test_size=0.2, random_state=seed, stratify=y[ref_idx])

    scaler = StandardScaler().fit(Xt[train_idx])
    Xtr = scaler.transform(Xt[train_idx])

    preds, models = {}, {}
    for name in model_names:
        model = GasClassifier(name, seed, probability).fit(Xtr, y[train_idx])
        p = {1: (y[test_idx], model.predict(scaler.transform(Xt[test_idx])))}
        for k in BATCHES[1:]:
            idx = np.where(batch == k)[0]
            p[k] = (y[idx], model.predict(scaler.transform(Xt[idx])))
        preds[name], models[name] = p, model
    return dict(preds=preds, models=models, comp=comp, scaler=scaler,
                Xt=Xt, train_idx=train_idx, test_idx=test_idx)


def batch_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y_true, y_pred),
                precision=p, recall=r, f1=f)


def metrics_table(preds_one_model):
    """preds_one_model: {batch: (y_true, y_pred)} -> DataFrame (Batch 1..10)."""
    rows = [dict(Batch=k, **batch_metrics(*preds_one_model[k])) for k in BATCHES]
    return pd.DataFrame(rows)


def pooled(preds_one_model, batches=BATCHES[1:]):
    yt = np.concatenate([preds_one_model[k][0] for k in batches])
    yp = np.concatenate([preds_one_model[k][1] for k in batches])
    return yt, yp


# ----------------------------------------------------------------------
# Drift detection
# ----------------------------------------------------------------------
def ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (0 = identical, 1 = disjoint)."""
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def mean_ks(A, B):
    return float(np.mean([ks_stat(A[:, j], B[:, j]) for j in range(A.shape[1])]))


def drift_table(X, y, batch, ref=REF_BATCH, min_class_samples=5):
    """Drift of every batch relative to the reference batch.

    mean_shift : mean_j |mu_bj - mu_refj| / sd_refj
    var_shift  : mean_j |log(var_bj / var_refj)|
    ks_pooled  : mean KS statistic over sensors (all gases pooled)
    drift_score: mean KS statistic over sensors AND gases, computed per gas so
                 that changes in gas mixture between batches are not mistaken
                 for sensor drift  (0 = no drift, 1 = complete drift)
    """
    X, y, batch = np.asarray(X, float), np.asarray(y), np.asarray(batch)
    Xr, yr = X[batch == ref], y[batch == ref]
    mu_r, sd_r, var_r = Xr.mean(0), Xr.std(0, ddof=1) + 1e-12, Xr.var(0, ddof=1) + 1e-12
    rows = []
    for b in BATCHES:
        Xb, yb = X[batch == b], y[batch == b]
        per_gas = []
        for g in GAS_LABELS:
            A, B = Xr[yr == g], Xb[yb == g]
            if len(A) >= min_class_samples and len(B) >= min_class_samples:
                per_gas.append(mean_ks(A, B))
        rows.append(dict(
            Batch=b,
            mean_shift=float(np.mean(np.abs(Xb.mean(0) - mu_r) / sd_r)),
            var_shift=float(np.mean(np.abs(np.log((Xb.var(0, ddof=1) + 1e-12) / var_r)))),
            ks_pooled=mean_ks(Xr, Xb),
            drift_score=float(np.mean(per_gas)) if per_gas else np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Statistical validation helpers
# ----------------------------------------------------------------------
def paired_stats(before, after):
    before, after = np.asarray(before, float), np.asarray(after, float)
    d = after - before
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
    t_stat, p_t = (stats.ttest_rel(after, before) if sd > 0 else (np.nan, np.nan))
    try:
        w_stat, p_w = stats.wilcoxon(after, before)
    except ValueError:
        w_stat, p_w = np.nan, np.nan
    return dict(n=n,
                mean_before=before.mean(), std_before=before.std(ddof=1),
                mean_after=after.mean(), std_after=after.std(ddof=1),
                mean_improvement=d.mean(), std_improvement=sd,
                ci_low=d.mean() - tcrit * se, ci_high=d.mean() + tcrit * se,
                t_stat=t_stat, p_ttest=p_t, w_stat=w_stat, p_wilcoxon=p_w,
                cohen_dz=(d.mean() / sd) if sd > 0 else np.nan,
                batches_improved=int((d > 0).sum()))


def mcnemar_exact(y, pred_before, pred_after):
    ok_b, ok_a = (y == pred_before), (y == pred_after)
    only_before = int((ok_b & ~ok_a).sum())   # compensation broke these
    only_after = int((~ok_b & ok_a).sum())    # compensation fixed these
    n = only_before + only_after
    p = stats.binomtest(min(only_before, only_after), n, 0.5).pvalue if n > 0 else 1.0
    return dict(fixed_by_compensation=only_after, broken_by_compensation=only_before,
                p_value=float(p))


def bootstrap_improvement(y, pred_before, pred_after, n_boot=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    diff = (y == pred_after).astype(float) - (y == pred_before).astype(float)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)), dtype=np.int32)
    boots = diff[idx].mean(axis=1)
    return dict(mean=float(boots.mean()),
                ci_low=float(np.percentile(boots, 2.5)),
                ci_high=float(np.percentile(boots, 97.5)))


def multi_seed_accuracy(X, y, batch, method, model_name, seeds=(0, 1, 2, 3, 4)):
    """Repeat the whole experiment with different random splits / model seeds.
    Returns array [n_seeds, 9] of accuracies for batches 2..10."""
    rows = []
    for s in seeds:
        res = run_setup_a(X, y, batch, method, [model_name], seed=s)
        rows.append([accuracy_score(*res["preds"][model_name][k]) for k in BATCHES[1:]])
    return np.array(rows)


# ----------------------------------------------------------------------
# Final deployable pipeline
# Input -> Preprocessing -> Drift Compensation -> Selected Model -> Gas Class
# ----------------------------------------------------------------------
class GasDriftPipeline:
    def __init__(self, comp, scaler, model, method, model_name):
        self.comp, self.scaler, self.model = comp, scaler, model
        self.method, self.model_name = method, model_name

    @staticmethod
    def preprocess(sensors):
        X = np.atleast_2d(np.asarray(sensors, dtype=float))
        if X.shape[1] != 128:
            raise ValueError("Exactly 128 sensor values per sample are required")
        if not np.isfinite(X).all():
            raise ValueError("Sensor values must be finite numbers")
        return X

    def predict(self, sensors, batch=None):
        X = self.preprocess(sensors)                                 # 1. preprocessing
        if batch is not None and self.method != "none":
            X = self.comp.transform(X, batch)                        # 2. drift compensation
        Xs = self.scaler.transform(X)                                # 3. scaling
        labels = self.model.predict(Xs)                              # 4. selected model
        proba = self.model.predict_proba(Xs)
        return labels, proba                                         # 5. gas class

# ======================================================================
# Flask application
# ======================================================================
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
CACHE_FILE = BASE_DIR / "artifacts" / "analysis_cache.json"
CACHE_VERSION = "drift-v1"

app = Flask(__name__)


def pct(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x) * 100, nd)


def num(x, nd=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_dataset():
    X_parts, y_parts, b_parts = [], [], []
    for b in BATCHES:
        path = RAW_DIR / f"batch{b}.dat"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Put the UCI Gas Sensor Array Drift files "
                f"batch1.dat ... batch10.dat inside the 'raw' folder next to app.py.")
        raw = pd.read_csv(path, sep=" ", header=None).dropna(axis=1, how="all")
        labels = raw[0].astype(str).str.split(";").str[0].astype(int).to_numpy()
        feats = np.column_stack([raw[i].astype(str).str.split(":").str[1].astype(float)
                                 for i in range(1, 129)])
        X_parts.append(feats), y_parts.append(labels)
        b_parts.append(np.full(len(labels), b))
    return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(b_parts)


def data_fingerprint():
    sizes = [(RAW_DIR / f"batch{b}.dat").stat().st_size for b in BATCHES]
    return {"version": CACHE_VERSION, "sizes": sizes}


# ----------------------------------------------------------------------
# Full analysis (everything shown on the dashboard)
# ----------------------------------------------------------------------
def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=GAS_LABELS).tolist()


def histogram_pair(ref, cur, bins=20):
    lo, hi = np.percentile(np.concatenate([ref, cur]), [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, bins + 1)
    h_ref, _ = np.histogram(ref, bins=edges, density=False)
    h_cur, _ = np.histogram(cur, bins=edges, density=False)
    return dict(edges=[num(e, 4) for e in edges],
                ref=(h_ref / max(len(ref), 1) * 100).round(2).tolist(),
                cur=(h_cur / max(len(cur), 1) * 100).round(2).tolist())


def pca_points(Xt, y, batch, shown=(1, 2, 5, 7, 10), per_batch=100):
    from sklearn.decomposition import PCA
    ref = batch == REF_BATCH
    sc = StandardScaler().fit(Xt[ref])
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(sc.transform(Xt[ref]))
    rng = np.random.default_rng(RANDOM_STATE)
    pts = []
    for b in shown:
        idx = np.where(batch == b)[0]
        idx = rng.choice(idx, size=min(per_batch, len(idx)), replace=False)
        P = pca.transform(sc.transform(Xt[idx]))
        pts += [dict(batch=int(b), gas=int(y[i]), x=num(p[0], 3), y=num(p[1], 3))
                for i, p in zip(idx, P)]
    return dict(explained=[num(v, 4) for v in pca.explained_variance_ratio_], points=pts)


def compute_analysis(X, y, batch):
    print("  [1/6] training models for every compensation method ...")
    runs = {m: run_setup_a(X, y, batch, m) for m in COMPENSATION_METHODS}
    base = runs["none"]

    # ---- baseline + model selection -----------------------------------
    base_tables = {m: metrics_table(base["preds"][m]) for m in MODEL_NAMES}
    baseline_metrics = {m: {k: pct(base_tables[m].loc[0, k])
                            for k in ("accuracy", "precision", "recall", "f1")}
                        for m in MODEL_NAMES}
    baseline_conf = {m: confusion(*base["preds"][m][1]) for m in MODEL_NAMES}
    selection = []
    for m in MODEL_NAMES:
        accs = base_tables[m]["accuracy"].to_numpy()
        selection.append(dict(model=m, baseline=pct(accs[0]), avg_cross=pct(accs[1:].mean()),
                              change=pct(accs[1:].mean() - accs[0]),
                              worst=pct(accs[1:].min())))
    selected = max(selection, key=lambda r: r["avg_cross"])["model"]
    cross_batch = [dict(batch=k, **{m: pct(base_tables[m].loc[k - 1, "accuracy"]) for m in MODEL_NAMES},
                        **{m + "_change": pct(base_tables[m].loc[k - 1, "accuracy"] - base_tables[m].loc[0, "accuracy"])
                           for m in MODEL_NAMES}) for k in BATCHES[1:]]

    # ---- compensation methods compared --------------------------------
    print("  [2/6] comparing compensation methods ...")
    method_rows, per_method_acc = [], {}
    for meth in COMPENSATION_METHODS:
        tabs = {m: metrics_table(runs[meth]["preds"][m]) for m in MODEL_NAMES}
        per_method_acc[meth] = tabs[selected]["accuracy"].to_numpy()
        method_rows.append(dict(
            method=meth, label=COMPENSATION_METHODS[meth],
            per_batch=[pct(a) for a in tabs[selected]["accuracy"]],
            avg_selected=pct(tabs[selected]["accuracy"].to_numpy()[1:].mean()),
            avg_all={m: pct(tabs[m]["accuracy"].to_numpy()[1:].mean()) for m in MODEL_NAMES}))
    best_method = max([m for m in COMPENSATION_METHODS if m != "none"],
                      key=lambda m: per_method_acc[m][1:].mean())
    after = runs[best_method]

    # ---- before / after for the selected model -------------------------
    tb, ta = metrics_table(base["preds"][selected]), metrics_table(after["preds"][selected])
    yb, pb = pooled(base["preds"][selected])
    ya, pa = pooled(after["preds"][selected])
    ps_b = precision_recall_fscore_support(yb, pb, labels=GAS_LABELS, zero_division=0)
    ps_a = precision_recall_fscore_support(ya, pa, labels=GAS_LABELS, zero_division=0)
    before_after = [dict(batch=k, acc_before=pct(tb.loc[k - 1, "accuracy"]), acc_after=pct(ta.loc[k - 1, "accuracy"]),
                         improvement=pct(ta.loc[k - 1, "accuracy"] - tb.loc[k - 1, "accuracy"]),
                         f1_before=pct(tb.loc[k - 1, "f1"]), f1_after=pct(ta.loc[k - 1, "f1"]))
                    for k in BATCHES]
    prf = {"before": batch_metrics(yb, pb), "after": batch_metrics(ya, pa)}
    prf = {s: {k: pct(v) for k, v in d.items()} for s, d in prf.items()}
    per_gas = [dict(gas=GAS_NAMES[g], recall_before=pct(ps_b[1][i]), recall_after=pct(ps_a[1][i]),
                    f1_before=pct(ps_b[2][i]), f1_after=pct(ps_a[2][i])) for i, g in enumerate(GAS_LABELS)]
    all_models_ba = {m: dict(
        before=[pct(a) for a in metrics_table(base["preds"][m])["accuracy"]],
        after=[pct(a) for a in metrics_table(after["preds"][m])["accuracy"]]) for m in MODEL_NAMES}

    # ---- drift detection ------------------------------------------------
    print("  [3/6] drift detection (KS tests) ...")
    d_raw = drift_table(X, y, batch)
    d_comp = {meth: drift_table(runs[meth]["Xt"], y, batch) for meth in COMPENSATION_METHODS if meth != "none"}
    mean_b = np.array([X[batch == b].mean(0) for b in BATCHES])
    std_b = np.array([X[batch == b].std(0, ddof=1) for b in BATCHES])
    acc_raw = tb["accuracy"].to_numpy()[1:]
    ds = d_raw["drift_score"].to_numpy()[1:]
    pear, spear = stats.pearsonr(ds, acc_raw), stats.spearmanr(ds, acc_raw)

    print("  [4/6] distribution shift + PCA ...")
    hist = {}
    for s in range(4):
        hist[f"sensor_{s + 1}"] = {}
        for k in BATCHES[1:]:
            r_, c_ = X[batch == REF_BATCH, s], X[batch == k, s]
            r2, c2 = after["Xt"][batch == REF_BATCH, s], after["Xt"][batch == k, s]
            hist[f"sensor_{s + 1}"][str(k)] = dict(raw=histogram_pair(r_, c_), comp=histogram_pair(r2, c2))
    pca = dict(before=pca_points(X, y, batch), after=pca_points(after["Xt"], y, batch))

    # ---- statistical validation ----------------------------------------
    print("  [5/6] statistical validation (multi-seed, tests, bootstrap) ...")
    PCT_KEYS = {"mean_before", "std_before", "mean_after", "std_after", "mean_improvement",
                "std_improvement", "ci_low", "ci_high"}      # accuracies -> percent

    def clean(d):
        return {k: (num(v * 100 if k in PCT_KEYS else v, 6) if isinstance(v, (float, np.floating))
                    else (int(v) if isinstance(v, (np.integer,)) else v))
                for k, v in d.items()}
    ps_main = paired_stats(acc_raw, ta["accuracy"].to_numpy()[1:])
    ps_methods = []
    for meth in COMPENSATION_METHODS:
        if meth == "none":
            continue
        s = paired_stats(acc_raw, per_method_acc[meth][1:])
        ps_methods.append(dict(method=COMPENSATION_METHODS[meth], **clean(s)))
    seeds = (0, 1, 2, 3, 4)
    ms_b = multi_seed_accuracy(X, y, batch, "none", selected, seeds)
    ms_a = multi_seed_accuracy(X, y, batch, best_method, selected, seeds)
    seed_stats = paired_stats(ms_b.mean(1), ms_a.mean(1))
    mcn = mcnemar_exact(yb, pb, pa)
    boot = bootstrap_improvement(yb, pb, pa)
    statistics = dict(
        batch_level=clean(ps_main), methods=ps_methods,
        seeds=dict(seeds=list(seeds), before=[pct(v) for v in ms_b.mean(1)], after=[pct(v) for v in ms_a.mean(1)],
                   before_mean=pct(ms_b.mean()), before_std=pct(ms_b.mean(1).std(ddof=1)),
                   after_mean=pct(ms_a.mean()), after_std=pct(ms_a.mean(1).std(ddof=1)),
                   per_batch_before_mean=[pct(v) for v in ms_b.mean(0)], per_batch_before_std=[pct(v) for v in ms_b.std(0, ddof=1)],
                   per_batch_after_mean=[pct(v) for v in ms_a.mean(0)], per_batch_after_std=[pct(v) for v in ms_a.std(0, ddof=1)],
                   test=clean(seed_stats)),
        mcnemar=dict(mcn, p_value=num(mcn["p_value"], 12), samples=int(len(yb))),
        bootstrap=dict(mean=pct(boot["mean"]), ci_low=pct(boot["ci_low"]), ci_high=pct(boot["ci_high"])),
        drift_vs_accuracy=dict(pearson_r=num(pear[0]), pearson_p=num(pear[1], 6),
                               spearman_rho=num(spear[0]), spearman_p=num(spear[1], 6)))

    print("  [6/6] assembling payload ...")
    drift_rows = []
    for i, b in enumerate(BATCHES):
        row = dict(batch=b, mean_shift=num(d_raw.loc[i, "mean_shift"]), var_shift=num(d_raw.loc[i, "var_shift"]),
                   ks_pooled=num(d_raw.loc[i, "ks_pooled"]), drift_score=num(d_raw.loc[i, "drift_score"]))
        for meth, dc in d_comp.items():
            row["drift_score_" + meth] = num(dc.loc[i, "drift_score"])
        drift_rows.append(row)

    final_table = [dict(r, drift_before=drift_rows[r["batch"] - 1]["drift_score"],
                        drift_after=drift_rows[r["batch"] - 1]["drift_score_" + best_method])
                   for r in before_after]
    gas_counts = pd.Series(y).value_counts().sort_index()
    return dict(
        overview=dict(samples=int(len(y)), sensors=128, batches=int(len(BATCHES)), gas_classes=len(GAS_NAMES),
                      gas_distribution=[dict(gas=GAS_NAMES[int(k)], count=int(v)) for k, v in gas_counts.items()],
                      batch_sizes=[int((batch == b).sum()) for b in BATCHES],
                      batch_gas={GAS_NAMES[g]: [int(((batch == b) & (y == g)).sum()) for b in BATCHES] for g in GAS_LABELS}),
        selected_model=selected, best_method=best_method, best_method_label=COMPENSATION_METHODS[best_method],
        baseline=dict(metrics=baseline_metrics, confusion=baseline_conf, selection=selection),
        cross_batch=cross_batch,
        drift=dict(rows=drift_rows, sensor_mean=mean_b.round(4).tolist(), sensor_std=std_b.round(4).tolist(),
                   hist=hist, pca=pca, vs_accuracy=[dict(batch=int(k), drift=num(d), acc=pct(a))
                                                    for k, d, a in zip(BATCHES[1:], ds, acc_raw)]),
        compensation=dict(methods=method_rows, before_after=before_after, prf=prf, per_gas=per_gas,
                          confusion_before=confusion(yb, pb), confusion_after=confusion(ya, pa),
                          all_models=all_models_ba,
                          avg_before=pct(acc_raw.mean()), avg_after=pct(ta["accuracy"].to_numpy()[1:].mean()),
                          improvement=pct(ta["accuracy"].to_numpy()[1:].mean() - acc_raw.mean()),
                          relative_error_reduction=pct(1 - (1 - ta["accuracy"].to_numpy()[1:].mean()) / (1 - acc_raw.mean()))),
        statistics=statistics, final_table=final_table)


def get_analysis(X, y, batch):
    fp = data_fingerprint()
    if CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            if cached.get("fingerprint") == fp:
                print("Loaded cached analysis (delete artifacts/analysis_cache.json to recompute).")
                return cached["analysis"]
        except Exception:
            pass
    print("Running full drift analysis (first start only, results are cached) ...")
    analysis = compute_analysis(X, y, batch)
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"fingerprint": fp, "analysis": analysis}))
    return analysis


# ----------------------------------------------------------------------
# Start-up
# ----------------------------------------------------------------------
print("Loading dataset ...")
X_ALL, Y_ALL, B_ALL = load_dataset()
print("Dataset shape:", X_ALL.shape)
ANALYSIS = get_analysis(X_ALL, Y_ALL, B_ALL)
SELECTED_MODEL = ANALYSIS["selected_model"]
BEST_METHOD = ANALYSIS["best_method"]

print(f"Building final pipelines ({SELECTED_MODEL} + {BEST_METHOD}) ...")
_raw_run = run_setup_a(X_ALL, Y_ALL, B_ALL, "none", [SELECTED_MODEL], probability=True)
_cmp_run = run_setup_a(X_ALL, Y_ALL, B_ALL, BEST_METHOD, [SELECTED_MODEL], probability=True)
PIPE_RAW = GasDriftPipeline(_raw_run["comp"], _raw_run["scaler"], _raw_run["models"][SELECTED_MODEL], "none", SELECTED_MODEL)
PIPE_COMP = GasDriftPipeline(_cmp_run["comp"], _cmp_run["scaler"], _cmp_run["models"][SELECTED_MODEL], BEST_METHOD, SELECTED_MODEL)
print("Ready.  Selected model:", SELECTED_MODEL, "| Compensation:", BEST_METHOD)


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/overview")
def overview():
    comp = ANALYSIS["compensation"]
    return jsonify(dict(ANALYSIS["overview"], selected_model=SELECTED_MODEL, best_method=BEST_METHOD,
                        best_method_label=ANALYSIS["best_method_label"], avg_before=comp["avg_before"],
                        avg_after=comp["avg_after"], improvement=comp["improvement"],
                        relative_error_reduction=comp["relative_error_reduction"],
                        baseline_accuracy=ANALYSIS["baseline"]["metrics"][SELECTED_MODEL]["accuracy"]))


@app.route("/api/models")
def models_api():
    return jsonify(dict(metrics=ANALYSIS["baseline"]["metrics"], selection=ANALYSIS["baseline"]["selection"],
                        selected_model=SELECTED_MODEL))


@app.route("/api/confusion/<model_name>")
def confusion_api(model_name):
    if model_name not in ANALYSIS["baseline"]["confusion"]:
        return jsonify({"error": "Invalid model"}), 400
    return jsonify(dict(labels=[GAS_NAMES[i] for i in GAS_LABELS],
                        matrix=ANALYSIS["baseline"]["confusion"][model_name]))


@app.route("/api/cross-batch")
def cross_batch_api():
    return jsonify(ANALYSIS["cross_batch"])


@app.route("/api/eda")
def eda_api():
    df_all = pd.DataFrame(X_ALL, columns=SENSOR_COLS)
    sensor_statistics = [dict(sensor=c, mean=num(df_all[c].mean()), std=num(df_all[c].std()),
                              min=num(df_all[c].min()), max=num(df_all[c].max())) for c in SENSOR_COLS]
    corr = df_all.corr().fillna(0).round(3).values.tolist()
    histogram = {}
    for c in SENSOR_COLS[:4]:
        counts, bins = np.histogram(df_all[c], bins=20)
        histogram[c] = dict(bins=[num(v, 4) for v in bins], counts=counts.tolist())
    return jsonify(dict(sensor_statistics=sensor_statistics, correlation=corr, histogram=histogram,
                        batch_sizes=ANALYSIS["overview"]["batch_sizes"], batch_gas=ANALYSIS["overview"]["batch_gas"]))


@app.route("/api/drift")
def drift_api():
    d = ANALYSIS["drift"]
    return jsonify(dict(rows=d["rows"], sensor_mean=d["sensor_mean"], sensor_std=d["sensor_std"], pca=d["pca"],
                        vs_accuracy=d["vs_accuracy"], best_method=BEST_METHOD,
                        correlation=ANALYSIS["statistics"]["drift_vs_accuracy"]))


@app.route("/api/drift/distribution")
def drift_distribution():
    sensor, batch_no = request.args.get("sensor", "sensor_1"), request.args.get("batch", "10")
    try:
        return jsonify(ANALYSIS["drift"]["hist"][sensor][batch_no])
    except KeyError:
        return jsonify({"error": "Distribution available for sensor_1..sensor_4 and batches 2..10"}), 400


@app.route("/api/compensation")
def compensation_api():
    return jsonify(dict(ANALYSIS["compensation"], selected_model=SELECTED_MODEL, best_method=BEST_METHOD,
                        best_method_label=ANALYSIS["best_method_label"],
                        labels=[GAS_NAMES[i] for i in GAS_LABELS]))


@app.route("/api/statistics")
def statistics_api():
    return jsonify(dict(ANALYSIS["statistics"], selected_model=SELECTED_MODEL,
                        best_method_label=ANALYSIS["best_method_label"]))


@app.route("/api/pipeline")
def pipeline_api():
    return jsonify(dict(selected_model=SELECTED_MODEL, method=BEST_METHOD, method_label=ANALYSIS["best_method_label"],
                        final_table=ANALYSIS["final_table"], compensation=dict(
                            avg_before=ANALYSIS["compensation"]["avg_before"], avg_after=ANALYSIS["compensation"]["avg_after"],
                            improvement=ANALYSIS["compensation"]["improvement"])))


@app.route("/api/sample")
def sample_api():
    try:
        b = int(request.args.get("batch", 1))
    except ValueError:
        return jsonify({"error": "batch must be an integer 1-10"}), 400
    if b not in BATCHES:
        return jsonify({"error": "batch must be between 1 and 10"}), 400
    idx = np.random.choice(np.where(B_ALL == b)[0])
    return jsonify(dict(sensors=[float(v) for v in X_ALL[idx]], batch=b,
                        true_label=int(Y_ALL[idx]), true_gas=GAS_NAMES[int(Y_ALL[idx])]))


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No input data received"}), 400
    values = data.get("sensors")
    if values is None:
        return jsonify({"error": "Sensor values are required"}), 400
    if not isinstance(values, list) or len(values) != 128:
        return jsonify({"error": "Exactly 128 sensor values are required"}), 400
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        return jsonify({"error": "Sensor values must be numeric"}), 400
    batch_no = data.get("batch")
    try:
        batch_no = int(batch_no) if batch_no not in (None, "", "none") else None
    except (TypeError, ValueError):
        return jsonify({"error": "batch must be an integer between 1 and 10"}), 400
    if batch_no is not None and batch_no not in BATCHES:
        return jsonify({"error": "batch must be between 1 and 10"}), 400
    try:
        raw_label, raw_proba = PIPE_RAW.predict(values)
        if batch_no is None:
            comp_label, comp_proba, used = raw_label, raw_proba, False
        else:
            comp_label, comp_proba, used = (*PIPE_COMP.predict(values, batch_no), True)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    def pack(label, proba):
        return dict(label=int(label[0]), gas=GAS_NAMES[int(label[0])], confidence=pct(float(np.max(proba[0]))) if proba is not None else None,
                    probabilities={GAS_NAMES[int(c)]: pct(float(p)) for c, p in zip(PIPE_RAW.model.le.classes_, proba[0])} if proba is not None else None)
    final = pack(comp_label, comp_proba)
    return jsonify(dict(model=SELECTED_MODEL, compensation=ANALYSIS["best_method_label"] if used else "Not applied (no batch selected)",
                        compensated=used, batch=batch_no, prediction=final["gas"], gas_label=final["label"],
                        confidence=final["confidence"], probabilities=final["probabilities"],
                        uncompensated=pack(raw_label, raw_proba)))


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
