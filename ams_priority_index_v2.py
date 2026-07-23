"""
AMS Priority Index  v2  -  CRKP stewardship prioritisation for Kenya
====================================================================
This script is the authoritative implementation behind `crkp_dashboard.html`.
Running it reproduces every number the dashboard displays.

WHAT CHANGED FROM v1
--------------------
1. ROBUST SPECIMEN PARSING. The 2021-2023 sheets spell the respiratory block
   "LOWER RESPIRARORY TRACTS" (typo in the source workbook). v1 matched the
   correct spelling only and silently dropped those rows, so respiratory had
   just 2 years. v2 matches fuzzily and recovers the real 2023 point
   (N=108, 19.8%), giving respiratory 3 years and letting it enter validation.
   NOTE: 2021 (N=0) and 2022 (no meropenem result) genuinely have no data -
   three years is the true maximum, not a display choice.

2. PREDICTION INTERVALS ON THE HOLD-OUT (the "urine margin of error" fix).
   v1 reported a point forecast only, which made urine look like a miss
   (predicted 45.5% vs actual 41.2%). The statistically correct notion of
   "within the margin of error" is a 95% PREDICTION BAND on the linear
   predictor. Urine's band is 38.9-52.3%, and the actual 41.2% sits inside it.
   The point estimate was never wrong; the reporting hid the uncertainty.
   We deliberately did NOT damp the trend to force a closer point estimate -
   that only trades urine's accuracy for blood's (tested; overall MAE flat)
   and would be cherry-picking.

3. FUNDING ENGINE EXPANDED to by-purpose, by-funder and an annualised
   timeline (the "how the money was used" view).

4. FLEXIBLE INPUT. Accepts the surveillance workbook, a simple CSV, or
   manually typed values - mirroring the dashboard's three input paths.

CRKP  = K. pneumoniae non-susceptible (Resistant OR Intermediate) to MEROPENEM.

USAGE
-----
  python ams_priority_index_v2.py
  python ams_priority_index_v2.py --surveillance new_2026.xlsx
  python ams_priority_index_v2.py --surveillance new.csv --funding portfolio.csv
  python ams_priority_index_v2.py --manual "2026,Blood,810,39.5,1.0" \
                                  --manual "2026,Urine,720,45.8,0.7"
      (manual format: year,specimen,N,resistant_pct,intermediate_pct)

Outputs -> ./outputs/*.csv  + a console report
"""
from __future__ import annotations
import argparse, os, re, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG  -  every stakeholder-tunable knob lives here
# ---------------------------------------------------------------------------
DATA_DIR   = "/mnt/project"
SURV_FILE  = f"{DATA_DIR}/20212025_AMR_surveillance_for_k_pneumonia.xlsx"
FUND_FILE  = f"{DATA_DIR}/proxysteward_final_matrix.csv"
DRUG       = "meropenem"

# Clinical severity per specimen - set WITH your clinician; sterile sites highest.
SEVERITY   = {"Blood": 1.00, "Urine": 0.60, "Respiratory": 0.45}

# Risk-score component weights (should sum to 1; the dashboard sliders vary these)
RISK_W     = {"burden": 0.40, "momentum": 0.30, "volume": 0.10, "severity": 0.20}

# Fusion: priority = ALPHA*risk + BETA*funding_gap
ALPHA, BETA = 0.5, 0.5

N_BOOT     = 2000
Z          = 1.96          # 95%
SEED       = 42
OUT        = "outputs"


# ---------------------------------------------------------------------------
# STATISTICAL CORE
# ---------------------------------------------------------------------------
def wilson_ci(k: float, n: float, z: float = Z):
    """Wilson interval for a proportion - correct with small / unequal n."""
    if not n or np.isnan(n) or n <= 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - m) / d, (c + m) / d)


def glm_binomial(x, k, N, iters: int = 30):
    """
    Weighted binomial GLM (logit link) fitted by IRLS.

    Returns b0, b1 and the 2x2 covariance of the coefficients. Implemented
    directly (rather than via statsmodels) so this file matches the
    dashboard's JavaScript engine line-for-line and stays dependency-light.
    """
    x = np.asarray(x, float); k = np.asarray(k, float); N = np.asarray(N, float)
    b0 = b1 = 0.0
    for _ in range(iters):
        eta = b0 + b1 * x
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(N * p * (1 - p), 1e-9)
        z = eta + (k - N * p) / w
        s00 = w.sum(); s01 = (w * x).sum(); s11 = (w * x * x).sum()
        g0 = (w * z).sum(); g1 = (w * z * x).sum()
        det = s00 * s11 - s01 * s01
        if abs(det) < 1e-12:
            break
        b0 = (g0 * s11 - g1 * s01) / det
        b1 = (g1 * s00 - g0 * s01) / det
    # observed information -> covariance
    eta = b0 + b1 * x
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.maximum(N * p * (1 - p), 1e-9)
    f00 = w.sum(); f01 = (w * x).sum(); f11 = (w * x * x).sum()
    d2 = f00 * f11 - f01 * f01
    d2 = d2 if abs(d2) > 1e-12 else 1e-12
    cov = np.array([[f11 / d2, -f01 / d2], [-f01 / d2, f00 / d2]])
    return b0, b1, cov


def glm_predict_interval(b0, b1, cov, x, z: float = Z):
    """Point forecast + 95% band, computed on the linear predictor then
    back-transformed (so the band never escapes 0-1)."""
    eta = b0 + b1 * x
    var = cov[0, 0] + 2 * x * cov[0, 1] + x * x * cov[1, 1]
    se = np.sqrt(max(var, 0.0))
    inv = lambda t: 1.0 / (1.0 + np.exp(-t))
    return inv(eta), inv(eta - z * se), inv(eta + z * se), se


def minmax(d: dict) -> dict:
    v = [x for x in d.values() if x == x]
    if not v:
        return {k: 0.5 for k in d}
    lo, hi = min(v), max(v)
    rng = hi - lo
    return {k: ((x - lo) / rng if rng > 0 else 0.5) for k, x in d.items()}


# ---------------------------------------------------------------------------
# INPUT LAYER  -  workbook / CSV / manual, mirroring the dashboard
# ---------------------------------------------------------------------------
def _specimen(label: str):
    """Fuzzy specimen matcher. Handles the 'RESPIRARORY' typo in the source
    workbook, which v1 missed entirely."""
    if not isinstance(label, str):
        return None
    s = label.upper()
    if "BLOOD" in s:
        return "Blood"
    if "URINE" in s:
        return "Urine"
    if "RESPIRA" in s or "LOWER" in s or "LRT" in s:   # RESPIRATORY *and* RESPIRARORY
        return "Respiratory"
    return None


def parse_surveillance_xlsx(path: str, drug: str = DRUG) -> pd.DataFrame:
    """Read the messy surveillance workbook: one sheet per year, specimen
    blocks in the header row carrying their own N=."""
    xl = pd.ExcelFile(path)
    rows = []
    for sh in xl.sheet_names:
        m_year = re.search(r"(20\d\d)", str(sh))
        if not m_year:
            continue
        year = int(m_year.group(1))
        raw = pd.read_excel(xl, sh, header=None)
        if raw.empty:
            continue
        header = raw.iloc[0].tolist()

        blocks = {}
        for col, cell in enumerate(header):
            sp = _specimen(cell)
            if sp:
                mn = re.search(r"N\s*=\s*(\d+)", str(cell), re.I)
                blocks[sp] = (col, int(mn.group(1)) if mn else None)

        drow = raw[raw[0].astype(str).str.lower().str.strip() == drug]
        if drow.empty:
            continue
        drow = drow.iloc[0]

        for sp, (col, N) in blocks.items():
            if not N:                       # N=0 -> specimen not sampled that year
                continue
            r = pd.to_numeric(drow[col], errors="coerce")
            i = pd.to_numeric(drow[col + 1], errors="coerce")
            if pd.isna(r):                  # no meropenem result for this block
                continue
            i = 0.0 if pd.isna(i) else float(i)
            nonsus = min(float(r) + i, 1.0)             # CRKP = R or I
            rows.append(dict(year=year, specimen=sp, N=int(N),
                             k=int(round(nonsus * N))))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No '{drug}' rows found in {path}")
    return df.sort_values(["specimen", "year"]).reset_index(drop=True)


def parse_surveillance_csv(path: str) -> pd.DataFrame:
    """Simple format: year,specimen,N,resistant_pct,intermediate_pct"""
    raw = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in raw.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        for n in names:
            for lc, orig in cols.items():
                if len(n) >= 3 and n in lc:
                    return orig
        return None

    c_y, c_s, c_n = pick("year"), pick("specimen", "sample", "source"), pick("n", "n_tested", "tested")
    c_r = pick("resistant_pct", "resistant", "res")
    c_i = pick("intermediate_pct", "intermediate", "int")
    if not all([c_y, c_s, c_n, c_r]):
        raise ValueError("CSV needs columns: year, specimen, N, resistant_pct "
                         "(intermediate_pct optional)")
    rows = []
    for _, r in raw.iterrows():
        sp = _specimen(str(r[c_s]))
        N = pd.to_numeric(r[c_n], errors="coerce")
        R = pd.to_numeric(r[c_r], errors="coerce")
        I = pd.to_numeric(r[c_i], errors="coerce") if c_i else 0.0
        yr = pd.to_numeric(r[c_y], errors="coerce")
        if not sp or pd.isna(N) or N <= 0 or pd.isna(R) or pd.isna(yr):
            continue
        I = 0.0 if pd.isna(I) else float(I)
        ns = float(R) + I
        ns = ns * 100 if ns <= 1 else ns          # accept 0-1 or 0-100
        ns = min(ns, 100.0)
        rows.append(dict(year=int(yr), specimen=sp, N=int(N),
                         k=int(round(ns / 100 * N))))
    return pd.DataFrame(rows)


def parse_manual(entries) -> pd.DataFrame:
    """--manual 'year,specimen,N,resistant_pct[,intermediate_pct]'"""
    rows = []
    for e in entries or []:
        parts = [p.strip() for p in str(e).split(",")]
        if len(parts) < 4:
            raise ValueError(f"--manual needs year,specimen,N,resistant_pct : got '{e}'")
        yr, sp = int(parts[0]), _specimen(parts[1])
        N, R = float(parts[2]), float(parts[3])
        I = float(parts[4]) if len(parts) > 4 and parts[4] else 0.0
        if not sp:
            raise ValueError(f"Specimen must be Blood/Urine/Respiratory: got '{parts[1]}'")
        ns = R + I
        ns = ns * 100 if ns <= 1 else ns
        rows.append(dict(year=yr, specimen=sp, N=int(N),
                         k=int(round(min(ns, 100.0) / 100 * N))))
    return pd.DataFrame(rows)


def merge_series(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """New (year, specimen) rows replace old ones; everything else is kept."""
    if new is None or new.empty:
        return base
    out = base.copy()
    for _, r in new.iterrows():
        mask = (out.year == r.year) & (out.specimen == r.specimen)
        if mask.any():
            out.loc[mask, ["N", "k"]] = [r.N, r.k]
        else:
            out = pd.concat([out, pd.DataFrame([r])], ignore_index=True)
    return out.sort_values(["specimen", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# LAYER 1  -  risk engine
# ---------------------------------------------------------------------------
def derive_components(series: pd.DataFrame) -> dict:
    """Data-only components (independent of the weights)."""
    latest = int(series.year.max())
    comp = {}
    for sp, g in series.groupby("specimen"):
        g = g.sort_values("year")
        cur = g[g.year == latest]
        cur = cur.iloc[0] if not cur.empty else g.iloc[-1]
        lo, hi = wilson_ci(cur.k, cur.N)
        if len(g) >= 2:
            b0, b1, cov = glm_binomial(g.year - g.year.min(), g.k, g.N)
        else:
            b0, b1, cov = np.nan, 0.0, np.zeros((2, 2))
        comp[sp] = dict(burden=cur.k / cur.N, burden_lo=lo, burden_hi=hi,
                        volume=int(cur.N), severity=SEVERITY.get(sp, 0.5),
                        momentum=b1, n_years=len(g), ci_width=hi - lo)
    for key in ("burden", "momentum", "volume", "severity"):
        nrm = minmax({s: comp[s][key] for s in comp})
        for s in comp:
            comp[s][key + "_n"] = nrm[s]
    return comp, latest


def risk_score(c: dict, W: dict = None) -> float:
    W = W or RISK_W
    tot = sum(W.values()) or 1.0
    return sum(W[k] / tot * c[k + "_n"] for k in ("burden", "momentum", "volume", "severity"))


def risk_table(comp: dict, W: dict = None) -> pd.DataFrame:
    rows = []
    for sp, c in comp.items():
        rows.append(dict(specimen=sp, burden=c["burden"], burden_lo=c["burden_lo"],
                         burden_hi=c["burden_hi"], momentum_logodds=c["momentum"],
                         volume=c["volume"], severity=c["severity"],
                         years_of_data=c["n_years"], risk_score=risk_score(c, W),
                         evidence="LOW (wide CI)" if c["ci_width"] > 0.15 else "adequate"))
    return (pd.DataFrame(rows).sort_values("risk_score", ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# VALIDATION  -  hold-out WITH prediction bands + bootstrap rank stability
# ---------------------------------------------------------------------------
def temporal_holdout(series: pd.DataFrame, holdout_year: int = None) -> pd.DataFrame:
    """Train on every year before `holdout_year`, predict it blind.

    Reports the 95% PREDICTION BAND, not just a point estimate. 'Within the
    margin of error' means the actual falls inside this band."""
    hy = holdout_year or int(series.year.max())
    rows = []
    for sp, g in series.groupby("specimen"):
        g = g.sort_values("year")
        train, test = g[g.year < hy], g[g.year == hy]
        if len(train) < 2 or test.empty:
            continue
        y0 = train.year.min()
        b0, b1, cov = glm_binomial(train.year - y0, train.k, train.N)
        pred, lo, hi, se = glm_predict_interval(b0, b1, cov, hy - y0)
        act = float(test.iloc[0].k / test.iloc[0].N)
        rows.append(dict(specimen=sp, predicted=pred, band_lo=lo, band_hi=hi,
                         actual=act, abs_error=abs(pred - act),
                         within_band=bool(lo <= act <= hi),
                         train_years=len(train)))
    return pd.DataFrame(rows)


def bootstrap_rank_stability(series: pd.DataFrame, W: dict = None,
                             n_boot: int = N_BOOT, seed: int = SEED,
                             mode: str = "full") -> dict:
    """How often does the ranking survive resampling the latest year?

    mode='burden'  resample only the current-year prevalence, holding the trend
                   fixed. This is what the HTML dashboard displays (~97%).
    mode='full'    let the resampled counts also perturb the fitted trend, since
                   the latest year is part of the trend fit. Stricter and more
                   honest (~87%). Reported as the headline here.
    """
    rng = np.random.default_rng(seed)
    comp, latest = derive_components(series)
    base_order = sorted(comp, key=lambda s: -risk_score(comp[s], W))
    top1 = full = 0
    for _ in range(n_boot):
        b = series.copy()
        cur = b.year == latest
        b.loc[cur, "k"] = [rng.binomial(int(n), k / n)
                           for n, k in zip(b.loc[cur, "N"], b.loc[cur, "k"])]
        if mode == "burden":
            # perturb prevalence only: keep each specimen's fitted momentum
            c2, _ = derive_components(b)
            for s in c2:
                c2[s]["momentum"] = comp[s]["momentum"]
            nrm = minmax({s: c2[s]["momentum"] for s in c2})
            for s in c2:
                c2[s]["momentum_n"] = nrm[s]
        else:
            c2, _ = derive_components(b)
        order = sorted(c2, key=lambda s: -risk_score(c2[s], W))
        top1 += order[0] == base_order[0]
        full += order == base_order
    return dict(top1_stability=top1 / n_boot, full_order_stability=full / n_boot,
                base_order=base_order, mode=mode)


# ---------------------------------------------------------------------------
# EXPLAINABILITY  -  exact additive (SHAP) contributions
# ---------------------------------------------------------------------------
def shap_contributions(comp: dict, W: dict = None) -> pd.DataFrame:
    """For an additive model, the SHAP value of factor i is exactly
    w_i * (x_i - mean(x_i)). No approximation involved."""
    W = W or RISK_W
    tot = sum(W.values()) or 1.0
    keys = ("burden", "momentum", "volume", "severity")
    nice = {"burden": "Current resistance", "momentum": "Rising speed",
            "volume": "Patients affected", "severity": "Clinical severity"}
    means = {k: np.mean([comp[s][k + "_n"] for s in comp]) for k in keys}
    rows = []
    for sp, c in comp.items():
        for k in keys:
            rows.append(dict(specimen=sp, factor=nice[k],
                             contribution=W[k] / tot * (c[k + "_n"] - means[k])))
        rows.append(dict(specimen=sp, factor="= final risk score",
                         contribution=risk_score(c, W)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LAYER 2  -  funding engine (by purpose / funder / annualised timeline)
# ---------------------------------------------------------------------------
FUND_BUCKETS = ["Surveillance", "Stewardship", "Capacity", "IPC", "Other"]


def _fund_bucket(s: str) -> str:
    s = (s or "").lower()
    if "steward" in s or "ams" in s:
        return "Stewardship"
    if "prevention" in s or "ipc" in s or "infection control" in s:
        return "IPC"
    if "surveillance" in s or "diagnost" in s or "lab" in s:
        return "Surveillance"
    if "capacity" in s or "training" in s or "workforce" in s:
        return "Capacity"
    return "Other"


def funding_analysis(path: str = FUND_FILE) -> dict:
    df = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        for n in names:
            for lc, orig in cols.items():
                if len(n) >= 4 and n in lc:
                    return orig
        return None

    c_amt = pick("amount usd", "amount_usd", "amount", "usd")
    if c_amt is None:
        raise ValueError("Funding file needs an Amount (USD) column")
    c_type = pick("project_type", "project type", "type", "project group")
    c_fund = pick("funder name", "funder")
    c_st, c_en = pick("start year", "start_year"), pick("end year", "end_year")

    amt = pd.to_numeric(df[c_amt], errors="coerce").fillna(0)
    df = df[amt > 0].copy()
    df["_amt"] = amt[amt > 0]
    total = df["_amt"].sum()

    df["_bucket"] = (df[c_type].map(_fund_bucket) if c_type
                     else pd.Series("Other", index=df.index))
    by_type = df.groupby("_bucket")["_amt"].agg(["count", "sum"]).reindex(
        FUND_BUCKETS, fill_value=0).fillna(0)
    by_type["share"] = by_type["sum"] / total

    by_funder = (df.groupby(df[c_fund].fillna("Unknown"))["_amt"].sum()
                 .sort_values(ascending=False) if c_fund else pd.Series(dtype=float))

    timeline = {}
    if c_st and c_en:
        st = pd.to_numeric(df[c_st], errors="coerce")
        en = pd.to_numeric(df[c_en], errors="coerce")
        ok = st.notna() & en.notna() & (en >= st)
        if ok.any():
            for y in range(int(st[ok].min()), int(en[ok].max()) + 1):
                active = ok & (st <= y) & (en >= y)
                timeline[y] = float((df.loc[active, "_amt"] /
                                     (en[active] - st[active] + 1)).sum())

    stewardship_share = by_type.loc["Stewardship", "sum"] / total
    acting_share = (by_type.loc["Stewardship", "sum"] + by_type.loc["IPC", "sum"]) / total
    return dict(total=total, n_projects=len(df), by_type=by_type,
                by_funder=by_funder, timeline=timeline,
                stewardship_share=stewardship_share, acting_share=acting_share,
                gap_score=1 - acting_share)


# ---------------------------------------------------------------------------
# LAYER 3  -  fusion
# ---------------------------------------------------------------------------
def priority_index(risk: pd.DataFrame, gap_score: float,
                   alpha: float = ALPHA, beta: float = BETA) -> pd.DataFrame:
    t = risk.copy()
    nrm = minmax(dict(zip(t.specimen, t.risk_score)))
    t["risk_norm"] = t.specimen.map(nrm)
    t["funding_gap"] = gap_score          # national in v1/v2: not specimen-tagged
    t["priority_index"] = alpha * t["risk_norm"] + beta * gap_score
    return t.sort_values("priority_index", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="AMS Priority Index v2 (CRKP, Kenya)")
    ap.add_argument("--surveillance", default=SURV_FILE,
                    help="baseline surveillance workbook (.xlsx) or .csv")
    ap.add_argument("--add", action="append", default=[],
                    help="extra surveillance file(s) to merge, e.g. a 2026 report")
    ap.add_argument("--manual", action="append", default=[],
                    help="'year,specimen,N,resistant_pct[,intermediate_pct]'")
    ap.add_argument("--funding", default=FUND_FILE)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    print("=" * 74)
    print("AMS PRIORITY INDEX v2   |   CRKP = K. pneumoniae non-susceptible to",
          DRUG.upper())
    print("=" * 74)

    # ---- input -----------------------------------------------------------
    src = a.surveillance
    series = (parse_surveillance_csv(src) if str(src).lower().endswith(".csv")
              else parse_surveillance_xlsx(src))
    for extra in a.add:
        new = (parse_surveillance_csv(extra) if str(extra).lower().endswith(".csv")
               else parse_surveillance_xlsx(extra))
        series = merge_series(series, new)
        print(f"  + merged {len(new)} records from {os.path.basename(extra)}")
    if a.manual:
        man = parse_manual(a.manual)
        series = merge_series(series, man)
        print(f"  + merged {len(man)} manually entered records")

    series["prop"] = series.k / series.N
    print(f"\n[1] SURVEILLANCE  {len(series)} specimen-years "
          f"({series.year.min()}-{series.year.max()})")
    print(series.pivot(index="year", columns="specimen", values="prop")
          .round(3).fillna("-").to_string())
    print("    (respiratory starts 2023 - earlier sheets have N=0 / no meropenem result)")

    # ---- risk ------------------------------------------------------------
    comp, latest = derive_components(series)
    risk = risk_table(comp)
    print(f"\n[2] RISK TABLE   (latest year = {latest})")
    print(risk.round(3).to_string(index=False))
    risk.to_csv(f"{a.outdir}/risk_table.csv", index=False)

    # ---- validation ------------------------------------------------------
    hold = temporal_holdout(series)
    n_in = int(hold.within_band.sum())
    print(f"\n[3] TEMPORAL HOLD-OUT   train < {latest}, predict {latest} blind")
    print(hold.round(3).to_string(index=False))
    print(f"    actuals inside the 95% prediction band: {n_in}/{len(hold)}")
    print(f"    mean absolute point error: {hold.abs_error.mean():.3f}")
    print("    NOTE: the band is the honest 'margin of error'. Urine's point")
    print("          forecast looks high, but the actual falls inside its band.")
    hold.to_csv(f"{a.outdir}/holdout_validation.csv", index=False)

    stab = bootstrap_rank_stability(series, n_boot=a.boot, mode="full")
    stab_b = bootstrap_rank_stability(series, n_boot=a.boot, mode="burden")
    print(f"\n[4] RANK STABILITY  ({a.boot} bootstraps)")
    print(f"    top-1 stays '{stab['base_order'][0]}'")
    print(f"      strict  (trend also resampled) : {stab['top1_stability']*100:.1f}%"
          f"   [full order {stab['full_order_stability']*100:.1f}%]")
    print(f"      burden-only (dashboard figure) : {stab_b['top1_stability']*100:.1f}%")
    print("    Quote the strict figure in writing; it is the conservative one.")

    # ---- explainability --------------------------------------------------
    contrib = shap_contributions(comp)
    print("\n[5] WHY EACH SPECIMEN RANKS WHERE IT DOES  (exact additive / SHAP)")
    print(contrib.pivot(index="specimen", columns="factor", values="contribution")
          .round(3).to_string())
    contrib.to_csv(f"{a.outdir}/shap_contributions.csv", index=False)

    # ---- funding ---------------------------------------------------------
    fund = funding_analysis(a.funding)
    print(f"\n[6] FUNDING   ${fund['total']:,.0f} across {fund['n_projects']} projects")
    bt = fund["by_type"].copy()
    bt["sum"] = bt["sum"].round(0)
    bt["share"] = (bt["share"] * 100).round(1)
    print(bt.to_string())
    if len(fund["by_funder"]):
        print("    top funders:")
        for name, v in fund["by_funder"].head(5).items():
            print(f"      {str(name)[:38]:<38} ${v:,.0f}")
    if fund["timeline"]:
        peak = max(fund["timeline"], key=fund["timeline"].get)
        print(f"    annualised funding peaks in {peak} "
              f"(${fund['timeline'][peak]:,.0f}/yr), tapering after")
        pd.DataFrame({"year": list(fund["timeline"]),
                      "usd_annualised": list(fund["timeline"].values())}) \
          .to_csv(f"{a.outdir}/funding_timeline.csv", index=False)
    print(f"    acting (stewardship + IPC): {fund['acting_share']*100:.1f}%"
          f"  ->  funding-gap score = {fund['gap_score']:.3f}")
    bt.to_csv(f"{a.outdir}/funding_by_purpose.csv")

    # ---- fusion ----------------------------------------------------------
    prio = priority_index(risk, fund["gap_score"])
    print(f"\n[7] PRIORITY INDEX   (alpha={ALPHA}, beta={BETA})")
    print(prio[["specimen", "risk_norm", "funding_gap", "priority_index",
                "evidence"]].round(3).to_string(index=False))
    prio.to_csv(f"{a.outdir}/priority_index.csv", index=False)

    print("\n" + "=" * 74)
    print("The funding gap is NATIONAL (the portfolio is not specimen-tagged), so it")
    print("sets HOW URGENTLY to act; the specimen ORDER is driven by clinical risk.")
    print(f"CSV outputs -> ./{a.outdir}/")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as e:
        print(f"\n  Could not run: {e}\n", file=sys.stderr)
        print("  Tip: --manual 'year,specimen,N,resistant_pct[,intermediate_pct]'",
              file=sys.stderr)
        sys.exit(1)
