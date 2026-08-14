# =============================================================================
# Atkins Consulting - Renewal Intelligence
# Renewal-Trajectory Cohort Agent - Interactive Prototype (Streamlit v2)
# =============================================================================
#
# WHAT THIS FILE IS
# -----------------
# A single-file, branded, multi-page version of the row-13 prototype from the
# project design summary. The MODEL LOGIC is unchanged from the batch/v1 app:
# same four-table synthetic data, same Layers 1-3 and the Layer-4 placeholder,
# same per-term modeling and provisional outcome bands. Everything new here is
# presentation: a company identity, real page-to-page navigation, red/amber/
# green risk signalling, hover explanations on columns, a full sortable account
# portfolio, and a single Assumptions page that carries every caveat.
#
# HOW IT IS ORGANISED (read top to bottom)
# ----------------------------------------
#   PART A - Configuration defaults + brand tokens + colour thresholds
#   PART B - The engine: pure functions, no web code (unchanged model logic)
#   PART C - Caching wrapper so navigation is instant
#   PART D - The web pages (branding, navigation, and each screen)
#
# You do NOT need to read PART B to change how it behaves. Use the controls at the top of the Data page.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
import base64, os


# =============================================================================
# PART A - CONFIGURATION DEFAULTS + BRAND + THRESHOLDS
# =============================================================================
# The synthetic-data settings (mirror "SECTION 0" of the batch script). These
# are only starting values; the Data-page controls change them live.

DEFAULT_N_ACCOUNTS    = 800            # historical accounts used to learn the pattern
DEFAULT_N_LIVE        = 100            # live accounts to score in the portfolio (starting point)
FEATURES              = [f"feature_{i+1}" for i in range(8)]
# Look-back time-frame horizons, in QUARTERS: 12mo (4q), 24mo (8q), 36mo (12q).
# We frame these as observation horizons, not contract terms - see Assumptions.
TERM_CHOICES          = [4, 8, 12]
TERM_LABEL            = {4: "12-month", 8: "24-month", 12: "36-month"}
TERM_LABEL_LONG       = {4: "12-month (4 quarters)", 8: "24-month (8 quarters)",
                         12: "36-month (12 quarters)"}
# Mix across horizons, weighted to the short end to reflect where AI-era buying is
# heading (shorter, committed-consumption horizons). 12mo / 24mo / 36mo.
DEFAULT_TERM_PROBS    = [0.55, 0.30, 0.15]
# Assumed years of usable renewal-trajectory history a company brings. Four years
# gives the 12mo band ~4 completed cycles, 24mo ~2, and 36mo at least ~1 rolling
# cycle that keeps firming up over time. Documented on the Assumptions page.
HISTORY_YEARS         = 4
DEFAULT_NOISE_SHARE   = 0.12           # outcome deliberately contradicts usage
DEFAULT_WINNER_SHARE  = 0.70           # winner vs faller usage-shape split
SEGMENTS              = ["SMB", "Mid", "Enterprise"]
DEFAULT_SEGMENT_PROBS = [0.50, 0.35, 0.15]
SEGMENT_COMMIT        = {"SMB": 50_000, "Mid": 200_000, "Enterprise": 800_000}
INDUSTRIES            = ["SaaS", "Fintech", "Healthcare", "Retail", "Logistics", "Media"]

# --- Per-page background photos ---------------------------------------------
# A different muted photo sits behind each page. Each value can be EITHER a local
# file path (recommended for a live demo - it gets embedded, so nothing loads
# from the internet) OR a full https URL. Missing/blank entries fall back to a
# plain wash, so the app still runs before you add any images.
#
# To use your own: make a folder called "images" next to app.py and drop in six
# royalty-free photos (Pexels.com or Unsplash.com) named as below. Landscape
# shots of people collaborating work best. Or paste a URL in place of the path.
PAGE_BACKGROUNDS = {
    "Home":         "images/home.jpg",
    "The Model":    "images/model.jpg",
    "Portfolio":    "images/portfolio.jpg",
    "Account Plan": "images/account.jpg",
    "Data":         "images/data.jpg",
    "Assumptions":  "images/assumptions.jpg",
}
# How strongly the photo is muted. 0.90 = a faint ghost behind the page.
# Lower it (e.g. 0.82) to show more of the photo; raise it (0.95) for subtler.
BG_SCRIM = 0.60

# Brand logos. MAIN_LOGO heads the first (Home) page; GENERAL_LOGO heads every
# other page. Put the two PNGs in the images/ folder next to this file. If a file
# is missing the app falls back to the text brand, so it still runs.
MAIN_LOGO    = "images/Main_Logo.png"
GENERAL_LOGO = "images/General_Logo.png"

# --- Brand tokens ------------------------------------------------------------
BRAND       = "Atkins Consulting"
BRAND_MARK  = "AC"
TAGLINE     = "Adoption and Value Realization for Customer Success and Forward Deployed Engineering"

INK    = "#0E2233"   # deep slate navy - primary
SURFACE= "#F4F6F8"   # cool off-white background
CARD   = "#FFFFFF"
ACCENT = "#0E7C86"   # restrained teal - the one brand accent
LINE   = "#DDE4EA"
MUTE   = "#5A6B78"
GREEN  = "#1E8E5A"   # on track to renew
AMBER  = "#C77D0A"   # needs attention
RED    = "#C0392B"   # at risk

# --- Red / Amber / Green thresholds -----------------------------------------
# Severity is measured in "spreads" (IQRs) outside the winning band. A driver is
# only recorded once it is more than 0.25 spreads off. These two lines decide a
# single DRIVER's colour in the action plan, and are adjustable (Assumptions).
YELLOW_SEV = 0.25    # at/above this and below RED = amber (needs attention)
RED_SEV    = 1.00    # at/above this = red (clearly off target)

# Account-level risk bands are calibrated to a realistic, hopeful operating book,
# not set by "any one signal is off". Positioning guideline (see Assumptions):
#   At risk (red)        ~6-10% of live accounts at any point
#   Needs attention (amber) ~8-15%
#   On track (green)     the remainder, ~75-86%
# We size the two off-target tiers to these target shares by ranking accounts on
# total gap; an account with no gap is always On track, so a healthier book shows
# fewer reds. Centre points below sit inside the ranges above.
TARGET_AT_RISK    = 0.08   # top ~8% by total gap -> At risk
TARGET_ATTENTION  = 0.12   # next ~12% -> Needs attention (rest On track)
# An account must be at least this far off in aggregate to be flagged at all, so
# reds are genuinely off (not merely the least-good in a healthy book) and a very
# healthy book shows fewer than the target share. Set below the off-target cluster
# of the gap distribution, so it does not bite in a normal book.
BAND_GAP_FLOOR    = 8.0

RISK_ORDER = {"At risk": 0, "Needs attention": 1, "On track": 2}
RISK_COLOR = {"At risk": RED, "Needs attention": AMBER, "On track": GREEN}
RISK_BADGE = {"At risk": "\U0001F534 At risk",
              "Needs attention": "\U0001F7E1 Needs attention",
              "On track": "\U0001F7E2 On track"}


# Provisional outcome bands (end-of-term consumption vs commitment). OPEN in the
# design summary; set here to move forward, not decided.
def label_outcome(end_ratio):
    if end_ratio >= 1.15:  return "expansion"
    if end_ratio >= 1.00:  return "full_renewal"
    if end_ratio >= 0.90:  return "90_99"
    if end_ratio >= 0.75:  return "under_90"
    return "churn"

WINNER_OUTCOMES = {"expansion", "full_renewal"}
LOSER_OUTCOMES  = {"under_90", "churn"}   # "90_99" is treated as neutral

# -----------------------------------------------------------------------------
# TRAJECTORY SIGNALS (formerly "usage metrics"). The full tracked set, with:
#   dir     +1 higher is better / -1 lower is better
#   tier    1 = trusted automatic telemetry, 2 = manual / maintained
#   cluster None, "engagement", or "relationship" (nested into one sub-score)
#   grain   "quarter" (a per-quarter trajectory) or "account" (a one-off scalar)
#   opt     True = absent-not-zero (Tier 2 that may be missing -> unknown, not bad)
# Order is display order on the Trajectory Signals pages.
SIGNALS = {
    # key                          label                                    dir tier cluster        grain      opt   why
    "consumption_vs_commit":     ("Consumption vs commitment",              +1, 1, None,          "quarter", False, "Using less than they bought signals contraction risk."),
    "consumption_concentration": ("Consumption concentration",              -1, 1, None,          "quarter", False, "Usage from one team collapses if that team changes."),
    "integrations_live":         ("Integrations (live count)",              +1, 1, None,          "quarter", False, "More live connections make the solution harder to drop."),
    "active_users":              ("Active users",                           +1, 1, "engagement",  "quarter", False, "Few active users means the product is not embedded."),
    "unique_logins":             ("Unique user logins",                     +1, 1, "engagement",  "quarter", False, "Distinct people in the product show real breadth of use."),
    "logins":                    ("Logins",                                 +1, 1, "engagement",  "quarter", False, "A coarse activity pulse; corroborates unique logins."),
    "activated_workflows":       ("Activated workflows-live",               +1, 1, "engagement",  "quarter", False, "Agents that truly caught on early tend to stick."),
    "workflow_breadth":          ("Breadth of active workflows",            +1, 1, "engagement",  "quarter", False, "Spread across the solution is stickier than one deep use."),
    "features_used":             ("Adoption breadth (features used)",       +1, 1, "engagement",  "quarter", False, "Narrow adoption is easier for a customer to walk away from."),
    "time_to_deploy":            ("Time-to-deploy (days)",                  -1, 1, None,          "account", False, "Slow delivery delays every downstream value signal."),
    "time_to_value":             ("Time-to-value (days)",                   -1, 1, None,          "account", False, "A slow first real use is the strongest early churn tell."),
    "grounding_fail_rate":       ("Grounding / hallucination-failure rate", -1, 1, None,          "quarter", False, "Answers not backed by the data erode trust fast."),
    "support_tickets":           ("Support tickets",                        -1, 1, None,          "quarter", False, "A heavy support load signals friction and frustration."),
    "escalations":               ("Escalations",                            -1, 1, None,          "quarter", False, "Escalations mean unresolved, renewal-threatening issues."),
    "eval_score":                ("Eval score",                             +1, 2, None,          "account", True,  "How well an FDE-built solution hits its baseline value."),
    "outcomes_produced":         ("Outcomes produced",                      +1, 2, None,          "quarter", False, "Fewer results means less proof of value at renewal."),
    "cost_per_outcome":          ("Cost per outcome",                       -1, 2, None,          "quarter", False, "Paying more per result erodes the value story."),
    "exec_sponsor_nps":          ("Exec-sponsor NPS",                       +1, 2, "relationship","account", True,  "The sponsor decides the renewal; no sponsor is itself risk."),
    "champion_present":          ("Champion present",                       +1, 2, "relationship","quarter", False, "Losing the internal champion is a top churn driver."),
    "exec_touch_recency":        ("Exec-touch recency (days)",              -1, 2, "relationship","quarter", False, "No recent executive contact weakens the relationship."),
}

# Derived views the rest of the app reads (keeps downstream code simple).
METRIC_DIRECTION = {k: v[1] for k, v in SIGNALS.items()}
METRIC_LABELS    = {k: v[0] for k, v in SIGNALS.items()}
METRIC_WHY       = {k: v[6] for k, v in SIGNALS.items()}
SIGNAL_TIER      = {k: v[2] for k, v in SIGNALS.items()}
SIGNAL_CLUSTER   = {k: v[3] for k, v in SIGNALS.items()}
SIGNAL_GRAIN     = {k: v[4] for k, v in SIGNALS.items()}
SIGNAL_OPTIONAL  = {k: v[5] for k, v in SIGNALS.items()}

QUARTER_SIGNALS = [k for k in SIGNALS if SIGNAL_GRAIN[k] == "quarter"]
ACCOUNT_SIGNALS = [k for k in SIGNALS if SIGNAL_GRAIN[k] == "account"]
TIER1_SIGNALS   = [k for k in SIGNALS if SIGNAL_TIER[k] == 1]
TIER2_SIGNALS   = [k for k in SIGNALS if SIGNAL_TIER[k] == 2]

# Cluster pseudo-drivers roll their members into one score so correlated signals
# do not each fire full-weight.
CLUSTER_LABELS = {"engagement": "Engagement (usage breadth & depth)",
                  "relationship": "Relationship health"}
CLUSTER_KEY = {"engagement": "__engagement__", "relationship": "__relationship__"}
for _c, _k in CLUSTER_KEY.items():
    METRIC_LABELS[_k] = CLUSTER_LABELS[_c]


# =============================================================================
# PART B - THE ENGINE  (pure functions; no Streamlit code; model unchanged)
# =============================================================================

def make_trajectory(shape, term, rng, faller_end_level=0.65):
    """Build one account's per-quarter usage numbers.

    Fallers peak mid-term then fade, so mid-term their consumption OVERLAPS
    winners. That is exactly why concentration and workflow breadth separate
    winners from losers earlier than raw consumption does. We bake that in on purpose.
    """
    rows = []
    for q in range(1, term + 1):
        frac = q / term
        if shape == "winner":
            cons = 0.55 + 0.60 * frac + rng.normal(0, 0.05)
            concentration = np.clip(0.38 + rng.normal(0, 0.06), 0, 1)
            features = min(len(FEATURES), int(round(3 + 4 * frac + rng.normal(0, 0.6))))
            champion = 1 if rng.random() > 0.05 else 0
            exec_recency = max(1, rng.normal(20, 8))
            cost_per_outcome = np.clip(1.00 - 0.15 * frac + rng.normal(0, 0.05), 0.4, 2.0)
            tickets = max(0, int(rng.normal(2, 1)))
            escal = 1 if rng.random() < 0.05 else 0
            integrations = max(1, int(round(3 + 4 * frac + rng.normal(0, 1.0))))
            activated = max(0, int(round(1 + 3 * frac + rng.normal(0, 0.7))))
            breadth = max(1, int(round(2 + 4 * frac + rng.normal(0, 0.8))))
            grounding = float(np.clip(0.05 + rng.normal(0, 0.02), 0, 1))
        else:  # faller
            start = 0.50
            peak = max(faller_end_level + 0.10, 0.85)
            if frac <= 0.45:
                cons = start + (peak - start) * (frac / 0.45)
            else:
                cons = peak + (faller_end_level - peak) * ((frac - 0.45) / 0.55)
            cons = cons + rng.normal(0, 0.04)
            concentration = np.clip(0.58 + 0.15 * frac + rng.normal(0, 0.07), 0, 1)
            features = min(len(FEATURES), int(round(2 + 2 * frac + rng.normal(0, 0.6))))
            champion = 0 if (frac > 0.5 and rng.random() < 0.5) else (1 if rng.random() > 0.2 else 0)
            exec_recency = max(1, rng.normal(55, 18) + 30 * frac)
            cost_per_outcome = np.clip(1.05 + 0.35 * frac + rng.normal(0, 0.07), 0.4, 3.0)
            tickets = max(0, int(rng.normal(5, 2)))
            escal = 1 if rng.random() < 0.30 else 0
            integrations = max(0, int(round(1 + 1.5 * frac + rng.normal(0, 0.8))))
            activated = max(0, int(round(0.5 + 1.0 * frac + rng.normal(0, 0.5))))
            breadth = max(1, int(round(2 + 1.0 * frac - 1.2 * max(0, frac - 0.5) + rng.normal(0, 0.6))))
            grounding = float(np.clip(0.12 + 0.15 * frac + rng.normal(0, 0.03), 0, 1))

        cons = max(0.05, cons)
        features = max(1, features)
        outcomes = max(0.1, cons * (1.0 + 0.1 * features) + rng.normal(0, 0.05))
        active_users = max(1, int(features * rng.normal(6, 1.5)))
        unique_logins = max(1, int(active_users * rng.uniform(0.5, 0.85)))
        logins = max(unique_logins, int(unique_logins * rng.normal(3.0, 0.6)))

        rows.append({
            "quarter_within_term":       q,
            "consumption_vs_commit":     round(cons, 4),
            "consumption_concentration": round(concentration, 4),
            "integrations_live":         integrations,
            "active_users":              active_users,
            "unique_logins":             unique_logins,
            "logins":                    logins,
            "activated_workflows":       activated,
            "workflow_breadth":          breadth,
            "features_used":             features,
            "grounding_fail_rate":       round(grounding, 4),
            "support_tickets":           tickets,
            "escalations":               escal,
            "outcomes_produced":         round(outcomes, 4),
            "cost_per_outcome":          round(cost_per_outcome, 4),
            "champion_present":          champion,
            "exec_touch_recency":        round(exec_recency, 1),
        })
    return rows


def generate_dataset(n_accounts, term_probs, segment_probs, noise_share,
                     winner_share, seed):
    """Create the four tables: accounts, account_quarter,
    account_quarter_feature, and a small event_log sample.
    """
    rng = np.random.default_rng(seed)
    acct_rows, aq_rows, feat_rows, event_rows = [], [], [], []

    for i in range(n_accounts):
        aid = f"ACC{i:04d}"
        segment = rng.choice(SEGMENTS, p=segment_probs)
        industry = rng.choice(INDUSTRIES)
        term = int(rng.choice(TERM_CHOICES, p=term_probs))
        start_quarter = f"Q{rng.integers(1,5)}-{rng.integers(2019,2024)}"

        base = SEGMENT_COMMIT[segment]
        committed = int(base * np.exp(rng.normal(0, 0.25)))

        shape = "winner" if rng.random() < winner_share else "faller"
        is_noise = rng.random() < noise_share

        faller_end_level = rng.uniform(0.45, 0.98)
        traj = make_trajectory(shape, term, rng, faller_end_level)
        end_ratio = traj[-1]["consumption_vs_commit"]
        outcome = label_outcome(end_ratio)

        if is_noise:
            if shape == "winner":
                outcome = "churn"
                reason = "budget_cut_or_acquisition"
            else:
                outcome = "full_renewal"
                reason = "switching_cost_retention"
        else:
            reason = {
                "expansion":    "expanded_usage",
                "full_renewal": "healthy_renewal",
                "90_99":        "soft_renewal",
                "under_90":     "under_target",
                "churn":        "did_not_renew",
            }[outcome]

        # Account-level (one-off) trajectory signals.
        is_win = (shape == "winner")
        t_deploy = float(max(3, rng.normal(35, 12) if is_win else rng.normal(70, 25)))
        t_value  = float(t_deploy + (rng.normal(20, 8) if is_win else rng.normal(55, 20)))
        # Eval score (absent-not-zero): only where an FDE built a solution.
        eval_present = rng.random() < (0.60 if is_win else 0.45)
        eval_score = (float(np.clip(rng.normal(0.90, 0.07) if is_win else rng.normal(0.60, 0.12),
                                    0.0, 1.3)) if eval_present else np.nan)
        # Exec-sponsor NPS: coverage flag first (no sponsor is itself risk).
        sponsor_present = rng.random() < (0.85 if is_win else 0.55)
        exec_sponsor_nps = (float(np.clip(rng.normal(45, 15) if is_win else rng.normal(10, 25),
                                          -100, 100)) if sponsor_present else np.nan)

        acct_rows.append({
            "account_id":                 aid,
            "segment":                    segment,
            "industry":                   industry,
            "start_quarter":              start_quarter,
            "term_quarters":              term,
            "committed_credits_quarter":  committed,
            "renewal_quarter":            term,
            "time_to_deploy":             round(t_deploy, 1),
            "time_to_value":              round(t_value, 1),
            "eval_present":               int(eval_present),
            "eval_score":                 (round(eval_score, 4) if eval_present else np.nan),
            "sponsor_present":            int(sponsor_present),
            "exec_sponsor_nps":           (round(exec_sponsor_nps, 1) if sponsor_present else np.nan),
            "outcome":                    outcome,
            "outcome_reason":             reason,
            "shape_internal":             shape,
            "is_noise_internal":          int(is_noise),
        })

        for r in traj:
            credits_consumed = int(committed * r["consumption_vs_commit"])
            aq_rows.append({
                "account_id":                aid,
                "term_quarters":             term,
                **r,
                "committed_credits":         committed,
                "credits_consumed":          credits_consumed,
            })
            conc = r["consumption_concentration"]
            weights = rng.random(len(FEATURES)) ** (1 + 4 * conc)
            weights = weights / weights.sum()
            for f, w in zip(FEATURES, weights):
                feat_rows.append({
                    "account_id":       aid,
                    "quarter_within_term": r["quarter_within_term"],
                    "feature":          f,
                    "credits_consumed": int(credits_consumed * w),
                    "outcomes_produced":round(r["outcomes_produced"] * w, 4),
                })

        if i < 3:
            for r in traj:
                for _ in range(40):
                    feat = rng.choice(FEATURES)
                    event_rows.append({
                        "account_id":     aid,
                        "user_id":        f"U{rng.integers(1, 50):03d}",
                        "quarter_within_term": r["quarter_within_term"],
                        "feature":        feat,
                        "event_type":     rng.choice(["run", "query", "export", "config"]),
                        "credits":        int(max(1, rng.normal(30, 10))),
                        "produced_outcome": int(rng.random() < 0.6),
                    })

    return {
        "accounts":                pd.DataFrame(acct_rows),
        "account_quarter":         pd.DataFrame(aq_rows),
        "account_quarter_feature": pd.DataFrame(feat_rows),
        "event_log":               pd.DataFrame(event_rows),
    }


def cohort_bands(aq, accounts, term, metric):
    """Layer 1. Median and middle-half band (25th-75th pct) at each
    quarter-within-term, for winners and losers separately.
    """
    outc = accounts.set_index("account_id")["outcome"]
    df = aq[aq["term_quarters"] == term].copy()
    df["group"] = df["account_id"].map(
        lambda a: "winning" if outc[a] in WINNER_OUTCOMES
        else ("losing" if outc[a] in LOSER_OUTCOMES else "neutral")
    )
    out = {}
    for grp in ("winning", "losing"):
        g = df[df["group"] == grp]
        stats = g.groupby("quarter_within_term")[metric].agg(
            median="median",
            lo=lambda s: s.quantile(0.25),
            hi=lambda s: s.quantile(0.75),
        ).reset_index()
        out[grp] = stats
    return out


def auc_score(x, y):
    """Rank-based AUC (Mann-Whitney). 0.5 = no signal, 1.0 = perfect."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    n1, n0 = int(y.sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(x).rank().to_numpy()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def indicator_auc(aq, accounts, term):
    """Layer 2. For each metric at each quarter, how well it separates
    winners from losers.
    """
    outc = accounts.set_index("account_id")["outcome"]
    df = aq[aq["term_quarters"] == term].copy()
    df["label"] = df["account_id"].map(
        lambda a: 1 if outc[a] in WINNER_OUTCOMES else (0 if outc[a] in LOSER_OUTCOMES else -1)
    )
    df = df[df["label"] >= 0]
    rows = []
    for q in sorted(df["quarter_within_term"].unique()):
        sub = df[df["quarter_within_term"] == q]
        for metric in QUARTER_SIGNALS:
            direction = METRIC_DIRECTION[metric]
            raw = auc_score(sub[metric] * direction, sub["label"])
            rows.append({"quarter_within_term": q, "metric": metric,
                         "auc": round(raw, 3) if pd.notna(raw) else np.nan})
    return pd.DataFrame(rows)


def severity_status(sev):
    """Turn a gap size (in spreads) into a red / amber / green word."""
    if sev >= RED_SEV:    return "At risk"
    if sev >= YELLOW_SEV: return "Needs attention"
    return "On track"


def account_risk_band(drivers):
    """Roll a list of off-target drivers up to one account-level colour.

    Rule (simple and adjustable): any clearly-off (red) driver makes the account
    At risk; otherwise one or more amber drivers make it Needs attention;
    nothing flagged is On track.
    """
    reds = sum(1 for d in drivers if d["severity"] >= RED_SEV)
    ambers = sum(1 for d in drivers if YELLOW_SEV <= d["severity"] < RED_SEV)
    if reds >= 1:
        return "At risk"
    if ambers >= 1:
        return "Needs attention"
    return "On track"


def driver_read(value, benchmark, direction):
    """Plain-language 'current vs winning' so 1.00 vs 0.94 is never ambiguous."""
    if direction > 0:
        tail = "below target" if value < benchmark else "at/above target"
    else:
        tail = "above target" if value > benchmark else "at/below target"
    return f"{value:.2f} vs winning {benchmark:.2f} ({tail})"


def score_live_accounts(dataset, term_probs, segment_probs, noise_share,
                        winner_share, seed, n_live):
    """Layer 3. Generate a fresh pool of live accounts observed partway through
    their term, hide the true outcome, and score each against the winning band.
    Returns (worklist_df, plans_dict).
    """
    aq_hist = dataset["account_quarter"]
    accts_hist = dataset["accounts"]
    outc = accts_hist.set_index("account_id")["outcome"]
    hist = aq_hist.copy()
    hist["is_winner"] = hist["account_id"].map(lambda a: outc[a] in WINNER_OUTCOMES)
    win = hist[hist["is_winner"]]

    # Quarter-grain winning bands (25th / median / 75th of winners at each quarter).
    band = {}
    for term in TERM_CHOICES:
        for q in range(1, term + 1):
            sub = win[(win["term_quarters"] == term) & (win["quarter_within_term"] == q)]
            if sub.empty:
                continue
            for metric in QUARTER_SIGNALS:
                s = sub[metric]
                band[(term, q, metric)] = (s.quantile(0.25), s.median(), s.quantile(0.75))

    # Account-grain winning bands (one-off signals, from winner accounts).
    win_ids = set(accts_hist[accts_hist["outcome"].isin(WINNER_OUTCOMES)]["account_id"])
    win_acct = accts_hist[accts_hist["account_id"].isin(win_ids)]
    aband = {}
    for metric in ACCOUNT_SIGNALS:
        s = win_acct[metric].dropna()
        if len(s) >= 3:
            aband[metric] = (s.quantile(0.25), s.median(), s.quantile(0.75))

    live = generate_dataset(n_live, term_probs, segment_probs, noise_share,
                            winner_share, seed + 999)
    live_acct = live["accounts"]
    live_aq = live["account_quarter"]

    rng = np.random.default_rng(seed + 7)
    work_rows, plans = [], {}

    def _sev(val, q25, med, q75, direction):
        iqr = max(q75 - q25, 1e-6)
        if direction > 0:
            s = max(0.0, (q25 - val) / iqr)
        else:
            s = max(0.0, (val - q75) / iqr)
        return min(s, 3.0)

    for _, a in live_acct.iterrows():
        term = a["term_quarters"]
        current_q = max(1, int(round(term * rng.uniform(0.40, 0.80))))
        q_to_renewal = term - current_q

        cur = live_aq[(live_aq["account_id"] == a["account_id"]) &
                      (live_aq["quarter_within_term"] == current_q)]
        if cur.empty:
            continue
        cur = cur.iloc[0]

        # One "read" per signal that could be scored (present + has a band).
        reads = []
        n_tracked = 0
        for metric in SIGNALS:
            direction = METRIC_DIRECTION[metric]
            cluster   = SIGNAL_CLUSTER[metric]
            optional  = SIGNAL_OPTIONAL[metric]

            if SIGNAL_GRAIN[metric] == "quarter":
                key = (term, current_q, metric)
                if key not in band:
                    continue
                q25, med, q75 = band[key]
                raw = cur[metric]
            else:
                raw = a[metric]
                q25, med, q75 = aband.get(metric, (None, None, None))

            # Absent-not-zero: a missing Tier 2 value is unknown, not bad.
            if optional and pd.isna(raw):
                if metric == "exec_sponsor_nps":     # no sponsor is itself a risk flag
                    reads.append({"metric": metric, "severity": RED_SEV, "value": np.nan,
                                  "benchmark": np.nan, "direction": direction, "cluster": cluster,
                                  "status": severity_status(RED_SEV),
                                  "detail": "No exec sponsor identified (coverage gap)"})
                    n_tracked += 1
                continue                              # eval score etc.: skip silently
            if q25 is None or pd.isna(raw):
                continue

            n_tracked += 1
            val = float(raw)
            sev = _sev(val, q25, med, q75, direction)
            reads.append({"metric": metric, "severity": round(sev, 2), "value": val,
                          "benchmark": float(med), "direction": direction, "cluster": cluster,
                          "status": severity_status(sev),
                          "detail": driver_read(val, float(med), direction)})

        # Individual drivers (non-clustered) + one rolled-up driver per cluster.
        drivers = [r for r in reads if r["cluster"] is None and r["severity"] > YELLOW_SEV]
        for cl, ckey in CLUSTER_KEY.items():
            members = [r for r in reads if r["cluster"] == cl]
            if not members:
                continue
            sev_list = [m["severity"] for m in members]
            csev = round(0.6 * float(np.mean(sev_list)) + 0.4 * float(np.max(sev_list)), 2)
            if csev > YELLOW_SEV:
                top = sorted(members, key=lambda m: -m["severity"])
                names = ", ".join(METRIC_LABELS[m["metric"]] for m in top[:2])
                drivers.append({"metric": ckey, "severity": csev, "value": np.nan,
                                "benchmark": np.nan, "direction": -1, "cluster": None,
                                "status": severity_status(csev),
                                "detail": f"{top[0]['detail']} (worst of: {names})",
                                "members": members})

        drivers = sorted(drivers, key=lambda d: -d["severity"])
        total_gap = round(sum(d["severity"] for d in drivers), 2)
        # Account band is assigned after all accounts are scored, by ranking on
        # total gap into the target operating tiers (see below). Placeholder here.
        risk_band = "On track"

        contract_value = int(a["committed_credits_quarter"]) * term
        value_weight = np.log10(contract_value) / 5.0
        urgency = 1.0 + (1.0 - q_to_renewal / term)
        priority = round(total_gap * value_weight * urgency, 2)

        work_rows.append({
            "account_id":                a["account_id"],
            "segment":                   a["segment"],
            "risk_band":                 risk_band,
            "term_label":                TERM_LABEL.get(term, f"{term}q"),
            "term_quarters":             term,
            "committed_credits_quarter": int(a["committed_credits_quarter"]),
            "contract_value":            contract_value,
            "current_quarter":           current_q,
            "quarters_to_renewal":       q_to_renewal,
            "off_target_count":          len(drivers),
            "top_drivers":               ", ".join(METRIC_LABELS[d["metric"]] for d in drivers[:2]) or "None",
            "total_gap":                 round(total_gap, 2),
            "priority_score":            priority,
            "true_outcome_hidden":       a["outcome"],
        })
        plans[a["account_id"]] = {"meta": a.to_dict(), "current_q": current_q,
                                  "q_to_renewal": q_to_renewal, "drivers": drivers,
                                  "risk_band": risk_band, "n_tracked": n_tracked}

    worklist = pd.DataFrame(work_rows)
    if not worklist.empty:
        # Calibrated bands: rank on total gap, then size the two off-target tiers
        # to the target operating shares. Any account with no gap stays On track,
        # so a healthier book naturally shows fewer reds.
        worklist = worklist.sort_values("total_gap", ascending=False).reset_index(drop=True)
        n = len(worklist)
        n_red = int(round(n * TARGET_AT_RISK))
        n_amber = int(round(n * TARGET_ATTENTION))
        bands, used = [], 0
        for gap in worklist["total_gap"]:
            if gap < BAND_GAP_FLOOR:
                bands.append("On track")
            elif used < n_red:
                bands.append("At risk"); used += 1
            elif used < n_red + n_amber:
                bands.append("Needs attention"); used += 1
            else:
                bands.append("On track")
        worklist["risk_band"] = bands
        for aid, band in zip(worklist["account_id"], bands):
            if aid in plans:
                plans[aid]["risk_band"] = band
        worklist = worklist.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return worklist, plans


# Owner + metric-to-move mapping (design summary 4.4 / Section 5).
DRIVER_TO_ACTION = {
    "consumption_vs_commit":     ("FDE + CSM",     "Consumption vs commitment %"),
    "consumption_concentration": ("FDE + CSM",     "Consumption concentration"),
    "integrations_live":         ("FDE",           "Live integration count"),
    "active_users":              ("CSM",           "Active-user ratio"),
    "unique_logins":             ("CSM",           "Unique logins vs seats"),
    "logins":                    ("CSM",           "Login activity"),
    "activated_workflows":       ("FDE + CSM",     "Activated workflows (50+/qtr)"),
    "workflow_breadth":          ("FDE + CSM",     "Distinct workflows in use"),
    "features_used":             ("CSM + FDE",     "Core features adopted"),
    "time_to_deploy":            ("FDE",           "Days to deploy"),
    "time_to_value":             ("FDE + CSM",     "Days to first real use"),
    "grounding_fail_rate":       ("FDE",           "Grounding-failure rate"),
    "support_tickets":           ("FDE + support", "Open ticket / severe count"),
    "escalations":               ("FDE + support", "Escalation rate"),
    "eval_score":                ("FDE",           "Eval attainment vs baseline"),
    "outcomes_produced":         ("FDE + CSM",     "Outcomes produced"),
    "cost_per_outcome":          ("FDE + CSM",     "Cost per outcome"),
    "exec_sponsor_nps":          ("CSM",           "Exec-sponsor NPS / coverage"),
    "champion_present":          ("CSM",           "Engaged stakeholders"),
    "exec_touch_recency":        ("CSM",           "Exec-touch recency"),
    "__engagement__":            ("CSM + FDE",     "Usage breadth & depth"),
    "__relationship__":          ("CSM",           "Sponsor / champion coverage"),
}
METRIC_WHY["__engagement__"]   = "Thin or narrowing usage across the solution weakens stickiness."
METRIC_WHY["__relationship__"] = "A weak or missing senior relationship puts the renewal at risk."


def build_action_plan(account_id, plan):
    """Layer 4 (placeholder). One row per off-target driver: colour, owner, the
    number to move, and current-vs-winning in plain words. 'Recommended play' is
    intentionally blank - that slot is filled by your team's play library.
    """
    steps = []
    for d in plan["drivers"]:
        owner, metric_to_move = DRIVER_TO_ACTION.get(d["metric"], ("CSM", d["metric"]))
        steps.append({
            "Status":             RISK_BADGE[d["status"]],
            "Risk area":          METRIC_LABELS.get(d["metric"], d["metric"]),
            "Why it matters":     METRIC_WHY.get(d["metric"], ""),
            "Owner":              owner,
            "Metric to move":     metric_to_move,
            "Current vs winning": d["detail"],
            "Recommended play":   "",   # <-- play library goes here
        })
    return {
        "account_id":          account_id,
        "segment":             plan["meta"]["segment"],
        "term_quarters":       plan["meta"]["term_quarters"],
        "current_quarter":     plan["current_q"],
        "quarters_to_renewal": plan["q_to_renewal"],
        "risk_band":           plan["risk_band"],
        "on_track_count":      plan["n_tracked"] - len(plan["drivers"]),
        "tracked_count":       plan["n_tracked"],
        "off_target_drivers":  steps,
    }


# =============================================================================
# PART C - CACHING  (so clicking around the app is instant)
# =============================================================================

@st.cache_data(show_spinner=False)
def run_pipeline(n_accounts, short_share, midterm_share, smb, mid, noise_share,
                 winner_share, seed, n_live):
    # Three look-back horizons (12 / 24 / 36 months), normalized to sum 1.
    long_share = max(0.0, 1 - short_share - midterm_share)
    tp = [short_share, midterm_share, long_share]
    tp_total = sum(tp) or 1.0
    term_probs = [p / tp_total for p in tp]
    ent = max(0.0, 1 - smb - mid)
    seg = [smb, mid, ent]
    total = sum(seg) or 1.0
    segment_probs = [p / total for p in seg]
    dataset = generate_dataset(n_accounts, term_probs, segment_probs,
                               noise_share, winner_share, seed)
    worklist, plans = score_live_accounts(dataset, term_probs, segment_probs,
                                           noise_share, winner_share, seed, n_live)
    return dataset, worklist, plans


# =============================================================================
# PART D - THE WEB PAGES
# =============================================================================

NAV = ["Home", "The Model", "Portfolio", "Account Plan", "Data", "Trajectory Signals", "Assumptions"]


def go(page, account=None):
    """Navigate. Used as an on_click callback so a single click is enough."""
    st.session_state.page = page
    st.session_state.portfolio_focus = None   # leave any Portfolio drill-down
    if account is not None:
        st.session_state.selected_account = account


# Resolve image paths relative to THIS script's folder, not the directory the app
# happens to be launched from. This is the usual reason a logo or background does
# not show: Streamlit reads relative paths from the current working directory.
try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_DIR = os.getcwd()


def _resolve_path(ref):
    """Turn a relative image path into an absolute one under the script folder,
    matching each folder/file name case-insensitively. Absolute paths and http(s)
    URLs are returned unchanged.

    This is deliberately forgiving: on a Mac the filesystem ignores case, but the
    Linux server that hosts the published app does not, so a folder committed as
    'Images' would break a lowercase 'images/...' path. Here we match either."""
    if not ref or ref.startswith("http://") or ref.startswith("https://") or os.path.isabs(ref):
        return ref
    exact = os.path.join(APP_DIR, ref)
    if os.path.exists(exact):
        return exact
    # Walk the path segment by segment, accepting a case-insensitive match at each.
    cur = APP_DIR
    for seg in ref.replace("\\", "/").split("/"):
        if not seg or seg == ".":
            continue
        candidate = os.path.join(cur, seg)
        if os.path.exists(candidate):
            cur = candidate
            continue
        try:
            entries = os.listdir(cur)
        except OSError:
            return exact
        match = next((e for e in entries if e.lower() == seg.lower()), None)
        if match is None:
            return exact          # give up; caller falls back to text/plain wash
        cur = os.path.join(cur, match)
    return cur


def _bg_source(ref):
    """Turn a page-background reference into a CSS url() value.

    Accepts a full http(s) URL (used directly) or a local file path (read and
    embedded as base64, so the published app needs no external image host).
    Returns None if the reference is blank or the file is missing, which tells
    the caller to fall back to a plain wash.
    """
    if not ref:
        return None
    if ref.startswith("http://") or ref.startswith("https://"):
        return f'url("{ref}")'
    ref = _resolve_path(ref)
    if os.path.exists(ref):
        ext = os.path.splitext(ref)[1].lstrip(".").lower() or "jpeg"
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        with open(ref, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f'url("data:image/{mime};base64,{b64}")'
    return None


def logo_img_tag(path, max_height_px):
    """Return an <img> tag with the logo embedded as base64, or None if missing.
    Transparent PNGs sit cleanly over the muted background, no white box."""
    path = _resolve_path(path)
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return (f'<img src="data:image/png;base64,{b64}" '
            f'style="max-height:{max_height_px}px;width:auto;display:block;" alt="Atkins Consulting">')


def inject_page_background(page):
    """Paint a muted, fixed, full-page photo behind the current page.

    A near-opaque scrim (the page's own base colour at BG_SCRIM alpha) sits ON
    TOP of the photo to mute it, and a faint dot grid adds texture. Cards, the
    hero, tables, and the sidebar are all solid white, so they stay in the
    foreground and text stays easy to read. If no image is available for the
    page, a plain wash is used instead.
    """
    src = _bg_source(PAGE_BACKGROUNDS.get(page))
    scrim = f"rgba(237,242,245,{BG_SCRIM})"
    dots = "radial-gradient(rgba(14,34,51,0.05) 1px, transparent 1.4px)"
    if src:
        image = f"{dots}, linear-gradient({scrim}, {scrim}), {src}"
        size  = "24px 24px, cover, cover"
        pos   = "0 0, center, center"
        rep   = "repeat, no-repeat, no-repeat"
    else:
        image = f"{dots}, linear-gradient(180deg, #F7F9FB 0%, #E9F0F4 100%)"
        size  = "24px 24px, auto"
        pos   = "0 0, 0 0"
        rep   = "repeat, no-repeat"
    st.markdown(f"""
    <style>
      .stApp {{
          background-color: #EDF2F5;
          background-image: {image};
          background-size: {size};
          background-position: {pos};
          background-repeat: {rep};
          background-attachment: fixed;
      }}
    </style>
    """, unsafe_allow_html=True)


def inject_css():
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

      .stApp {{ background-color: #EDF2F5; }}
      html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {INK}; }}
      h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif; color: {INK}; letter-spacing: -0.01em; }}
      .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1180px; }}

      /* Brand header */
      .ac-brandbar {{ display:flex; align-items:center; gap:14px; margin: 0 0 4px 0; }}
      .ac-mark {{ width:40px; height:40px; border-radius:10px; background:{INK};
                  color:#fff; font-family:'Space Grotesk'; font-weight:700; font-size:18px;
                  display:flex; align-items:center; justify-content:center; letter-spacing:0.02em; }}
      .ac-word {{ font-family:'Space Grotesk'; font-weight:700; font-size:20px; color:{INK}; line-height:1; }}
      .ac-tag  {{ font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:0.04em;
                  text-transform:uppercase; color:{MUTE}; margin-top:3px; }}
      .ac-rule {{ height:3px; width:100%; background:linear-gradient(90deg,{INK} 0%,{ACCENT} 55%,transparent 100%);
                  border-radius:2px; margin:8px 0 14px 0; }}

      .ac-eyebrow {{ font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:0.08em;
                     text-transform:uppercase; color:{ACCENT}; margin-bottom:6px; }}

      /* Nav buttons: render as a flat segmented bar */
      div[data-testid="stHorizontalBlock"] .stButton button {{
          border-radius:8px; border:1px solid {LINE}; background:{CARD}; color:{INK};
          font-weight:600; font-size:0.9rem; padding:0.45rem 0.4rem; }}
      div[data-testid="stHorizontalBlock"] .stButton button:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
      .stButton button[kind="primary"], .stButton button[data-testid="baseButton-primary"] {{
          background:{INK}; border:1px solid {INK}; color:#fff; }}
      .stButton button[kind="primary"]:hover {{ background:{ACCENT}; border-color:{ACCENT}; color:#fff; }}

      /* Cell (filter) buttons: compact, narrower, text left-aligned and indented.
         Scoped by the st-key-cellbtn_ class Streamlit adds from the widget key. */
      [class*="st-key-cellbtn_"] button {{
          justify-content:flex-start; text-align:left;
          font-size:0.78rem; line-height:1.05;
          min-height:0; height:auto;
          padding:0.14rem 0.4rem 0.14rem 1.8ch;
          max-width:250px; }}

      /* Hero */
      .ac-hero {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px;
                  padding:34px 36px; box-shadow:0 1px 2px rgba(14,34,51,0.04); }}
      .ac-hero h1 {{ font-size:2.15rem; margin:0 0 10px 0; line-height:1.12; }}
      .ac-hero p  {{ font-size:1.02rem; color:{MUTE}; max-width:640px; margin:0; line-height:1.55; }}

      /* Cards */
      .ac-card {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:18px 20px;
                  height:100%; box-shadow:0 1px 2px rgba(14,34,51,0.04); }}
      .ac-card .step {{ font-family:'IBM Plex Mono', monospace; font-size:12px; color:{ACCENT}; }}
      .ac-card h4 {{ margin:6px 0 6px 0; font-size:1.02rem; }}
      .ac-card p  {{ margin:0; color:{MUTE}; font-size:0.9rem; line-height:1.5; }}

      /* Legend pills */
      .ac-legend {{ display:flex; gap:10px; flex-wrap:wrap; margin:2px 0 10px 0; }}
      .ac-pill {{ display:inline-flex; align-items:center; gap:7px; background:{CARD};
                  border:1px solid {LINE}; border-radius:999px; padding:5px 12px; font-size:0.82rem; color:{INK}; }}
      .ac-dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}

      .ac-note {{ background:#EEF4F5; border-left:3px solid {ACCENT}; border-radius:8px;
                  padding:12px 16px; color:{INK}; font-size:0.9rem; line-height:1.5; }}

      [data-testid="stMetric"] {{ background:{CARD}; border:1px solid {LINE}; border-radius:12px;
                                   padding:12px 16px; }}
      section[data-testid="stSidebar"] {{ background:{CARD}; border-right:1px solid {LINE}; }}

      a, a:visited {{ color:{ACCENT}; }}
      :focus-visible {{ outline:2px solid {ACCENT}; outline-offset:2px; }}
      @media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; transition:none !important; }} }}

      /* Stronger contrast for text that sits on the page background (not in boxes).
         Boxed/table/tier text sets its own colour and is unaffected. */
      h1, h2, h3, h4 {{ color:{INK} !important; }}
      [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] {{ color:#33424C !important; }}
      .stMarkdown p, .stMarkdown li {{ color:#22303A; }}

      /* Hide Streamlit's own header, toolbar, menu and footer for a cleaner page */
      header[data-testid="stHeader"] {{display:none;}}
      [data-testid="stToolbar"] {{display:none;}}
      footer {{visibility:hidden;}}
      #MainMenu {{visibility:hidden;}}

      /* The scenario controls now live at the top of the Data page, so the
         Streamlit sidebar is removed entirely. */
      [data-testid="stSidebar"] {{ display:none !important; }}
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="collapsedControl"],
      [data-testid="stSidebarCollapseButton"] {{ display:none !important; }}

      /* Render the keyed "Signal Library" button as a plain blue text link,
         right-aligned above the picker. Scoped by its widget key so no other
         button is affected. */
      .st-key-ts_link {{ text-align:right; margin-bottom:-0.35rem; }}
      .st-key-ts_link button {{
        background:transparent !important; border:none !important; box-shadow:none !important;
        padding:0 !important; min-height:0 !important; width:auto !important; display:inline;
        color:#1A73E8 !important; font-weight:600;
      }}
      .st-key-ts_link button:hover {{ background:transparent !important; text-decoration:underline !important; color:#1558B0 !important; }}
      .st-key-ts_link button p, .st-key-ts_link button div {{ color:#1A73E8 !important; }}
    </style>
    """, unsafe_allow_html=True)


def render_header_and_nav():
    is_home = st.session_state.page == "Home"
    logo = logo_img_tag(MAIN_LOGO if is_home else GENERAL_LOGO, 88 if is_home else 54)
    if logo:
        # Soft light plate keeps the (dark navy) logo readable on any background.
        st.markdown(
            f'<div style="padding:10px 0 6px;">'
            f'<span style="display:inline-block;background:rgba(255,255,255,0.62);'
            f'padding:8px 16px;border-radius:12px;">{logo}</span></div>'
            f'<div class="ac-rule"></div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
          <div class="ac-brandbar">
            <div class="ac-mark">{BRAND_MARK}</div>
            <div>
              <div class="ac-word">{BRAND}</div>
              <div class="ac-tag">{TAGLINE}</div>
            </div>
          </div>
          <div class="ac-rule"></div>
        """, unsafe_allow_html=True)

    cols = st.columns(len(NAV))
    for i, name in enumerate(NAV):
        active = st.session_state.page == name
        cols[i].button(name, key=f"nav_{name}", use_container_width=True,
                       type="primary" if active else "secondary",
                       on_click=go, args=(name,))
    st.write("")


def risk_legend():
    st.markdown(f"""
      <div class="ac-legend">
        <span class="ac-pill"><span class="ac-dot" style="background:{RED}"></span> At risk - act now</span>
        <span class="ac-pill"><span class="ac-dot" style="background:{AMBER}"></span> Needs attention</span>
        <span class="ac-pill"><span class="ac-dot" style="background:{GREEN}"></span> On track to renew</span>
      </div>
    """, unsafe_allow_html=True)


def brand_font():
    """Use a clean matplotlib font; fall back silently if unavailable."""
    for fam in ["DejaVu Sans"]:
        try:
            fm.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = fam
            break
        except Exception:
            pass


# ---------- SCENARIO CONTROLS (top of the Data page) -------------------------
# The pipeline reads the *_val keys below. These are plain session_state keys, NOT
# widget keys, so Streamlit never garbage-collects them when we navigate to a page
# that does not render the sliders. Each slider is initialized FROM its value key
# every render (value=...) and writes its change back through an on_change
# callback. Result: the controls always mirror the data set in use - on first
# open, after a change, and after navigating away and back.
_CTRL_DEFAULTS = {
    "n_accounts_val": DEFAULT_N_ACCOUNTS,
    "n_live_val":     DEFAULT_N_LIVE,
    "winner_val":     DEFAULT_WINNER_SHARE,
    "noise_val":      DEFAULT_NOISE_SHARE,
    "short_val":      DEFAULT_TERM_PROBS[0],
    "midterm_val":    DEFAULT_TERM_PROBS[1],
    "smb_val":        DEFAULT_SEGMENT_PROBS[0],
    "mid_val":        DEFAULT_SEGMENT_PROBS[1],
    "seed_val":       42,
}


def init_settings():
    """Seed the persistent control values once, so the pipeline can read them from
    the first run - before the Data page ever renders the widgets."""
    for k, v in _CTRL_DEFAULTS.items():
        st.session_state.setdefault(k, v)


def read_settings():
    s = st.session_state
    return dict(n_accounts=int(s["n_accounts_val"]), n_live=int(s["n_live_val"]),
                winner_share=float(s["winner_val"]), noise_share=float(s["noise_val"]),
                short_share=float(s["short_val"]), midterm_share=float(s["midterm_val"]),
                smb=float(s["smb_val"]),
                mid=float(s["mid_val"]), seed=int(s["seed_val"]))


def _sync(widget_key, value_key):
    """Copy a control's new value into its persistent value key."""
    st.session_state[value_key] = st.session_state[widget_key]


def render_data_controls():
    """The scenario controls across the top of the Data page. Each slider shows the
    value currently driving the data (value=...), and writes changes back to a
    persistent key (on_change), so the controls and the data set never fall out of
    step - including after leaving this page and returning."""
    st.markdown("<div class='ac-eyebrow'>Scenario controls</div>", unsafe_allow_html=True)
    st.caption("These controls reflect the data set currently in use. Move any control "
               "and every page recomputes; the tables below update to match.")
    ss = st.session_state

    r1 = st.columns(4)
    r1[0].slider("Historical accounts", 200, 1500, value=int(ss["n_accounts_val"]), step=100,
                 key="n_accounts_w", on_change=_sync, args=("n_accounts_w", "n_accounts_val"),
                 help="Past accounts used to learn the winning pattern. Higher keeps the "
                      "longer horizons well-populated.")
    r1[1].slider("Live accounts to score", 10, 200, value=int(ss["n_live_val"]), step=5,
                 key="n_live_w", on_change=_sync, args=("n_live_w", "n_live_val"),
                 help="How many current accounts fill the Portfolio. Starting point is 100.")
    r1[2].slider("Healthy-account share", 0.40, 0.95, value=float(ss["winner_val"]), step=0.05,
                 key="winner_w", on_change=_sync, args=("winner_w", "winner_val"),
                 help="Higher means fewer churners. The main dial on the risk mix.")
    r1[3].slider("Noise share", 0.0, 0.30, value=float(ss["noise_val"]), step=0.02,
                 key="noise_w", on_change=_sync, args=("noise_w", "noise_val"),
                 help="Accounts built to fool the model (outcome contradicts usage).")

    r2 = st.columns(4)
    r2[0].slider("12-month share", 0.0, 1.0, value=float(ss["short_val"]), step=0.05,
                 key="short_w", on_change=_sync, args=("short_w", "short_val"),
                 help="Share of accounts on the 12-month look-back horizon.")
    r2[1].slider("24-month share", 0.0, 1.0, value=float(ss["midterm_val"]), step=0.05,
                 key="midterm_w", on_change=_sync, args=("midterm_w", "midterm_val"),
                 help="Share on the 24-month horizon. 36-month fills the remainder.")
    r2[2].slider("SMB share", 0.0, 1.0, value=float(ss["smb_val"]), step=0.05,
                 key="smb_w", on_change=_sync, args=("smb_w", "smb_val"))
    r2[3].slider("Mid-market share", 0.0, 1.0, value=float(ss["mid_val"]), step=0.05,
                 key="mid_w", on_change=_sync, args=("mid_w", "mid_val"))

    r3 = st.columns(4)
    r3[0].number_input("Random seed", 0, 9999, value=int(ss["seed_val"]), step=1,
                       key="seed_w", on_change=_sync, args=("seed_w", "seed_val"),
                       help="Same seed reproduces the same data.")

    sh = float(ss["short_val"]); mt = float(ss["midterm_val"])
    tf = [sh, mt, max(0.0, 1.0 - sh - mt)]; tft = sum(tf) or 1.0
    smb = float(ss["smb_val"]); mid = float(ss["mid_val"])
    seg = [smb, mid, max(0.0, 1.0 - smb - mid)]; tot = sum(seg) or 1.0
    st.caption(f"Time-frame mix (normalized): 12-month {tf[0]/tft:.0%}, "
               f"24-month {tf[1]/tft:.0%}, 36-month {tf[2]/tft:.0%}.  |  "
               f"Segment mix (normalized): SMB {seg[0]/tot:.0%}, "
               f"Mid-market {seg[1]/tot:.0%}, Enterprise {seg[2]/tot:.0%}.")


# ---------- HOME -------------------------------------------------------------
def page_home(worklist):
    at_risk = int((worklist["risk_band"] == "At risk").sum()) if not worklist.empty else 0
    soon = int(((worklist["risk_band"] == "At risk") &
                (worklist["quarters_to_renewal"] <= 2)).sum()) if not worklist.empty else 0

    st.markdown(f"""
      <div class="ac-hero">
        <h1>See which customers are drifting off the adoption and value realization path, multiple quarters before the contract discussion.</h1>
        <p>This Agent learns what a winning account looks like in real time evaluating multiple signals across all accounts historically.  It will learn when, what and how to take actions
        to materially have customers succeed and realize value.  The solution places every live account on that curve, and turns the gap into a ranked, owner-assigned 
        action plan for forward deployed engineers and customer success teams.</p>
      </div>
    """, unsafe_allow_html=True)
    st.write("")

    c1, c2, c3 = st.columns(3)
    c1.metric("Live accounts scored", 0 if worklist.empty else len(worklist))
    c2.metric("Flagged at risk", at_risk)
    c3.metric("At risk and renewing soon", soon,
              help="At risk with two or fewer quarters to renewal.")
    st.write("")

    st.markdown("<div class='ac-eyebrow'>How it works</div>", unsafe_allow_html=True)
    cards = [
        ("01", "Learn the winning shape",
         "From history, plot how usage evolved for accounts that renewed or expanded versus those that fell short. That is the benchmark band."),
        ("02", "Find the early signals",
         "Measure which usage signals actually separated winners from losers, and at which quarter. Earlier and sharper is more useful."),
        ("03", "Place every live account",
         "Score each current account against the winning band at its own quarter, and rank the book by risk, size, and time to renewal."),
        ("04", "Hand over an action plan",
         "For each off-track account, produce a plan: the areas behind target, who owns each, and the number to move before renewal."),
    ]
    cc = st.columns(4)
    for col, (step, title, body) in zip(cc, cards):
        col.markdown(f"""<div class="ac-card"><div class="step">{step}</div>
                     <h4>{title}</h4><p>{body}</p></div>""", unsafe_allow_html=True)
    st.write("")

    b1, b2, b3 = st.columns(3)
    b1.button("Open the account portfolio", use_container_width=True, type="primary",
              on_click=go, args=("Portfolio",))
    b2.button("See how the model learns", use_container_width=True,
              on_click=go, args=("The Model",))
    b3.button("Read the assumptions", use_container_width=True,
              on_click=go, args=("Assumptions",))


# ---------- THE MODEL --------------------------------------------------------
def page_model(dataset):
    accounts, aq = dataset["accounts"], dataset["account_quarter"]
    st.header("How the model learns the adoption and value realization path")

    st.caption("Each account is scored on its Trajectory Signals - the factors that "
               "reveal where it is heading before renewal.")
    st.markdown(
        "<div class='ac-note'>The <b>time frame</b> is the look-back horizon the model "
        "learns over - 12, 24, or 36 months - not a contract term. This showcase is "
        "look-back only: it places each account on the learned path. Forward forecasting "
        "(projecting the path ahead) is the natural next step, not built here. As AI-era "
        "pricing moves toward committed consumption over a horizon, we frame this as a "
        "time frame rather than a fixed term.</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Winning vs losing path", "Which signals predict, and when"])

    with tab1:
        st.write("Pick a time frame and a Trajectory Signal. The green band is the "
                 "range for accounts that renewed or expanded; the red band is the "
                 "range for accounts that fell short. The moment they separate is the "
                 "moment this signal can tell a winner from a loser.")
        col_a, col_b = st.columns(2)
        _lbl = "font-size:0.875rem;font-weight:600;color:%s;margin:0 0 0.35rem;line-height:1.4;" % INK
        col_a.markdown(f'<div style="{_lbl}">Time frame (look-back)</div>', unsafe_allow_html=True)
        term_sel = col_a.selectbox("Time frame", TERM_CHOICES,
                                   format_func=lambda t: TERM_LABEL_LONG.get(t, f"{t}q"),
                                   label_visibility="collapsed")
        # Inline label: plain text, with only the parenthetical a link to the library.
        col_b.markdown(
            f'<div style="{_lbl}">Trajectory Signals '
            f'<span style="font-weight:400;">(<a href="?nav=signals" target="_self" '
            f'style="color:#1A73E8;text-decoration:none;">See Signal Library Definitions</a>)</span>'
            f'</div>', unsafe_allow_html=True)
        # Only per-quarter signals have a quarter-by-quarter band to draw.
        metric_sel = col_b.selectbox("Trajectory Signal", QUARTER_SIGNALS,
                                     format_func=lambda m: METRIC_LABELS[m],
                                     label_visibility="collapsed")
        bands = cohort_bands(aq, accounts, term_sel, metric_sel)
        fig, ax = plt.subplots(figsize=(9, 4.1))
        for grp, color, lbl in (("winning", GREEN, "Renewed / expanded"),
                                ("losing", RED, "Fell short")):
            s = bands[grp]
            if s.empty:
                continue
            ax.plot(s["quarter_within_term"], s["median"], color=color, linewidth=2.4, label=lbl)
            ax.fill_between(s["quarter_within_term"], s["lo"], s["hi"], color=color, alpha=0.13)
        ax.set_xlabel("Quarter within time frame")
        ax.set_ylabel(METRIC_LABELS[metric_sel])
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        st.pyplot(fig)

    with tab2:
        st.write("Each cell scores one signal at one quarter. Read it as separation "
                 "power: 0.50 means the signal tells you nothing yet, 1.00 means it "
                 "perfectly splits winners from losers. Greener and further left is "
                 "better, because it is a clearer signal that arrives earlier.")
        term_auc = st.selectbox("Time frame ", TERM_CHOICES,
                                format_func=lambda t: TERM_LABEL_LONG.get(t, f"{t}q"),
                                key="auc_term")
        auc_df = indicator_auc(aq, accounts, term_auc)
        pivot = auc_df.pivot(index="metric", columns="quarter_within_term", values="auc")
        pivot.index = [METRIC_LABELS.get(m, m) for m in pivot.index]
        pivot.index.name = "Signal"
        pivot.columns = [f"Q{c}" for c in pivot.columns]
        st.dataframe(pivot.style.background_gradient(cmap="Greens", vmin=0.5, vmax=1.0)
                     .format("{:.2f}"), use_container_width=True)
        st.markdown("<div class='ac-note'>In this synthetic data, consumption "
                    "concentration and workflow breadth separate winners from losers "
                    "earlier than raw consumption, because a single consumption reading "
                    "lags the trend. To be re-checked on real data.</div>", unsafe_allow_html=True)


# ---------- PORTFOLIO --------------------------------------------------------
SOON_MAX_QTRS = 2   # "renewing soon" = this many quarters or fewer to renewal
ZONE_BASE     = {"At risk": 0, "Needs attention": 1, "On track": 2}  # chart rows
ZONE_DOT      = {"At risk": "\U0001F534", "Needs attention": "\U0001F7E1",
                 "On track": "\U0001F7E2"}


def _classify_cells(df):
    """Tag each account with its time half: renewing soon vs more time."""
    df = df.copy()
    df["_soon"] = df["quarters_to_renewal"] <= SOON_MAX_QTRS
    return df


def _sort_risk_then_renewal(df):
    """Default order: risk band (red, amber, green), then soonest renewal first."""
    d = df.copy()
    d["_r"] = d["risk_band"].map(RISK_ORDER)
    d = d.sort_values(["_r", "quarters_to_renewal"], ascending=[True, True])
    return d.drop(columns="_r")


def _quadrant_chart(df, big=False):
    """Risk (vertical zones) vs time-to-renewal (horizontal). No account labels.

    Vertical position is driven by the RISK BAND, not the raw gap, so colour and
    height always agree: green in the top zone, amber in the middle, red at the
    bottom, regardless of time to renewal. Within a zone a bigger gap sits a
    little lower; tiny jitter stops bubbles overlapping.
    """
    fig, ax = plt.subplots(figsize=(10.4, 5.2) if big else (7.8, 3.7))
    xmax = max(df["quarters_to_renewal"].max(), 3) + 0.5

    jrng = np.random.default_rng(7)
    d = df.copy()
    d["_xj"] = d["quarters_to_renewal"] + jrng.normal(0, 0.07, len(d))

    def _yval(sub):
        g = sub["total_gap"].to_numpy(dtype=float)
        lo, hi = g.min(), g.max()
        norm = (g - lo) / (hi - lo) if hi > lo else np.full_like(g, 0.5)
        within = 0.15 + 0.70 * (1.0 - norm)            # more gap -> lower in the zone
        base = sub["risk_band"].map(ZONE_BASE).to_numpy(dtype=float)
        return base + within + jrng.normal(0, 0.015, len(sub))

    d["_yj"] = 0.0
    for band in ZONE_BASE:
        m = d["risk_band"] == band
        if m.any():
            d.loc[m, "_yj"] = _yval(d[m])

    ax.axhspan(2, 3, color=GREEN, alpha=0.05)
    ax.axhspan(1, 2, color=AMBER, alpha=0.05)
    ax.axhspan(0, 1, color=RED,   alpha=0.06)
    ax.axvline(SOON_MAX_QTRS + 0.5, color=INK, alpha=0.15, linestyle="--", linewidth=1)

    smax = max(d["contract_value"].max(), 1)
    for band in ["On track", "Needs attention", "At risk"]:
        sub = d[d["risk_band"] == band]
        if sub.empty:
            continue
        sizes = 40 + (sub["contract_value"] / smax) * (460 if big else 320)
        ax.scatter(sub["_xj"], sub["_yj"], s=sizes,
                   c=RISK_COLOR[band], alpha=0.85, edgecolor="white", linewidth=1.2)

    ax.text(0.0, 0.12, "Act now", color=RED, fontsize=10, va="bottom",
            fontfamily="monospace")
    ax.set_xlim(-0.5, xmax)
    ax.set_ylim(0, 3)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["At risk", "Needs attention", "On track"])
    ax.set_xlabel("Time to renewal:  sooner \u2190          \u2192 later")
    ax.grid(axis="x", alpha=0.15)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    return fig


def _portfolio_table(df_sorted, key):
    """Clickable account list; clicking a row opens that account's plan."""
    show = df_sorted.reset_index(drop=True).copy()
    show["risk_band"] = show["risk_band"].map(RISK_BADGE)
    show = show[["account_id", "risk_band", "segment", "term_label",
                 "current_quarter", "quarters_to_renewal", "off_target_count",
                 "top_drivers", "contract_value", "priority_score",
                 "true_outcome_hidden"]]

    st.caption("Click any row to open that account's plan.")
    event = st.dataframe(
        show, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key=key,
        column_config={
            "account_id":          st.column_config.TextColumn("Account", help="Account identifier"),
            "risk_band":           st.column_config.TextColumn("Risk", help="Overall renewal risk"),
            "segment":             st.column_config.TextColumn("Segment", help="Customer size tier"),
            "term_label":          st.column_config.TextColumn("Term", help="Contract length"),
            "current_quarter":     st.column_config.NumberColumn("Now Q", help="Current quarter in term"),
            "quarters_to_renewal": st.column_config.NumberColumn("Qtrs to renewal", help="Quarters until renewal"),
            "off_target_count":    st.column_config.NumberColumn("Off-target", help="Signals behind target"),
            "top_drivers":         st.column_config.TextColumn("Main gaps", help="Largest off-target signals"),
            "contract_value":      st.column_config.NumberColumn("Contract value", help="Credits over full term", format="%d"),
            "priority_score":      st.column_config.NumberColumn("Priority", help="Higher means act sooner", format="%.2f"),
            "true_outcome_hidden": st.column_config.TextColumn("Actual (hidden)", help="Real result, model never sees it"),
        })

    rows = event.selection["rows"]
    if rows:
        picked_id = show.iloc[rows[0]]["account_id"]
        st.session_state.pop(key, None)     # clear selection so it doesn't re-fire
        go("Account Plan", picked_id)
        st.rerun()

    st.caption("The last column is the true outcome, hidden from the model and shown "
               "only so you can sanity-check the ranking by eye.")


def _focus_cell(band, soon):
    """Callback: drill into one chart cell (risk band x time half)."""
    st.session_state.portfolio_focus = (band, bool(soon))
    st.session_state.pop("portfolio_table", None)


def _clear_focus():
    st.session_state.portfolio_focus = None
    st.session_state.pop("portfolio_focus_table", None)


def _cell_label(band, soon):
    return f"{band} \u00b7 {'renewing soon' if soon else 'more time'}"


def page_portfolio(worklist):
    st.header("Account portfolio")

    if worklist.empty:
        st.info("No live accounts to show. Raise 'Live accounts to score' in the controls at the top of the Data page.")
        return

    # A clicked chart cell drills into just those accounts, then stops here.
    focus = st.session_state.get("portfolio_focus")
    if focus:
        _portfolio_focus_view(worklist, focus)
        return

    st.write("Every live account, scored against the winning path at its own quarter. "
             "Use this to find the accounts that are both high risk and close to renewal.")

    # ---- controls ----
    c1, c2 = st.columns(2)
    sort_choice = c1.selectbox("Sort by", [
        "Risk, then soonest renewal", "Soonest renewal first",
        "Highest priority first", "Largest accounts first"])
    bands_pick = c2.multiselect("Show risk levels",
                                ["At risk", "Needs attention", "On track"],
                                default=["At risk", "Needs attention", "On track"])

    df = worklist[worklist["risk_band"].isin(bands_pick)].copy()
    if df.empty:
        st.info("No accounts match the selected risk levels. Add one back in "
                "'Show risk levels' above.")
        return
    df = _classify_cells(df)

    # ---- chart (left) + clickable cells (right) ----
    st.markdown("<div class='ac-eyebrow' style='margin-top:6px'>Risk vs time to renewal</div>",
                unsafe_allow_html=True)
    left, right = st.columns([2, 1])
    with left:
        st.pyplot(_quadrant_chart(df))
    with right:
        st.markdown("<div class='ac-eyebrow'>Open a cell</div>", unsafe_allow_html=True)
        st.caption("Click a cell to see just those accounts, expanded.")
        for band in ["On track", "Needs attention", "At risk"]:
            for soon in (True, False):
                n = int(((df["risk_band"] == band) & (df["_soon"] == soon)).sum())
                when = "soon" if soon else "later"
                slug = band.replace(" ", "").lower()
                st.button(f"{ZONE_DOT[band]} {band} \u00b7 {when} ({n})",
                          key=f"cellbtn_{slug}_{'soon' if soon else 'later'}",
                          use_container_width=True, disabled=(n == 0),
                          on_click=_focus_cell, args=(band, soon))

    st.caption("Vertical is risk, horizontal is time to renewal (sooner on the left). "
               "Bubble size is contract value; the dashed line marks about two quarters "
               "out, so the lower-left cell - at risk and renewing soon - is act-now.")

    # ---- sortable table ----
    if sort_choice == "Risk, then soonest renewal":
        df = _sort_risk_then_renewal(df)
    elif sort_choice == "Soonest renewal first":
        df = df.sort_values(["quarters_to_renewal", "total_gap"], ascending=[True, False])
    elif sort_choice == "Largest accounts first":
        df = df.sort_values("contract_value", ascending=False)
    else:
        df = df.sort_values("priority_score", ascending=False)

    _portfolio_table(df, key="portfolio_table")


def _portfolio_focus_view(worklist, focus):
    """Expanded chart + list for a single clicked cell (risk band x time half)."""
    band, soon = focus
    df = _classify_cells(worklist)
    sub = df[(df["risk_band"] == band) & (df["_soon"] == soon)].copy()

    top = st.columns([3, 1])
    top[0].markdown(f"<div class='ac-eyebrow'>Focused cell</div>"
                    f"<h3 style='margin:2px 0 0 0'>{_cell_label(band, soon)} "
                    f"&middot; {len(sub)} account(s)</h3>", unsafe_allow_html=True)
    top[1].button("\u2190 Back to full portfolio", use_container_width=True,
                  on_click=_clear_focus)

    if sub.empty:
        st.info("No accounts in this cell right now.")
        return

    st.pyplot(_quadrant_chart(sub, big=True))
    st.caption("Only the accounts in this cell are plotted, on the same axes as the "
               "full portfolio so you can see where they sit.")

    _portfolio_table(_sort_risk_then_renewal(sub), key="portfolio_focus_table")


# ---------- ACCOUNT PLAN -----------------------------------------------------
def page_account_plan(worklist, plans):
    st.header("Account action plan")

    if worklist.empty or not plans:
        st.info("No accounts to plan yet. Open the Portfolio tab first.")
        return

    ids = worklist["account_id"].tolist()
    default_id = st.session_state.get("selected_account")
    if default_id not in ids:
        default_id = ids[0]

    c1, c2 = st.columns([2, 1])
    chosen = c1.selectbox("Account", ids, index=ids.index(default_id))
    c2.button("\u2190 Back to portfolio", use_container_width=True,
              on_click=go, args=("Portfolio",))
    st.session_state.selected_account = chosen

    plan = build_action_plan(chosen, plans[chosen])
    risk = plan["risk_band"]

    st.markdown(f"""
      <div class="ac-card" style="border-left:5px solid {RISK_COLOR[risk]}; margin-bottom:14px">
        <div class="step" style="color:{RISK_COLOR[risk]}">{RISK_BADGE[risk]}</div>
        <h4 style="margin-top:4px">{chosen} &middot; {plan['segment']}</h4>
        <p>Quarter {plan['current_quarter']} of {plan['term_quarters']} &middot;
        {plan['quarters_to_renewal']} quarter(s) to renewal &middot;
        {plan['on_track_count']} of {plan['tracked_count']} tracked signals on track.</p>
      </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='ac-note'>This plan lists only the areas where the account "
                "is behind the accounts that renewed. Each row names the team that should "
                "act, the number to improve, and how far the account is from the target "
                "today. The <b>Recommended play</b> column is empty on purpose: that is "
                "where your team's proven step goes (for example, \"run an executive "
                "business review\"). We leave it blank because the right play depends on "
                "your company's own playbook, which a real deployment would load in.</div>",
                unsafe_allow_html=True)
    st.write("")

    if plan["off_target_drivers"]:
        pdf = pd.DataFrame(plan["off_target_drivers"])
        st.dataframe(pdf, use_container_width=True, hide_index=True, column_config={
            "Status":             st.column_config.TextColumn("Status", help="Red, amber or green"),
            "Risk area":          st.column_config.TextColumn("Risk area", help="Signal that is off target"),
            "Why it matters":     st.column_config.TextColumn("Why it matters", help="Plain reason it predicts churn"),
            "Owner":              st.column_config.TextColumn("Owner", help="Team responsible to act"),
            "Metric to move":     st.column_config.TextColumn("Metric to move", help="The number to improve"),
            "Current vs winning": st.column_config.TextColumn("Current vs winning", help="Now versus the winning benchmark"),
            "Recommended play":   st.column_config.TextColumn("Recommended play", help="Your playbook step (blank)"),
        })
    else:
        st.success("No areas off target at this quarter. This account is tracking the "
                   "winning path and is on course to renew.")

    with st.expander("Raw action plan (JSON, as the batch prototype writes it)"):
        st.json(plan)


# ---------- DATA -------------------------------------------------------------
# ---------- TRAJECTORY SIGNALS (library + two tier pages) --------------------
# Per-signal one-line definition, the calculation in plain English, and (Tier 2
# only) the short manual/upkeep note. Kept tight on purpose.
SIGNAL_PAGE = {
    "consumption_vs_commit":     ("Are they using what they bought?",
                                  "Usage divided by commitment each quarter, watched as a trend.", None),
    "consumption_concentration": ("Is usage spread out or riding on one team?",
                                  "Share of total usage coming from the single biggest user or team.", None),
    "integrations_live":         ("How many live connections into their systems.",
                                  "A straight count of active integrations read from the environment.", None),
    "active_users":              ("How many people actually work in the product.",
                                  "Count of distinct active users each quarter; rolls into engagement.", None),
    "unique_logins":             ("How many different people show up, not just how often.",
                                  "Count of distinct people logging in; rolls into engagement.", None),
    "logins":                    ("Raw login activity.",
                                  "Total logins per period, used only to back up unique logins.", None),
    "activated_workflows":       ("Did a deployed agent truly catch on early?",
                                  "Agents past 50 real uses in their first quarter live; growth rolls into engagement.", None),
    "workflow_breadth":          ("How many different parts of the solution are in use.",
                                  "Count of distinct workflows in use each period; shrinking is an early warning.", None),
    "features_used":             ("How widely the product's features are used.",
                                  "Count of distinct features touched each period; rolls into engagement.", None),
    "time_to_deploy":            ("How fast the FDE got it live.",
                                  "Days from contract start to the agent going live.", None),
    "time_to_value":             ("How fast the customer actually started using it.",
                                  "Days from start to the first real, non-test use.", None),
    "grounding_fail_rate":       ("How often answers aren't backed by the source data.",
                                  "Share of checked outputs that fail a grounding test, watched as a trend.", None),
    "support_tickets":           ("Support load.",
                                  "Ticket count and severity per period; weak on its own.", None),
    "escalations":               ("Serious, unresolved issues.",
                                  "Count of escalated issues per period; rare but meaningful.", None),
    "eval_score":                ("How well an FDE-built solution hits the value it was built for.",
                                  "Its score against the customer's own test set, as percent of the agreed target reached.",
                                  "Needs a value/outcome baseline set with the customer and reviewed each period."),
    "outcomes_produced":         ("How many defined business results it delivered.",
                                  "Count of outcome events per period.",
                                  "Needs a per-account definition of \u201Can outcome\u201D where the product doesn't emit one."),
    "cost_per_outcome":          ("What each result costs.",
                                  "Period cost divided by outcomes produced.",
                                  "Only as solid as the outcome definition beneath it."),
    "exec_sponsor_nps":          ("How the exec sponsor rates us, and whether we have one.",
                                  "The sponsor's recommend score rolled to an account read; no sponsor at all is its own risk flag.",
                                  "Needs the sponsor identified and kept current."),
    "champion_present":          ("Is there an engaged internal advocate?",
                                  "A maintained yes/no flag; feeds relationship health.",
                                  "Degrades silently if not updated."),
    "exec_touch_recency":        ("How long since we engaged an executive.",
                                  "Days since the last logged exec touch; feeds relationship health.",
                                  "Only as good as the touch-logging."),
}


def _signal_rows_html(tier):
    rows = []
    for k in SIGNALS:
        if SIGNAL_TIER[k] != tier:
            continue
        definition, calc, note = SIGNAL_PAGE[k]
        up = METRIC_DIRECTION[k] > 0
        arrow = (f'<span style="color:{GREEN};font-weight:700;">&#8593;</span>' if up
                 else f'<span style="color:{RED};font-weight:700;">&#8595;</span>')
        note_html = (f'<div style="color:{AMBER};font-size:0.86rem;margin-top:2px;">'
                     f'Manual: {note}</div>') if note else ""
        rows.append(
            f'<div style="padding:9px 0;border-bottom:1px solid {LINE};">'
            f'<div style="color:{INK};font-weight:700;">{arrow}&nbsp; {METRIC_LABELS[k]}'
            f'<span style="color:{MUTE};font-weight:500;"> &mdash; {definition}</span></div>'
            f'<div style="color:#33424C;font-size:0.9rem;margin-top:2px;"><i>How the model reads it:</i> {calc}</div>'
            f'{note_html}</div>')
    return "".join(rows)


def page_trajectory_library():
    st.header("CS/FDE Trajectory Signal Library")
    st.write("The signals the model weighs to see where an account is heading - toward "
             "renewal and expansion, or toward churn - a quarter or more before the "
             "renewal conversation. Open a tier below for the full list.")

    subs = [
        ("Direction", "&#8593; means higher is better; &#8595; means lower is better."),
        ("Tier 1 - Trusted automatic telemetry",
         "Signals that come straight from product and environment data with low upkeep. The load-bearing ones."),
        ("Tier 2 - Manual maintained",
         "Signals that need people to keep them current. A missing value is treated as unknown, never as bad."),
        ("Tags",
         "Some Tier 2 signals need a value/outcome baseline set with the customer, or depend on CRM hygiene; those are flagged in-line."),
        ("Sub-scores",
         "Engagement signals (usage, logins, activation, breadth) roll into one score, and relationship signals "
         "(sponsor, champion, exec-touch) into another, so correlated signals don't each count separately."),
        ("Governing principle",
         "No single signal is treated as true. The model learns which signals, and in what combination, actually "
         "correlate with renewal and expansion, and revisits those weights over time."),
    ]
    cards = "".join(
        f'<div style="padding:8px 0;border-bottom:1px solid {LINE};">'
        f'<div style="color:{INK};font-weight:700;">{t}</div>'
        f'<div style="color:#33424C;font-size:0.92rem;margin-top:2px;">{b}</div></div>'
        for t, b in subs)
    st.markdown(f'<div class="ac-card">{cards}</div>', unsafe_allow_html=True)
    st.write("")

    b1, b2 = st.columns(2)
    b1.button("Open Tier 1 - Trusted automatic telemetry", use_container_width=True,
              type="primary", on_click=go, args=("Signals - Tier 1",))
    b2.button("Open Tier 2 - Manual maintained", use_container_width=True,
              on_click=go, args=("Signals - Tier 2",))


def page_signals_tier(tier):
    title = "Tier 1 - Trusted automatic telemetry" if tier == 1 else "Tier 2 - Manual maintained"
    st.header(title)
    if tier == 1:
        st.caption("From product and environment data. Low upkeep - the load-bearing signals.")
    else:
        st.caption("Need people to keep them current. A missing value means unknown, not bad.")
    st.markdown(f'<div class="ac-card">{_signal_rows_html(tier)}</div>', unsafe_allow_html=True)
    st.write("")
    st.button("\u2190 Back to the Signal Library", on_click=go, args=("Trajectory Signals",))


def page_data(dataset):
    st.header("Data")
    render_data_controls()
    st.markdown("<hr style='border:none;border-top:1px solid %s;margin:8px 0 16px'>" % LINE,
                unsafe_allow_html=True)
    st.write("These four tables are worked examples of the exact shapes your real data "
             "must match to run this on live accounts. Download any of them to see the "
             "columns. The tables reflect the scenario controls above.")
    labels = {
        "accounts": "One row per account: who they are, and how their renewal ended.",
        "account_quarter": "One row per account per quarter: every usage signal we track.",
        "account_quarter_feature": "Usage split by feature, for consumption breakdowns.",
        "event_log": "A sample of raw events for a few accounts, the atomic grain.",
    }
    for name in ["accounts", "account_quarter", "account_quarter_feature", "event_log"]:
        df = dataset[name]
        st.markdown(f"**{name}** &nbsp;<span style='color:{MUTE}'>&mdash; {labels[name]} "
                    f"({len(df):,} rows)</span>", unsafe_allow_html=True)
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)
        st.download_button(f"Download {name}.csv",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name=f"{name}.csv", mime="text/csv", key=f"dl_{name}")
        st.write("")


# ---------- ASSUMPTIONS ------------------------------------------------------
def page_assumptions():
    st.header("Assumptions and how to read the results")

    st.markdown("<div class='ac-note'>This is an <b>exploration prototype</b> running on "
                "<b>synthetic (invented) data</b>. Its job is to prove the logic before "
                "real data is gathered, not to make live renewal decisions. The outcome "
                "bands and the way accounts are grouped are provisional choices, set so we "
                "can move forward.</div>", unsafe_allow_html=True)
    st.write("")

    st.subheader("What the colours mean")
    risk_legend()
    st.write("Two different things carry a colour, and they are set differently on purpose.")
    st.markdown(
        f"**A single signal** inside an account's plan is coloured by how far it sits "
        f"outside the winning range, measured in \"spreads\" (the normal range of the "
        f"winning group): <b style='color:{GREEN}'>green</b> within or better than the "
        f"range, <b style='color:{AMBER}'>amber</b> modestly outside, and "
        f"<b style='color:{RED}'>red</b> more than one full spread outside.",
        unsafe_allow_html=True)
    st.markdown(
        f"**An account's overall band** is deliberately not set by any single signal - one "
        f"wobbly metric should not condemn a whole account. Accounts are ranked by their "
        f"total gap and grouped into the operating tiers below. So an account can be "
        f"<b style='color:{GREEN}'>green</b> overall while still showing a "
        f"<b style='color:{RED}'>red</b> signal or two to work on in its plan.",
        unsafe_allow_html=True)
    st.write('"Current vs winning" reads as the account\'s number versus the winning '
             'benchmark, with plain words for direction. For example, "0.78 vs winning '
             '0.94 (below target)" means the account is behind; "1.00 vs winning 0.94 '
             '(at/above target)" means it is fine on that signal.')

    st.subheader("Positioning: what a healthy book looks like")
    st.write("Account bands are calibrated to the ranges a well-run book would show at any "
             "point in the customer journey, rather than letting one off metric inflate the "
             "risk count. Anything above roughly 20% at risk reads as a company off the "
             "rails; these ranges keep the picture realistic and hopeful. They are the "
             "go-in guidelines for the synthetic data and are adjustable.")
    st.table(pd.DataFrame({
        "Band": ["At risk (red)", "Needs attention (amber)", "On track (green)"],
        "Target share of the live book": ["6-10%", "8-15%", "75-86%"],
        "How it is set": [
            "Highest total-gap accounts (about the top 8%) that are genuinely off.",
            "The next tier by total gap (about 12%).",
            "The remainder - within or close to the winning range.",
        ],
    }))
    st.caption("A genuinely healthier book shows fewer flags than the target; the "
               "healthy-account share control at the top of the Data page moves this.")

    st.subheader("Provisional outcome bands")
    st.write("An account's fate is labelled from where its end-of-term consumption lands "
             "against what it committed to buy.")
    st.table(pd.DataFrame({
        "End-of-term consumption vs commitment": ["115% or more", "100% to 115%",
                                                  "90% to 100%", "75% to 90%", "below 75%"],
        "Outcome": ["Expansion", "Full renewal", "90-99% (neutral)", "Under 90%", "Churn"],
    }))

    st.subheader("Time frames, not contract terms")
    st.write("The model reasons over a look-back time frame - the window it learns a "
             "winning trajectory across and places a live account on - rather than a "
             "signed contract term. This is deliberate: as AI-era pricing moves toward "
             "committed consumption over a horizon (use-it-or-lose-it, so the vendor can "
             "recognize revenue), the commercial wrapper is getting shorter and less "
             "stable, while the model only needs a horizon to reason over. The showcase "
             "is look-back only; forward forecasting is the natural next step and is not "
             "built here. We show three horizons - 12, 24, and 36 months - to demonstrate "
             "the method generalizes across spans.")
    st.write("Each horizon is learned separately, on a quarter-within-horizon axis, "
             "because a 12-month account has to reach healthy usage far faster than a "
             "36-month one. The winning and losing paths separate at about the same "
             "fraction of the horizon for each, which in real time arrives sooner for the "
             f"shorter ones. The horizon mix defaults short ({int(DEFAULT_TERM_PROBS[0]*100)}% "
             f"/ {int(DEFAULT_TERM_PROBS[1]*100)}% / {int(DEFAULT_TERM_PROBS[2]*100)}%) to "
             "reflect where buying is heading.")

    st.subheader(f"How much history this assumes ({HISTORY_YEARS} years)")
    st.write(f"The bands are calibrated assuming a company brings about {HISTORY_YEARS} "
             "years of usable adoption history. That gives the 12-month horizon "
             "roughly four completed cycles, the 24-month about two, and the 36-month at "
             "least one full rolling cycle - which keeps firming up as more cycles accrue, "
             "so the model is weakest on day one and compounds from there. Companies with "
             "materially less history get an illustrative walkthrough rather than an "
             "immediately actionable read.")
    st.markdown(
        "**Two-speed history.** The adoption and value realization backbone - consumption, engagement, "
        "relationship, support - can genuinely reach back four-plus years. The newer AI-native "
        "signals (grounding-failure rate, eval score, activated workflows, time-to-value "
        "on agents) cannot yet, for anyone: embedded enterprise AI mostly landed in "
        "2023-2024, so almost no one has more than two to three years on them. Those "
        "signals therefore carry less weight early and strengthen as deployments age. "
        "This is an accurate account of where the data is in 2026, not a flaw to hide.",
        unsafe_allow_html=True)
    st.caption("The synthetic data does not simulate a real calendar; it stands in for a "
               "book of completed histories at these horizons.")

    st.subheader("The noise accounts")
    st.write("About one in eight accounts is deliberate \"noise\": strong usage that still "
             "churns (a budget cut or acquisition), or weak usage that renews anyway (too "
             "costly to switch). They are included so we can later test how often the model "
             "is fooled, rather than assuming it never is.")

    st.subheader("Honest limitations")
    st.markdown(
        "- The synthetic data is cleaner than a real book, so real bands will be noisier.\n"
        "- The score is a first-pass rule, not a trained model, and it depends on which quarter you look at.\n"
        "- Priority is weighted by account value, so a large under-performing account can outrank a small churning one. That is a deliberate triage choice, and it is adjustable.\n"
        "- Only the first time frame is modelled, so survivorship is handled lightly.\n"
        "- Forward forecasting (look-forward) is out of scope here; this is a look-back, at-risk-detection model.\n"
        "- The play library (the specific recommended actions) is the one piece only your organisation can supply.")

    st.write("")
    st.button("Back to home", type="primary", on_click=go, args=("Home",))


# =============================================================================
# ROUTER
# =============================================================================
def main():
    st.set_page_config(page_title=f"{BRAND} | Adoption and Value Realization",
                       page_icon="\u25C9", layout="wide",
                       initial_sidebar_state="collapsed")
    inject_css()
    brand_font()

    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "selected_account" not in st.session_state:
        st.session_state.selected_account = None
    if "portfolio_focus" not in st.session_state:
        st.session_state.portfolio_focus = None

    # In-text links (e.g. "See Signal Library Definitions" on The Model page)
    # navigate by setting a URL query parameter; handle it, then clear it.
    if st.query_params.get("nav") == "signals":
        st.session_state.page = "Trajectory Signals"
        st.query_params.clear()

    init_settings()
    settings = read_settings()
    dataset, worklist, plans = run_pipeline(
        settings["n_accounts"], settings["short_share"], settings["midterm_share"],
        settings["smb"], settings["mid"], settings["noise_share"], settings["winner_share"],
        settings["seed"], settings["n_live"])

    inject_page_background(st.session_state.page)
    render_header_and_nav()

    page = st.session_state.page
    if page == "Home":
        page_home(worklist)
    elif page == "The Model":
        page_model(dataset)
    elif page == "Trajectory Signals":
        page_trajectory_library()
    elif page == "Signals - Tier 1":
        page_signals_tier(1)
    elif page == "Signals - Tier 2":
        page_signals_tier(2)
    elif page == "Portfolio":
        page_portfolio(worklist)
    elif page == "Account Plan":
        page_account_plan(worklist, plans)
    elif page == "Data":
        page_data(dataset)
    elif page == "Assumptions":
        page_assumptions()


if __name__ == "__main__":
    main()
