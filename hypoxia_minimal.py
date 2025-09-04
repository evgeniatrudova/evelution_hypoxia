#   streamlit run hypoxia_minimal.py


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="EVelution — Hypoxia EV (Single File)", layout="wide")

# --- Math / helpers -----------------------------------------------------------
def logistic(x): return 1.0/(1.0+np.exp(-x))

def lowO2_driver(O2, theta, sigma):
    x = np.maximum(0.0, theta - O2)
    return 1.0 - np.exp(-0.5 * (x**2) / (sigma**2))   # smooth 0→1 when O2<θ

def simulate(T, r_T, kW, kS, ATP, ROS, S, hyp_scale, hyp_theta, hyp_sigma,
             o2_high, o2_low, t_on, t_off, hyp_on=True, noise=0.4, seed=42):
    t = np.arange(T, dtype=float)
    O2 = np.full_like(t, o2_high, dtype=float)
    if hyp_on:
        O2[(t >= t_on) & (t <= t_off)] = o2_low
    W   = 0.6*ATP + 0.4*(1.0-ROS)
    gW  = logistic(kW*(W-0.5))
    gS  = logistic(kS*(S-0.5))
    base = r_T * gW * gS
    d        = lowO2_driver(O2, hyp_theta, hyp_sigma)
    lam_hyp  = 1.0 + (hyp_scale - 1.0) * d
    rate     = base * lam_hyp
    rng = np.random.default_rng(seed)
    if noise > 0:
        alpha = 1.0/(noise+1e-9)
        lam   = rng.gamma(shape=alpha, scale=rate/alpha)
        counts = rng.poisson(lam)
    else:
        counts = rng.poisson(rate)
    return pd.DataFrame({
        "time_min": t, "O2_pct": O2, "driver_d": d,
        "gW": np.full_like(t, gW, dtype=float),
        "gS": np.full_like(t, gS, dtype=float),
        "lambda_hyp": lam_hyp, "rate": rate, "counts": counts
    })

def fit_params(y, base_const, O2, theta_grid, sigma_grid):
    best, y_over_b = None, y/np.maximum(1e-9, base_const)
    for theta in theta_grid:
        for sigma in sigma_grid:
            d = lowO2_driver(O2, theta, sigma)
            num = np.sum(d * (y_over_b - 1.0))
            den = np.sum(d*d) + 1e-12
            a  = max(0.0, num/den)              # ensures λ_max ≥ 1
            lam = base_const * (1.0 + a * d)
            err = np.mean((y - lam)**2)
            if (best is None) or (err < best["err"]):
                best = {"theta":theta, "sigma":sigma, "a":a, "err":err, "lam":lam, "d":d}
    lam_max_hat = 1.0 + best["a"]
    m, v = float(np.mean(y)), float(np.var(y, ddof=1)) if len(y)>1 else 0.0
    phi  = max(0.0, (v - m) / max(1e-9, m*m))         # overdispersion moment estimate
    return {"theta":best["theta"], "sigma":best["sigma"], "lam_max":lam_max_hat,
            "phi":phi, "lam_fit":best["lam"], "d_fit":best["d"]}

def kalman_counts(y, init=0.0, q=50.0, r_scale=1.0):
    x, P, out = init, 1e6, []
    for obs in y:
        x_pred, P_pred = x, P + q
        R = r_scale * max(1.0, obs)
        K = P_pred / (P_pred + R)
        x = x_pred + K*(obs - x_pred)
        P = (1 - K)*P_pred
        out.append(x)
    return np.array(out)

def make_ts(x, ys, names, ytitle, win=None):
    fig = go.Figure()
    for y, name in zip(ys, names):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    layout = dict(
        margin=dict(l=10,r=10,t=30,b=10), height=280,
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=30, label="30m", step="minute", stepmode="backward"),
                dict(count=60, label="1h",  step="minute", stepmode="backward"),
                dict(step="all")
            ]),
            rangeslider=dict(visible=True),
            title="time (min)"
        ),
        yaxis=dict(title=ytitle),
        legend=dict(orientation="h", y=1.02, x=0)
    )
    if win is not None:
        layout["xaxis"]["range"] = [win[0], win[1]]
    fig.update_layout(**layout).update_traces(hovertemplate="t=%{x} min<br>%{y}")
    fig.update_layout(dragmode="pan")
    return fig

# --- Defaults / state ---------------------------------------------------------
DEFAULTS = dict(
    T=240, r_T=2000.0, kW=6.0, kS=6.0, ATP=0.7, ROS=0.3, S=0.6,
    hyp_scale=1.4, hyp_theta=5.0, hyp_sigma=3.0,
    o2_high=21.0, o2_low=1.0, t_on=60, t_off=180, hyp_on=True, noise=0.4
)
for k,v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# --- UI: tabs -----------------------------------------------------------------
st.title("EVelution — Hypoxia EV (Single File)")

tab_learn, tab_preview, tab_tweak, tab_lab, tab_report = st.tabs(
    ["1) Learn", "2) Preview", "3) Tweak", "4) Lab Data", "5) Report"]
)

# 1) Learn
with tab_learn:
    st.markdown("""
**Hypothesis.** EVs are **stress-responsive**. Under **hypoxia** (low %O₂) cells often increase EV output.
EVelution simulates this, blends **wet-lab** and **in-silico** data, and explains parameters.

**Model**
\\[
r(t) = r_T\\, g_S(S)\\, g_W(W)\\, \\lambda_{hyp}(O_2(t)),\\qquad
W = 0.6\\,ATP + 0.4\\,(1-ROS)
\\]
\\[
g_X(x) = \\frac{1}{1+e^{-k_X(x-0.5)}},\\quad
\\lambda_{hyp} = 1 + (\\lambda_{max}-1)\\,d,\\quad
d = 1-\\exp\\!\\left(-\\tfrac{1}{2}\\tfrac{\\max(0,\\theta-O_2)^2}{\\sigma^2}\\right)
\\]
Counts are Poisson or Gamma–Poisson (overdispersion).

**ML**
- Fit **θ, σ, λ_max** by least squares on \\( y \\approx b(1+a d) \\) with closed-form \\( a \\) → \\( \\lambda_{max}=1+a \\).  
- Bootstrap CIs.  
- Kalman filter smooths noisy counts to a latent rate.

**What to measure (CSV)**
- Required: `time_min,counts`
- Optional: `O2_pct,ATP_norm,ROS_norm,ESCRT_norm` (normalized 0–1).
""")

# 2) Preview
with tab_preview:
    st.subheader("Preview (defaults)")
    sim_df = simulate(**DEFAULTS)
    time_min, time_max = int(sim_df["time_min"].min()), int(sim_df["time_min"].max())
    win_prev = st.slider("Visible time range", time_min, time_max, (time_min, time_max), 1, key="win_prev")
    st.plotly_chart(make_ts(sim_df["time_min"], [sim_df["O2_pct"]], ["O₂"], "%O₂", win_prev),
                    use_container_width=True, key="prev_o2_chart")
    st.plotly_chart(make_ts(sim_df["time_min"], [sim_df["rate"]],   ["rate"], "EV rate (particles/min)", win_prev),
                    use_container_width=True, key="prev_rate_chart")
    st.plotly_chart(make_ts(sim_df["time_min"], [sim_df["counts"]], ["counts"], "EV counts/min", win_prev),
                    use_container_width=True, key="prev_counts_chart")

# 3) Tweak
with tab_tweak:
    st.subheader("Adjust parameters → see changes")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.session_state.r_T = st.number_input("Baseline rₜ (particles/min)", 1.0, float(1e7), st.session_state.r_T, 100.0, format="%.1f")
        st.session_state.kW  = st.slider("k (g_W)", 1.0, 12.0, st.session_state.kW, 0.5)
        st.session_state.kS  = st.slider("k (g_S)", 1.0, 12.0, st.session_state.kS, 0.5)
    with c2:
        st.session_state.ATP = st.slider("ATP (0–1)", 0.0, 1.0, st.session_state.ATP, 0.01)
        st.session_state.ROS = st.slider("ROS (0–1)", 0.0, 1.0, st.session_state.ROS, 0.01)
        st.session_state.S   = st.slider("ESCRT S (0–1)", 0.0, 1.0, st.session_state.S, 0.01)
    with c3:
        st.session_state.hyp_scale = st.slider("λ_max", 1.0, 2.5, st.session_state.hyp_scale, 0.05)
        st.session_state.hyp_theta = st.slider("θ (%O₂)", 0.0, 21.0, st.session_state.hyp_theta, 0.5)
        st.session_state.hyp_sigma = st.slider("σ (%O₂)", 0.5, 10.0, st.session_state.hyp_sigma, 0.5)
        st.session_state.noise     = st.slider("Overdispersion", 0.0, 2.0, st.session_state.noise, 0.05)
    with c4:
        st.session_state.T       = st.slider("Duration (min)", 10, 720, st.session_state.T, 5)
        st.session_state.hyp_on  = st.checkbox("Add hypoxia pulse", value=st.session_state.hyp_on)
        st.session_state.o2_high = st.slider("Normoxia %O₂", 0.0, 21.0, st.session_state.o2_high, 0.5)
        st.session_state.o2_low  = st.slider("Hypoxia %O₂",  0.0, 10.0, st.session_state.o2_low, 0.5)
        st.session_state.t_on    = st.slider("Pulse start", 0, max(1, st.session_state.T-1), st.session_state.t_on, 1)
        st.session_state.t_off   = st.slider("Pulse end",   1, st.session_state.T,           st.session_state.t_off, 1)

    sim_df = simulate(
        T=st.session_state.T, r_T=st.session_state.r_T, kW=st.session_state.kW, kS=st.session_state.kS,
        ATP=st.session_state.ATP, ROS=st.session_state.ROS, S=st.session_state.S,
        hyp_scale=st.session_state.hyp_scale, hyp_theta=st.session_state.hyp_theta, hyp_sigma=st.session_state.hyp_sigma,
        o2_high=st.session_state.o2_high, o2_low=st.session_state.o2_low,
        t_on=st.session_state.t_on, t_off=st.session_state.t_off,
        hyp_on=st.session_state.hyp_on, noise=st.session_state.noise
    )
    win_tweak = st.slider("Visible time range", int(sim_df["time_min"].min()), int(sim_df["time_min"].max()),
                          (int(sim_df["time_min"].min()), int(sim_df["time_min"].max())), 1, key="win_tweak")
    st.plotly_chart(make_ts(sim_df["time_min"], [sim_df["O2_pct"]], ["O₂"], "%O₂", win_tweak),
                    use_container_width=True, key="tweak_o2_chart")
    st.plotly_chart(make_ts(sim_df["time_min"], [sim_df["rate"]],   ["rate"], "EV rate (particles/min)", win_tweak),
                    use_container_width=True, key="tweak_rate_chart")
    st.plotly_chart(make_ts(sim_df["time_min"], [sim_df["counts"]], ["counts"], "EV counts/min", win_tweak),
                    use_container_width=True, key="tweak_counts_chart")

    # Explain-Why snapshot
    t_idx = int(win_tweak[1])
    gW_v = float(sim_df["gW"].iloc[0]); gS_v = float(sim_df["gS"].iloc[0])
    lam_v = float(sim_df["lambda_hyp"].iloc[min(t_idx, len(sim_df)-1)])
    explain = {"log g_S": np.log(max(1e-9, gS_v)), "log g_W": np.log(max(1e-9, gW_v)), "log λ_hyp": np.log(max(1e-9, lam_v))}
    figE = go.Figure(data=[go.Bar(x=list(explain.keys()), y=list(explain.values()))])
    figE.update_layout(height=240, margin=dict(l=10,r=10,t=30,b=10),
                       yaxis_title="log contribution", title="log r = log rₜ + Σ log(terms)")
    st.plotly_chart(figE, use_container_width=True, key="tweak_explain_chart")

# 4) Lab Data
with tab_lab:
    st.subheader("Upload wet-lab CSV → overlay → fit")
    st.download_button("Download CSV template",
        data="time_min,counts,O2_pct,ATP_norm,ROS_norm,ESCRT_norm\n0,2100,21,0.7,0.3,0.6\n1,1981,21,0.7,0.3,0.6\n",
        file_name="evelution_hypoxia_template.csv", mime="text/csv")

    lab = st.file_uploader("Upload lab CSV (requires time_min,counts)", type=["csv"])
    if lab:
        try:
            lab_df = pd.read_csv(lab).sort_values("time_min").reset_index(drop=True)
            if not {"time_min","counts"}.issubset(lab_df.columns):
                st.error("CSV must include at least: time_min, counts")
            else:
                st.success(f"Loaded {len(lab_df)} rows.")
                with st.expander("Preview"):
                    st.dataframe(lab_df.head(20), use_container_width=True)

                # Align lab to a clean minute grid from a default sim
                base_df = simulate(**DEFAULTS)
                t = base_df["time_min"].to_numpy()
                merged = pd.merge_asof(base_df[["time_min"]], lab_df, on="time_min",
                                       direction="nearest", tolerance=0.5)

                # Build inputs (fallbacks if cols missing)
                O2_vec  = (merged["O2_pct"]   if "O2_pct"   in merged else pd.Series(np.full_like(t, st.session_state.o2_high))).to_numpy()
                ATP_m   = float(np.nanmedian(merged["ATP_norm"])) if "ATP_norm" in merged else st.session_state.ATP
                ROS_m   = float(np.nanmedian(merged["ROS_norm"])) if "ROS_norm" in merged else st.session_state.ROS
                S_m     = float(np.nanmedian(merged["ESCRT_norm"])) if "ESCRT_norm" in merged else st.session_state.S

                # Model using current tweak k, θ, σ, λ_max and medians of ATP/ROS/S
                W  = 0.6*ATP_m + 0.4*(1.0-ROS_m)
                gW = logistic(st.session_state.kW*(W-0.5))
                gS = logistic(st.session_state.kS*(S_m-0.5))
                base = st.session_state.r_T * gW * gS
                d     = lowO2_driver(O2_vec, st.session_state.hyp_theta, st.session_state.hyp_sigma)
                lam   = base * (1.0 + (st.session_state.hyp_scale-1.0)*d)

                st.plotly_chart(make_ts(t, [lam, merged["counts"]], ["model rate","lab counts"], "rate / counts"),
                                use_container_width=True, key="lab_overlay_chart")

                # Fit θ, σ, λ_max to lab counts on this base
                fit = fit_params(
                    merged["counts"].fillna(method="ffill").fillna(method="bfill").to_numpy(),
                    base*np.ones_like(t), O2_vec,
                    theta_grid=np.linspace(0,21,22), sigma_grid=np.linspace(0.5,8.0,16)
                )
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("θ̂", f"{fit['theta']:.2f} %O₂")
                m2.metric("σ̂", f"{fit['sigma']:.2f} %O₂")
                m3.metric("λ̂_max", f"{fit['lam_max']:.2f}")
                m4.metric("φ̂", f"{fit['phi']:.3f}")

                st.plotly_chart(
                    make_ts(t, [merged["counts"], fit["lam_fit"]], ["lab counts","fitted mean"], "counts/min"),
                    use_container_width=True, key="lab_fit_chart"
                )

                # Bootstrap CIs
                with st.expander("Uncertainty (bootstrap)"):
                    B = st.slider("Bootstrap draws", 10, 200, 50, 10, key="boot")
                    thetas, sigmas, lammax = [], [], []
                    idx = np.arange(len(t))
                    y = merged["counts"].fillna(method="ffill").fillna(method="bfill").to_numpy()
                    for _ in range(B):
                        samp = np.random.choice(idx, size=len(idx), replace=True)
                        fb = fit_params(y[samp], base*np.ones_like(t)[samp], O2_vec[samp],
                                        theta_grid=np.linspace(0,21,22), sigma_grid=np.linspace(0.5,8.0,16))
                        thetas.append(fb["theta"]); sigmas.append(fb["sigma"]); lammax.append(fb["lam_max"])
                    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
                    st.write(f"θ 95% CI: {q(thetas)[0]:.2f}–{q(thetas)[1]:.2f}")
                    st.write(f"σ 95% CI: {q(sigmas)[0]:.2f}–{q(sigmas)[1]:.2f}")
                    st.write(f"λ_max 95% CI: {q(lammax)[0]:.2f}–{q(lammax)[1]:.2f}")

                if st.button("Adopt fitted θ, σ, λ_max into Tweak"):
                    st.session_state.hyp_theta = float(fit["theta"])
                    st.session_state.hyp_sigma = float(fit["sigma"])
                    st.session_state.hyp_scale = float(fit["lam_max"])
                    st.success("Adopted. Check the Tweak tab.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

# 5) Report
with tab_report:
    st.subheader("Quick report + data export")
    sim_df = simulate(
        T=st.session_state.T, r_T=st.session_state.r_T, kW=st.session_state.kW, kS=st.session_state.kS,
        ATP=st.session_state.ATP, ROS=st.session_state.ROS, S=st.session_state.S,
        hyp_scale=st.session_state.hyp_scale, hyp_theta=st.session_state.hyp_theta, hyp_sigma=st.session_state.hyp_sigma,
        o2_high=st.session_state.o2_high, o2_low=st.session_state.o2_low,
        t_on=st.session_state.t_on, t_off=st.session_state.t_off,
        hyp_on=st.session_state.hyp_on, noise=st.session_state.noise
    )
    kf = kalman_counts(sim_df["counts"].values, init=sim_df["rate"].iloc[0], q=50.0, r_scale=1.0)
    lines = [
        "EVelution — Hypoxia EV Report",
        "-"*34,
        f"r_T={st.session_state.r_T:.1f}, ATP={st.session_state.ATP:.2f}, ROS={st.session_state.ROS:.2f}, S={st.session_state.S:.2f}",
        f"kW={st.session_state.kW:.2f}, kS={st.session_state.kS:.2f}",
        f"θ={st.session_state.hyp_theta:.2f} %O2, σ={st.session_state.hyp_sigma:.2f} %O2, λ_max={st.session_state.hyp_scale:.2f}",
        f"Duration={st.session_state.T} min; pulse {st.session_state.t_on}→{st.session_state.t_off} @ {st.session_state.o2_low}% (normoxia {st.session_state.o2_high}%)",
        f"Mean counts={sim_df['counts'].mean():.1f}, SD={sim_df['counts'].std():.1f}"
    ]
    txt = "\n".join(lines)
    st.text_area("Preview", txt, height=180, key="report_textarea")
    st.download_button("Download report.txt", data=txt, file_name="evelution_hypoxia_report.txt", mime="text/plain", key="report_dl")
    st.download_button("Download full sim CSV",
        data=sim_df.assign(rate_kalman=kf).to_csv(index=False),
        file_name="evelution_hypoxia_full.csv", mime="text/csv", key="csv_dl")
