"""CreditGuard - Credit Card Fraud Detection dashboard (educational demo)."""
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent  # repo root, so paths work locally and on Streamlit Cloud

st.set_page_config(page_title="CreditGuard Fraud Detection", page_icon="🛡️", layout="wide")


# ---------- Loading (cached so it only happens once) ----------
@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "fraud_detection_model.pkl")


@st.cache_data
def load_csv(name):
    return pd.read_csv(ROOT / name)


bundle = load_bundle()
model = bundle["model"]
model_name = bundle["model_name"]
scaler = bundle["scaler"]  # None for our saved model
features = bundle["feature_names"]  # Time, V1..V28, Amount


def score(df):
    """Return fraud probability for each row. Class column is ignored."""
    X = df[features]  # keeps only the model's columns (drops Class)
    if scaler is not None:  # only scale if a scaler was saved
        X = scaler.transform(X)
    return model.predict_proba(X)[:, 1]


# ---------- Header ----------
st.title("🛡️ CreditGuard: Credit Card Fraud Detection")
st.caption(f"Best model: **{model_name}** (highest PR-AUC). Educational demo, not for real use.")

tab1, tab2, tab3 = st.tabs(["📊 Model results", "🔴 Live monitor", "📁 Upload CSV"])

# ---------- Tab 1: Model results ----------
with tab1:
    st.subheader("Model comparison")
    st.write(
        "Only about 0.17% of transactions are fraud, so **accuracy is misleading**. "
        "We use **PR-AUC** as the main metric."
    )
    results = load_csv("model_comparison.csv")
    st.dataframe(results, use_container_width=True, hide_index=True)

    metrics = st.multiselect(
        "Metrics to chart",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"],
        default=["Precision", "Recall", "PR-AUC"],
    )
    if metrics:
        st.bar_chart(results.set_index("Model")[metrics])

    st.subheader("Top features (Random Forest importance)")
    try:
        imp = pd.read_csv(ROOT / "feature_importance.csv", index_col=0)
        st.bar_chart(imp["importance"].sort_values(ascending=False).head(10))
    except Exception:
        st.info("feature_importance.csv not found or in an unexpected format.")

# ---------- Tab 2: Live monitor ----------
with tab2:
    st.subheader("Live transaction monitor")
    st.write("Replays held-out test transactions one batch at a time, as if they were arriving live.")

    c1, c2, c3 = st.columns(3)
    threshold = c1.slider("Fraud threshold", 0.05, 0.95, 0.5, 0.05,
                          help="Lower = catch more fraud but more false alarms. Higher = fewer false alarms but miss more fraud.")
    delay = c2.slider("Delay between batches (s)", 0.0, 1.0, 0.1, 0.05)
    n_rows = c3.slider("Transactions to replay", 100, 2000, 500, 100)

    if st.button("▶ Start replay"):
        demo = load_csv("demo_transactions.csv").head(n_rows).reset_index(drop=True)
        probs = score(demo)
        truth = demo["Class"].values
        flagged_all = probs >= threshold

        step = 20  # rows per update
        counters = st.empty()
        bar = st.progress(0)
        table_box = st.empty()

        for end in range(step, len(demo) + step, step):
            end = min(end, len(demo))
            f = flagged_all[:end]
            t = truth[:end]
            caught = int(((f == 1) & (t == 1)).sum())
            missed = int(((f == 0) & (t == 1)).sum())
            false_alarms = int(((f == 1) & (t == 0)).sum())

            with counters.container():
                k = st.columns(5)
                k[0].metric("Checked", end)
                k[1].metric("Flagged", int(f.sum()))
                k[2].metric("Frauds caught", caught)
                k[3].metric("Frauds missed", missed)
                k[4].metric("False alarms", false_alarms)

            bar.progress(end / len(demo))
            shown = demo.iloc[:end][f].copy()
            shown.insert(0, "fraud_prob", probs[:end][f].round(3))
            shown = shown[["fraud_prob", "Time", "Amount", "Class"]].rename(columns={"Class": "true_label"})
            table_box.dataframe(shown.tail(15).iloc[::-1], use_container_width=True, hide_index=True)
            time.sleep(delay)

        st.success("Replay finished.")

# ---------- Tab 3: Upload CSV ----------
with tab3:
    st.subheader("Score your own transactions")
    st.write("Upload a CSV with columns: " + ", ".join(features[:3]) + ", ... , " + features[-1])
    up = st.file_uploader("CSV file", type="csv")
    thr2 = st.slider("Fraud threshold ", 0.05, 0.95, 0.5, 0.05, key="thr_upload")

    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        else:
            df["fraud_prob"] = score(df)
            df["flagged"] = df["fraud_prob"] >= thr2
            st.write(f"Scored **{len(df)}** rows, flagged **{int(df['flagged'].sum())}**.")
            flagged = df[df["flagged"]].sort_values("fraud_prob", ascending=False)
            st.dataframe(flagged, use_container_width=True)
            st.download_button("Download flagged rows", flagged.to_csv(index=False).encode(),
                               "flagged_transactions.csv", "text/csv")
