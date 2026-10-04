import os
import sys
import json
import sqlite3
import datetime as dt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_loader import (
    load_data,
    add_features,
    FEATURES,
    NUM,
    RAW_NUM,
    TARGET,
    DATA_PATH,
)
from prediction import load_model, predict, ModelNotAvailable
from explainability import global_importance, DISCLAIMER


st.set_page_config(
    page_title="PredictX AI",
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b1020; color: #e8ecf7; }
    [data-testid="stSidebar"] { background: #080d19; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
    .hero {
        background: linear-gradient(135deg, #151d35, #10172b);
        border: 1px solid #293657;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 22px;
    }
    .hero-title { font-size: 2.4rem; font-weight: 800; margin-bottom: 4px; }
    .hero-subtitle { color: #9aa6c4; font-size: 1rem; }
    .card {
        background: #151c30;
        border: 1px solid #26304d;
        border-radius: 12px;
        padding: 17px;
        min-height: 95px;
    }
    .k {
        color: #8995b5;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .05em;
    }
    .v { font-size: 1.55rem; font-weight: 750; margin-top: 5px; }
    .recommendation {
        background: #151d32;
        border-left: 4px solid #6c8cff;
        border-radius: 10px;
        padding: 16px;
        margin-top: 15px;
    }
    .factor-box {
        background: #121a2e;
        border: 1px solid #263454;
        border-radius: 10px;
        padding: 12px;
        margin: 6px 0;
    }
    .status-online { color: #35d07f; font-weight: 700; }
    .status-offline { color: #ff5b5b; font-weight: 700; }
    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 18px;
        margin-bottom: 8px;
    }
    .risk-normal {
        background: #121d25;
        border-left: 4px solid #35d07f;
        border-radius: 10px;
        padding: 13px;
        margin: 6px 0;
    }
    .risk-warning {
        background: #241f14;
        border-left: 4px solid #f2c94c;
        border-radius: 10px;
        padding: 13px;
        margin: 6px 0;
    }
    .risk-danger {
        background: #281619;
        border-left: 4px solid #ff5b5b;
        border-radius: 10px;
        padding: 13px;
        margin: 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(__file__)
DB = os.path.join(BASE_DIR, "predictions.db")
RES = os.path.join(BASE_DIR, "reports", "model_results.json")


@st.cache_data
def get_df():
    try:
        return add_features(load_data()), None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def get_model():
    try:
        return load_model(), None
    except ModelNotAvailable as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def db():
    connection = sqlite3.connect(DB)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS h (
            ts TEXT,
            inputs TEXT,
            pred INTEGER,
            prob REAL,
            risk TEXT,
            version TEXT
        )
        """
    )
    connection.commit()
    return connection


def card(label, value):
    st.markdown(
        f"""
        <div class="card">
            <div class="k">{label}</div>
            <div class="v">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_results():
    if not os.path.exists(RES):
        return None
    try:
        with open(RES, "r") as f:
            return json.load(f)
    except Exception:
        return None


def history_table(data):
    if data is None or len(data) == 0:
        st.info("No prediction history yet.")
        return
    st.markdown(data.to_html(index=False), unsafe_allow_html=True)


def machine_condition_analysis(inp, data):
    analysis = []
    feature_names = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    for feature in feature_names:
        if feature not in data.columns:
            continue

        value = float(inp[feature])
        q1 = float(data[feature].quantile(0.25))
        q3 = float(data[feature].quantile(0.75))
        median = float(data[feature].median())

        if value < q1:
            status = "LOW"
            score = abs(value - median) / max(abs(median - q1), 0.0001)
        elif value > q3:
            status = "HIGH"
            score = abs(value - median) / max(abs(q3 - median), 0.0001)
        else:
            status = "NORMAL"
            score = 0

        analysis.append({
            "feature": feature,
            "value": value,
            "q1": q1,
            "q3": q3,
            "status": status,
            "score": score,
        })

    return analysis


def show_machine_analysis(inp, data):
    st.markdown(
        '<div class="section-title">🧠 Machine Risk Analysis</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "This compares the selected machine's operating conditions "
        "with the historical dataset distribution."
    )

    analysis = machine_condition_analysis(inp, data)

    abnormal = [item for item in analysis if item["status"] != "NORMAL"]

    if not abnormal:
        st.success(
            "🟢 All monitored operating conditions are within the "
            "central historical range."
        )
    else:
        st.warning(
            f"⚠️ {len(abnormal)} monitored condition(s) fall outside "
            "the central historical range."
        )

    for item in analysis:
        feature = item["feature"]
        value = item["value"]
        status = item["status"]

        if status == "NORMAL":
            st.markdown(
                f"""
                <div class="risk-normal">
                    🟢 <b>{feature}</b><br>
                    Current value: <b>{value:.2f}</b>
                    &nbsp; — &nbsp; Within typical historical range
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif status == "HIGH":
            st.markdown(
                f"""
                <div class="risk-danger">
                    🔴 <b>{feature}</b><br>
                    Current value: <b>{value:.2f}</b>
                    &nbsp; — &nbsp; Above the central historical range
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="risk-warning">
                    🟡 <b>{feature}</b><br>
                    Current value: <b>{value:.2f}</b>
                    &nbsp; — &nbsp; Below the central historical range
                </div>
                """,
                unsafe_allow_html=True,
            )

    return analysis


df, df_err = get_df()
mm, m_err = get_model()
results = safe_results()


st.sidebar.title("⚙️ PredictX AI")
st.sidebar.caption("Predictive Maintenance Intelligence")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Executive Overview",
        "🔮 Prediction Studio",
        "📊 Dataset Intelligence",
        "🤖 Model Lab",
        "🔍 Explainability",
        "📋 Prediction History",
        "ℹ️ Methodology",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### SYSTEM STATUS")
st.sidebar.markdown(
    f"""
    {'🟢' if df is not None else '🔴'} Dataset loaded

    {'🟢' if mm else '🔴'} Model loaded

    {'🟢' if mm else '🔴'} Prediction engine
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("ALG-DATA-02 • Predict What Happens Next")

if df_err:
    st.error(df_err)

if m_err:
    st.warning(m_err)


def presets():
    model, meta = mm
    d = df.copy()

    d["p"] = model.predict_proba(d[FEATURES])[:, 1]

    healthy = (
        d[d[TARGET] == 0]
        .assign(
            dist=lambda x:
            (x["Torque [Nm]"] - x["Torque [Nm]"].median()).abs()
        )
        .nsmallest(1, "dist")
        .iloc[0]
    )

    warning = d[d[TARGET] == 0].nlargest(1, "p").iloc[0]
    high_risk = d[d[TARGET] == 1].nlargest(1, "p").iloc[0]

    return {
        "🟢 HEALTHY": healthy,
        "🟡 WARNING": warning,
        "🔴 HIGH-RISK": high_risk,
    }


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page.startswith("🏠"):

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">⚙️ PredictX AI</div>
            <div class="hero-subtitle">
                Predictive Maintenance Intelligence
            </div>
            <br>
            <b>Predict failure before the machine stops.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mm and df is not None:
        st.markdown(
            '<p class="status-online">● AI SYSTEM ONLINE</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="status-offline">● SYSTEM DEGRADED</p>',
            unsafe_allow_html=True,
        )

    if df is not None:
        connection = db()
        try:
            hist = pd.read_sql("SELECT prob FROM h", connection)
        finally:
            connection.close()

        cols = st.columns(6)

        values = [
            ("Records", f"{len(df):,}"),
            ("Failure cases", f"{int(df[TARGET].sum())}"),
            ("Failure rate", f"{100 * df[TARGET].mean():.2f}%"),
            (
                "Best model",
                mm[1]["model_name"] if mm else "—",
            ),
            (
                "Test ROC-AUC",
                f"{mm[1]['metrics_test']['roc_auc']:.3f}" if mm else "—",
            ),
            (
                "Avg predicted risk",
                f"{100 * hist.prob.mean():.1f}%" if len(hist) else "—",
            ),
        ]

        for col, (label, value) in zip(cols, values):
            with col:
                card(label, value)

        st.markdown(
            '<div class="section-title">System Intelligence</div>',
            unsafe_allow_html=True,
        )

        a, b = st.columns(2)

        a.plotly_chart(
            px.pie(
                df,
                names=df[TARGET].map({0: "Normal", 1: "Failure"}),
                title="Failure overview",
                hole=0.55,
            ),
            use_container_width=True,
        )

        if results:
            model_data = pd.DataFrame(
                {
                    name: model_result["test"]
                    for name, model_result in results["models"].items()
                }
            ).T.reset_index().melt("index")

            b.plotly_chart(
                px.bar(
                    model_data,
                    x="index",
                    y="value",
                    color="variable",
                    barmode="group",
                    title="Model performance — test set",
                ),
                use_container_width=True,
            )

            st.markdown(
                "**Key dataset insights** (associations, not causation)"
            )

            insights = results.get("insights", {})

            st.write(
                "Failure rate by Type (%): "
                f"{insights.get('failure_rate_by_type_pct', '—')}"
                " | Tool-wear quartiles (%): "
                f"{insights.get('failure_rate_by_toolwear_quartile_pct', '—')}"
            )

            importance = global_importance()

            if importance:
                st.plotly_chart(
                    px.bar(
                        x=list(importance.values()),
                        y=list(importance.keys()),
                        orientation="h",
                        title="Most influential features",
                    ),
                    use_container_width=True,
                )

        st.markdown(
            '<div class="section-title">Recent predictions</div>',
            unsafe_allow_html=True,
        )

        connection = db()
        try:
            recent = pd.read_sql(
                """
                SELECT ts, pred, prob, risk
                FROM h
                ORDER BY ts DESC
                LIMIT 5
                """,
                connection,
            )
        finally:
            connection.close()

        history_table(recent)


# ============================================================
# PREDICTION STUDIO
# ============================================================

elif page.startswith("🔮"):

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🔮 Prediction Studio</div>
            <div class="hero-subtitle">
                Simulate a machine and estimate its failure risk.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not mm or df is None:
        st.stop()

    meta = mm[1]

    st.markdown(
        '<div class="section-title">⚡ Quick Demo Scenarios</div>',
        unsafe_allow_html=True,
    )

    st.caption("These scenarios use real rows from the AI4I dataset.")

    P = presets()
    cols = st.columns(3)

    for col, (name, row) in zip(cols, P.items()):
        with col:
            if st.button(f"DEMO: {name}", use_container_width=True):
                st.session_state["vals"] = {
                    **{c: float(row[c]) for c in RAW_NUM},
                    "Type": row["Type"],
                }
                st.session_state["scenario"] = name

    current = st.session_state.get(
        "vals",
        {
            **{c: meta["feature_ranges"][c][2] for c in RAW_NUM},
            "Type": meta["types"][0],
        },
    )

    scenario = st.session_state.get("scenario", "CUSTOM MACHINE")

    if scenario != "CUSTOM MACHINE":
        st.info(f"Active scenario: **{scenario}**")

    st.markdown(
        '<div class="section-title">Machine Operating Conditions</div>',
        unsafe_allow_html=True,
    )

    inp = {
        "Type": st.selectbox(
            "Machine Type",
            meta["types"],
            index=meta["types"].index(
                current.get("Type", meta["types"][0])
            ),
        )
    }

    input_cols = st.columns(2)

    for i, c in enumerate(RAW_NUM):
        lo, hi, default = meta["feature_ranges"][c]

        try:
            value = float(current.get(c, default))
        except Exception:
            value = float(default)

        value = min(max(value, lo), hi)

        with input_cols[i % 2]:
            inp[c] = st.slider(
                c,
                min_value=lo,
                max_value=hi,
                value=value,
                key=f"input_{c}",
            )

    st.markdown("---")

    if st.button(
        "⚡ ANALYZE MACHINE RISK",
        type="primary",
        use_container_width=True,
    ):

        try:
            r = predict(inp, *mm)
            probability = float(r["probability"])
            risk = r["risk_level"]
            label = r["label"]

            st.markdown("## 🎯 MACHINE FAILURE RISK")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "font": {"size": 42}},
                    title={"text": "Model-estimated failure probability"},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {"thickness": 0.3},
                        "steps": [
                            {"range": [0, 30]},
                            {"range": [30, 60]},
                            {"range": [60, 100]},
                        ],
                        "threshold": {
                            "line": {"width": 4},
                            "thickness": 0.8,
                            "value": probability * 100,
                        },
                    },
                )
            )

            gauge.update_layout(
                height=300,
                margin={"l": 20, "r": 20, "t": 60, "b": 20},
            )

            st.plotly_chart(gauge, use_container_width=True)

            if risk == "LOW":
                st.success(
                    f"🟢 **LOW RISK**  \n\n"
                    f"Model prediction: **{label}**"
                )
            elif risk == "MEDIUM":
                st.warning(
                    f"🟡 **MEDIUM RISK**  \n\n"
                    f"Model prediction: **{label}**"
                )
            else:
                st.error(
                    f"🔴 **HIGH RISK**  \n\n"
                    f"Model prediction: **{label}**"
                )

            st.markdown(
                '<div class="section-title">🏭 Machine Condition Snapshot</div>',
                unsafe_allow_html=True,
            )

            s1, s2, s3 = st.columns(3)

            with s1:
                st.metric("Machine Type", inp["Type"])
            with s2:
                st.metric(
                    "Air Temperature",
                    f'{inp["Air temperature [K]"]:.1f} K',
                )
            with s3:
                st.metric(
                    "Process Temperature",
                    f'{inp["Process temperature [K]"]:.1f} K',
                )

            s4, s5, s6 = st.columns(3)

            with s4:
                st.metric(
                    "Rotational Speed",
                    f'{inp["Rotational speed [rpm]"]:.0f} rpm',
                )
            with s5:
                st.metric("Torque", f'{inp["Torque [Nm]"]:.1f} Nm')
            with s6:
                st.metric("Tool Wear", f'{inp["Tool wear [min]"]:.0f} min')

            show_machine_analysis(inp, df)

            c1, c2, c3 = st.columns(3)

            with c1:
                card("Failure probability", f"{100 * probability:.1f}%")
            with c2:
                card("Risk classification", risk)
            with c3:
                card(
                    "Model",
                    meta.get("model_name", "Predictive model"),
                )

            st.markdown(
                '<div class="section-title">🧠 Why this matters</div>',
                unsafe_allow_html=True,
            )

            if risk == "LOW":
                st.success(
                    "The model currently estimates a relatively low failure "
                    "risk. Continue normal monitoring."
                )
            elif risk == "MEDIUM":
                st.warning(
                    "The model detects elevated failure risk. Consider "
                    "increased monitoring and inspection."
                )
            else:
                st.error(
                    "The model estimates high failure risk. Preventive "
                    "inspection should be prioritised before continued operation."
                )

            importance = global_importance() if results else {}

            st.markdown(
                '<div class="section-title">🔍 Model Factors</div>',
                unsafe_allow_html=True,
            )

            if importance:
                top_features = list(importance.keys())[:5]

                for i, feature in enumerate(top_features, 1):
                    value = importance[feature]

                    st.markdown(
                        f"""
                        <div class="factor-box">
                            <b>{i}. {feature}</b>
                            <br>
                            <span style="color:#8995b5;">
                                Relative model importance: {value:.4f}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown(
                f"""
                <div class="recommendation">
                    <b>🔧 Recommended next action</b>
                    <br><br>
                    {r["recommendation"]}
                    <br><br>
                    <small>
                    Decision support only — this prediction is not a guarantee
                    of physical failure.
                    </small>
                </div>
                """,
                unsafe_allow_html=True,
            )

            connection = db()

            try:
                connection.execute(
                    """
                    INSERT INTO h
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dt.datetime.now().isoformat(timespec="seconds"),
                        json.dumps(inp),
                        r["predicted_class"],
                        probability,
                        risk,
                        meta["version"],
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            st.success(
                "✓ Prediction analyzed and saved to Prediction History."
            )

        except ValueError as e:
            st.error(str(e))
        except Exception:
            st.error(
                "Prediction failed. Please check the machine inputs."
            )


# ============================================================
# DATASET INTELLIGENCE — EXECUTIVE UPGRADE
# ============================================================

elif page.startswith("📊"):

    st.markdown(
        '<div class="section-title">📊 Dataset Intelligence</div>',
        unsafe_allow_html=True,
    )

    if df is None:
        st.stop()

    st.caption(
        "Explore failure patterns, machine conditions, and feature "
        "relationships across the AI4I 2020 Predictive Maintenance dataset."
    )

    total_records = len(df)
    failure_cases = int(df[TARGET].sum())
    failure_rate = (
        failure_cases / total_records * 100
        if total_records
        else 0
    )

    type_rates = (
        df.groupby("Type")[TARGET]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        if "Type" in df.columns
        else pd.Series(dtype=float)
    )

    highest_type = type_rates.index[0] if len(type_rates) else "N/A"
    highest_type_rate = type_rates.iloc[0] if len(type_rates) else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card("Total records", f"{total_records:,}")

    with c2:
        card("Failure cases", f"{failure_cases:,}")

    with c3:
        card("Overall failure rate", f"{failure_rate:.2f}%")

    with c4:
        card(
            "Highest-risk type",
            f"{highest_type} ({highest_type_rate:.2f}%)",
        )

    st.markdown("---")

    # Failure rate by Type
    st.markdown("### 🏭 Failure Rate by Machine Type")

    if len(type_rates):
        type_df = type_rates.reset_index()
        type_df.columns = ["Type", "Failure Rate (%)"]

        fig_type = px.bar(
            type_df,
            x="Type",
            y="Failure Rate (%)",
            text="Failure Rate (%)",
            title="Historical Failure Rate by Machine Type",
        )

        fig_type.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )

        fig_type.update_layout(
            height=420,
            yaxis_range=[
                0,
                max(float(type_df["Failure Rate (%)"].max()) * 1.25, 5),
            ],
        )

        st.plotly_chart(fig_type, use_container_width=True)

        st.caption(
            "This describes an observed historical association. "
            "It does not prove machine type causes failure."
        )

    # Tool wear
    st.markdown("### 🛠️ Failure Rate vs Tool Wear")

    tool_summary = None

    if "Tool wear [min]" in df.columns:
        try:
            tool_work = df[["Tool wear [min]", TARGET]].copy()

            tool_work["Tool Wear Band"] = pd.qcut(
                tool_work["Tool wear [min]"],
                q=4,
                duplicates="drop",
            )

            tool_summary = (
                tool_work.groupby(
                    "Tool Wear Band",
                    observed=False,
                )[TARGET]
                .agg(["count", "sum", "mean"])
                .reset_index()
            )

            tool_summary["Failure Rate (%)"] = (
                tool_summary["mean"] * 100
            )

            fig_tool = px.bar(
                tool_summary,
                x="Tool Wear Band",
                y="Failure Rate (%)",
                text="Failure Rate (%)",
                title="Historical Failure Rate Across Tool-Wear Quartiles",
            )

            fig_tool.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
            )

            fig_tool.update_layout(
                height=420,
                yaxis_range=[
                    0,
                    max(
                        float(tool_summary["Failure Rate (%)"].max()) * 1.25,
                        5,
                    ),
                ],
            )

            st.plotly_chart(fig_tool, use_container_width=True)

            max_tool = tool_summary.loc[
                tool_summary["Failure Rate (%)"].idxmax()
            ]

            st.info(
                f"Highest historical tool-wear band: "
                f"{max_tool['Tool Wear Band']} — "
                f"{max_tool['Failure Rate (%)']:.2f}% failure rate."
            )

        except Exception:
            st.warning("Tool-wear analysis could not be calculated.")

    # Torque
    st.markdown("### ⚙️ Failure Rate vs Torque")

    torque_summary = None

    if "Torque [Nm]" in df.columns:
        try:
            torque_work = df[["Torque [Nm]", TARGET]].copy()

            torque_work["Torque Band"] = pd.qcut(
                torque_work["Torque [Nm]"],
                q=4,
                duplicates="drop",
            )

            torque_summary = (
                torque_work.groupby(
                    "Torque Band",
                    observed=False,
                )[TARGET]
                .agg(["count", "sum", "mean"])
                .reset_index()
            )

            torque_summary["Failure Rate (%)"] = (
                torque_summary["mean"] * 100
            )

            fig_torque = px.bar(
                torque_summary,
                x="Torque Band",
                y="Failure Rate (%)",
                text="Failure Rate (%)",
                title="Historical Failure Rate Across Torque Quartiles",
            )

            fig_torque.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
            )

            fig_torque.update_layout(
                height=420,
                yaxis_range=[
                    0,
                    max(
                        float(torque_summary["Failure Rate (%)"].max()) * 1.25,
                        5,
                    ),
                ],
            )

            st.plotly_chart(fig_torque, use_container_width=True)

            max_torque = torque_summary.loc[
                torque_summary["Failure Rate (%)"].idxmax()
            ]

            st.info(
                f"Highest historical torque band: "
                f"{max_torque['Torque Band']} — "
                f"{max_torque['Failure Rate (%)']:.2f}% failure rate."
            )

        except Exception:
            st.warning("Torque analysis could not be calculated.")

    # Temperature relationship
    st.markdown("### 🌡️ Temperature Relationship")

    air_temp = "Air temperature [K]"
    process_temp = "Process temperature [K]"

    if air_temp in df.columns and process_temp in df.columns:
        temp_plot = df[[air_temp, process_temp, TARGET]].copy()
        temp_plot["Status"] = temp_plot[TARGET].map(
            {0: "Normal", 1: "Failure"}
        )

        fig_temp = px.scatter(
            temp_plot,
            x=air_temp,
            y=process_temp,
            color="Status",
            opacity=0.65,
            title="Air Temperature vs Process Temperature",
            labels={
                air_temp: "Air Temperature (K)",
                process_temp: "Process Temperature (K)",
            },
            color_discrete_map={
                "Normal": "green",
                "Failure": "red",
            },
        )

        fig_temp.update_layout(height=480)

        st.plotly_chart(
            fig_temp,
            use_container_width=True,
        )

        st.caption(
            "Failure points help identify regions of the historical "
            "temperature distribution where failures occurred."
        )

    # Interactive feature explorer
    st.markdown("### 🔎 Interactive Feature Explorer")

    numeric_features = [
        c for c in NUM
        if c in df.columns
    ]

    if numeric_features:
        selected_feature = st.selectbox(
            "Choose a numerical feature",
            numeric_features,
            key="dataset_feature_selector",
        )

        selected_plot = df[[selected_feature, TARGET]].copy()
        selected_plot["Status"] = selected_plot[TARGET].map(
            {0: "Normal", 1: "Failure"}
        )

        fig_feature = px.histogram(
            selected_plot,
            x=selected_feature,
            color="Status",
            nbins=35,
            marginal="box",
            opacity=0.7,
            barmode="overlay",
            title=f"{selected_feature} Distribution by Failure Status",
            color_discrete_map={
                "Normal": "green",
                "Failure": "red",
            },
        )

        fig_feature.update_layout(height=500)

        st.plotly_chart(
            fig_feature,
            use_container_width=True,
        )

    # Data quality
    st.markdown("### 🧹 Data Quality")

    missing_total = int(df.isna().sum().sum())
    duplicate_total = int(df.duplicated().sum())

    q1, q2, q3 = st.columns(3)

    with q1:
        card("Columns", len(df.columns))

    with q2:
        card("Missing values", missing_total)

    with q3:
        card("Duplicate rows", duplicate_total)

    # Failure mode breakdown
    st.markdown("### ⚠️ Failure Mode Breakdown")

    modes = [
        c
        for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]
        if c in df.columns
    ]

    if modes:
        mode_df = pd.DataFrame(
            {
                "Failure Mode": modes,
                "Cases": [int(df[c].sum()) for c in modes],
            }
        ).sort_values("Cases", ascending=False)

        fig_modes = px.bar(
            mode_df,
            x="Failure Mode",
            y="Cases",
            text="Cases",
            title="Failure-Mode Counts",
        )

        fig_modes.update_traces(textposition="outside")
        fig_modes.update_layout(height=420)

        st.plotly_chart(
            fig_modes,
            use_container_width=True,
        )
    else:
        st.info(
            "No individual failure-mode columns were detected."
        )

    # Automatic insights
    st.markdown("### 🧠 Automatic Dataset Insights")

    insights = []

    insights.append(
        f"The dataset contains {total_records:,} machine records "
        f"with {failure_cases:,} historical failure cases "
        f"({failure_rate:.2f}%)."
    )

    if len(type_rates):
        insights.append(
            f"Machine Type {highest_type} has the highest observed "
            f"failure rate at {highest_type_rate:.2f}%."
        )

    if tool_summary is not None and len(tool_summary):
        max_tool = tool_summary.loc[
            tool_summary["Failure Rate (%)"].idxmax()
        ]

        insights.append(
            f"The tool-wear band {max_tool['Tool Wear Band']} "
            f"has the highest observed failure rate among the "
            f"tool-wear quartiles ({max_tool['Failure Rate (%)']:.2f}%)."
        )

    if torque_summary is not None and len(torque_summary):
        max_torque = torque_summary.loc[
            torque_summary["Failure Rate (%)"].idxmax()
        ]

        insights.append(
            f"The torque band {max_torque['Torque Band']} "
            f"has the highest observed failure rate among the "
            f"torque quartiles ({max_torque['Failure Rate (%)']:.2f}%)."
        )

    if numeric_features:
        try:
            corr = df[numeric_features + [TARGET]].corr()[TARGET]
            corr = corr.drop(labels=[TARGET], errors="ignore").dropna()

            if len(corr):
                strongest = corr.abs().idxmax()
                strongest_value = corr[strongest]

                insights.append(
                    f"The strongest linear correlation with the failure "
                    f"target among the numerical features is "
                    f"{strongest} ({strongest_value:.3f})."
                )
        except Exception:
            pass

    for i, insight in enumerate(insights, 1):
        st.markdown(
            f'<div class="factor-box"><b>{i}.</b> {insight}</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "⚠️ These are historical associations, not proof of causation. "
        "They are intended for exploratory analysis and decision support."
    )


# ============================================================
# MODEL LAB
# ============================================================

elif page.startswith("🤖"):

    st.title("🤖 Model Lab")

    if not results:
        st.warning("Run python src/train.py first.")
        st.stop()

    rows = []

    for name, result in results["models"].items():
        row = {
            "Model": name,
            "CV recall": result["cv_recall"],
            "CV F1": result["cv_f1"],
            "CV ROC-AUC": result["cv_roc_auc"],
        }

        for key, value in result["test"].items():
            row[f"Test {key}"] = value

        rows.append(row)

    history_table(pd.DataFrame(rows))

    st.success(
        f"Selected model: {results['selected']} "
        "(selected using validation performance)."
    )

    selected = results["models"][results["selected"]]

    st.plotly_chart(
        px.imshow(
            selected["confusion_matrix"],
            text_auto=True,
            x=["Pred normal", "Pred failure"],
            y=["Actual normal", "Actual failure"],
            title="Confusion matrix — test set",
        ),
        use_container_width=True,
    )

    figure = go.Figure(
        go.Scatter(
            x=selected["roc"]["fpr"],
            y=selected["roc"]["tpr"],
            name="ROC",
        )
    )

    figure.update_layout(
        title="ROC curve — test set",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    st.caption(
        "Failure recall is prioritised because missing a real failure "
        "can be more costly than triggering an additional inspection."
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page.startswith("🔍"):

    st.title("🔍 Explainability")

    st.caption(
        "Understanding which features influence the model overall."
    )

    if not results:
        st.stop()

    importance = global_importance()

    if importance:
        st.plotly_chart(
            px.bar(
                x=list(importance.values())[::-1],
                y=list(importance.keys())[::-1],
                orientation="h",
                title="Top predictive factors",
            ),
            use_container_width=True,
        )

        for i, key in enumerate(list(importance)[:5], 1):
            st.write(f"**{i}.** {key}")

    st.warning(DISCLAIMER)


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page.startswith("📋"):

    st.title("📋 Prediction History")

    connection = db()

    try:
        history = pd.read_sql(
            """
            SELECT *
            FROM h
            ORDER BY ts DESC
            """,
            connection,
        )
    finally:
        connection.close()

    st.caption(f"{len(history)} prediction(s) recorded.")

    levels = st.multiselect(
        "Risk level",
        ["LOW", "MEDIUM", "HIGH"],
        ["LOW", "MEDIUM", "HIGH"],
    )

    filtered = history[history.risk.isin(levels)]
    history_table(filtered)


# ============================================================
# METHODOLOGY
# ============================================================

else:

    st.title("ℹ️ Methodology")

    st.markdown(
        """
        ### Problem

        **ALG-DATA-02 — Predict What Happens Next**

        Predictive maintenance using historical machine data.

        ### Dataset

        UCI AI4I 2020 Predictive Maintenance Dataset.

        ### Preparation

        IDs are dropped.

        `Machine failure` is used as the target.

        Failure-mode columns are excluded from model features
        to avoid target leakage.

        The dataset is split using a stratified train/test split.

        ### Feature Engineering

        The pipeline uses raw machine sensor features together
        with engineered temperature and mechanical-power features.

        ### Models

        The project evaluates:

        - Logistic Regression
        - Random Forest
        - HistGradientBoosting

        ### Validation

        Five-fold stratified cross-validation is performed on
        the training data.

        Model selection is based on validation performance.

        The test set is reserved for final evaluation.

        ### Explainability

        Permutation importance is used to understand feature
        influence at the model level.

        ### Limitations

        The AI4I dataset is synthetic.

        Strong test performance on this dataset does not guarantee
        equivalent performance on real industrial equipment.

        Model probabilities are not necessarily calibrated.

        Feature importance indicates model association, not causation.

        ### Future Improvements

        - Real sensor streams
        - Time-series prediction
        - Remaining useful life estimation
        - Probability calibration
        - SHAP-based explanations
        - Cost-sensitive thresholds
        - Real-time alerting

        **PredictX AI is a decision-support prototype, not a
        guaranteed physical-failure prediction system.**
        """
    )
