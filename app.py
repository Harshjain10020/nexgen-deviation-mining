import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NexGen Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("NexGen Intelligence")
st.caption("Process Mining & Value Realization Platform")

# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data
def load_data():
    orders = pd.read_csv("data/orders.csv")
    delivery = pd.read_csv("data/delivery_performance.csv")
    routes = pd.read_csv("data/routes_distance.csv")
    cost = pd.read_csv("data/cost_breakdown.csv")
    feedback = pd.read_csv("data/customer_feedback.csv")
    
    # Helper to normalize and rename columns
    def clean_df(df, mapping=None):
        # Normalize to snake_case: Order_ID -> order_id
        df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
        if mapping:
            df.rename(columns=mapping, inplace=True)
        return df

    # Apply cleaning and specific renames to match app logic
    orders = clean_df(orders, {'order_value_inr': 'order_value'})
    
    # Delivery: delivery_status -> status, delivery_cost_inr -> delivery_cost, customer_rating -> rating
    delivery = clean_df(delivery, {
        'delivery_status': 'status', 
        'delivery_cost_inr': 'delivery_cost', 
        'customer_rating': 'rating'
    })
    
    # Routes: traffic_delay_minutes -> traffic_delay
    routes = clean_df(routes, {'traffic_delay_minutes': 'traffic_delay'})
    
    cost = clean_df(cost)
    
    # Feedback: rating -> feedback_rating (to avoid collision with delivery rating)
    feedback = clean_df(feedback, {'rating': 'feedback_rating'})
    
    return orders, delivery, routes, cost, feedback

orders, delivery, routes, cost, feedback = load_data()

# -----------------------------
# PROCESS TABLE
# -----------------------------
df = orders.merge(delivery, on="order_id", how="left")
df = df.merge(routes, on="order_id", how="left")
df = df.merge(cost, on="order_id", how="left")
df = df.merge(feedback, on="order_id", how="left")

df.fillna(0, inplace=True)

# -----------------------------
# DEVIATION LOGIC
# -----------------------------
df["delay"] = df["actual_delivery_days"] > df["promised_delivery_days"]
df["damage"] = df["status"].isin(["Damaged", "Wrong Item"])
df["cost_overrun"] = df["delivery_cost"] > df["delivery_cost"].median()

df["deviation"] = df["delay"] | df["damage"] | df["cost_overrun"]

# -----------------------------
# VALUE LEAKAGE
# -----------------------------
df["value_leakage"] = 0
df.loc[df["delay"], "value_leakage"] += 0.1 * df["order_value"]
df.loc[df["damage"], "value_leakage"] += 0.2 * df["order_value"]
df.loc[df["cost_overrun"], "value_leakage"] += (
    df["delivery_cost"] - df["delivery_cost"].median()
)

# -----------------------------
# ROOT CAUSE TAGGING
# -----------------------------
df["root_cause"] = "Normal"
df.loc[df["traffic_delay"] > 2, "root_cause"] = "Traffic"
df.loc[df["carrier"].isin(["GlobalTransit"]), "root_cause"] = "Carrier Reliability"
df.loc[df["special_handling"] != "None", "root_cause"] = "Handling Complexity"

# -----------------------------
# RISK SCORING (Predictive Layer)
# -----------------------------
df["risk_score"] = (
    0.4 * (df["actual_delivery_days"] / (df["promised_delivery_days"] + 1)) +
    0.3 * (df["delivery_cost"] / (df["delivery_cost"].mean() + 1)) +
    0.3 * (df["traffic_delay"] / (df["traffic_delay"].max() + 1))
)

# -----------------------------
# SIDEBAR
# -----------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Executive Dashboard", "Process Deviation Mining", "Value Leakage Simulator", "Raw Data"]
)

# ======================================================
# EXECUTIVE DASHBOARD
# ======================================================
if page == "Executive Dashboard":

    st.subheader("Operational Health Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Value Leakage", f"₹{int(df['value_leakage'].sum())}")
    col2.metric("Perfect Order Rate", f"{round((~df['deviation']).mean()*100,1)}%")
    col3.metric("Avg Customer Rating", round(df["rating"].mean(),2))
    col4.metric("Active Deviations", int(df["deviation"].sum()))

    st.divider()

    st.subheader("Leakage by Deviation Type")
    st.bar_chart(df.groupby("status")["value_leakage"].sum())

    st.subheader("Carrier Cost vs Leakage")
    st.scatter_chart(
        df.groupby("carrier")[["delivery_cost", "value_leakage"]].mean()
    )

    st.subheader("Top Recommended Actions")
    actions = df.groupby("carrier")["value_leakage"].sum().sort_values(ascending=False).head(3)
    for carrier, loss in actions.items():
        st.warning(f"Shift volume away from **{carrier}** → Potential saving ₹{int(loss*0.3)}")

# ======================================================
# PROCESS DEVIATION MINING
# ======================================================
elif page == "Process Deviation Mining":

    st.subheader("Process Flow: Priority → Carrier → Outcome")

    flow = df.groupby(["priority", "carrier", "status"]).size().reset_index(name="count")

    labels = list(pd.unique(flow[["priority", "carrier", "status"]].values.ravel()))
    label_index = {label: i for i, label in enumerate(labels)}

    source = [label_index[p] for p in flow["priority"]] + \
             [label_index[c] for c in flow["carrier"]]

    target = [label_index[c] for c in flow["carrier"]] + \
             [label_index[s] for s in flow["status"]]

    values = list(flow["count"]) * 2

    fig = go.Figure(go.Sankey(
        node=dict(label=labels),
        link=dict(source=source, target=target, value=values)
    ))

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# VALUE LEAKAGE SIMULATOR
# ======================================================
elif page == "Value Leakage Simulator":

    st.subheader("Predictive Value & Simulation")

    delay_reduction = st.slider("Reduce Delay Frequency (%)", 0, 50, 20)
    damage_reduction = st.slider("Reduce Damage Rate (%)", 0, 50, 10)
    route_saving = st.slider("Route Optimization Saving (%)", 0, 30, 15)

    base_leakage = df["value_leakage"].sum()

    simulated_saving = (
        base_leakage * (delay_reduction / 100 * 0.4 +
                        damage_reduction / 100 * 0.4 +
                        route_saving / 100 * 0.2)
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Projected Monthly Savings", f"₹{int(simulated_saving)}")
    col2.metric("New Leakage Baseline", f"₹{int(base_leakage - simulated_saving)}")
    col3.metric("ROI Horizon", f"{round(100000/simulated_saving,1)} Months")

# ======================================================
# RAW DATA
# ======================================================
else:
    st.subheader("Raw Data Explorer")
    st.dataframe(df)
