from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from helpers.cosmos_db_client import container
import streamlit as st

items = list(
    container.query_items(
        query="SELECT * FROM c",
        enable_cross_partition_query=True
    )
)

df = pd.DataFrame(items)

total_tokens = df["total_tokens"].sum()
total_calls = len(df)
avg_tokens = round(total_tokens / total_calls, 2)

c1, c2, c3 = st.columns(3)

c1.metric("Total Calls", f"{total_calls:,}")
c2.metric("Total Tokens", f"{total_tokens:,}")
c3.metric("Avg Tokens/Call", f"{avg_tokens:,}")

st.subheader("Tokens by Agent")
agent_usage = (
    df.groupby("agent")["total_tokens"]
      .sum()
      .reset_index()
)
st.bar_chart(
    data=agent_usage,
    x="agent",
    y="total_tokens"
)

st.subheader("Tokens by Model")
agent_usage = (
    df.groupby("model")["total_tokens"]
      .sum()
      .reset_index()
)
st.bar_chart(
    data=agent_usage,
    x="model",
    y="total_tokens"
)

st.subheader("Calls by Agent")
calls = (
    df.groupby("agent")
      .size()
      .reset_index(name="calls")
)

st.bar_chart(
    calls,
    x="agent",
    y="calls"
)

agent_cost_view = (
    df.groupby("agent")["total_tokens"]
      .sum()
      .reset_index()
)

agent_cost_view["%"] = (
    agent_cost_view["total_tokens"]
    /
    agent_cost_view["total_tokens"].sum()
    * 100
)

st.dataframe(agent_cost_view)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df["date"] = df["timestamp"].dt.date

daily = (
    df.groupby("date")["total_tokens"]
      .sum()
      .reset_index()
)

st.line_chart(
    daily,
    x="date",
    y="total_tokens"
)

threads = (
    df.groupby("thread_id")["total_tokens"]
      .sum()
      .sort_values(ascending=False)
      .reset_index()
)

st.subheader("Top Conversations")

st.dataframe(
    threads.head(10)
)


