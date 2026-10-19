from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from helpers.cosmos_db_client import container
import streamlit as st

items = list(
    container.query_items(
        query="SELECT * FROM c WHERE IS_DEFINED(c.groundedness) AND IS_DEFINED(c.relevance) AND IS_DEFINED(c.completeness)",
        enable_cross_partition_query=True
    )
)

df = pd.DataFrame(items)

groundedness = df["groundedness"].mean()
relevance = df["relevance"].mean()
completeness = df["completeness"].mean()

c1, c2, c3 = st.columns(3)

c1.metric(
    "Groundedness",
    round(groundedness, 2)
)

c2.metric(
    "Relevance",
    round(relevance, 2)
)

c3.metric(
    "Completeness",
    round(completeness, 2)
)

# Score comparison
st.subheader("Scores by category")
score_df = pd.DataFrame({
    "Metric": [
        "Groundedness",
        "Relevance",
        "Completeness"
    ],
    "Score": [
        groundedness,
        relevance,
        completeness
    ]
})

st.bar_chart(
    score_df,
    x="Metric",
    y="Score"
)

# Daily quality trend
st.subheader("Daily quality trend")
df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

daily = (
    df.groupby(
        df["timestamp"].dt.date
    )[[
        "groundedness",
        "relevance",
        "completeness"
    ]]
    .mean()
    .reset_index()
)

st.line_chart(
    daily.set_index("timestamp")
)

# Low scores 
st.subheader("Low score threads")
low_scores = df[
    (df["groundedness"] < 4)
    | (df["relevance"] < 4)
    | (df["completeness"] < 4)
]

st.dataframe(
    low_scores[
        [
            "timestamp",
            "thread_id",
            "groundedness",
            "relevance",
            "completeness"
        ]
    ]
)

#### Read application insights data and build performace metrics (latency)
# import os
# from dotenv import load_dotenv
# load_dotenv()
# from azure.identity import DefaultAzureCredential
# from azure.monitor.query import LogsQueryClient
# from azure.monitor.query import LogsQueryStatus
# import pandas as pd
# from datetime import timedelta


# workspace_id=os.getenv("WORKSPACE_ID")

# client = LogsQueryClient(
#     credential=DefaultAzureCredential()
# )

# def get_avg_span_duration():

#     query = """
#     dependencies
#     | where timestamp > ago(24h)
#     | summarize
#         AvgDurationMs = avg(duration)
#       by name
#     | order by AvgDurationMs desc
#     """
#     print("query", query)
#     result = client.query_workspace(
#         workspace_id=workspace_id,
#         query=query,
#         timespan=timedelta(hours=24)
#     )
#     print("result", result)

#     if result.status == LogsQueryStatus.SUCCESS:

#         table = result.tables[0]

#         return pd.DataFrame(
#             table.rows,
#             columns=[c.name for c in table.columns]
#         )

#     return pd.DataFrame()

# get_avg_span_duration()