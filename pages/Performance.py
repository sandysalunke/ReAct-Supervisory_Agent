import os
from dotenv import load_dotenv
load_dotenv()
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient
from azure.monitor.query import LogsQueryStatus
import pandas as pd
from datetime import timedelta


workspace_id=os.getenv("WORKSPACE_ID")

client = LogsQueryClient(
    credential=DefaultAzureCredential()
)

def get_avg_span_duration():

    query = """
    dependencies
    | where timestamp > ago(24h)
    | summarize
        AvgDurationMs = avg(duration)
      by name
    | order by AvgDurationMs desc
    """
    print("query", query)
    result = client.query_workspace(
        workspace_id=workspace_id,
        query=query,
        timespan=timedelta(hours=24)
    )
    print("result", result)

    if result.status == LogsQueryStatus.SUCCESS:

        table = result.tables[0]

        return pd.DataFrame(
            table.rows,
            columns=[c.name for c in table.columns]
        )

    return pd.DataFrame()

get_avg_span_duration()