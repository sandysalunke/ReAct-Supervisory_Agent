import json
from langchain.tools import tool
from langchain_core.utils.uuid import uuid7
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config.azure_config import llm

def create_excel_tools(data_frame, dependency_results):
    @tool
    def get_schema() -> str:
        """Get workbook structure."""

        prompt = f"""
            Columns:
            {list(data_frame.columns)}

            Row count:
            {len(data_frame)}
        """

        return prompt

    @tool
    def analyze_data(question: str) -> str:
        """Analyze dataframe data."""
        
        prompt = f"""
        Results from previous steps:
        {json.dumps(dependency_results, indent=2)}

        DataFrame columns:
        {list(data_frame.columns)}

        Write pandas code to answer:

        {question}

        STRICT RULES:
        - Use results from previous steps if they help answer the question.
        - Only return valid Python code
        - No explanations
        - No markdown (no ``` blocks)
        - The Excel file is already loaded into a pandas DataFrame named data_frame.
        - Never call pd.read_excel().
        - Never read files from disk.
        - Never assume a filename.
        - Never create sample data.
        - Only use the provided data_frame variable.
        - Generate only analysis code.
        - Store final result in variable 'result'
        """

        local_vars = {"data_frame": data_frame}
        safe_globals = {"pd": __import__("pandas")}
        code = llm.invoke(prompt).content

        exec(code, safe_globals, local_vars)

        return local_vars.get("result", None)

    @tool
    def create_chart(question: str) -> str:
        """Create chart from dataframe."""

        prompt = f"""
            Results from previous steps:
            {json.dumps(dependency_results, indent=2)}

            DataFrame columns:
            {list(data_frame.columns)}

            Create a chart for:

            {question}

            STRICT RULES:
            - Use results from previous steps if they help answer the question.
            - Only return valid Python code
            - No explanations
            - No markdown (no ``` blocks)
            - Use dataframe variable name: data_frame
            - Use matplotlib.pyplot as plt
            - Save the chart to:

            chart_path = "./data/chart.png"

            - The final code MUST define:

            chart_path = "./data/images/{uuid7()}.png"

            - Do NOT print anything
            - Do NOT explain anything
            """

        local_vars = {"data_frame": data_frame}
        safe_globals = {"pd": __import__("pandas")}
        code = llm.invoke(prompt).content
        
        exec(code, safe_globals, local_vars)

        return {
            "status": "success",
            "chart_path": local_vars.get("chart_path", None)
        }
    
    return [get_schema, analyze_data, create_chart]
