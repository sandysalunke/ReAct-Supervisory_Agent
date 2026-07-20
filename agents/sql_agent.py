from agent_state.agent_state import AgentState
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from config.azure_config import llm

# Function to run the user prompt through LLM
# Uses SQL agent from langchanin agent_toolkits
# Langchain SQL agent is capable of generating SQL queries 
# and translate the result in human language 
def ask_sql_llm(state: AgentState):

    prompt = state.get("user_input")
    agent_result = state.get("agent_result")

    # SQLite DB connection
    db = SQLDatabase.from_uri(
        "sqlite:///sales_multi.db",
        include_tables=["customers", "orders", "products"]
    )

    # Set Rules
    custom_prefix = """
        You are a SQL expert.

        Tables:
        - customers(customer_id, name, city)
        - products(product_id, product_name, category, price)
        - orders(order_id, customer_id, product_id, quantity, total_amount, order_date)

        Rules:
        - Use JOIN when needed
        - Only SELECT queries
        - Use correct column names
    """

    custom_suffix = f"""
        Results from previous steps:
        {agent_result}
    """

    # SQL Agent
    sql_agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        prefix=custom_prefix,
        suffix=custom_suffix
    )

    response = sql_agent.run(prompt)

    return response

# sql_agent - Added as node in supervisor agent graph
def sql_agent(state: AgentState) -> AgentState:
    """Use this agent for Database operations, database search, SQL generation"""
    
    print("\n===== SQL AGENT =====")

    result = ask_sql_llm(state)

    return {
        "agent_result": [result],
        "completed_steps": ["database_search"]
    }



