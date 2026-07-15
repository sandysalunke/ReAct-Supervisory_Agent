from typing import Optional, Annotated
from agent_state.agent_state import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from agents import chat_agent, rag_agent, sql_agent, ocr_agent, meeting_agent, image_agent, excel_agent, summary_agent
from nodes.router import route_agent
from nodes.planner import create_pan

def invoke_graph(request: Annotated) -> AgentState:
    prompt = request["user_input"]
    uploaded_file = request["uploaded_file"]

    # Create execution plan for user input
    execution_plan = create_pan(request["user_input"])
    print("========execution_plan=====", execution_plan)
    
    # Initiate a graph
    graph = StateGraph(AgentState)

    # Map of agents linked to intent returned by planner in execution plan
    agent_mapp = {
        "chat": chat_agent.chat_agent,
        "image_generation": image_agent.image_agent,
        "file_reasoning": rag_agent.rag_agent,
        "database_search": sql_agent.sql_agent,
        "image_to_text": ocr_agent.ocr_agent,
        "meeting_assistant": meeting_agent.meeting_agent,
        "excel_analysis": excel_agent.excel_agent,
    }

    # Add independent nodes to graph 
    for task in execution_plan["tasks"]:
        graph.add_node(
            task["id"],
            agent_mapp[task["intent"]]
        )

    # Add summary_agent node that will always be the last node of the workflow
    graph.add_node("summarize_result", summary_agent.summary_agent)

    # Add edges to the graph
    for task in execution_plan["tasks"]:
        if not task["dependencies"]:
            graph.add_edge(START, task["id"])   # adds edges next to start (this could add parellel nodes)

        for dep in task["dependencies"]:
            graph.add_edge(dep, task["id"])     # adds edges to link dependent nodes

    # Add edges from NON Dependent nodes to summary_agent
    all_task_ids = {t["id"] for t in execution_plan["tasks"]}

    dependency_ids = {
        dep
        for t in execution_plan["tasks"]
        for dep in t["dependencies"]
    }

    leaf_nodes = all_task_ids - dependency_ids

    for leaf in leaf_nodes:
        graph.add_edge(leaf, "summarize_result")    # All leaf nodes are linked to summary_agent

    # Add edge from summary_agent to END
    graph.add_edge("summarize_result", END)

    # Compile graph
    app = graph.compile()

    # display(Image(app.get_graph().draw_mermaid_png(max_retries=5)))
    app.get_graph().print_ascii()

    result  = app.invoke(
        {
            "user_input": prompt,
            "uploaded_file": uploaded_file,
            "execution_plan": execution_plan
        }
    )
    
    # print("=========STATE========: ", result)

    return result