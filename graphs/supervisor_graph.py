####################################
import json
from typing import Annotated
from agent_state.agent_state import AgentState
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from agents import chat_agent, hybrid_rag_agent, rag_agent, sql_agent, ocr_agent, meeting_agent, image_agent, excel_agent, summary_agent, ppt_agent
from nodes.planner import create_plan, get_execution_plan, delete_execution_plan
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from utils.tracing import tracer
from opentelemetry import trace

####################################
# This variable to be replaced with a persistent checkpointer in future. Currently, 
# it is a memory checkpointer that will be lost when the server restarts.
GLOBAL_CHECKPOINTER = MemorySaver()

class WorkflowService:

    request: Annotated
    execution_plan: Annotated
    thread_id: str

    ###############################################
    # function to initialize the WorkflowService with request data and build the graph
    def __init__(self, request: Annotated):
        self.request = request
        self.thread_id = self.request.get("thread_id") or str(uuid7())
        self.checkpointer = GLOBAL_CHECKPOINTER
        # Trace graph construction per request/thread
        with tracer.start_as_current_span("workflow.init", attributes={"thread_id": self.thread_id}):
            self.graph = self.build_graph()

    ###############################################
    # function to build the graph based on the execution plan
    def build_graph(self) -> AgentState:
        prompt = self.request["user_input"]
        uploaded_file = self.request["uploaded_file"]
        chat_history = self.request["chat_history"]
    
        # Create execution plan for user input
        with tracer.start_as_current_span("workflow.build_graph", attributes={"thread_id": self.thread_id, "prompt_length": len(str(prompt) or "")}):
            if self.request.get("workflow_status") == "INTERRUPTED":
                self.execution_plan = get_execution_plan(self.thread_id)
            else:
                self.execution_plan = create_plan(self.thread_id, prompt, uploaded_file)
            try:
                task_count = len(self.execution_plan.get("tasks", []))
            except Exception:
                task_count = 0
            trace.get_current_span().set_attribute("workflow.tasks.count", task_count)
            trace.get_current_span().set_attribute("workflow.execution_plan", json.dumps(self.execution_plan))
            print("EXECUTION PLAN: ", json.dumps(self.execution_plan, indent=2))
        
        # Initiate a graph
        graph = StateGraph(AgentState)
    
        # Map of agents linked to intent returned by planner in execution plan
        agent_mapp = {
            "chat": chat_agent.chat_agent,
            "image_generation": image_agent.image_agent,
            "file_reasoning": rag_agent.rag_agent,
            "knowledge_search": hybrid_rag_agent.hybrid_rag_agent,
            "database_search": sql_agent.sql_agent,
            "image_to_text": ocr_agent.ocr_agent,
            "meeting_assistant": meeting_agent.meeting_agent,
            "excel_analysis": excel_agent.excel_agent,
            "ppt_creation": ppt_agent.ppt_agent
        }
    
        # Add independent nodes to graph 
        for task in self.execution_plan["tasks"]:
            graph.add_node(
                task["id"],
                self.create_task_node(
                    agent_mapp[task["intent"]],
                    task
                )
            )
    
        # Add summary_agent node that will always be the last node of the workflow
        graph.add_node("summarize_result", summary_agent.summary_agent)
    
        # Add edges to the graph
        for task in self.execution_plan["tasks"]:
            if not task["dependencies"]:
                graph.add_edge(START, task["id"])   # adds edges next to start (this could add parellel nodes)
    
            for dep in task["dependencies"]:
                graph.add_edge(dep, task["id"])     # adds edges to link dependent nodes
    
        # Add edges from NON Dependent nodes to summary_agent
        all_task_ids = {t["id"] for t in self.execution_plan["tasks"]}
    
        dependency_ids = {
            dep
            for t in self.execution_plan["tasks"]
            for dep in t["dependencies"]
        }
    
        leaf_nodes = all_task_ids - dependency_ids
    
        for leaf in leaf_nodes:
            graph.add_edge(leaf, "summarize_result")    # All leaf nodes are linked to summary_agent
    
        # Add edge from summary_agent to END
        graph.add_edge("summarize_result", END)
    
        # Compile graph
        app = graph.compile(checkpointer=self.checkpointer)
    
        # display(Image(app.get_graph().draw_mermaid_png(max_retries=5)))
        app.get_graph().print_ascii()

        return app

    ##############################################
    # function to create a task node for the graph
    def create_task_node(self, agent_fn, task):

        def node(state):

            with tracer.start_as_current_span(f"task.{task.get("intent")}", attributes={"task_id": task.get("id"), "intent": task.get("intent")}):
                return agent_fn({
                    **state,
                    "current_task": task,
                    "task_prompt": task["prompt"],
                })

        return node

    ##############################################
    # function to invoke the graph with input data and thread_id
    def invoke(self, input_data):
        print("----Graph Invoked----")
        with tracer.start_as_current_span("graph.invoke", attributes={"thread_id": self.thread_id}):
            result = self.graph.invoke(
                {
                    **input_data,
                    "thread_id": self.thread_id,
                    "execution_plan": self.execution_plan
                },
                config={
                    "configurable": {
                        "thread_id": self.thread_id
                    }
                }
            )
        return self._build_response(self.thread_id, result)

    ##############################################
    # function to resume the graph with user response and thread_id
    def resume(self, user_response, thread_id):
        print("----Graph Resumed----")
        with tracer.start_as_current_span("graph.resume", attributes={"thread_id": thread_id}):
            result = self.graph.invoke(
                Command(resume=user_response),
                config={
                    "configurable": {
                        "thread_id": thread_id
                    }
                }
            )
            delete_execution_plan(thread_id)
        return self._build_response(thread_id, result)

    ##############################################
    # function to build the response based on the result of the graph execution
    def _build_response(self, thread_id, result):
        with tracer.start_as_current_span("workflow.build_response", attributes={"thread_id": thread_id}):
            if "__interrupt__" in result:
                trace.get_current_span().set_attribute("workflow.status", "INTERRUPTED")
                print("===========Inturrupt==========")
                return {
                    "thread_id": thread_id,
                    "workflow_status": "INTERRUPTED",
                    "interrupt": result["__interrupt__"],
                    "data": {
                        "messages": [
                            HumanMessage(content=self.request["user_input"]),
                            SystemMessage(
                                content=result["__interrupt__"][0].value,
                                additional_kwargs={
                                    "thread_id": thread_id,
                                    "workflow_status": "INTERRUPTED"
                                }
                            )
                        ]
                    }
                }

            trace.get_current_span().set_attribute("workflow.status", "COMPLETED")
            return {
                "thread_id": thread_id,
                "workflow_status": "COMPLETED",
                "data": result
            }