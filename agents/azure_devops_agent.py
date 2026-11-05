import asyncio
import copy
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from typing import Any
import uuid

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent_state.agent_state import AgentState
from config.azure_config import llm
from helpers.cosmos_db_client import async_container
from helpers.dependency_manager import get_dependency_results
from utils.tracing import tracer
from opentelemetry import trace


_clients: dict[str, Client] = {}
_client_locks: dict[str, asyncio.Lock] = {}
_MAX_TOOL_ROUNDS = 6
_TOOL_DOMAINS = (
    "core",
    "work",
    "work-items",
    "repositories",
    "pipelines",
    "wiki",
    "search",
)
_READ_ONLY_ACTIONS = {
    "work": {
        "list_iterations",
        "list_team_iterations",
        "get_team_settings",
        "get_team_capacity",
        "get_iteration_capacities",
    },
    "wit_work_item": {
        "get",
        "get_batch",
        "list_comments",
        "my",
        "list_revisions",
        "list_for_iteration",
        "get_type",
    },
    "wit_query": {"get", "get_results", "wiql"},
    "wit_backlog": {"list", "list_work_items"},
    "repo_repository": {"get", "list"},
    "repo_pull_request": {"get", "list", "list_by_commits"},
    "repo_pull_request_thread": {"list", "list_comments"},
    "repo_branch": {"get", "list", "list_mine"},
    "repo_file": {"get_content", "list_directory"},
    "pipelines_build": {"list", "get_status", "get_changes"},
    "pipelines_build_log": {"list", "get_content"},
    "pipelines_definition": {"list", "list_revisions"},
    "pipelines_run": {"get", "list"},
    "pipelines_artifact": {"list"},
    "wiki": {"list_wikis", "get_wiki", "list_pages", "get_page", "get_page_content"},
}
_READ_ONLY_TOOLS = {
    "core_list_projects",
    "core_list_project_teams",
    "core_get_identity_ids",
    "repo_pull_request_org",
    "repo_search_commits",
    "search_code",
    "search_wiki",
    "search_workitem",
}


def _get_client() -> tuple[str, Client]:
    organization = os.getenv("AZURE_DEVOPS_ORG", "selection-navigator").strip()
    if not organization:
        raise RuntimeError(
            "Azure DevOps is not configured. Set AZURE_DEVOPS_ORG to your organization name."
        )

    npx = shutil.which("npx")
    if not npx:
        raise RuntimeError(
            "The local Azure DevOps MCP server requires Node.js 20 or later. "
            "Install Node.js, reopen the terminal running Streamlit, and try again."
        )
    if not shutil.which("az"):
        raise RuntimeError(
            "Azure DevOps MCP is configured to authenticate with Azure CLI. "
            "Install Azure CLI, run `az login`, and try again."
        )

    client_key = f"stdio:{organization}"
    client = _clients.get(client_key)
    if client is None:
        arguments = [
            "-y",
            "@azure-devops/mcp",
            organization,
            "--authentication",
            "azcli",
            "-d",
            *_TOOL_DOMAINS,
        ]
        if sys.platform == "win32" and npx.lower().endswith((".cmd", ".bat")):
            transport = StdioTransport(
                command=os.environ.get("COMSPEC", "cmd.exe"),
                args=["/d", "/c", "npx", *arguments],
            )
        else:
            transport = StdioTransport(command=npx, args=arguments)
        client = Client(
            transport,
            name="Azure DevOps read-only agent",
        )
        _clients[client_key] = client
    return client_key, client


def _read_only_tools(tools: list[Any]) -> list[Any]:
    allowed = []
    for tool in tools:
        if tool.name in _READ_ONLY_TOOLS:
            allowed.append(tool)
            continue

        actions = _READ_ONLY_ACTIONS.get(tool.name)
        if not actions:
            continue

        schema = copy.deepcopy(tool.input_schema)
        action_schema = schema.get("properties", {}).get("action")
        if not action_schema:
            continue

        if "enum" in action_schema:
            action_schema["enum"] = [
                action for action in action_schema["enum"] if action in actions
            ]
            if not action_schema["enum"]:
                continue
        allowed.append(tool.model_copy(update={"input_schema": schema}))
    return allowed


def _tool_schema(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


def _tool_result_text(result: Any) -> str:
    text_parts = [
        item.text
        for item in getattr(result, "content", [])
        if getattr(item, "text", None) is not None
    ]
    if text_parts:
        return "\n".join(text_parts)
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(mode="json"), default=str)
    return str(result)


async def _record_model_usage(response: Any, thread_id: str | None) -> None:
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = metadata.get("token_usage")
    if not usage:
        return

    model_name = metadata.get("model_name", "unknown")
    trace.get_current_span().set_attribute("llm.model", model_name)
    trace.get_current_span().set_attribute("llm.prompt_tokens", usage["prompt_tokens"])
    trace.get_current_span().set_attribute(
        "llm.completion_tokens", usage["completion_tokens"]
    )
    trace.get_current_span().set_attribute("llm.total_tokens", usage["total_tokens"])
    await async_container.create_item(
        {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "agent": "azure_devops_agent",
            "thread_id": thread_id,
            "model": model_name,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "finish_reason": metadata.get("finish_reason"),
        }
    )


async def azure_devops_agent(state: AgentState) -> AgentState:
    """Answer Azure DevOps questions using read-only MCP tools."""
    current_task = state.get("current_task", {})
    task_prompt = state.get("task_prompt", state.get("user_input", ""))
    task_id = current_task.get("id", "azure-devops")
    thread_id = state.get("thread_id")
    endpoint, client = _get_client()
    lock = _client_locks.setdefault(endpoint, asyncio.Lock())

    with tracer.start_as_current_span(
        "agent.azure_devops",
        attributes={
            "thread_id": thread_id or "",
            "task_id": task_id,
            "intent": "azure_devops",
        },
    ):
        async with lock:
            async with client:
                available_tools = _read_only_tools(await client.list_tools())
                if not available_tools:
                    raise RuntimeError(
                        "The Azure DevOps MCP server returned no tools. Check the organization and access permissions."
                    )

                tool_by_name = {tool.name: tool for tool in available_tools}
                model = llm.bind_tools(
                    [_tool_schema(tool) for tool in available_tools]
                )
                dependency_results = get_dependency_results(
                    current_task, state.get("task_results", {})
                )
                messages = [
                    SystemMessage(
                        content=(
                            "You are an Azure DevOps read-only assistant. Use the "
                            "available Azure DevOps tools to retrieve requested data. "
                            "Do not attempt to create, update, delete, approve, or trigger "
                            "anything. Treat tool results as the only source of Azure "
                            "DevOps facts. Clearly say when data is unavailable, and "
                            "include relevant IDs, project names, and links returned by tools."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"User request:\n{task_prompt}\n\n"
                            "Results from prerequisite steps:\n"
                            f"{json.dumps(dependency_results, indent=2, default=str)}"
                        )
                    ),
                ]

                for _ in range(_MAX_TOOL_ROUNDS):
                    response = await model.ainvoke(messages)
                    await _record_model_usage(response, thread_id)
                    messages.append(response)

                    if not response.tool_calls:
                        answer = response.content
                        break

                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        if tool_name not in tool_by_name:
                            raise RuntimeError(
                                f"The model requested an Azure DevOps tool that was not provided: {tool_name}"
                            )
                        with tracer.start_as_current_span(
                            "agent.azure_devops.tool",
                            attributes={"tool_name": tool_name},
                        ):
                            result = await client.call_tool(
                                tool_name, tool_call.get("args", {})
                            )
                        messages.append(
                            ToolMessage(
                                content=_tool_result_text(result),
                                tool_call_id=tool_call["id"],
                                name=tool_name,
                            )
                        )
                else:
                    raise RuntimeError(
                        "Azure DevOps tool execution exceeded the maximum number of rounds."
                    )

        if not isinstance(answer, str):
            answer = json.dumps(answer, default=str)
        return {
            "agent_result": [answer],
            "completed_steps": ["azure_devops"],
            "task_results": {
                task_id: {
                    "intent": "azure_devops",
                    "result": answer,
                }
            },
        }
