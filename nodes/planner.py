import asyncio
import json
import os
import re
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from config.azure_config import llm
from helpers.cosmos_db_client import async_container
from utils.tracing import tracer
from opentelemetry import trace


SUPPORTED_INTENTS = frozenset(
    {
        "chat",
        "image_generation",
        "file_reasoning",
        "knowledge_search",
        "database_search",
        "image_to_text",
        "meeting_assistant",
        "excel_analysis",
        "ppt_creation",
        "azure_devops",
    }
)
MAX_PLAN_TASKS = 8
MAX_TASK_PROMPT_LENGTH = 4000
MAX_TASK_ID_LENGTH = 64
TASK_ID_PATTERN = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9_-]{{0,{MAX_TASK_ID_LENGTH - 1}}}\Z"
)
RESERVED_TASK_IDS = frozenset({"summarize_result"})


def normalize_execution_plan(
    execution_plan: object,
    allowed_intents: Optional[set[str]] = None,
) -> dict:
    if isinstance(execution_plan, list):
        execution_plan = {"tasks": execution_plan}

    if not isinstance(execution_plan, dict):
        raise ValueError(
            "Execution plan must be an object containing a 'tasks' list "
            "or a list of tasks."
        )

    if set(execution_plan) != {"tasks"}:
        raise ValueError("Execution plan must contain only a 'tasks' field.")

    tasks = execution_plan["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Execution plan 'tasks' must be a non-empty list.")
    if len(tasks) > MAX_PLAN_TASKS:
        raise ValueError(
            f"Execution plan may contain at most {MAX_PLAN_TASKS} tasks."
        )

    intents = SUPPORTED_INTENTS if allowed_intents is None else allowed_intents
    required_fields = {"id", "intent", "prompt"}
    allowed_fields = required_fields | {"dependencies"}
    normalized_tasks = []
    task_ids = set()

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Execution plan task {index} must be an object.")
        if not required_fields.issubset(task):
            raise ValueError(
                f"Execution plan task {index} must include 'id', 'intent', and 'prompt'."
            )
        if not set(task).issubset(allowed_fields):
            raise ValueError(
                f"Execution plan task {index} contains unsupported fields."
            )

        task_id = task["id"]
        if (
            not isinstance(task_id, str)
            or len(task_id) > MAX_TASK_ID_LENGTH
            or not TASK_ID_PATTERN.fullmatch(task_id)
        ):
            raise ValueError(
                f"Execution plan task {index} has an invalid id; use 1-{MAX_TASK_ID_LENGTH} "
                "letters, digits, hyphens, or underscores, starting with a letter or digit."
            )
        if task_id in RESERVED_TASK_IDS:
            raise ValueError(
                f"Execution plan task id {task_id!r} is reserved by the workflow."
            )
        if task_id in task_ids:
            raise ValueError(f"Execution plan contains duplicate task id {task_id!r}.")
        task_ids.add(task_id)

        intent = task["intent"]
        if not isinstance(intent, str) or intent not in intents:
            raise ValueError(
                f"Execution plan task {task_id!r} has unsupported intent {intent!r}."
            )

        task_prompt = task["prompt"]
        if (
            not isinstance(task_prompt, str)
            or not task_prompt.strip()
            or len(task_prompt) > MAX_TASK_PROMPT_LENGTH
        ):
            raise ValueError(
                f"Execution plan task {task_id!r} prompt must contain 1-"
                f"{MAX_TASK_PROMPT_LENGTH} characters."
            )

        dependencies = task.get("dependencies", [])
        if dependencies is None:
            dependencies = []
        if not isinstance(dependencies, list) or any(
            not isinstance(dependency, str) for dependency in dependencies
        ):
            raise ValueError(
                f"Execution plan task {task_id!r} dependencies must be a list of task ids."
            )
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(
                f"Execution plan task {task_id!r} contains duplicate dependencies."
            )
        if task_id in dependencies:
            raise ValueError(
                f"Execution plan task {task_id!r} cannot depend on itself."
            )

        normalized_tasks.append(
            {
                "id": task_id,
                "intent": intent,
                "prompt": task_prompt,
                "dependencies": dependencies,
            }
        )

    for task in normalized_tasks:
        for dependency in task["dependencies"]:
            if dependency not in task_ids:
                raise ValueError(
                    f"Execution plan task {task['id']!r} references unknown dependency "
                    f"{dependency!r}."
                )

    dependents = {task_id: [] for task_id in task_ids}
    in_degrees = {}
    for task in normalized_tasks:
        in_degrees[task["id"]] = len(task["dependencies"])
        for dependency in task["dependencies"]:
            dependents[dependency].append(task["id"])

    ready = deque(
        task_id for task_id, in_degree in in_degrees.items() if in_degree == 0
    )
    visited = 0
    while ready:
        task_id = ready.popleft()
        visited += 1
        for dependent in dependents[task_id]:
            in_degrees[dependent] -= 1
            if in_degrees[dependent] == 0:
                ready.append(dependent)

    if visited != len(normalized_tasks):
        raise ValueError("Execution plan dependencies must form an acyclic graph.")

    return {"tasks": normalized_tasks}


def build_planner_prompt(user_goal: str, uploaded_file: Optional[str]) -> str:
    return f"""
        Create a workflow DAG.

        USER GOAL:
        {user_goal}

        UPLOADED FILE PATH:
        {uploaded_file}
        
        AVAILABLE INTENTS:
            chat
                - General/public knowledge that is NOT organization-specific.
                - Conversational and generative requests such as writing, rewriting, brainstorming, explanations, and recommendations.
                - Do NOT use for questions about organizational policies, responsibilities, procedures, processes, standards, products
                or internal documentation.
                - When uncertain between chat and knowledge_search for an organization-related factual question, choose knowledge_search.
            database_search
                - Retrieve data from database.
            file_reasoning
                - Reasoning based on uploaded text file content.
                - Allowed file extensions: ".txt", ".pdf", ".docx", ".doc", ".ppt", ".pptx"
            knowledge_search
                - A user does NOT need to explicitly mention "enterprise knowledge", "knowledge base", "company documents", or "search".
                - Questions asking about organizational responsibilities, ownership, policies, procedures, processes, standards, products, business rules, internal terminology, or "who is responsible for X" should be treated as enterprise knowledge questions unless the user explicitly refers to an uploaded file.
                - For ambiguous factual questions that could reasonably refer to the user's organization, prefer knowledge_search over chat.
                - Use chat only when the question is clearly general/public knowledge or is a generative/conversational task.
            excel_analysis
                - Analyze uploaded spreadsheet/excel data.
                - Allowed file extensions: ".xls", ".xlsx", ".xlm", ".xlsb"
            image_to_text
                - Extract information from images, use for OCR.
                - Allowed file extensions: ".png", ".jpg", ".jpeg", ".gif", ".webp"
            image_generation
                - Create images.
            meeting_assistant
                - Handle meeting-related tasks.
                - Allowed file extensions: ".mp4", ".mkv", ".webm", ".avi", ".mov", ".wmv", ".m4v", ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma"
            ppt_creation
                - Create powerpoint presentation deck
            azure_devops
                - Read Azure DevOps projects, work items, repositories and pull requests,
                  pipelines and builds, wiki pages, and team or iteration information.
                - Use for requests that explicitly ask about Azure DevOps, ADO, work items,
                  pull requests, repositories, builds, pipelines, or Azure DevOps wikis.
                - This capability is read-only.

        Rules:
            - Determine the tasks required to achieve user goal.
            - Return exactly one JSON object with this shape:
              {{"tasks":[{{"id":"task-1","intent":"chat","prompt":"...","dependencies":[]}}]}}
            - Return no more than {MAX_PLAN_TASKS} tasks.
            - Each task must have an id, intent, prompt, and optional dependencies field; do not add other fields.
            - Task ids must be unique, at most {MAX_TASK_ID_LENGTH} characters, and start with a letter or digit.
            - Each task prompt must be non-empty and no longer than {MAX_TASK_PROMPT_LENGTH} characters.
            - Route explicit Azure DevOps or ADO data requests to azure_devops, not chat,
              database_search, or knowledge_search.
            - Route from the current USER GOAL first. The uploaded file path only indicates that a file is available; it does not mean the user is asking about that file.
            - Choose file_reasoning only when the current USER GOAL explicitly asks to read, summarize, analyze, or answer from the uploaded document, or uses a clear reference such as "this file" or "the document I uploaded".
            - If the current USER GOAL asks about organizational policies, procedures, responsibilities, products, or internal business rules, choose knowledge_search even when an unrelated document is attached. Do not select file_reasoning just because an uploaded file path is present.
            - When a request explicitly asks to compare or combine the uploaded document with organization knowledge, include both file_reasoning and knowledge_search as separate tasks and set dependencies only when one task needs the other's result.
            - Choose file-type-specific intents such as excel_analysis, image_to_text, or meeting_assistant only when the current USER GOAL requests work on the uploaded file and its extension matches that intent.
            - Assign unique ids.
            - Specify the prompt
            - Specify dependencies.
            - Tasks that can run in parallel must have empty dependencies.
            - Do not include markup
            - Return JSON only.
            - Do NOT wrap the JSON in ```json or ``` code fences. 
            - Do not include explanations, markdown, or any text outside the JSON object.
            - If the user goal can be fulfilled directly by a single available intent, return tasks DAG containing exactly one task.
            - Do not create unnecessary intermediate tasks.
            - Use chat directly for general knowledge questions that do not require
            database access, files, meetings, images, or spreadsheets.
        """


async def create_plan(
    thread_id: str, prompt: str, uploaded_file: Optional[str]
) -> dict:
    planner_prompt = build_planner_prompt(prompt, uploaded_file)

    with tracer.start_as_current_span("planner.create_plan", attributes={"llm_used": True, "thread_id": thread_id}):
    
        # Invoke LLM
        result = await llm.ainvoke(planner_prompt)

        execution_plan = normalize_execution_plan(json.loads(result.content))

        # Add traces in AI for observability and monitoring
        usage = result.response_metadata["token_usage"]
        trace.get_current_span().set_attribute("llm.model", result.response_metadata["model_name"])
        trace.get_current_span().set_attribute("llm.prompt_tokens", usage["prompt_tokens"])
        trace.get_current_span().set_attribute("llm.completion_tokens", usage["completion_tokens"])
        trace.get_current_span().set_attribute("llm.total_tokens", usage["total_tokens"])
        trace.get_current_span().set_attribute("llm.finish_reason", result.response_metadata["finish_reason"])
        trace.get_current_span().set_attribute("execution_plan", execution_plan)

        # Add traces in Cosmos DB for cost observability and monitoring
        await async_container.upsert_item(
            {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
                "agent": "planner",
                "thread_id": thread_id,
                "model": result.response_metadata["model_name"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "finish_reason": result.response_metadata["finish_reason"],
            }
        )

        await asyncio.to_thread(
            save_execution_plan, thread_id, execution_plan
        )

        return execution_plan

def save_execution_plan(thread_id: str, execution_plan: dict):
    """
    Save execution plan for a thread.
    """

    file_path = f"./data/plan/{thread_id}.json"

    data = {
        "thread_id": thread_id,
        "execution_plan": execution_plan
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def get_execution_plan(thread_id: str):
    """
    Retrieve execution plan for a thread.
    """

    file_path = Path(f"./data/plan/{thread_id}.json")

    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("execution_plan")

def delete_execution_plan(thread_id: str):
    """
    Delete execution plan after workflow completion.
    """

    file_path = Path(f"./data/plan/{thread_id}.json")

    if file_path.exists():
        os.remove(file_path)