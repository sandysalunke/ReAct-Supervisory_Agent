import asyncio
from config.azure_config import llm
import json
import os
from pathlib import Path
from typing import Optional
import uuid
from datetime import datetime, UTC
from helpers.cosmos_db_client import async_container
from utils.tracing import tracer
from opentelemetry import trace

async def create_plan(
    thread_id: str, prompt: str, uploaded_file: Optional[str]
) -> dict:
    
    prompt = f"""
        Create a workflow DAG.

        USER GOAL:
        {prompt}

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

        Rules:
            - Determine the tasks required to achieve user goal.
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

    with tracer.start_as_current_span("planner.create_plan", attributes={"llm_used": True, "thread_id": thread_id}):
    
        # Invoke LLM
        result = await llm.ainvoke(prompt)

        execution_plan = json.loads(result.content)

        # Add traces in AI for observability and monitoring
        usage = result.response_metadata["token_usage"]
        trace.get_current_span().set_attribute("llm.model", result.response_metadata["model_name"])
        trace.get_current_span().set_attribute("llm.prompt_tokens", usage["prompt_tokens"])
        trace.get_current_span().set_attribute("llm.completion_tokens", usage["completion_tokens"])
        trace.get_current_span().set_attribute("llm.total_tokens", usage["total_tokens"])
        trace.get_current_span().set_attribute("llm.finish_reason", result.response_metadata["finish_reason"])
        trace.get_current_span().set_attribute("execution_plan", execution_plan)

        # Add traces in Cosmos DB for cost observability and monitoring
        await async_container.create_item(
            {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
                "agent": "chat_agent",
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