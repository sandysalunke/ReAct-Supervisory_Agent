from config.azure_config import llm
import json
import os
from pathlib import Path

def create_plan(thread_id: str, prompt: str, uploaded_file: str) -> dict:
    
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
                - When uncertain between chat and knowledge_base for a factual question, choose knowledge_base.
            database_search
                - Retrieve data from database.
            file_reasoning
                - Reasoning based on uploaded text file content.
                - Allowed file extensions: ".txt", ".pdf", ".docx", ".doc", ".ppt", ".pptx"
            knowledge_search
                - A user does NOT need to explicitly mention "enterprise knowledge", "knowledge base", "company documents", or "search".
                - Questions asking about organizational responsibilities, ownership, policies, procedures, processes, standards, products, business rules, internal terminology, or "who is responsible for X" should be treated as enterprise knowledge questions unless the user explicitly refers to an uploaded file.
                - For ambiguous factual questions that could reasonably refer to the user's organization, prefer knowledge_base over chat.
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
            - Choose appropriate tasks (intents) based on file extension when file path is provided 
            - Assign unique ids.
            - Specify the prompt
            - Specify dependencies.
            - Tasks that can run in parallel must have empty dependencies.
            - Do not include markup
            - Return JSON only.
            - If the user goal can be fulfilled directly by a single available intent, return tasks DAG containing exactly one task.
            - Do not create unnecessary intermediate tasks.
            - Use chat directly for general knowledge questions that do not require
            database access, files, meetings, images, or spreadsheets.
        """
    
    # Invoke LLM
    result = llm.invoke(prompt).content

    execution_plan = json.loads(result)

    save_execution_plan(thread_id, execution_plan)  # Save the execution plan for the thread

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

    file_path = f"./data/plan/{thread_id}.json"

    if file_path.exists():
        os.remove(file_path)