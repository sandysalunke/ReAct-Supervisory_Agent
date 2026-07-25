from config.azure_config import llm
import json

def create_pan(prompt: str, uploaded_file: str) -> dict:
    
    prompt = f"""
        Create a workflow DAG.

        USER GOAL:
        {prompt}

        UPLOADED FILE PATH:
        {uploaded_file}
        
        AVAILABLE INTENTS:
            chat
                - Public knowledge, General Knowledge, Generate explanations, summaries, recommendations, or final responses.
            database_search
                - Retrieve data from databases.
            file_reasoning
                - Reasoning based on uploaded text file content.
                - Allowed file extensions: ".txt", ".pdf", ".docx", ".doc", ".ppt", ".pptx" 
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

    return execution_plan

