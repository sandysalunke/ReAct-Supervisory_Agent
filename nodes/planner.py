from config.azure_config import llm
import json

def create_pan(prompt: str) -> dict:
    
    prompt = f"""
        Create a workflow DAG.

        USER GOAL:
        {prompt}
        
        AVAILABLE INTENTS:
            chat
                - Public knowledge, General Knowledge, Generate explanations, summaries, recommendations, or final responses.
            database_search
                - Retrieve data from databases.
            file_reasoning
                - Reasoning based on uploaded file content.
            excel_analysis
                - Analyze uploaded spreadsheet/excel data.
            image_to_text
                - Extract information from images, use for OCR.
            image_generation
                - Create images.
            meeting_assistant
                - Handle meeting-related tasks.

        Rules:
            1. Determine the tasks required to achieve user goal.
            2. Assign unique ids.
            3. Specify the prompt
            4. Specify dependencies.
            5. Tasks that can run in parallel must have empty dependencies.
            6. Do not include markup
            7. Return JSON only.
            8. If the user goal can be fulfilled directly by a single available intent, return tasks DAG containing exactly one task.
            9. Do not create unnecessary intermediate tasks.
            10. Use chat directly for general knowledge questions that do not require
            database access, files, meetings, images, or spreadsheets.
        """
    
    # print("======== Prompt: ", prompt)

    # Invoke LLM
    result = llm.invoke(prompt).content

    execution_plan = json.loads(result)
    # print("========execution_plan=====", execution_plan)

    return execution_plan

