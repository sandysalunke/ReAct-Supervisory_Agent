from config.azure_config import llm
from agent_state.agent_state import AgentState

def classify_intent(state: AgentState) -> AgentState:
    
    prompt = f"""
        You are a workflow planner.

        ORIGINAL USER GOAL:
        {state["user_input"]}

        COMPLETED INTENTS:
        {state.get("completed_steps", [])}

        PREVIOUS RESULTS:
        {state.get("agent_result", [])}

        AVAILABLE INTENTS:

        chat
        - Generate explanations, summaries, recommendations, or final responses.

        database_search
        - Retrieve information from databases.

        file_reasoning
        - Read and reason over uploaded files.

        excel_analysis
        - Analyze spreadsheet data.

        image_to_text
        - Extract information from images.

        image_generation
        - Create images.

        meeting_assistant
        - Handle meeting-related tasks.

        END
        - Use only when the original user goal has been completely satisfied.

        PLANNING RULES:

        1. Analyze the original user goal.
        2. Break the goal into required sub-tasks.
        3. Determine which sub-tasks have already been completed using COMPLETED INTENTS and PREVIOUS RESULTS.
        4. Determine the next unfinished sub-task.
        5. Return the intent needed for that next unfinished sub-task.
        6. Never repeat an intent unless previous results indicate it failed or additional information is still required.
        7. If chat has already been executed and its output provides the final answer required by the original user goal, return END.
        8. Return chat only when information gathering and analysis are complete and
        a user-facing summary/explanation still needs to be generated.
        9. After chat generates that final summary, return END.

        Output exactly one intent name and nothing else.
        """
    
    # Invoke LLM
    intent = llm.invoke(prompt).content

    state["intent"] = intent

    return state