from agent_state.agent_state import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from pathlib import Path
from config.azure_config import client, CHAT_MODEL, WHISPER_MODEL

# Whisper LLM to read transcript
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model=WHISPER_MODEL   # or your Azure deployment name
        )
    return transcript.text

# LLM to process transcript and return structured response 
def process_meeting(transcript):
    response = client.chat.completions.create(
        model=CHAT_MODEL,  # your deployment name
        messages=[
            {
                "role": "system",
                "content": "You are an AI meeting assistant."
            },
            {
                "role": "user",
                "content": f"""
                Analyze this meeting transcript:

                Provide:
                1. Summary
                2. Key Points
                3. Action Items (with owner if possible)
                4. List of attendees (if mentioned)

                Transcript:
                {transcript}
                """
            }
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# meeting_agent - Added as node in supervisor agent graph
def meeting_agent(state: AgentState) -> AgentState:
    """Use this agent for meeting assistance"""
    print("\n===== MEETING AGENT =====")

    file_path = state["uploaded_file"]
    file_extension = Path(file_path).suffix
    file_extensions = { ".mp4", ".mkv", ".webm", ".avi", ".mov", ".wmv", ".m4v", ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma"}

    if file_path and file_extension in file_extensions:
        transcript = transcribe_audio(file_path)
        result = process_meeting(transcript)
    else:
        result = "I can not find the image, please upload the image to extract the text."

    return {
        "agent_result": [result],
        "completed_steps": ["meeting_assistant"]
    }