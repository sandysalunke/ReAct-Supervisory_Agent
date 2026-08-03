from agent_state.agent_state import AgentState
from pathlib import Path
from config.azure_config import client, CHAT_MODEL, WHISPER_MODEL
import glob
import imageio_ffmpeg
import os
import subprocess

def chunk_audio_file(file_path):
    file_size = os.path.getsize(file_path)
    MAX_SIZE = 25 * 1024 * 1024
    
    if file_size <= MAX_SIZE:
        return [file_path]

    chunks = []
    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg,
            "-i", file_path,
            "-vn",
            "-f", "segment",
            "-segment_time", "300",
            "-c:a", "libmp3lame",
            "./data/uploads/audio/chunk_%03d.mp3"
        ], check=True)
        chunks = sorted(glob.glob("./data/uploads/audio/chunk_*.mp3"))
    except Exception:
        import traceback
        traceback.print_exc()
            
    return chunks

# Whisper LLM to read transcript
def transcribe_audio(file_path):
    chunks = chunk_audio_file(file_path)
    transcripts = []

    for chunk_file in chunks:

        response = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=open(chunk_file, "rb")
        )

        transcripts.append(response.text)

    full_transcript = "\n".join(transcripts)
    return full_transcript

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

    current_task = state.get("current_task", {})
    file_path = state.get("uploaded_file", "")
    file_extension = Path(file_path).suffix
    file_extensions = { ".mp4", ".mkv", ".webm", ".avi", ".mov", ".wmv", ".m4v", ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma"}
    file = Path(file_path)

    if file.is_file() and file_extension in file_extensions:
        transcript = transcribe_audio(file_path)
        result = process_meeting(transcript)
    else:
        result = "I can not find the appropriate file, please try uploading a file again."

    return {
        "agent_result": [result],
        "completed_steps": ["meeting_assistant"],
        "task_results": {
            current_task["id"] : {
                "intent": current_task["intent"],
                "result": result
            }
        }
    }