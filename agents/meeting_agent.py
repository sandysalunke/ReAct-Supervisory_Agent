import asyncio
from agent_state.agent_state import AgentState
from pathlib import Path
from config.azure_config import async_client, CHAT_MODEL, WHISPER_MODEL
import glob
import imageio_ffmpeg
import os
from uuid import uuid4


async def chunk_audio_file(file_path):
    file_size = await asyncio.to_thread(os.path.getsize, file_path)
    MAX_SIZE = 25 * 1024 * 1024
    
    if file_size <= MAX_SIZE:
        return [file_path]

    output_dir = Path("./data/uploads/audio")
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    output_pattern = output_dir / f"{uuid4().hex}_chunk_%03d.mp3"
    ffmpeg = await asyncio.to_thread(imageio_ffmpeg.get_ffmpeg_exe)
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-i", file_path,
        "-vn",
        "-f", "segment",
        "-segment_time", "300",
        "-c:a", "libmp3lame",
        str(output_pattern),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(
            f"FFmpeg failed to chunk meeting audio: {stderr.decode(errors='replace')}"
        )
    return await asyncio.to_thread(
        lambda: sorted(glob.glob(str(output_pattern).replace("%03d", "*")))
    )

# Whisper LLM to read transcript
async def transcribe_audio(file_path):
    chunks = await chunk_audio_file(file_path)
    transcripts = []

    try:
        for chunk_file in chunks:
            chunk_bytes = await asyncio.to_thread(Path(chunk_file).read_bytes)

            response = await async_client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=(Path(chunk_file).name, chunk_bytes)
            )

            transcripts.append(response.text)
    finally:
        if file_path not in chunks:
            await asyncio.gather(
                *(asyncio.to_thread(Path(chunk).unlink, missing_ok=True) for chunk in chunks)
            )

    full_transcript = "\n".join(transcripts)
    return full_transcript

# LLM to process transcript and return structured response 
async def process_meeting(transcript):
    response = await async_client.chat.completions.create(
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
async def meeting_agent(state: AgentState) -> AgentState:
    """Use this agent for meeting assistance"""
    print("\n===== MEETING AGENT =====")

    current_task = state.get("current_task", {})
    file_path = state.get("uploaded_file", "")
    file_extension = Path(file_path).suffix
    file_extensions = { ".mp4", ".mkv", ".webm", ".avi", ".mov", ".wmv", ".m4v", ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma"}
    file = Path(file_path)

    if file.is_file() and file_extension in file_extensions:
        transcript = await transcribe_audio(file_path)
        result = await process_meeting(transcript)
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