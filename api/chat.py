import asyncio
import logging
import uuid
from dataclasses import dataclass, field

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from api.files import image_url
from utils.image_display import split_local_image_paths

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@dataclass
class ChatSession:
    """Per-browser-session state (what Streamlit kept in st.session_state)."""

    chat_history: list[BaseMessage] = field(default_factory=list)
    # Remembered so the user can keep referencing a file without re-uploading it.
    uploaded_file_path: str = ""
    uploaded_file_name: str = ""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# In-memory, like the workflow checkpointer: sessions are lost on restart.
_sessions: dict[str, ChatSession] = {}


def _is_interrupted(message: BaseMessage) -> bool:
    return message.additional_kwargs.get("workflow_status") == "INTERRUPTED"


def _content_text(content) -> str:
    return content if isinstance(content, str) else str(content)


def serialize_message(message: BaseMessage) -> dict:
    text = _content_text(message.content)

    if isinstance(message, AIMessage):
        segments = []
        for segment_text, image_path in split_local_image_paths(text):
            if image_path is None:
                if segment_text:
                    segments.append({"type": "text", "text": segment_text})
                continue
            url = image_url(image_path)
            if url is None:
                # Only images under ./data are served; show other paths as text.
                segments.append({"type": "text", "text": str(image_path)})
            else:
                segments.append({"type": "image", "url": url})
        return {"role": "assistant", "segments": segments}

    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, ToolMessage):
        role = "tool"
    elif isinstance(message, SystemMessage):
        role = "system"
    else:
        role = message.type

    return {
        "role": role,
        "segments": [{"type": "text", "text": text}],
        "interrupted": _is_interrupted(message),
    }


def _session_payload(session_id: str, session: ChatSession) -> dict:
    return {
        "session_id": session_id,
        "messages": [serialize_message(message) for message in session.chat_history],
        "uploaded_file_name": session.uploaded_file_name,
    }


def _save_upload(filename: str, file_object) -> str:
    # Imported here, in a worker thread: loading the document parsers takes
    # several seconds and must not block the event loop.
    from helpers import file_operations

    return file_operations.save_uploaded_file(filename, file_object)


async def _run_workflow(session: ChatSession, prompt: str) -> dict:
    from graphs.supervisor_graph import WorkflowService
    from utils.tracing import tracer

    chat_history = session.chat_history
    with tracer.start_as_current_span(
        "User workflow", attributes={"user.workflow": "Begins"}
    ):
        if chat_history and _is_interrupted(chat_history[-1]):
            # Resume the interrupted workflow with the user's answer.
            interrupted_kwargs = chat_history[-1].additional_kwargs
            thread_id = interrupted_kwargs.get("thread_id")
            request = {
                "thread_id": thread_id,
                "workflow_status": "INTERRUPTED",
                "interrupt_ids": interrupted_kwargs.get("interrupt_ids", []),
                "user_input": prompt,
                "uploaded_file": session.uploaded_file_path,
                "chat_history": chat_history,
            }
            return await WorkflowService(request).resume(prompt, thread_id)

        request = {
            "user_input": prompt,
            "uploaded_file": session.uploaded_file_path,
            "chat_history": chat_history,
        }
        return await WorkflowService(request).invoke(request)


@router.post("/chat")
async def send_message(
    message: str = Form(...),
    session_id: str | None = Form(None),
    file: UploadFile | None = File(None),
):
    prompt = message.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="The message must not be empty.")

    if session_id is None or session_id not in _sessions:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = ChatSession()
    session = _sessions[session_id]

    # One workflow at a time per session, so history stays in order.
    async with session.lock:
        if file is not None and file.filename:
            session.uploaded_file_path = await asyncio.to_thread(
                _save_upload, file.filename, file.file
            )
            session.uploaded_file_name = file.filename

        try:
            result = await _run_workflow(session, prompt)
        except Exception as error:
            logger.exception("Workflow failed for session %s", session_id)
            raise HTTPException(
                status_code=500, detail=f"The workflow failed: {error}"
            ) from error

        new_messages = result["data"]["messages"]
        session.chat_history = session.chat_history + new_messages

    return {
        "session_id": session_id,
        "thread_id": result["thread_id"],
        "workflow_status": result["workflow_status"],
        "messages": [serialize_message(new_message) for new_message in new_messages],
        "uploaded_file_name": session.uploaded_file_name,
    }


@router.get("/chat/{session_id}")
async def get_session(session_id: str):
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return _session_payload(session_id, session)


@router.delete("/chat/{session_id}", status_code=204)
async def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return Response(status_code=204)
