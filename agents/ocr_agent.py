from agent_state.agent_state import AgentState
import asyncio
import base64
from config.azure_config import async_client, CHAT_MODEL
from pathlib import Path

# OCR LLM to extract text from image 
async def image_to_text(image_path, task_prompt):
    image_base64 = await asyncio.to_thread(_encode_image, image_path)

    response = await async_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": task_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content


def _encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ocr_agent - Added as node in supervisor agent graph
async def ocr_agent(state: AgentState) -> AgentState:
    """Use this agent for OCR - Image to text conversion"""
    print("\n===== OCR AGENT =====")

    user_input = state.get("user_input")
    task_prompt = state.get("task_prompt", user_input)
    current_task = state.get("current_task", {})
    file_path = state.get("uploaded_file", "")
    file_extension = Path(file_path).suffix
    file_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    file = Path(file_path)

    if file.is_file() and file_extension in file_extensions:
        result = await image_to_text(file_path, task_prompt)
    else:
        result = "I can not find the image, please upload the image to extract the text."

    return {
        "agent_result": [result],
        "completed_steps": ["image_to_text"],
        "task_results": {
            current_task["id"] : {
                "intent": current_task["intent"],
                "result": result
            }
        }
    }