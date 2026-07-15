from agent_state.agent_state import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
import base64
from config.azure_config import client, CHAT_MODEL

# OCR LLM to extract text from image 
def image_to_text(image_path):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Extract text and describe this image"
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

def ocr_agent(state: AgentState) -> AgentState:
    """Use this agent for OCR - Image to text conversion"""
    print("\n===== OCR AGENT =====")
    
    image_path = state["uploaded_file"]

    if image_path:
        result = image_to_text(image_path)
    else:
        result = "I can not find the image, please upload the image to extract the text."

    return {
        "agent_result": [result],
        "completed_steps": ["image_to_text"]
    }