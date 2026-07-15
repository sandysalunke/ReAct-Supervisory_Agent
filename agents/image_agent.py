from agent_state.agent_state import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from config.azure_config import client, IMAGE_MODEL
import base64
from datetime import datetime

# LLM to generate image and return base64 data
def generate_image(state: AgentState):

    prompt = state.get("user_input")
    agent_result = state.get("agent_result")

    result = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024"
    )

    img_base64 = base64.b64decode(result.data[0].b64_json)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./images/{timestamp}.png"
    if img_base64:
        with open(filename, "wb") as f:
            f.write(img_base64)

    return filename

# image_agent - Added as node in supervisor agent graph
def image_agent(state: AgentState) -> AgentState:
    """Use this agent for Image generation"""
    print("\n===== IMAGE GENERATION AGENT =====")

    result = generate_image(state)
    
    return {
        "agent_result": [result],
        "completed_steps": ["image_generation"]
    }