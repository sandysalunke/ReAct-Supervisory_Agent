import json
from agent_state.agent_state import AgentState
from config.azure_config import llm, client, IMAGE_MODEL
from utils.tracing import tracer
from opentelemetry import trace
import base64
from datetime import datetime
from helpers.dependency_manager import get_dependency_results

# Generate the prompt that considers dependency_results and chat_history
def generate_prompt(task_prompt, dependency_results, chat_history):
    prompt = f"""
        You are an Image Generation Agent.

        Current Task:
        {task_prompt}

        Dependency Results:
        {json.dumps(dependency_results, indent=2)}

        Chat Context:
        {chat_history}

        Your responsibility:
        - Create an optimized image generation prompt.
        - Use dependency results when relevant.
        - Use chat context only if necessary to resolve references.
        - Infer visual details from task outputs when available.

        STRICT RULES:
        - Do not answer the user directly.
        - Do not explain your reasoning.
        - Do not summarize data.
        - Return only the final image generation prompt.
        - If dependency results contain analytical findings, convert them into a visual representation.
        - If the task requests an infographic, include the key findings from dependency results.
        - If the task requests a chart, indicate that chart generation should be handled by the analytics agent instead.
        - Create professional, detailed, visually rich prompts.
        """
    with tracer.start_as_current_span("agent.image.generate_prompt", attributes={"llm_used": True}):
        response = llm.invoke(prompt)

        usage = response.response_metadata["token_usage"]
        trace.get_current_span().set_attribute("llm.model", response.response_metadata["model_name"])
        trace.get_current_span().set_attribute("llm.prompt_tokens", usage["prompt_tokens"])
        trace.get_current_span().set_attribute("llm.completion_tokens", usage["completion_tokens"])
        trace.get_current_span().set_attribute("llm.total_tokens", usage["total_tokens"])
        trace.get_current_span().set_attribute("llm.finish_reason", response.response_metadata["finish_reason"])
        trace.get_current_span().set_attribute("prompt_length", len(response.content))

    return response.content;

# LLM to generate image and return base64 data
def generate_image(task_prompt):
    with tracer.start_as_current_span("agent.image.generate_image", attributes={
        "llm_used": True, 
        "llm_model": IMAGE_MODEL,
        "prompt_length": len(task_prompt)
        }):
        
        result = client.images.generate(
            model=IMAGE_MODEL,
            prompt=task_prompt,
            size="1024x1024"
        )

        usage = result.usage
        trace.get_current_span().set_attribute("image.input_tokens", usage.input_tokens)
        trace.get_current_span().set_attribute("image.output_tokens", usage.output_tokens)
        trace.get_current_span().set_attribute("image.total_tokens", usage.total_tokens)

        img_base64 = base64.b64decode(result.data[0].b64_json)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./data/images/{timestamp}.png"
        if img_base64:
            with open(filename, "wb") as f:
                f.write(img_base64)
            trace.get_current_span().set_attribute("image.generated", True)
            trace.get_current_span().set_attribute("image.filename", filename)
            
        else:
            trace.get_current_span().set_attribute("image.generated", False)
            return "Sorry. I can not generate an image at the moment, please try again later"

        return filename

# image_agent - Added as node in supervisor agent graph
def image_agent(state: AgentState) -> AgentState:
    """Use this agent for Image generation"""
    print("\n===== IMAGE GENERATION AGENT =====")
    
    user_input = state.get("user_input")
    task_prompt = state.get("task_prompt", user_input)
    current_task = state.get("current_task", {})
    task_results = state.get("task_results",{})
    chat_history = state.get("chat_history",[])

    with tracer.start_as_current_span("agent.image", attributes={
            "thread_id": state.get("thread_id"),
            "task_id": current_task.get("id"),
            "intent": current_task.get("intent")
        }):
        dependency_results = get_dependency_results(current_task, task_results)
        task_prompt = generate_prompt(task_prompt, dependency_results, chat_history)
        result = generate_image(task_prompt)
        
        return {
            "agent_result": [result],
            "completed_steps": ["image_generation"],
            "task_results": {
                current_task["id"] : {
                    "intent": current_task["intent"],
                    "result": result
                }
            }
        }