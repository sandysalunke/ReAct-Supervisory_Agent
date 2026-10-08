from utils.tracing import tracer
from graphs.supervisor_graph import WorkflowService
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
import streamlit as st
from pathlib import Path
from helpers import file_operations

is_streamlit_app = True
# is_streamlit_app = False

if is_streamlit_app:
    st.set_page_config(page_title="GenAI Copilot", layout="wide")
    st.title("🧠 GenAI Copilot")

    # Display chat
    def display_chat():
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

        for msg in st.session_state.chat_history:
            
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)

            elif isinstance(msg, AIMessage):
                message = msg.content
                if (
                    Path(message).exists()
                    and Path(message).suffix.lower() in image_extensions
                ):
                    st.write("Here is your generated image:")
                    st.image(message, width=400)
                else:
                    st.chat_message("assistant").write(message)

            elif isinstance(msg, ToolMessage):
                st.chat_message("Tool").write(msg.content)

            elif isinstance(msg, SystemMessage):
                st.chat_message("System").write(msg.content)

    # Initialize chat history in session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

    # Initialize uploaded_file_path in session_state
    # This will help remember that the file is already uploaded
    # User need not upload the file again and can continue referencing it for further queries
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = ""

    # Display chat history
    display_chat()

    uploaded_file = st.file_uploader("Upload a file", key=f"file_uploader_{st.session_state.uploader_key}")
    prompt = st.chat_input("Ask a question (type 'exit' to reset)...")

    if prompt:
        if uploaded_file:
            # save file here
            uploaded_file_path = file_operations.save_uploaded_file(uploaded_file)
            st.session_state.uploaded_file_path = uploaded_file_path
        
        st.chat_message("user").write(prompt)

        spinner_placeholder = st.empty()

        with spinner_placeholder:
            with st.spinner("Thinking..."):
                with tracer.start_as_current_span("User workflow", attributes={"user.workflow": "Begins"}):
                    if (st.session_state.chat_history and st.session_state.chat_history[-1].additional_kwargs.get("workflow_status") == "INTERRUPTED"):
                        thread_id = st.session_state.chat_history[-1].additional_kwargs.get("thread_id")
                        # Resume Agent
                        request = {
                            "thread_id": thread_id,
                            "workflow_status": "INTERRUPTED",
                            "user_input": prompt,
                            "uploaded_file": st.session_state.uploaded_file_path,
                            "chat_history": st.session_state.chat_history
                        }
                        workflow_service = WorkflowService(request)
                        result = workflow_service.resume(prompt, thread_id)
                    else:
                        # Invoke Agent
                        request = {
                            "user_input": prompt,
                            "uploaded_file": st.session_state.uploaded_file_path,
                            "chat_history": st.session_state.chat_history
                        }
                        workflow_service = WorkflowService(request)
                        result = workflow_service.invoke(
                            request
                        )
                
                    # Append result to chat history 
                    st.session_state.chat_history = st.session_state.chat_history + result["data"]["messages"]
    
        # Clear spinner (optional)
        spinner_placeholder.empty()
        
        # ✅ Reset uploader by changing key
        st.session_state.uploader_key += 1

        # Rerun the streamlit app
        st.rerun()
else:
    # ==========================================================
    print("\n ===== MAIN =====")

    user_input = input("\nHow can I help you? ")
    # uploaded_file_path = "./data/uploads/sample_sales_data.xlsx"
    # uploaded_file_path = "./data/uploads/test.ppt"
    uploaded_file_path = "./data/uploads/test2.mp4"
    request = {
        "user_input": user_input,
        "uploaded_file": uploaded_file_path,
        "chat_history": []
    }
    workflow_service = WorkflowService(request)
    
    result = workflow_service.invoke(
        request
    )
    print("\n ===== MAIN RESULT =====")
    print("\n messages: ", result)
