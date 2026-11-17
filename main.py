import asyncio
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import streamlit as st
from utils.async_runner import get_async_runner
from utils.image_display import split_local_image_paths

is_streamlit_app = True
# is_streamlit_app = False

if is_streamlit_app:
    st.set_page_config(page_title="GenAI Copilot", layout="wide")
    st.title("🧠 GenAI Copilot")

    # Display chat
    def display_chat():
        for msg in st.session_state.chat_history:
            
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)

            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    for text, image_path in split_local_image_paths(msg.content):
                        if image_path is not None:
                            st.image(str(image_path), width=400)
                        elif text:
                            st.markdown(text)

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
            from helpers import file_operations

            uploaded_file_path = get_async_runner().run(
                asyncio.to_thread(file_operations.save_uploaded_file, uploaded_file)
            )
            st.session_state.uploaded_file_path = uploaded_file_path
        
        st.chat_message("user").write(prompt)

        spinner_placeholder = st.empty()

        with spinner_placeholder:
            with st.spinner("Thinking..."):
                from graphs.supervisor_graph import WorkflowService
                from utils.tracing import tracer

                with tracer.start_as_current_span("User workflow", attributes={"user.workflow": "Begins"}):
                    if (st.session_state.chat_history and st.session_state.chat_history[-1].additional_kwargs.get("workflow_status") == "INTERRUPTED"):
                        thread_id = st.session_state.chat_history[-1].additional_kwargs.get("thread_id")
                        # Resume Agent
                        request = {
                            "thread_id": thread_id,
                            "workflow_status": "INTERRUPTED",
                            "interrupt_ids": st.session_state.chat_history[-1].additional_kwargs.get("interrupt_ids", []),
                            "user_input": prompt,
                            "uploaded_file": st.session_state.uploaded_file_path,
                            "chat_history": st.session_state.chat_history
                        }
                        workflow_service = WorkflowService(request)
                        result = get_async_runner().run(
                            workflow_service.resume(prompt, thread_id)
                        )
                    else:
                        # Invoke Agent
                        request = {
                            "user_input": prompt,
                            "uploaded_file": st.session_state.uploaded_file_path,
                            "chat_history": st.session_state.chat_history
                        }
                        workflow_service = WorkflowService(request)
                        result = get_async_runner().run(
                            workflow_service.invoke(request)
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
    from graphs.supervisor_graph import WorkflowService

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
    
    result = asyncio.run(
        workflow_service.invoke(request)
    )
    print("\n ===== MAIN RESULT =====")
    print("\n messages: ", result)
