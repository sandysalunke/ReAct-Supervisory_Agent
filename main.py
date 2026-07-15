from graphs.supervisor_graph import invoke_graph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
import streamlit as st

# # Display chat
# def display_chat():
#     for msg in st.session_state.chat_history:
        
#         if isinstance(msg, HumanMessage):
#             # print("Human:", msg.content)
#             st.chat_message("user").write(msg.content)

#         elif isinstance(msg, AIMessage):
#             # print("AI:", msg.content)
#             st.chat_message("assistant").write(msg.content)

#         elif isinstance(msg, ToolMessage):
#             # print("Tool:", msg.content)
#             st.chat_message("Tool").write(msg.content)

#         elif isinstance(msg, SystemMessage):
#             # print("System:", msg.content)
#             st.chat_message("System").write(msg.content)

# # Initialize chat history in session_state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Display chat history
# display_chat()

# prompt = st.chat_input("Ask a question (type 'exit' to reset)...")

# if prompt:
#     st.chat_message("user").write(prompt)

#     spinner_placeholder = st.empty()

#     with spinner_placeholder:
#         with st.spinner("Thinking..."):
#             # Invoke Agent
#             result = invoke_graph(prompt)
#             print("\n messages: ", result)
            
#             # Append result to chat history 
#             st.session_state.chat_history = st.session_state.chat_history + result["messages"]
    
#     # Clear spinner (optional)
#     spinner_placeholder.empty()
    
#     # Rerun the streamlit app
#     st.rerun()
    
# ==========================================================
print("\n ===== MAIN =====")

user_input = input("\nHow can I help you? ")

result = invoke_graph(user_input)

print("\n ===== MAIN RESULT =====")
print("\n messages: ", result)
