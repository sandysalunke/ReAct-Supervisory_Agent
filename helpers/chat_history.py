from langchain_core.messages import BaseMessage, HumanMessage


def recent_chat_history(
    chat_history: list[BaseMessage], max_turns: int = 10
) -> list[BaseMessage]:
    human_message_indices = [
        index
        for index, message in enumerate(chat_history)
        if isinstance(message, HumanMessage)
    ]
    if len(human_message_indices) <= max_turns:
        return chat_history
    return chat_history[human_message_indices[-max_turns]:]
