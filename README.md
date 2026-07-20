# Copilot like AI Agent

### Building a copilot like AI agent using Langgraph that can perform following operations. 
- Text chat
- Image creation
- Audio meeting assistant 
- image to text conversion
- Database search
- Text reasoning based on uploaded PDF or any text file
- Excel file analysis

### Architecture:
                    +------------------+
                    |   User Interface |
                    | (Web/Mobile/Teams)|
                    +---------+--------+
                              |
                              v
                    +------------------+
                    |  LangGraph Router|
                    |  Supervisor Agent|
                    +---------+--------+
                              |
      -------------------------------------------------------------------
      |         |          |         |          |           |           |
      v         v          v         v          v           v           v

+---------+ +--------+ +--------+ +--------+ +--------+ +--------+ +--------+
| Chat    | | Image  | | Audio  | | OCR    | | RAG    | | DB     | | Excel  |
| Agent   | | Agent  | | Agent  | | Agent  | | Agent  | | Agent  | | Agent  |
+---------+ +--------+ +--------+ +--------+ +--------+ +--------+ +--------+

      |         |          |         |        |         |
      --------------------------------------------------
                              |
                              v
                    +------------------+
                    |  Response Builder |
                    +------------------+

### Folder Structure:

app/
│
├── agents/
│   ├── chat_agent.py
│   ├── image_agent.py
│   ├── pdf_agent.py
│   ├── sql_agent.py
│   ├── ocr_agent.py
│   ├── meeting_agent.py
|   └── excel_agent.py
│
├── graphs/
│   └── supervisor_graph.py
│
├── tools/
│   ├── search_tool.py
│   ├── sql_tool.py
│   ├── vector_tool.py
│   └── ocr_tool.py
│
├── memory/
│   ├── redis_memory.py
│   └── long_term_memory.py
│
├── api/
│   └── fastapi_server.py
│
├── ui/
│   └── react_app
│
└── storage/
