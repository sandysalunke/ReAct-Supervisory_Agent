# Copilot like AI Agent

### Building a copilot like AI agent using Langgraph that can perform following operations. 
- Text chat
- Image creation
- Audio meeting assistant 
- Image to text conversion
- Database search
- Text reasoning based on uploaded PDF or any text file
- Excel file analysis

### Architecture:
```python
                            +--------------------+
                            |   User Interface   |
                            | (Web/Mobile/Teams) |
                            +---------+----------+
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
      |         |           |          |          |          |          |
      -------------------------------------------------------------------
                                       |
                                       v
                              +-------------------+
                              |  Response Builder |
                              +-------------------+
```

### Folder Structure:
```python
app/
│
├── agent_state/
│   ├── agent_state.py
│
├── agents/
│   ├── chat_agent.py
│   ├── image_agent.py
│   ├── rag_agent.py
│   ├── sql_agent.py
│   ├── ocr_agent.py
│   ├── meeting_agent.py
|   └── excel_agent.py
|   └── summary_agent.py
│
├── cofig/
│   └── azure_config.py
│
├── data/
│   └── /images
│   └── /uplaods
│   └── /vectorDB
│
├── graphs/
│   └── supervisor_graph.py
│
├── helpers/
│   ├── chunk_helper.py
│   ├── embeddings.py
│   ├── file_operations.py
│   └── rag_helper.py
│   └── SQLite.py
│   └── vector_db.py
│   └── createSQLiteMultipleTableDB.py (run thus file to create SQLite DB)
│
├── nodes/
│   ├── classifier.py
│   ├── planner.py
│   ├── router.py
│
├── tools/
│   └── excel_tools.py
│
└── main.py
```