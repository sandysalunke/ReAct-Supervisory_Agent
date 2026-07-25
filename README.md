# Copilot like AI Agent

### Building a copilot like AI agent using Langgraph that can perform following operations. 
- Text reasoning/chat
- Image creation
- Audio/meeting assistant
- OCR - Image to text conversion
- SQL Analytics
- Multi-document RAG
- Excel Analytics
- PowerPoint agent

### TBD:
- Document Comparison Agent
- Search from knowledgeBase - Multiple documents (RAG can be used to search across multiple collections OR vector DB)
- Evaluation & Telemetry Dashboard (Agent accuracy)

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
      ----------------------------------------------------------------------------
      |         |          |         |          |           |           |         |
      v         v          v         v          v           v           v         v
+---------+ +--------+ +--------+ +--------+ +--------+ +--------+ +--------+ +--------+
| Chat    | | Image  | | Audio  | | OCR    | | RAG    | | DB     | | Excel  | | PPT  |
| Agent   | | Agent  | | Agent  | | Agent  | | Agent  | | Agent  | | Agent  | | Agent  |
+---------+ +--------+ +--------+ +--------+ +--------+ +--------+ +--------+ +--------+
      |         |           |          |          |          |          |         |
      -----------------------------------------------------------------------------
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