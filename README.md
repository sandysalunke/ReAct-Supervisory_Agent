# Copilot like AI Agent

### Building a copilot like AI agent using Langgraph that can perform following operations. 
- Text reasoning/chat
- Image creation
- Audio/meeting assistant - This uses ffmpeg to chunk large files
- OCR - Image to text conversion
- SQL Analytics
- Multi-document RAG
- Excel Analytics
- PowerPoint agent

### TBD:
- Document Comparison Agent
- Search from knowledgeBase - Multiple documents (RAG can be used to search across multiple collections OR vector DB)
- Evaluation & Telemetry Dashboard (Agent accuracy)
- Human in the loop
- 

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
      -----------------------------------------------------------------------------
      |         |          |         |          |           |           |         |
      v         v          v         v          v           v           v         v
+---------+ +--------+ +--------+ +--------+ +--------+ +--------+ +--------+ +--------+
| Chat    | | Image  | | Audio  | | OCR    | | RAG    | | DB     | | Excel  | | PPT    |
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
│   ├── dependency_manager.py
│   ├── file_operations.py
│   ├── rag_helper.py
│   ├── SQLite.py
│   ├── vector_db.py
│   └── createSQLiteMultipleTableDB.py (run thus file to create SQLite DB)
│
├── nodes/
│   ├── classifier.py
│   ├── planner.py
│   └── router.py
│
├── tools/
│   └── excel_tools.py
│
└── main.py
```

### Sample Prompts:
Excel + SQL + Chat
Compare the region-wise revenue from the uploaded Excel report with the system records. Identify discrepancies, explain the likely causes, and summarize the findings.

Excel + SQL + Chart + PowerPoint
Analyze revenue trends from the uploaded Excel report and compare them with system records. Create visualizations for major variances and generate an executive PowerPoint presentation summarizing your findings.

Meeting Audio + Action Items + PowerPoint
Analyze the uploaded meeting recording, extract all action items with owners and due dates, identify key risks discussed, and create a management presentation.

OCR + RAG + Summary
Extract the text from the image, compare it with the uploaded policy document, identify inconsistencies, and provide a summary.

Image + PowerPoint
Generate a professional infographic showing AI adoption trends and then create a 5-slide executive presentation explaining the infographic.