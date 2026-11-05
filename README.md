# Copilot like AI Agent

### Building a copilot like AI agent using Langgraph that can perform following operations. 
- Text reasoning/chat
- Image creation
- Audio/meeting assistant - This uses ffmpeg to chunk large files
- OCR - Image to text conversion
- SQL Analytics
- Multi-document RAG
- Hybrid RAG - Using Azure Search (Keyword + Vector).      
- Excel Analytics
- PowerPoint agent
- Human in the loop (In Chat Agent)
- Cost Evaluation Dashboard
- Performance Monitoring Dashboard
- MCP client-server sample

### TBD:
- Document Comparison Agent
- Webserach Agent

### Azure DevOps read-only integration
- Azure DevOps requests are routed to a dedicated agent that launches Microsoft's local Azure DevOps MCP server over stdio. The hosted MCP endpoint is not used because its Entra dynamic client registration is incompatible with this generic FastMCP client.
- Configure `AZURE_DEVOPS_ORG` with the organization name (defaults to `selection-navigator`).
- Install Node.js 20 or later and Azure CLI, then sign in as the user whose access should be used with `az login`. The local server uses `--authentication azcli`; it does not initiate a separate browser sign-in.
- The MCP server loads only the core, work, work-item, repository, pipeline, wiki, and search domains. The agent additionally filters the tool list and allowed action values to read-only operations.
- The Azure CLI identity must have permission to read the requested organization, projects, and resources. In a hosted or multi-user deployment, this uses the app host's Azure CLI identity; do not treat it as per-Streamlit-user authentication.
- Azure DevOps MCP uses the FastMCP client dependency declared in `requirements.txt`.

### Async workflow
- The Streamlit UI remains synchronous, but the LangGraph workflow uses async graph invocation (`ainvoke`) and async task nodes.
- Streamlit uses a shared background event loop so shared async SDK clients stay on one loop across UI reruns.
- Planner, chat, uploaded-document and hybrid RAG, SQL, Excel, OCR, meeting, image-generation, PowerPoint, classifier, and Cosmos usage/dashboard paths now use async APIs.
- Blocking document/image/audio I/O, vector-store and SQLite setup, local cache access, PowerPoint rendering, and dataframe/code execution run in worker threads; large audio chunking uses an async FFmpeg subprocess.
- The sync-only UI submits workflow coroutines to the shared background event loop and waits for their results.
- Summary/state assembly remains synchronous because it does no network or blocking file work.
      
### Architecture:
```python
                            +--------------------+
                            |   User Interface   |
                            | (Web/Mobile/Teams) |
                            +---------+----------+
                                      |
                                      v
                             +-------------------+
                             |  LangGraph Router |
                             |  Supervisor Agent |
                             +---------+---------+
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

### Steps to setup Azure AI search for Hybrid RAG
- Refer this [video](https://www.youtube.com/watch?v=yu4M7OKjnR4) for Azure AI Search setup 
- Create Blob Storage Account
- Create container
- Upload documents
- Configure Vecror profile, Vector algorithm, Vectorizer
- Create Search Service (Foundry IQ)
- Create Index
- Create Indexer
- Create Skillset
- Add field to the index to store vector data
- Add application insights to log traces for Performance Dashboards
- Add cosmos DB to store usage information for Analytics (Usage/Cost/Performace/Quality) Dashboards

### Sample Prompts:
Excel + SQL + Chat
- Compare the region-wise revenue from the uploaded Excel report with the system records. Identify discrepancies, explain the likely causes, and summarize the findings.

Excel + SQL + Chart + PowerPoint
- Analyze revenue trends from the uploaded Excel report and compare them with system records. Create visualizations for major variances and generate an executive PowerPoint presentation summarizing your findings.

Meeting Audio + Action Items + PowerPoint
- Analyze the uploaded meeting recording, extract all action items with owners and due dates, identify key risks discussed, and create a management presentation.

OCR + RAG + Summary
- Extract the text from the image, compare it with the uploaded policy document, identify inconsistencies, and provide a summary.

Image + PowerPoint
- Generate a professional infographic showing AI adoption trends and then create a 5-slide executive presentation explaining the infographic.