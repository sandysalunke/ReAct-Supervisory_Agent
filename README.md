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
- The Azure CLI identity must have permission to read the requested organization, projects, and resources. In a hosted or multi-user deployment, this uses the app host's Azure CLI identity; do not treat it as per-user authentication.
- Azure DevOps MCP uses the FastMCP client dependency declared in `requirements.txt`.

### Running the app
The UI is an Angular app in `frontend/`; the backend is a FastAPI app in `api/` that wraps the LangGraph workflow.

Backend (from the project root, serves on http://127.0.0.1:8001):
```
pip install -r requirements.txt
python main.py
```

Frontend in development (serves on http://localhost:4200 and proxies `/api` to the backend):
```
cd frontend
npm install
npm start
```

Single-process mode: run `npm run build` in `frontend/` once, then `python main.py` also serves the built UI at http://127.0.0.1:8001.

- Set `API_HOST` / `API_PORT` to change where the backend listens (the default port is 8001 because the sample Math MCP server uses 8000). If you change the port, update `frontend/proxy.conf.json` too.
- Run the API tests with `python scripts	est_api.py` (the workflow and Cosmos DB are mocked).

### API
| Endpoint | Purpose |
| --- | --- |
| `POST /api/chat` | Multipart form: `message`, optional `session_id`, optional `file`. Runs (or resumes, after a human-in-the-loop interrupt) the workflow and returns the new messages. |
| `GET /api/chat/{session_id}` / `DELETE /api/chat/{session_id}` | Read or reset a chat session. |
| `GET /api/files/{path}` | Serves generated images from `data/`. |
| `GET /api/usage`, `GET /api/performance` | Aggregated Cosmos DB data for the Usage and Performance dashboards. |
| `GET /api/evaluation/reports`, `GET /api/evaluation/reports/{name}`, `POST /api/evaluation/reports` | List, view, or upload-and-view planner evaluation reports. |
| `POST /api/mcp/add-numbers` | Runs the sample MCP client. |

- Chat sessions (history and the last uploaded file) are kept in server memory, like the workflow checkpointer, and are lost when the backend restarts. The browser keeps its session ID per tab.
- Interactive API docs are available at http://127.0.0.1:8001/docs.

### Async workflow
- FastAPI runs the LangGraph workflow directly on its event loop using async graph invocation (`ainvoke`) and async task nodes, so shared async SDK clients stay on one loop.
- Planner, chat, uploaded-document and hybrid RAG, SQL, Excel, OCR, meeting, image-generation, PowerPoint, classifier, and Cosmos usage/dashboard paths use async APIs.
- Blocking document/image/audio I/O, vector-store and SQLite setup, local cache access, PowerPoint rendering, and dataframe/code execution run in worker threads; large audio chunking uses an async FFmpeg subprocess.
- Summary/state assembly remains synchronous because it does no network or blocking file work.

### Planner execution-plan limits
- Planner output is validated before it is saved or used to build a workflow graph.
- Plans must contain 1-8 tasks, use supported intents and unique task IDs, and form an acyclic dependency graph with valid references.
- Each task prompt is limited to 4,000 characters; invalid plans fail explicitly instead of being truncated or partially executed.

### Planner routing evaluation
- `evals/planner_routing.jsonl` contains 60 golden prompts, including single-intent, attached-file, and multi-agent cases.
- Run `python scripts\evaluate_planner.py` to evaluate the configured chat model. The report includes exact intent-set accuracy, micro precision/recall/F1, per-intent scores, and mismatches.
- Use `python scripts\evaluate_planner.py --limit 5` for a small smoke run. Each evaluated prompt invokes the configured model; the evaluator does not write usage records to Cosmos DB or save workflow plans.
- Progress is printed for each prompt; the final report is printed as JSON. Add `--output evals\planner-results.json` to save a copy.
- Add labeled examples to the JSONL dataset when routing behavior changes; `expected_intents` should list every intent the plan is expected to contain.
- Open the **Evaluation** page in the UI to review saved `evals\planner-results*.json` reports, per-intent scores, and mismatches. You can also upload a report JSON directly.
      
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