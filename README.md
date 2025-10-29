# EchoPilot
Customer Success Copilot - an angentic AI RAG that does intent classification, autonomous decision making and tool calling 

## Demo Video
[![watch the demo video](knowledgeBase/images/video-demo.png)](https://www.loom.com/embed/52d3f6b3a6ac48318bb19dfad2ddfcff?sid=7f6c8b12-b8cd-4287-a79f-c3ac76a6ac77)

# setup commands:
1. run FastAPI server for SDK: python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
2. run streamlit app: streamlit run app.py

## Tech-Stack
python
langchain and langGraph
chromaDB
streamlit for UI

## Layers:
Data ingestion - data_ingestion.py
main graph and nodes decleration - echo.py
ticket creation - jira_tool.py
loading and saving chat history - chat_mgmt.py
embedding model and vector store service - services.py
multi-modal input processing - multiModalInputService.py

It is an AI agentic workflow for has RAG and tool calling. It has components for data ingestion @data_ingestion.py  , jira tool @jira_tool.py  , @services.py for vector database and embedding , @rag_scoring.py  for rag scoring and more services in @services/ .  @echo.py contains the main exitsing flow. 

## Expectations:
1. user can ingest files to vector db that persists data locally
2. RAG framework with tool calling like data retrieval and ticket creation
3. internal analysis flow: intent analysis -> RAG -> relevant tool calling
4. chat memory is stored
5. user can add files like image and pdf in the user query
