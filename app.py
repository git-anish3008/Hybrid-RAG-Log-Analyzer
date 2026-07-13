import os
import re
import asyncio
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict

# Import Streamlit for the Web UI
import streamlit as st

# LlamaIndex Components
from llama_index.core import Document
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

# Infrastructure Components
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

# LangGraph Components
from langgraph.graph import StateGraph, END

# ==============================================================================
# DATA LAYER: INGESTION & RETRIEVAL AGENTS
# ==============================================================================
class IngestionAgent:
    def _detect_phase(self, text: str) -> str:
        lower = text.lower()
        if any(kw in lower for kw in ["requirement", "applicability"]): return "1. Applicability"
        if any(kw in lower for kw in ["detect", "isdiscovered", "detectionmanager"]): return "2. Detection"
        if any(kw in lower for kw in ["download", "bits", "dojob"]): return "3. Download"
        if any(kw in lower for kw in ["decrypt", "unzip", "extract"]): return "4. Decryption"
        if any(kw in lower for kw in ["execute", "installing", "process"]): return "5. Execution"
        if any(kw in lower for kw in ["report", "status", "statusservice"]): return "6. Reporting"
        return "0. Initialization"
    
    # Process files and extract log chunks
    def process_files(self, file_paths: List[str]) -> List[Document]:
        nodes = []

        #Define a regex pattern to match GUIDs (app IDs)
        app_id_pattern = re.compile(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', re.IGNORECASE)
        
        # Iterate through each file path provided
        for path in file_paths:
            if not os.path.exists(path): continue
            file_name = Path(path).name
            
            if path.lower().endswith(".evtx"):
                raw_text = subprocess.run(["wevtutil", "qe", path, "/lf:true", "/f:text"], capture_output=True, text=True).stdout
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()

            # Split the raw text into chunks based on log entry patterns
            chunks = re.split(r'(?=\<!\[LOG\[|\n\d{4}-\d{2}-\d{2}|\n\[[A-Za-z0-9])', raw_text)
            
            # Iterate through each chunk and create Document objects with metadata
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) > 20: 
                    all_guids = app_id_pattern.findall(chunk)
                    timestamp = "unknown"

                    # Attempt to extract timestamp from the chunk using regex patterns
                    time_match = re.search(r'time="([^"]+)"\s+date="([^"]+)"', chunk)
                    if time_match:
                        timestamp = f"{time_match.group(2)} {time_match.group(1)}"
                    else:
                        # Try to find a timestamp in the format YYYY-MM-DD HH:MM:SS
                        ts_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', chunk)
                        if ts_match: timestamp = ts_match.group(1)
                    
                    # Create a Document object for each chunk with relevant metadata
                    nodes.append(Document(
                        text=chunk,
                        metadata={
                            "file_name": file_name,
                            "phase": self._detect_phase(chunk),
                            "app_ids": [g.lower() for g in all_guids],
                            "timestamp": timestamp,
                        }
                    ))
        return nodes

# ==============================================================================
# DATA LAYER: RETRIEVAL & KNOWLEDGE 
# ==============================================================================

class RetrievalEngine:
    def __init__(self, qdrant_client: QdrantClient): # <-- Add parameter
        self.collection_name = "intune_logs"
        self.qdrant = qdrant_client                  # <-- Use shared client
        self.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.bm25 = None
        self.internal_nodes = []
    
    # Build the Qdrant index and BM25 index from the ingested nodes
    def build_index(self, nodes: List[Document]):

        # Store the nodes internally for later retrieval
        self.internal_nodes = nodes

        # Determine the embedding dimension size and create a new Qdrant collection
        dim_size = len(self.embed_model.get_text_embedding("probe"))
        if self.qdrant.collection_exists(self.collection_name):

            # Delete the existing collection to avoid conflicts
            self.qdrant.delete_collection(self.collection_name)

        # Create a new collection with the appropriate vector configuration    
        self.qdrant.create_collection(self.collection_name, vectors_config=VectorParams(size=dim_size, distance=Distance.COSINE))
        
        # Create points for Qdrant and prepare corpus for BM25
        points, corpus = [], []
        for idx, node in enumerate(nodes):

            # Generate embeddings for each node and create a PointStruct for Qdrant
            vector = self.embed_model.get_text_embedding(node.text)
            
            # Store the text, internal index, and metadata in the payload for later retrieval
            points.append(PointStruct(id=idx, vector=vector, payload={"text": node.text, "internal_index": idx, **node.metadata}))
            
            # Prepare the text for BM25 by cleaning and tokenizing it
            corpus.append(re.sub(r'[^a-z0-9]', ' ', node.text.lower()).split())
        
        # Batch upsert to Qdrant for efficiency
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.qdrant.upsert(self.collection_name, points=points[i:i + batch_size])
        self.bm25 = BM25Okapi(corpus)
    
    # Search function that combines exact match, semantic search, and BM25 ranking
    def search(self, query: str, exact_match_keyword: str = None, top_k: int = 3) -> str:
        top_ids = []
        if exact_match_keyword:
            search_str = exact_match_keyword.lower()
            clean_search_str = re.sub(r'[^a-z0-9]', '', search_str)

            # Search for exact matches in the internal nodes based on text or app_ids but backwards to prioritize recent logs
            for idx in range(len(self.internal_nodes)-1, -1, -1):
                node = self.internal_nodes[idx]
                if clean_search_str in re.sub(r'[^a-z0-9]', '', node.text.lower()) or search_str in node.metadata.get("app_ids", []):
                    top_ids.append(idx)

                    # Accept only the top_k results for exact matches
                    if len(top_ids) >= top_k: break
        
        # If no exact matches found, perform semantic and BM25 search
        if not top_ids:
            query_vec = self.embed_model.get_text_embedding(query)

            # Get the top 15 results from Qdrant semantic search
            sem_res = self.qdrant.query_points(self.collection_name, query=query_vec, limit=15)
            # Extract the IDs of the points returned by Qdrant
            sem_ids = [p.id for p in sem_res.points]

            clean_query = re.sub(r'[^a-z0-9]', ' ', query.lower())
            bm25_scores = self.bm25.get_scores(clean_query.split())

            # Get the top 15 results from BM25 scoring, filtering out any with a score of 0
            bm25_ids = [int(idx) for idx in np.argsort(bm25_scores)[::-1][:15] if bm25_scores[idx] > 0]

            # Fuse the results with RRF (Reciprocal Rank Fusion) to combine semantic and BM25 rankings
            fused = {}
            for r_list in [sem_ids, bm25_ids]:
                for rank, doc_id in enumerate(r_list):
                    # The formula 1 / (k + rank + 1) is used to give higher weight to higher-ranked results, with k=60 to reduce the influence of lower-ranked results
                    fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (60 + rank + 1)
                    
            # Sort the fused results and take the top_k IDs        
            top_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:top_k]
        
        # Expand the top IDs to include adjacent log entries from the same file for better context
        expanded_ids = set()   # Use a set to avoid duplicates
        for doc_id in top_ids:
            expanded_ids.add(doc_id)

            # Check the previous 3 entries for the same file
            for i in range(1, 4):
                if doc_id + i < len(self.internal_nodes):
                    if self.internal_nodes[doc_id + i].metadata.get('file_name') == self.internal_nodes[doc_id].metadata.get('file_name'):
                        expanded_ids.add(doc_id + i)
        
        # Compile the final context blocks to return
        context_blocks = []
        for doc_id in sorted(list(expanded_ids)):
            # Include metadata in the context for better traceability
            m = self.internal_nodes[doc_id].metadata
            # Append the text along with its metadata to the context blocks
            context_blocks.append(f"[File: {m.get('file_name')}] [Phase: {m.get('phase')}] [Time: {m.get('timestamp')}]\n{self.internal_nodes[doc_id].text}")
        return "\n\n".join(context_blocks) if context_blocks else "No matching device logs found."


# ==============================================================================
# DATA LAYER: KNOWLEDGE AGENT
# ==============================================================================

class KnowledgeAgent:
    def __init__(self, qdrant_client: QdrantClient): # <-- Add parameter
        self.collection_name = "intune_runbooks"
        self.qdrant = qdrant_client                  # <-- Use shared client
        self.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.internal_docs = []
    
    # Ingest runbook files, chunk them, and store them in Qdrant for semantic search
    def ingest_runbooks(self, file_paths: List[str]):
        dim_size = len(self.embed_model.get_text_embedding("probe"))
        if self.qdrant.collection_exists(self.collection_name):
            self.qdrant.delete_collection(self.collection_name)
        self.qdrant.create_collection(self.collection_name, vectors_config=VectorParams(size=dim_size, distance=Distance.COSINE))
        
        # Prepare points for Qdrant and store the text chunks internally
        points = []
        doc_id = 0
        for path in file_paths:
            if not os.path.exists(path): continue
            with open(path, "r", encoding="utf-8") as f:
                paragraphs = f.read().split("\n\n")
                for para in paragraphs:
                    if len(para.strip()) > 10:
                        self.internal_docs.append(para.strip())
                        vector = self.embed_model.get_text_embedding(para.strip())
                        points.append(PointStruct(id=doc_id, vector=vector, payload={"text": para.strip()}))
                        doc_id += 1
        if points: self.qdrant.upsert(self.collection_name, points=points)
    
    # Search the ingested runbooks for relevant guidance based on the user query
    def search_knowledge(self, query: str, top_k: int = 2) -> str:
        query_vec = self.embed_model.get_text_embedding(query)
        sem_res = self.qdrant.query_points(self.collection_name, query=query_vec, limit=top_k)
        docs = [p.payload["text"] for p in sem_res.points]
        return "\n\n".join(docs) if docs else "No specific runbook guidance found."

# ==============================================================================
# ORCHESTRATION LAYER: GRAPH STATE MACHINE
# ==============================================================================
class DiagnosticState(TypedDict):
    user_query: str
    target_app_id: Optional[str]
    log_evidence: str
    runbook_rules: str
    chat_history: List[Dict[str, str]]
    final_response: str

class LangGraphOrchestrator:
    def __init__(self, log_paths: List[str], runbook_paths: List[str]):
        # 0. Initialize the Qdrant client for local storage
        # Note: The QdrantClient is initialized once and shared across agents to avoid multiple connections and potential conflicts.
        # 1. Create the MASTER connection here
        master_qdrant_client = QdrantClient(path="./local_intune_db")
        
        self.ingestion = IngestionAgent()
        
        # 2. Pass the master connection down to the agents
        self.retrieval = RetrievalEngine(qdrant_client=master_qdrant_client)
        self.knowledge = KnowledgeAgent(qdrant_client=master_qdrant_client)

        nodes = self.ingestion.process_files(log_paths)
        if nodes: self.retrieval.build_index(nodes)
        self.knowledge.ingest_runbooks(runbook_paths)
        
        # 3. Initialize the Ollama LLM for generating responses
        self.llm = Ollama(model="qwen2.5:3b", 
                          request_timeout=300.0, 
                          temperature=0.0, 
                          context_window=4096, 
                          additional_kwargs={"num_thread": 8}
                          )
        self.session_chat_history = []

        # 4. Define the LangGraph workflow for orchestrating the diagnostic process
        workflow = StateGraph(DiagnosticState)
        workflow.add_node("triage_query", self._node_triage_query)
        workflow.add_node("harvest_device_logs", self._node_harvest_device_logs)
        workflow.add_node("fetch_runbook_rules", self._node_fetch_runbook_rules)
        workflow.add_node("synthesize_rca_report", self._node_synthesize_rca_report)
        
        # Define the edges between nodes to create a linear workflow
        workflow.set_entry_point("triage_query")
        workflow.add_edge("triage_query", "harvest_device_logs")
        workflow.add_edge("harvest_device_logs", "fetch_runbook_rules")
        workflow.add_edge("fetch_runbook_rules", "synthesize_rca_report")
        workflow.add_edge("synthesize_rca_report", END)
        
        self.compiled_graph = workflow.compile()
    
    # As per the user query, finds the target app ID and returns it for the next node to use in log retrieval
    def _node_triage_query(self, state: DiagnosticState) -> Dict:
        query = state["user_query"]
        app_id_pattern = re.compile(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', re.IGNORECASE)
        match = app_id_pattern.search(query)
        return {"target_app_id": match.group(1).lower() if match else None}
    
    # This one does the heavy lifting of searching the logs based on the user query and optional target app ID, returning relevant log evidence for the next node to use
    def _node_harvest_device_logs(self, state: DiagnosticState) -> Dict:
        target_id = state.get("target_app_id")
        query = state["user_query"]
        if target_id:
            evidence = self.retrieval.search(query=query, exact_match_keyword=target_id, top_k=2)
        else:
            evidence = self.retrieval.search(query=query, top_k=5)
        return {"log_evidence": evidence}

    # This one fetches relevant runbook rules based on the user query
    def _node_fetch_runbook_rules(self, state: DiagnosticState) -> Dict:
        return {"runbook_rules": self.knowledge.search_knowledge(state["user_query"])}

    # This one synthesizes the final RCA report based on the collected evidence and runbook rules
    def _node_synthesize_rca_report(self, state: DiagnosticState) -> Dict:
        history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in state["chat_history"][-4:]])
        
        prompt = f"""You are an elite Intune Support Engineer.

=== RECENT CONVERSATION HISTORY ===
{history_text if history_text else "No previous conversation history."}

=== INTUNE RUNBOOK GUIDELINES ===
{state["runbook_rules"]}

=== USER DEVICE LOGS ===
{state["log_evidence"]}

User Query: {state["user_query"]}

Response:"""
        
        response_accumulator = ""
        # Removed print streaming since Streamlit handles the UI display now
        for token in self.llm.stream_complete(prompt):
            response_accumulator += token.delta
            
        return {"final_response": response_accumulator.strip()}

    def execute_session_turn(self, query: str) -> str:
        initial_state: DiagnosticState = {
            "user_query": query, "target_app_id": None, "log_evidence": "", "runbook_rules": "",
            "chat_history": self.session_chat_history, "final_response": ""
        }
        
        final_computed_state = self.compiled_graph.invoke(initial_state)
        response_text = final_computed_state["final_response"]
        
        self.session_chat_history.append({"role": "User", "content": query})
        self.session_chat_history.append({"role": "Agent", "content": response_text})
        
        return response_text # Added return statement for the UI

# ==============================================================================
# WEB UI LAYER (Streamlit)
# ==============================================================================
def render_ui():
    st.set_page_config(page_title="Intune AI Diagnostics", layout="wide")
    st.title("🛡️ Intune Local RAG Agent")
    st.markdown("Analyze Win32 App & Autopilot failures locally with Qwen 2.5.")

    # 1. Initialize memory variables for the web app
    if "app_engine" not in st.session_state:
        st.session_state.app_engine = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 2. Sidebar for File Uploads
    with st.sidebar:
        st.header("📂 Upload Telemetry")
        uploaded_logs = st.file_uploader("Upload Intune Logs (.log)", accept_multiple_files=True)
        uploaded_runbooks = st.file_uploader("Upload Runbook Manuals (.txt)", accept_multiple_files=True)

        if st.button("Initialize Agent", type="primary"):
            if not uploaded_logs and not uploaded_runbooks:
                st.warning("Please upload at least one log or runbook.")
            else:
                with st.spinner("Embedding data into local Qdrant database..."):
                    # Create temporary directories to save the uploaded web files
                    os.makedirs("temp_data/logs", exist_ok=True)
                    os.makedirs("temp_data/runbooks", exist_ok=True)

                    log_paths = []
                    for file in uploaded_logs:
                        path = os.path.join("temp_data/logs", file.name)
                        with open(path, "wb") as f: f.write(file.read())
                        log_paths.append(path)

                    runbook_paths = []
                    for file in uploaded_runbooks:
                        path = os.path.join("temp_data/runbooks", file.name)
                        with open(path, "wb") as f: f.write(file.read())
                        runbook_paths.append(path)

                    # Boot up your Orchestrator with the saved files
                    st.session_state.app_engine = LangGraphOrchestrator(log_paths, runbook_paths)
                    st.success("Indexing Complete! Agent is ready.")

    # 3. Main Chat Interface
    # Render previous messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Box
    if user_query := st.chat_input("Enter App ID or ask a diagnostic question..."):
        if st.session_state.app_engine is None:
            st.error("⚠️ Please upload files and click 'Initialize Agent' first.")
            return

        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Show AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing telemetry..."):
                # Call the LangGraph orchestrator
                response = st.session_state.app_engine.execute_session_turn(user_query)
                st.markdown(response)
                
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    render_ui()
