import os
import re
import asyncio
import numpy as np
from typing import List, Dict, Any

# LlamaIndex Core Components
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext

from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Qdrant & BM25 Backend
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi

# ==============================================================================
# 1. INTUNE WORKLOAD LOG PARSER & CHUNKER
# ==============================================================================
def parse_intune_workload_logs(file_path: str) -> List[Document]:
    """
    Parses Intune-style or standard app deployment logs.
    Captures timestamps, App IDs (if present), and constructs logical chunks.
    """
    documents = []
    
    # Common regex patterns to hunt for Win32 application identifiers in Intune
    app_id_pattern = re.compile(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', re.IGNORECASE)
    
    # Basic timestamp extraction fallback if it's not a standard CCM log format
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\,\d{3}|\d{2}:\d{2}:\d{2})')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find log file at: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    print(f"[+] Loaded {len(lines)} raw log lines.")

    # Process logs dynamically line by line to build structured documents
    for idx, line in enumerate(lines):
        line_str = line.strip()
        if not line_str:
            continue
            
        # Extract metadata fields safely
        app_id_match = app_id_pattern.search(line_str)
        ts_match = timestamp_pattern.search(line_str)
        
        app_id = app_id_match.group(1) if app_id_match else "unknown_app_id"
        timestamp = ts_match.group(1) if ts_match else f"line_{idx}"
        
        # Determine gravity of log line
        level = "INFO"
        if "error" in line_str.lower() or "failed" in line_str.lower() or "exit code" in line_str.lower():
            level = "ERROR"
        elif "warning" in line_str.lower():
            level = "WARN"

        # Construct LlamaIndex Document with explicit metadata for filtering
        doc = Document(
            text=line_str,
            metadata={
                "line_id": idx,
                "app_id": app_id,
                "timestamp": timestamp,
                "level": level
            }
        )
        documents.append(doc)

    print(f"[+] Successfully converted logs into {len(documents)} structured nodes.")
    return documents

# ==============================================================================
# 2. LOCAL HYBRID PIPELINE MANAGER (Qdrant + BM25 + RRF)
# ==============================================================================
class IntuneLogHybridRAG:
    def __init__(self, collection_name="intune_win32_logs"):
        self.collection_name = collection_name
        
        # Setup clients to local instances
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        
        # Using on-device Ollama models as requested
        self.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")
        self.llm = Ollama(model="qwen2.5:3b", request_timeout=180.0)
        
        self.bm25 = None
        self.raw_nodes = []

    def initialize_storage(self, documents: List[Document]):
        """Splits logs via a sliding window structure, runs embeddings, indexes Qdrant + BM25."""
        # Chunking Strategy: Maintain close sequential text block overlapping
        splitter = SentenceSplitter(chunk_size=256, chunk_overlap=30)
        self.raw_nodes = splitter.get_nodes_from_documents(documents)
        
        # Get baseline dimension size
        sample_emb = self.embed_model.get_text_embedding("test configuration dimension size")
        dim_size = len(sample_emb)

        # Re-create Qdrant Collection safely
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)
            
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim_size, distance=Distance.COSINE)
        )

        points = []
        tokenized_corpus = []

        print("[+] Embedding log segments and preparing indexes. Please wait...")
        for idx, node in enumerate(self.raw_nodes):
            # Compute embeddings
            vector = self.embed_model.get_text_embedding(node.text)
            
            # Pack payload for metadata tracking inside Qdrant
            payload = node.metadata.copy()
            payload["text"] = node.text

            points.append(
                PointStruct(id=idx, vector=vector, payload=payload)
            )
            # Prep items for standard BM25 keyword matching engine
            tokenized_corpus.append(node.text.lower().split())

        # Massive batch injection directly to Qdrant
        # Safely upload to Qdrant in batches to prevent WinError 10053
        batch_size = 100
        print(f"[+] Uploading data to Qdrant in chunks of {batch_size}...")
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name, 
                points=batch
            )
        
        # Initialize internal BM25 indexes
        print("[+] Initializing keyword index matrix...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize internal BM25 indexes
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[+] Storage pipeline operational. Loaded {len(points)} vector chunks into system memory.")

    def reciprocal_rank_fusion(self, semantic_ids: List[int], bm25_ids: List[int], k: int = 60) -> List[int]:
        """Calculates fused score priority values across vector and keyword retrieval methods."""
        fused_scores = {}
        
        # Rank points for dense results
        for rank, doc_id in enumerate(semantic_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            
        # Rank points for sparse results
        for rank, doc_id in enumerate(bm25_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            
        # Return sorted IDs by highest fused visibility score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return sorted_ids

    def execute_hybrid_search(self, query: str, app_id: str = None, top_n: int = 5) -> str:
        """Runs a metadata-filtered semantic and keyword hybrid query with RRF merging."""
        # 1. Dense Semantic Extraction via Qdrant
        query_vector = self.embed_model.get_text_embedding(query)
        
        # Build filter layer dynamically if user queries a targeted Win32 Application ID
        qdrant_filter = None
        if app_id and app_id != "unknown_app_id":
            qdrant_filter = Filter(
                must=[FieldCondition(key="app_id", match=MatchValue(value=app_id))]
            )

        semantic_response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=20  # Over-fetch to filter down via RRF
        )
        semantic_ids = [point.id for point in semantic_response.points]

        # 2. Sparse Retrieval via BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices globally
        top_bm25_idx = np.argsort(bm25_scores)[::-1]
        
        # Filter down by app_id if requested manually
        bm25_ids = []
        for idx in top_bm25_idx:
            if len(bm25_ids) >= 20:
                break
            if app_id and app_id != "unknown_app_id":
                if self.raw_nodes[idx].metadata.get("app_id") == app_id:
                    bm25_ids.append(int(idx))
            else:
                bm25_ids.append(int(idx))

        # 3. Apply Reciprocal Rank Fusion
        fused_final_ids = self.reciprocal_rank_fusion(semantic_ids, bm25_ids, k=60)[:top_n]

        # Assemble the clean ordered context for the LLM
        context_blocks = []
        for doc_id in fused_final_ids:
            meta = self.raw_nodes[doc_id].metadata
            context_blocks.append(
                f"[Timestamp: {meta['timestamp']}] [App ID: {meta['app_id']}] [Level: {meta['level']}]\n"
                f"Log Content: {self.raw_nodes[doc_id].text}\n"
                f"---"
            )
        
        return "\n".join(context_blocks)

    def chat_with_logs(self, user_query: str):
        """Intelligently routes queries based on whether a specific App ID is detected."""
        
        # 1. Regex to sniff out if the user included an App ID in their question
        app_id_pattern = re.compile(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', re.IGNORECASE)
        match = app_id_pattern.search(user_query)
        target_app_id = match.group(1) if match else None

        # 2. Dynamic Search Routing
        if target_app_id:
            print(f"\n[!] Targeted Trace Detected for App ID: {target_app_id}")
            # If they asked a general question but included an ID, combine them for context
            search_query = f"{user_query} unzipping extraction verification validation installation script failure error"
            context = self.execute_hybrid_search(query=search_query, app_id=target_app_id, top_n=7)
        else:
            print(f"\n[!] General Analysis Detected. Searching across ALL logs for: '{user_query}'")
            # Pass None as the app_id to search the entire global log timeline
            context = self.execute_hybrid_search(query=user_query, app_id=None, top_n=10)

        if not context.strip():
            print("[-] Zero related log lines found matching your query.")
            return

        # 3. Dynamic Prompting Strategy
        evaluation_prompt = f"""You are a specialized Microsoft Intune Win32 Application Deployment Engineer.
The user has asked the following question regarding the deployment logs:
"{user_query}"

Analyze the following parsed context extracts extracted via hybrid RRF search from the logs to answer their question. 

Context Log Entries:
{context}

Guidelines for your response:
- If they asked about a specific application, provide the Failure Phase, Chronological Sequence, and Root Cause.
- If they asked a general question (e.g., "what is failing?", "track something suspicious"), identify the specific App IDs experiencing issues, the exact error codes, and summarize the overall health of the system based on the context.
- Be highly technical, concise, and structured.
"""
        print("[+] Invoking Local Qwen2.5:3B Inference Engine...")
        response = self.llm.complete(evaluation_prompt)
        
        print("\n" + "="*80)
        print("                            LOG ANALYSIS REPORT                          ")
        print("="*80)
        print(response.text)
        print("="*80)

# ==============================================================================
# 3. MAIN RUNTIME INVOCATION BLOCK
# ==============================================================================
async def main():
    # Update this path to your exact local desktop log file destination location
    log_path = r"C:\Users\Aniss\Downloads\DiagLogs-UMGC-P4F25173GT-20260527T093008Z\(64) FoldersFiles ProgramData_Microsoft_IntuneManagementExtension_Logs\intunemanagementextension.log"
    
    # Initialize implementation object
    rag_system = IntuneLogHybridRAG()
    
    # 1. Parse raw deployment text structures 
    parsed_docs = parse_intune_workload_logs(log_path)
    
    # 2. Build out internal Qdrant Collections and compute BM25 vectors locally
    rag_system.initialize_storage(parsed_docs)
    
    # 3. Run diagnostics via continuous command loop
    # 3. Run diagnostics via continuous command loop
    print("\n[+] System Ready. You can ask general questions or investigate specific App IDs.")
    print("    Examples:")
    print("    - 'Why did app b9876543-12ab-34cd-56ef-1234567890ab fail?'")
    print("    - 'Are there any suspicious network timeouts?'")
    print("    - 'What app is failing the most right now?'\n")

    while True:
        user_input = input("Agent Query (or type 'exit'): ").strip()
        
        if user_input.lower() == 'exit':
            break
        if not user_input:
            continue
            
        try:
            rag_system.chat_with_logs(user_input)
        except Exception as e:
            print(f"[-] Execution issue encountered: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())