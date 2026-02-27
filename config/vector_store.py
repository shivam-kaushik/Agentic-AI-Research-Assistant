"""
Vector Store Integration for Context Management

Supports GCP Vertex AI Vector Search and Firestore for:
- Storing conversation history embeddings
- Storing research findings embeddings  
- Semantic search for context retrieval
- RAG-based question answering
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
    from google.cloud import firestore
    from google.cloud import aiplatform
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

from config.gcp_config import config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector embeddings for conversation context and research findings.
    Uses Firestore for storage and Vertex AI for embeddings.
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.embedding_model = None
        self.db = None
        
        if not VECTOR_SEARCH_AVAILABLE:
            logger.warning("Vector search libraries not available. Using basic storage only.")
            return
        
        try:
            # Initialize Vertex AI with quota project
            import os
            quota_project = os.getenv('GOOGLE_CLOUD_QUOTA_PROJECT', config.project_id)
            
            vertexai.init(project=config.project_id, location=config.location)
            
            # Initialize embedding model (requires AI Platform API enabled)
            self.embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
            
            # Initialize Firestore
            self.db = firestore.Client(project=config.project_id)
            
            logger.info(f"Vector store initialized for session {session_id}")
        except Exception as e:
            logger.warning(f"Vector store unavailable (will use memory-only mode): {str(e)[:100]}")
            self.embedding_model = None
            self.db = None
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text."""
        if not self.embedding_model:
            return None
        
        try:
            embeddings = self.embedding_model.get_embeddings([text])
            return embeddings[0].values if embeddings else None
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def store_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Store a conversation message with its embedding.
        
        Returns:
            Document ID
        """
        if not self.db:
            return ""
        
        try:
            # Generate embedding
            embedding = self.generate_embedding(content)
            
            # Create document
            doc_data = {
                "session_id": self.session_id,
                "role": role,
                "content": content,
                "timestamp": datetime.now(),
                "metadata": metadata or {},
            }
            
            if embedding:
                doc_data["embedding"] = embedding
            
            # Store in Firestore
            doc_ref = self.db.collection("conversation_history").document()
            doc_ref.set(doc_data)
            
            logger.debug(f"Stored message: {doc_ref.id}")
            return doc_ref.id
            
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
            return ""
    
    def store_research_finding(self, 
                               task_id: str, 
                               data_source: str, 
                               content: str,
                               structured_data: Dict[str, Any] = None) -> str:
        """
        Store research findings with embeddings for semantic search.
        
        Returns:
            Document ID
        """
        if not self.db:
            return ""
        
        try:
            # Generate embedding
            embedding = self.generate_embedding(content)
            
            # Create document
            doc_data = {
                "session_id": self.session_id,
                "task_id": task_id,
                "data_source": data_source,
                "content": content,
                "structured_data": structured_data or {},
                "timestamp": datetime.now(),
            }
            
            if embedding:
                doc_data["embedding"] = embedding
            
            # Store in Firestore
            doc_ref = self.db.collection("research_findings").document()
            doc_ref.set(doc_data)
            
            logger.debug(f"Stored research finding: {doc_ref.id}")
            return doc_ref.id
            
        except Exception as e:
            logger.error(f"Failed to store research finding: {e}")
            return ""
    
    def semantic_search_messages(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search conversation history using semantic similarity.
        
        Returns:
            List of relevant messages with similarity scores
        """
        if not self.db or not self.embedding_model:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return []
            
            # Query Firestore for messages in this session
            messages_ref = self.db.collection("conversation_history")
            query_docs = messages_ref.where("session_id", "==", self.session_id).stream()
            
            # Calculate cosine similarity
            results = []
            for doc in query_docs:
                doc_data = doc.to_dict()
                if "embedding" in doc_data:
                    similarity = self._cosine_similarity(query_embedding, doc_data["embedding"])
                    results.append({
                        "doc_id": doc.id,
                        "role": doc_data.get("role"),
                        "content": doc_data.get("content"),
                        "timestamp": doc_data.get("timestamp"),
                        "similarity": similarity
                    })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to search messages: {e}")
            return []
    
    def semantic_search_findings(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search research findings using semantic similarity.
        
        Returns:
            List of relevant findings with similarity scores
        """
        if not self.db or not self.embedding_model:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return []
            
            # Query Firestore for findings in this session
            findings_ref = self.db.collection("research_findings")
            query_docs = findings_ref.where("session_id", "==", self.session_id).stream()
            
            # Calculate cosine similarity
            results = []
            for doc in query_docs:
                doc_data = doc.to_dict()
                if "embedding" in doc_data:
                    similarity = self._cosine_similarity(query_embedding, doc_data["embedding"])
                    results.append({
                        "doc_id": doc.id,
                        "task_id": doc_data.get("task_id"),
                        "data_source": doc_data.get("data_source"),
                        "content": doc_data.get("content"),
                        "structured_data": doc_data.get("structured_data"),
                        "timestamp": doc_data.get("timestamp"),
                        "similarity": similarity
                    })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to search findings: {e}")
            return []
    
    def get_relevant_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        Retrieve relevant context for answering a user query.
        Combines conversation history and research findings.
        
        Returns:
            Formatted context string suitable for RAG
        """
        # Search both messages and findings
        relevant_messages = self.semantic_search_messages(query, top_k=3)
        relevant_findings = self.semantic_search_findings(query, top_k=5)
        
        # Build context string
        context_parts = []
        
        if relevant_messages:
            context_parts.append("## Relevant Conversation History\n")
            for msg in relevant_messages:
                context_parts.append(
                    f"**{msg['role'].title()}** ({msg['timestamp'].strftime('%H:%M:%S') if msg.get('timestamp') else 'N/A'}): "
                    f"{msg['content'][:300]}...\n"
                )
        
        if relevant_findings:
            context_parts.append("\n## Relevant Research Findings\n")
            for finding in relevant_findings:
                context_parts.append(
                    f"**{finding['data_source']}** (Task: {finding['task_id']}): "
                    f"{finding['content'][:300]}...\n"
                )
        
        context = "\n".join(context_parts)
        
        # Truncate if too long (simple token estimation: ~4 chars per token)
        if len(context) > max_tokens * 4:
            context = context[:max_tokens * 4] + "\n...[context truncated]"
        
        return context if context else "No relevant context found."
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def clear_session_data(self):
        """Delete all data for this session."""
        if not self.db:
            return
        
        try:
            # Delete conversation history
            messages_ref = self.db.collection("conversation_history")
            query_docs = messages_ref.where("session_id", "==", self.session_id).stream()
            for doc in query_docs:
                doc.reference.delete()
            
            # Delete research findings
            findings_ref = self.db.collection("research_findings")
            query_docs = findings_ref.where("session_id", "==", self.session_id).stream()
            for doc in query_docs:
                doc.reference.delete()
            
            logger.info(f"Cleared all data for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to clear session data: {e}")
