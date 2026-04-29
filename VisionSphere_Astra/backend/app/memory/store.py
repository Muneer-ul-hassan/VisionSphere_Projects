import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
from datetime import datetime
import json
from app.config import settings


class MemoryStore:
    """Vector-based memory storage for conversation history and visual memories"""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.memory_db_path,
            settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False)
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=settings.memory_collection,
            metadata={"hnsw:space": "cosine"}
        )

    def add_conversation(self, user_message: str, assistant_response: str,
                         context: Optional[str] = None) -> str:
        """Store a conversation turn in memory"""
        doc_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        self.collection.add(
            documents=[f"User: {user_message}\nAssistant: {assistant_response}"],
            metadatas=[{
                "type": "conversation",
                "timestamp": datetime.now().isoformat(),
                "context": context or ""
            }],
            ids=[doc_id]
        )

        return doc_id

    def add_visual_memory(self, description: str, objects: List[str],
                          location: Optional[str] = None) -> str:
        """Store a visual observation in memory"""
        doc_id = f"visual_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        self.collection.add(
            documents=[description],
            metadatas=[{
                "type": "visual",
                "timestamp": datetime.now().isoformat(),
                "objects": json.dumps(objects),
                "location": location or ""
            }],
            ids=[doc_id]
        )

        return doc_id

    def search_memories(self, query: str, n_results: int = 5,
                        memory_type: Optional[str] = None) -> List[Dict]:
        """Search memories by semantic similarity"""
        where_clause = None
        if memory_type:
            where_clause = {"type": memory_type}

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )

        memories = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                memory = {
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                }
                memories.append(memory)

        return memories

    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """Get most recent conversation turns"""
        # ChromaDB doesn't have native sorting by metadata, so we fetch and sort
        results = self.collection.get(
            where={"type": "conversation"},
            limit=limit * 2  # Get extra to ensure we have enough after sorting
        )

        conversations = []
        if results['documents']:
            for i, doc in enumerate(results['documents']):
                conversations.append({
                    "content": doc,
                    "metadata": results['metadatas'][i] if results['metadatas'] else {},
                    "id": results['ids'][i]
                })

        # Sort by timestamp descending
        conversations.sort(
            key=lambda x: x['metadata'].get('timestamp', ''),
            reverse=True
        )

        return conversations[:limit]

    def get_visual_memories(self, limit: int = 20) -> List[Dict]:
        """Get recent visual memories"""
        results = self.collection.get(
            where={"type": "visual"},
            limit=limit
        )

        memories = []
        if results['documents']:
            for i, doc in enumerate(results['documents']):
                memories.append({
                    "description": doc,
                    "metadata": results['metadatas'][i] if results['metadatas'] else {},
                    "id": results['ids'][i]
                })

        # Sort by timestamp descending
        memories.sort(
            key=lambda x: x['metadata'].get('timestamp', ''),
            reverse=True
        )

        return memories

    def clear_memories(self, memory_type: Optional[str] = None):
        """Clear memories (optionally by type)"""
        if memory_type:
            # Get all IDs of this type and delete
            results = self.collection.get(where={"type": memory_type})
            if results['ids']:
                self.collection.delete(ids=results['ids'])
        else:
            self.collection.delete(where={})


# Global instance
memory_store = MemoryStore()
