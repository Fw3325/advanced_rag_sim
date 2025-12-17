"""
Knowledge Graph Retriever: Retrieves chunks based on entity matching and graph traversal.
Optimized with caching.
"""
from typing import List, Set, Optional
from langchain_core.documents import Document
from .kg_builder import KnowledgeGraphBuilder
from .entity_extractor import EntityExtractor

class KGRetriever:
    """
    Retrieves document chunks using knowledge graph.
    Optimized with entity extraction caching.
    """
    
    def __init__(self, kg_builder: KnowledgeGraphBuilder, chunks: List[Document], entity_extractor: EntityExtractor):
        """
        Args:
            kg_builder: KnowledgeGraphBuilder instance with loaded graph
            chunks: List of all Document chunks (for lookup by chunk_key)
            entity_extractor: EntityExtractor for extracting entities from queries
        """
        self.kg_builder = kg_builder
        self.chunks = chunks
        self.entity_extractor = entity_extractor
        self.chunk_key_to_doc = {self._get_chunk_key(chunk): chunk for chunk in chunks}
        self._entity_cache = {}  # Simple cache for entity extraction
    
    def search(self, query: str, k: int = 10, use_graph_expansion: bool = True, max_hops: int = 2) -> List[Document]:
        """
        Search for chunks using knowledge graph.
        Optimized with caching.
        
        Args:
            query: User query text
            k: Number of chunks to return
            use_graph_expansion: If True, expand to related entities
            max_hops: Maximum graph hops for expansion
        
        Returns:
            List of Document objects
        """
        # Cache entity extraction (simple hash-based cache)
        query_hash = hash(query.lower().strip())
        if query_hash in self._entity_cache:
            query_entities = self._entity_cache[query_hash]
        else:
            query_entities = self.entity_extractor.extract_entities(query)
            self._entity_cache[query_hash] = query_entities
            # Limit cache size
            if len(self._entity_cache) > 100:
                self._entity_cache.clear()
        
        if not query_entities:
            return []
        
        # Get chunks that mention these entities
        chunk_keys = self.kg_builder.get_chunks_for_entities(list(query_entities), max_chunks=k * 2)
        
        # Graph expansion: get related entities and their chunks
        if use_graph_expansion:
            related_entities = set()
            for entity in query_entities:
                related = self.kg_builder.get_related_entities(entity, max_hops=max_hops, max_entities=5)
                related_entities.update(related)
            
            if related_entities:
                related_chunk_keys = self.kg_builder.get_chunks_for_entities(list(related_entities), max_chunks=k)
                chunk_keys.update(related_chunk_keys)
        
        # Convert chunk keys to Document objects
        results = []
        for chunk_key in list(chunk_keys)[:k]:
            if chunk_key in self.chunk_key_to_doc:
                results.append(self.chunk_key_to_doc[chunk_key])
        
        return results
    
    def _get_chunk_key(self, chunk: Document) -> str:
        """Generate a unique key for a chunk."""
        source = chunk.metadata.get("source", "unknown")
        chunk_num = chunk.metadata.get("chunk_number", 0)
        return f"{source}:{chunk_num}"
