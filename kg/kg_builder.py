"""
Knowledge Graph Builder: Creates and stores a knowledge graph from document chunks.
Uses NetworkX for graph storage (lightweight, no external DB required).
"""
import os
import pickle
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from langchain_core.documents import Document
from .entity_extractor import EntityExtractor

class KnowledgeGraphBuilder:
    """
    Builds a knowledge graph from document chunks.
    Graph structure:
    - Nodes: Entities (with name attribute)
    - Edges: Relationships between entities (with relation_type attribute)
    - Entity-Chunk mapping: Stores which chunks mention which entities
    """
    
    def __init__(self, entity_extractor: EntityExtractor, save_path: str = "kg_store/"):
        """
        Args:
            entity_extractor: EntityExtractor instance
            save_path: Directory to save/load the graph
        """
        self.entity_extractor = entity_extractor
        self.save_path = save_path
        self.graph = nx.MultiDiGraph()  # Directed multigraph (multiple edges allowed)
        self.entity_to_chunks: Dict[str, Set[str]] = {}  # entity_name -> set of chunk_keys
        self.chunk_to_entities: Dict[str, Set[str]] = {}  # chunk_key -> set of entity_names
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)
    
    def build_from_chunks(self, chunks: List[Document], rebuild: bool = False):
        """
        Build knowledge graph from a list of document chunks.
        
        Args:
            chunks: List of Document objects with metadata
            rebuild: If True, clear existing graph before building
        """
        if rebuild:
            self.graph.clear()
            self.entity_to_chunks.clear()
            self.chunk_to_entities.clear()
        
        print(f"🔨 Building knowledge graph from {len(chunks)} chunks...")
        
        for chunk in chunks:
            chunk_key = self._get_chunk_key(chunk)
            text = chunk.page_content
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(text, chunk_key)
            
            if not entities:
                continue
            
            # Store entity-chunk mappings
            for entity in entities:
                # Normalize entity name (lowercase for matching, but keep original)
                entity_normalized = entity.lower().strip()
                
                if entity_normalized not in self.entity_to_chunks:
                    self.entity_to_chunks[entity_normalized] = set()
                self.entity_to_chunks[entity_normalized].add(chunk_key)
                
                if chunk_key not in self.chunk_to_entities:
                    self.chunk_to_entities[chunk_key] = set()
                self.chunk_to_entities[chunk_key].add(entity_normalized)
                
                # Add entity node to graph (if not exists)
                if not self.graph.has_node(entity_normalized):
                    self.graph.add_node(entity_normalized, name=entity, original_name=entity)
            
            # Extract relations
            relations = self.entity_extractor.extract_relations(text, entities)
            
            for head, relation, tail in relations:
                head_norm = head.lower().strip()
                tail_norm = tail.lower().strip()
                
                # Ensure nodes exist
                if not self.graph.has_node(head_norm):
                    self.graph.add_node(head_norm, name=head, original_name=head)
                if not self.graph.has_node(tail_norm):
                    self.graph.add_node(tail_norm, name=tail, original_name=tail)
                
                # Add edge with relation type
                self.graph.add_edge(head_norm, tail_norm, relation_type=relation, chunk_key=chunk_key)
        
        print(f"✅ Knowledge graph built: {len(self.graph.nodes())} entities, {len(self.graph.edges())} relations")
        print(f"   Entity-chunk mappings: {len(self.entity_to_chunks)} entities mapped to chunks")
    
    def get_chunks_for_entities(self, entity_names: List[str], max_chunks: int = 20) -> Set[str]:
        """
        Get chunk keys that mention any of the given entities.
        
        Args:
            entity_names: List of entity names (will be normalized)
            max_chunks: Maximum number of chunk keys to return
        
        Returns:
            Set of chunk keys
        """
        chunk_keys = set()
        
        for entity in entity_names:
            entity_norm = entity.lower().strip()
            if entity_norm in self.entity_to_chunks:
                chunk_keys.update(self.entity_to_chunks[entity_norm])
        
        # Return limited set
        return set(list(chunk_keys)[:max_chunks])
    
    def get_related_entities(self, entity_name: str, max_hops: int = 2, max_entities: int = 10) -> Set[str]:
        """
        Get entities related to the given entity within max_hops.
        
        Args:
            entity_name: Starting entity name
            max_hops: Maximum graph distance (1 = direct neighbors, 2 = neighbors of neighbors)
            max_entities: Maximum number of related entities to return
        
        Returns:
            Set of related entity names
        """
        entity_norm = entity_name.lower().strip()
        
        if not self.graph.has_node(entity_norm):
            return set()
        
        related = set()
        
        # Get neighbors within max_hops
        if max_hops >= 1:
            # Direct neighbors
            related.update(self.graph.successors(entity_norm))
            related.update(self.graph.predecessors(entity_norm))
        
        if max_hops >= 2:
            # 2-hop neighbors
            for neighbor in list(self.graph.successors(entity_norm)) + list(self.graph.predecessors(entity_norm)):
                related.update(self.graph.successors(neighbor))
                related.update(self.graph.predecessors(neighbor))
        
        # Remove the original entity
        related.discard(entity_norm)
        
        return set(list(related)[:max_entities])
    
    def save(self):
        """Save the knowledge graph to disk."""
        graph_file = os.path.join(self.save_path, "kg_graph.pkl")
        mappings_file = os.path.join(self.save_path, "kg_mappings.pkl")
        
        with open(graph_file, "wb") as f:
            pickle.dump(self.graph, f)
        
        with open(mappings_file, "wb") as f:
            pickle.dump({
                "entity_to_chunks": self.entity_to_chunks,
                "chunk_to_entities": self.chunk_to_entities
            }, f)
        
        print(f"✅ Knowledge graph saved to {self.save_path}")
    
    def load(self) -> bool:
        """
        Load the knowledge graph from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        graph_file = os.path.join(self.save_path, "kg_graph.pkl")
        mappings_file = os.path.join(self.save_path, "kg_mappings.pkl")
        
        if not (os.path.exists(graph_file) and os.path.exists(mappings_file)):
            return False
        
        try:
            with open(graph_file, "rb") as f:
                self.graph = pickle.load(f)
            
            with open(mappings_file, "rb") as f:
                mappings = pickle.load(f)
                self.entity_to_chunks = mappings["entity_to_chunks"]
                self.chunk_to_entities = mappings["chunk_to_entities"]
            
            print(f"✅ Knowledge graph loaded: {len(self.graph.nodes())} entities, {len(self.graph.edges())} relations")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load knowledge graph: {e}")
            return False
    
    def _get_chunk_key(self, chunk: Document) -> str:
        """Generate a unique key for a chunk."""
        source = chunk.metadata.get("source", "unknown")
        chunk_num = chunk.metadata.get("chunk_number", 0)
        return f"{source}:{chunk_num}"
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge graph."""
        return {
            "num_entities": len(self.graph.nodes()),
            "num_relations": len(self.graph.edges()),
            "num_entity_chunk_mappings": len(self.entity_to_chunks),
            "avg_entities_per_chunk": sum(len(v) for v in self.chunk_to_entities.values()) / max(len(self.chunk_to_entities), 1)
        }
