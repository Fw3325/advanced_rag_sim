# Advanced RAG System

## Overview
Semantic chunking → sub-query rewrite → multi-retriever fusion (BM25 + vector) → BGE rerank + MMR diversity

Result: maximum recall, maximum precision, minimum hallucination.

## Features

### 1. Knowledge Graph Integration
- **Entity Extraction**: Extracts entities from documents using spaCy NER or LLM-based extraction
- **Graph Building**: Creates a knowledge graph with entities and relationships
- **Graph-based Retrieval**: Uses entity matching and graph traversal to find relevant chunks
- **Graph Expansion**: Expands queries to related entities for better recall

### 2. Performance Optimizations
- **Parallel Retrieval**: Concurrent execution of dense, BM25, and KG searches
- **Batch Reranking**: Single reranking pass instead of per sub-query
- **Query Rewrite Caching**: Caches query rewrites to avoid repeated LLM calls
- **Fast Mode**: Skips expensive operations (query rewriting, reranking, MMR) for maximum speed
- **Ultra-Fast Retrieval**: Direct retrieval without sub-queries for fastest response times

## Usage

### Standard Mode
```python
response = llm_models.generate_response(llm, vectorstore, chunks, query)
```

### With Knowledge Graph
```python
from kg import KnowledgeGraphBuilder, KGRetriever, EntityExtractor

entity_extractor = EntityExtractor(method="spacy", llm=llm)
kg_builder = KnowledgeGraphBuilder(entity_extractor, save_path="kg_store/")
kg_builder.build_from_chunks(chunks, rebuild=True)
kg_builder.save()

kg_retriever = KGRetriever(kg_builder, chunks, entity_extractor)
response = llm_models.generate_response(
    llm, vectorstore, chunks, query, 
    kg_retriever=kg_retriever, enable_kg=True
)
```

### Fast Mode
```python
response = llm_models.generate_response(
    llm, vectorstore, chunks, query,
    fast_mode=True  # Skip reranking, MMR, and query rewriting
)
```
