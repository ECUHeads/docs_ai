# Hybrid Search Implementation

This document describes the hybrid search functionality that combines GraphDB (Neo4j) and VectorDB (Qdrant) capabilities.

## Overview

The hybrid search system combines two different approaches to search:
1. **GraphRAG** - Uses Neo4j to perform graph-based reasoning and relationship discovery
2. **Vector Search** - Uses Qdrant to perform semantic similarity search using embeddings

## Architecture

### Configuration System
The system uses a configuration module (`config.py`) that allows for flexible configuration of both database parameters:

- **GraphDB (Neo4j)**: Configurable URI, credentials, and graphRAG parameters
- **VectorDB (Qdrant)**: Configurable URI, collection names, and vector search parameters
- **Hybrid Search**: Configurable weights and search parameters

### Core Components

1. **GraphDB Integration** (`graph_db.py`)
   - Neo4j driver initialization
   - Document node creation
   - GraphRAG search implementation
   - Relationship discovery

2. **VectorDB Integration** (`vector_db.py`)
   - Qdrant client initialization
   - Sentence transformer model for embeddings
   - Document embedding storage
   - Vector search implementation

3. **Hybrid Search Engine** (`hybrid_search.py`)
   - Combines graph and vector search results
   - Weighted scoring of results
   - Document storage in both databases

### Microservice Endpoints

The updated microservice now includes:

1. **`/health`** - Health check endpoint
2. **`/convert`** - PDF to Markdown conversion (existing)
3. **`/search/hybrid`** - Hybrid search using both GraphDB and VectorDB
4. **`/document/store`** - Store documents in both databases

## Usage

### Configuration

Set environment variables for database connections:

```bash
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export QDRANT_URI="localhost:6333"
export GRAPH_WEIGHT=0.5
export VECTOR_WEIGHT=0.5
```

### Hybrid Search API

To perform a hybrid search:
```bash
curl -X POST http://localhost:8080/search/hybrid \
  -H "Content-Type: text/plain" \
  -d "Your search query here"
```

To store a document for hybrid search:
```bash
curl -X POST http://localhost:8080/document/store \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "document123",
    "content": "Document content here...",
    "metadata": {"source": "pdf", "author": "John Doe"}
  }'
```

## Implementation Details

### GraphRAG Approach
The graphRAG implementation:
- Creates document nodes in Neo4j
- Establishes relationships between related documents
- Performs graph traversal to find related content
- Uses relationship strength for ranking

### Vector Search Approach
The vector search implementation:
- Uses sentence transformers to create embeddings
- Stores document chunks in Qdrant
- Performs semantic similarity search
- Returns semantically similar content

### Hybrid Scoring
Results are combined using weighted scoring:
- Graph score (based on relationship strength)
- Vector score (based on semantic similarity)
- Combined score = (graph_weight × graph_score) + (vector_weight × vector_score)

## Benefits

1. **Enhanced Retrieval**: Combines structural knowledge (graph) with semantic understanding (vectors)
2. **Improved Accuracy**: Leverages both relationship information and content similarity
3. **Flexible Configuration**: Parameters can be tuned for specific use cases
4. **Scalable**: Both Neo4j and Qdrant are designed for scalability

## Future Enhancements

1. **Advanced GraphRAG**: Implement more sophisticated graph reasoning algorithms
2. **Dynamic Weighting**: Adjust weights based on query type or domain
3. **Caching**: Implement caching for frequently accessed results
4. **Batch Processing**: Optimize for processing large document collections