# Project 14: Knowledge Graph–Enhanced RAG (Wikipedia + Neo4j)

## Situation / Objective
Pure vector search can miss structured relationships, while pure knowledge graphs can miss semantic similarity. The objective is to implement a **hybrid RAG system** that uses Wikipedia as a source corpus and Neo4j for graph + vector retrieval.

## Task
- Ingest Wikipedia content.
- Construct a knowledge graph representation.
- Create embeddings and a vector index for semantic retrieval.
- Answer user questions using retrieved context + an LLM.

## Actions
- Used Wikipedia loaders to pull documents and applied chunking/token-based splitting.
- Built a Neo4j-backed graph representation and a Neo4j vector index for hybrid retrieval.
- Orchestrated the pipeline using LangChain components (retrieval, graph queries, prompting, response generation).

## Results / Summary
- Implemented an end-to-end KG-RAG workflow demonstrating how graph structure and embeddings can complement each other.
- Produced a reproducible notebook pipeline for ingestion, indexing, and question-answering.

## Security / Secrets Note (Important)
- This project requires credentials (e.g., LLM API keys and Neo4j connection details).
- **Do not hardcode secrets** in notebooks or code.
- Use environment variables instead, e.g.:
  - `GOOGLE_API_KEY`
  - `NEO4J_URI`
  - `NEO4J_USERNAME`
  - `NEO4J_PASSWORD`

## Repository contents
- `kg_rag_wikipedia_neo4j.ipynb`
