# Project 17: NLP — Transformers, RAG, Web Scraping & Text-to-SQL

## Situation / Objective
NLP systems in practice combine multiple capabilities: classification, retrieval-augmented generation (RAG), evaluation, and building structured datasets from the web. The objective is to complete an NLP assignment spanning these applied components.

## Task
- Build a transformer-based text classification pipeline.
- Build a Wikipedia-backed index for retrieval.
- Evaluate RAG answer quality.
- Extract structured data via web scraping.
- Enable natural-language querying over SQL (Text-to-SQL-style workflow).

## Actions
- Implemented BERT-based sequence classification (tokenization, model loading/training/evaluation).
- Built Wikipedia ingestion + indexing pipelines using LlamaIndex-style components (cleaning, chunking, embeddings).
- Implemented RAG evaluation using similarity-based scoring and compared top-K retrieval settings.
- Scraped web data (ESPN scorecard) and stored it in SQLite for structured querying.
- Built an NL-to-SQL workflow using an LLM interface over SQL tables.

## Results / Summary
- Delivered a complete set of notebooks demonstrating multiple real-world NLP building blocks.
- Showed how retrieval quality impacts generation quality and how structured storage enables richer downstream querying.

## Repository contents
- `bert_text_classification.ipynb`
- `wikipedia_index_prime_ministers_llamaindex.ipynb`
- `rag_evaluation_prime_ministers.ipynb`
- `web_scraping_espn_scorecard_to_sqlite.ipynb`
- `text_to_sql_llamaindex_groq.ipynb`
