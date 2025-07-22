# LegalDocumentAnalyzer
This project is an advanced Legal Document QA system that uses RAG (Retrieval-Augmented Generation) to answer legal questions from PDFs, DOCX, images, and text files. It also supports semantic chunking, clause reference extraction, and Cohere reranking to improve relevance and trustworthiness of answers.
Multi-format support: Extracts legal text from .pdf, .docx, .txt, .jpg, .png

Semantic chunking: Intelligent sentence-level chunking with overlap

FAISS vector store: Stores chunked embeddings for retrieval

MMR-based retriever: Uses Maximal Marginal Relevance for diverse retrieval

Cohere Reranker: Boosts top-k relevant results from retriever

Clause reference extraction: Extracts and appends clause numbers to answers

LLM-based answering: Uses microsoft/phi-2 for answer generation

Secure API key handling: via .env and python-dotenv


🚀 How It Works
Extract Text

Uses fitz, docx, or pytesseract for OCR depending on file type

Semantic Chunking
Breaks text into ~1000 character overlapping chunks using nltk sentence tokenization

Vectorization
Chunks are converted to embeddings using all-MiniLM-L6-v2 and stored in FAISS

Retriever & Reranker
Relevant chunks are retrieved via MMR and reranked using Cohere

Prompt + Generation
Context is injected into a prompt and sent to phi-2 using transformers pipeline

Postprocessing
Extracts referenced clauses using regex and appends to the final answer.
