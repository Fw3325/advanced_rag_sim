import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import re

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")

'''
chunking: Sentence Transformers + Semantic Chunking + Recursive Splitter
'''

# ✅ Sentence Transformer for semantic-aware splitting (fast & accurate)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight + good quality

# ✅ spaCy model for sentence segmentation
_nlp_model = None

def _get_spacy_model():
    """Lazy load spaCy model."""
    global _nlp_model
    if _nlp_model is None and SPACY_AVAILABLE:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            print("   Falling back to simple sentence splitting.")
            _nlp_model = False  # Mark as unavailable
    return _nlp_model

def text_to_sentences(text: str) -> str:
    """
    Split text into sentences using spaCy.
    Returns sentences joined by newlines (compatible with blingfire format).
    """
    nlp = _get_spacy_model()
    
    if nlp and nlp is not False:
        # Use spaCy for sentence segmentation
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return "\n".join(sentences)
    else:
        # Fallback to simple regex-based sentence splitting
        # Split on sentence endings followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return "\n".join(sentences)

def semantic_chunk_sentences(pages, threshold=0.55): # update threshold as needed 
    """
    Pages = list[Document], each containing a page of text.
    Group sentences into semantic chunks based on embedding similarity.
    """
    semantic_chunks = []

    for page in pages:
        text = page.page_content
        try:
            page_num = int(page.metadata.get("page", 1))
        except:
            page_num = None 
        # (sentence, page_num) pairs
        raw_sentences = text_to_sentences(text).split("\n")
        raw_sentences = [s.strip() for s in raw_sentences if s.strip()]
        sentences = [(s, page_num) for s in raw_sentences]
        
        if not sentences:
            continue

        # Embed only the text portion
        embeddings = sentence_model.encode([s for s, _ in sentences])

        current_chunk = []
        current_embedding = None
        current_pages = set()

        for (sentence, pg), embedding in zip(sentences, embeddings):
            # first sentence initializes the chunk
            if not current_chunk:
                current_chunk = [sentence]
                current_embedding = embedding
                current_pages.add(pg)
                continue

            # semantic similarity - Cosine Similarity
            similarity = np.dot(current_embedding, embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(embedding)
            )
            
            # same topic -> append to current chunk
            if similarity >= threshold:
                current_chunk.append(sentence)
                # update embedding center
                current_embedding = (current_embedding + embedding) / 2.0
                current_pages.add(pg)
            else:
                # new semantic topic
                semantic_chunks.append({"text": ". ".join(current_chunk), 
                                        "page": sorted(p for p in current_pages if p is not None) or ["Unknown"]
                                        })
                current_chunk = [sentence]
                current_embedding = embedding
                current_pages = {pg}
        
        if current_chunk:  # flush remaining chunk
            semantic_chunks.append({"text": ". ".join(current_chunk), 
                                    "page": sorted(p for p in current_pages if p is not None) or ["Unknown"]
                                    })
    return semantic_chunks

# ===== PDF Processing Pipeline ===== 
def process_pdf_from_local(pdf_path, chunk_size=512, chunk_overlap=64):
    """
    Load a local PDF, perform semantic chunking + token-aware splitting.
    Returns: list[Document] with metadata (source, pages, chunk_number, word_count) and page_content
    e.g.:
    Document(
        metadata={'pages': [2], 'source': 'fitness_nutrition_guide.pdf', 'chunk_number': 5, 'word_count': 69}, 
        page_content='xxxxxx.'
        )
    """
    filename = os.path.basename(pdf_path)
    print(f"Loading local PDF: {filename}")
    
    # 1) Load PDF as page-level documents
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()  # returns a list of Documents, each page has metadata={'page'}
    
    if not pages:
        print(f"No content extracted from PDF: {filename}")
        return []

    # page number normalization (page always starts from 1)
    min_page = min(int(p.metadata.get("page", 1)) for p in pages)
    normalize_offset = 1 - min_page if min_page < 1 else 0

    # Step 1: semantic sentence-based chunk grouping
    sentence_chunks = semantic_chunk_sentences(pages, threshold=0.55)

    # texts and base metas for next splitter
    texts = [c["text"] for c in sentence_chunks]
    base_metas = [{"pages": c["page"]} for c in sentence_chunks]  # inherit page info

    # Step 2: Fallback safety split by token count
    final_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". "],  # priority: paragraph → sentence → whitespace
        keep_separator=True
    )

    docs = final_splitter.create_documents(texts, metadatas=base_metas)
    
    for i, chunk in enumerate(docs, start=1):
        chunk.metadata["source"] = filename
        chunk.metadata["chunk_number"] = i
        chunk.metadata["word_count"] = len(chunk.page_content.split())
        
        orig_pages = chunk.metadata.get("pages", "Unknown")
        if isinstance(orig_pages, list) and orig_pages and isinstance(orig_pages[0], int):
            chunk.metadata["pages"] = [p + normalize_offset for p in orig_pages]
        else:
            chunk.metadata["pages"] = orig_pages 

    print(f"✅ Processed {len(docs)} semantic chunks from {filename}")

    return docs

def process_pdfs_from_local_dir(dir_path):
    """
    Process all PDF files in a local directory.
    Returns: list[Document]
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f"❌ Folder doesn't exist: {dir_path}")

    all_chunks = []
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(dir_path, filename)
            chunks = process_pdf_from_local(pdf_path)
            all_chunks.extend(chunks)
    print(f"📚 Total chunks from directory: {len(all_chunks)}")
    return all_chunks
