'''
Advanced RAG System: 
Semantic chunking → sub-query rewrite → multi-retriever fusion (BM25 + vector) → BGE rerank + MMR diversity
Result: maximum recall, maximum precision, minimum hallucination.
'''
import os
import shutil
import streamlit as st
import hashlib
import json

from chunking.chunking import process_pdfs_from_local_dir
from chunking.vector_store import create_faiss_index, load_faiss_index
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llm import llm_models
from kg import KnowledgeGraphBuilder, KGRetriever, EntityExtractor

def main():
    # Streamlit app only (no CLI pre-run)
    st.set_page_config(page_title="Health & Fitness Guide", layout="wide")

    @st.cache_data(show_spinner=False)
    def load_chunks(path="input/"):
        return process_pdfs_from_local_dir(path)

    @st.cache_resource(hash_funcs={list: id})
    def get_vectorstore(all_chunks, embedding_model="BAAI/bge-large-en-v1.5", save_path="faiss_store/"):
        if os.path.exists(save_path):
            return load_faiss_index(embedding_model=embedding_model, save_path=save_path)
        else:
            return create_faiss_index(all_chunks, embedding_model=embedding_model, save_path=save_path)
    
    def chunks_fingerprint(all_chunks) -> str:
        """Make a stable representation for caching."""
        payload = [
            {
                "text": c.page_content,
                "source": c.metadata.get("source"),
                "chunk_number": c.metadata.get("chunk_number"),
            }
            for c in all_chunks
        ]
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @st.cache_resource
    def get_kg_retriever(_all_chunks, chunks_key=None, kg_save_path="kg_store/"):
        """Build or load knowledge graph and return retriever."""
        llm = llm_models.llm_model()
        entity_extractor = EntityExtractor(method="spacy", llm=llm)
        kg_builder = KnowledgeGraphBuilder(entity_extractor, save_path=kg_save_path)
        
        # Try to load existing KG
        if not kg_builder.load():
            # Build new KG
            print("🔨 Building knowledge graph...")
            kg_builder.build_from_chunks(_all_chunks, rebuild=True)
            kg_builder.save()
        
        return KGRetriever(kg_builder, _all_chunks, entity_extractor)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts: {"role": "user"|"assistant", "text": "..."}
    if "resources_loaded" not in st.session_state:
        st.session_state.resources_loaded = False
    if "enable_kg" not in st.session_state:
        st.session_state.enable_kg = True
    if "fast_mode" not in st.session_state:
        st.session_state.fast_mode = False

    st.title("Supplement & Fitness Guide Q&A")
    st.markdown("Type a question about the PDFs placed in the `input/` folder. The app uses a RAG pipeline with Knowledge Graph and performance optimizations.")

    with st.sidebar:
        st.header("Controls")
        if st.button("Clear conversation"):
            st.session_state.messages = []
        if st.button("Rebuild index (delete & recreate)"):
            if os.path.exists("faiss_store/"):
                shutil.rmtree("faiss_store/")
            if os.path.exists("kg_store/"):
                shutil.rmtree("kg_store/")
            # clear cache resource so it rebuilds on next access
            try:
                del st.session_state["__get_vectorstore_cache__"]
                del st.session_state["__get_kg_retriever_cache__"]
            except Exception:
                pass
            st.session_state.resources_loaded = False
            st.rerun()
        
        st.header("Settings")
        # Use session state to persist checkbox values and avoid reruns
        enable_kg = st.checkbox(
            "Enable Knowledge Graph", 
            value=st.session_state.enable_kg, 
            help="Use knowledge graph for entity-based retrieval (improves recall)",
            key="enable_kg_checkbox"
        )
        fast_mode = st.checkbox(
            "Fast Mode", 
            value=st.session_state.fast_mode, 
            help="Skip reranking and MMR for faster responses (lower precision)",
            key="fast_mode_checkbox"
        )
        
        # Update session state only if changed
        if enable_kg != st.session_state.enable_kg:
            st.session_state.enable_kg = enable_kg
        if fast_mode != st.session_state.fast_mode:
            st.session_state.fast_mode = fast_mode

    # Load resources only once and cache in session state
    if not st.session_state.resources_loaded:
        with st.spinner("Loading documents and index..."):
            all_chunks = load_chunks("input/")
            vectorstore = get_vectorstore(all_chunks)
            llm = llm_models.llm_model()
            key = chunks_fingerprint(all_chunks)
            
            # Store in session state
            st.session_state.all_chunks = all_chunks
            st.session_state.vectorstore = vectorstore
            st.session_state.llm = llm
            st.session_state.chunks_key = key
            st.session_state.resources_loaded = True
    
    # Retrieve from session state (no reloading)
    all_chunks = st.session_state.all_chunks
    vectorstore = st.session_state.vectorstore
    llm = st.session_state.llm
    
    # Load KG retriever only if enabled and not already loaded
    if st.session_state.enable_kg:
        if "kg_retriever" not in st.session_state:
            # Only show spinner on first load, not on checkbox toggle
            if not st.session_state.get("kg_loading_shown", False):
                with st.spinner("Loading knowledge graph..."):
                    kg_retriever = get_kg_retriever(all_chunks, chunks_key=st.session_state.chunks_key)
                    st.session_state.kg_retriever = kg_retriever
                    st.session_state.kg_loading_shown = True
            else:
                kg_retriever = get_kg_retriever(all_chunks, chunks_key=st.session_state.chunks_key)
                st.session_state.kg_retriever = kg_retriever
        else:
            kg_retriever = st.session_state.kg_retriever
    else:
        kg_retriever = None

    # Input form
    with st.form(key="query_form", clear_on_submit=True):
        user_query = st.text_input("Your question", placeholder="Ask about the PDFs, e.g. 'Should I take supplements?'")
        submitted = st.form_submit_button("Ask")
        if submitted and user_query.strip():
            st.session_state.messages.append({"role": "user", "text": user_query})
            with st.spinner("Generating response..."):
                response = llm_models.generate_response(
                    llm, vectorstore, all_chunks, user_query, 
                    kg_retriever=kg_retriever, 
                    enable_kg=st.session_state.enable_kg, 
                    fast_mode=st.session_state.fast_mode
                )
                # support objects with .answer or plain string
                answer = getattr(response, "answer", None) or str(response)
                timing = getattr(response, "_timing", None)
            st.session_state.messages.append({
                "role": "assistant",
                "text": answer,
                "timing": timing,
            })

    # Display conversation
    chat_container = st.container()
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_container.markdown(f"**You:** {msg['text']}")
        else:
            chat_container.markdown(f"**Assistant:** {msg['text']}")
            # Show timing if available
            timing = msg.get("timing")
            if timing:
                rt = timing.get("retrieval_time", 0.0)
                gt = timing.get("generation_time", 0.0)
                tt = timing.get("total_time", 0.0)
                fast = timing.get("fast_mode", False)
                enable_kg = timing.get("enable_kg", False)
                chat_container.markdown(
                    f"*Timings — retrieval: {rt:.2f}s, generation: {gt:.2f}s, "
                    f"total: {tt:.2f}s "
                    f"(fast_mode={fast}, enable_kg={enable_kg})*"
                )



if __name__ == "__main__":
    main()
