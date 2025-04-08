import os
import gc
import uuid
import tempfile
import base64
from dotenv import load_dotenv
from rag_code import Transcribe, EmbedData, QdrantVDB_QB, Retriever, RAG
import streamlit as st
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
session_id = st.session_state.id
collection_name = "chat with audios"
batch_size = 32
load_dotenv()
def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()
def process_audio_file(file_path, file_name):
    file_key = f"{session_id}-{file_name}"
    
    if file_key not in st.session_state.get('file_cache', {}):
        transcriber = Transcribe(api_key=os.getenv("ASSEMBLYAI_API_KEY"))
        transcripts = transcriber.transcribe_audio(file_path)
        st.session_state.transcripts = transcripts
        
        documents = [f"Speaker {t['speaker']}: {t['text']}" for t in transcripts]
        embeddata = EmbedData(embed_model_name="BAAI/bge-large-en-v1.5", batch_size=batch_size)
        embeddata.embed(documents)
        qdrant_vdb = QdrantVDB_QB(collection_name=collection_name,
                              batch_size=batch_size,
                              vector_dim=1024)
        qdrant_vdb.define_client()
        qdrant_vdb.create_collection()
        qdrant_vdb.ingest_data(embeddata=embeddata)
        retriever = Retriever(vector_db=qdrant_vdb, embeddata=embeddata)
        query_engine = RAG(retriever=retriever, llm_name="DeepSeek-R1-Distill-Llama-70B")
        st.session_state.file_cache[file_key] = query_engine
        return query_engine, transcripts
    else:
        query_engine = st.session_state.file_cache[file_key]
        return query_engine, st.session_state.transcripts
with st.sidebar:
    st.header("Add your audio file!")
    st.subheader("Sample Files")
    sample_col1, sample_col2 = st.columns(2)
    
    with sample_col1:
        if st.button("Stephen Schwarzman"):
            sample_path = os.path.join("samples", "Stephen Schwarzman_AI_Podcast_Clip.mp3")
            sample_name = "Stephen Schwarzman_AI_Podcast_Clip"
            
            st.write("Processing sample file...")
            try:
                query_engine, transcripts = process_audio_file(sample_path, sample_name)
                st.session_state.query_engine = query_engine
                st.session_state.transcripts = transcripts
                st.session_state.current_file = sample_name
                st.success("Ready to Chat!")
            except Exception as e:
                st.error(f"Error processing sample: {e}")
    
    with sample_col2:
        if st.button("Harvard Talk"):
            sample_path = os.path.join("samples", "harvard.wav")
            sample_name = "harvard.wav"
            
            st.write("Processing sample file...")
            try:
                query_engine, transcripts = process_audio_file(sample_path, sample_name)
                st.session_state.query_engine = query_engine
                st.session_state.transcripts = transcripts
                st.session_state.current_file = sample_name
                st.success("Ready to Chat!")
            except Exception as e:
                st.error(f"Error processing sample: {e}")
    
    st.divider()
    
    # Original file uploader
    uploaded_file = st.file_uploader("Or upload your own audio file", type=["mp3", "wav", "m4a"])
    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                st.write("Transcribing with AssemblyAI and storing in vector database...")
                query_engine, transcripts = process_audio_file(file_path, uploaded_file.name)
                st.session_state.query_engine = query_engine
                st.session_state.current_file = uploaded_file.name
                
                st.success("Ready to Chat!")
                st.audio(uploaded_file)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()
    # Display transcript if available
    if hasattr(st.session_state, 'transcripts') and hasattr(st.session_state, 'current_file'):
        st.subheader(f"Current File: {st.session_state.current_file}")
        with st.expander("Show full transcript", expanded=True):
            for t in st.session_state.transcripts:
                st.text(f"Speaker {t['speaker']}: {t['text']}")
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("""
    # RAG over Audio""") 
with col2:
    st.button("Clear ↺", on_click=reset_chat)
# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Ask about the audio conversation..."):
    if not hasattr(st.session_state, 'query_engine'):
        st.error("Please upload an audio file or select a sample first.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""        
        streaming_response = st.session_state.query_engine.query(prompt)
        
        for chunk in streaming_response:
            try:
                new_text = chunk.raw["choices"][0]["delta"]["content"]
                full_response += new_text
                message_placeholder.markdown(full_response + "▌")
            except:
                pass
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})