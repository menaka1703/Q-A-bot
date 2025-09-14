import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import textwrap
from langchain.llms import Ollama
from langchain import PromptTemplate
import os
import fitz 
import pytesseract
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any
import time


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'document_name' not in st.session_state:
    st.session_state.document_name = None

def load_pdf_data(file_path):
    try:

        loader = PyMuPDFLoader(file_path=file_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No documents could be loaded from the PDF")
        
        
        pdf = fitz.open(file_path)
        
        for doc in docs:
            page_num = doc.metadata['page'] 
            page = pdf[page_num]  
            
            
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]  
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                
                image_path = os.path.join("temp_dir", f"image_{page_num}_{img_index}.{image_ext}")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                try:
                    
                    image_text = pytesseract.image_to_string(image_path, lang='eng')
                    if image_text.strip():  
                        doc.page_content += "\n\n[Image Text]\n" + image_text
                except Exception as e:
                    st.warning(f"OCR failed for image on page {page_num}: {e}")
                finally:
                    
                    if os.path.exists(image_path):
                        os.remove(image_path)
        
        pdf.close()  
        return docs
    
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return None


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents=documents)
    return chunks


def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': normalize_embedding}
    )


def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(storing_path)
    return vectorstore


def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )


def get_streaming_response(query, chain):
    try:
        response = chain({'query': query})
        return response['result'], response.get('source_documents', [])
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None, []


def stream_text(text, delay=0.05):
    """Simulate streaming text word by word"""
    words = text.split()
    for i, word in enumerate(words):
        yield word + " "
        time.sleep(delay)


def add_to_conversation(role, content, sources=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "sources": sources or []
    }
    st.session_state.conversation_history.append(message)


def display_conversation():
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["sources"] and message["role"] == "assistant":
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"][:3]):  
                        st.write(f"**Source {i+1}:**")
                        st.write(f"Page: {source.metadata.get('page', 'Unknown')}")
                        st.write(f"Content: {source.page_content[:200]}...")


def clear_conversation():
    st.session_state.conversation_history = []
    st.rerun()


def export_conversation():
    if not st.session_state.conversation_history:
        return None
    
    
    serializable_conversation = []
    for message in st.session_state.conversation_history:
        serializable_message = {
            "role": message["role"],
            "content": message["content"],
            "timestamp": message["timestamp"],
            "sources": []
        }
        
        
        if message.get("sources"):
            for source in message["sources"]:
                serializable_source = {
                    "page": source.metadata.get('page', 'Unknown'),
                    "content_preview": source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content,
                    "metadata": source.metadata
                }
                serializable_message["sources"].append(serializable_source)
        
        serializable_conversation.append(serializable_message)
    
    export_data = {
        "document_name": st.session_state.document_name,
        "export_timestamp": datetime.now().isoformat(),
        "conversation": serializable_conversation
    }
    return json.dumps(export_data, indent=2)


def get_document_summary(docs):
    if not docs:
        return "No document available for summary."
    
    
    total_chunks = len(docs)
    sample_text = ""
    for doc in docs[:3]:  
        sample_text += doc.page_content[:500] + " "
    
    return f"Document contains {total_chunks} text chunks. Sample content: {sample_text[:300]}..."


def main():
    
    
    with st.sidebar:
        st.markdown("### 📁 Upload Document")
        
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload a PDF file to start chatting with it",
            label_visibility="collapsed"
        )
        
        
        if uploaded_file is not None and not st.session_state.document_processed:
            with st.spinner('🔄 Processing PDF...'):
                try:
                    
                    temp_file_path = os.path.join("temp_dir", uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    
                    docs = load_pdf_data(file_path=temp_file_path)
                    if docs is None:
                        st.error("Failed to load PDF. Please try a different file.")
                        return
                    
                    documents = split_docs(documents=docs, chunk_size=1000, chunk_overlap=20)
                    
                    
                    with st.spinner('Creating embeddings...'):
                        embed = load_embedding_model(model_path="all-MiniLM-L6-v2")
                        vectorstore = create_embeddings(documents, embed)
                        st.session_state.vectorstore = vectorstore

                    
                    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

                    
                    template = """
                    ### System:
                    You are a helpful and knowledgeable assistant. Answer the user's questions using only the context provided from the document. 
                    Be concise but comprehensive. If you don't know the answer based on the context, say so clearly.
                    Maintain context from previous questions in the conversation when relevant.

                    ### Context:
                    {context}

                    ### User Question:
                    {question}

                    ### Response:
                    """
                    prompt = PromptTemplate.from_template(template)

                    
                    llm = Ollama(base_url='http://localhost:11434', model="llama3:8b", temperature=0.0)

                    
                    chain = load_qa_chain(retriever, llm, prompt)
                    st.session_state.chain = chain
                    st.session_state.document_processed = True
                    st.session_state.document_name = uploaded_file.name
                    
                   
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    
                    st.success("✅ PDF processed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    return
        
        st.markdown("---")
        st.markdown("### 🎛️ Controls")
        
        
        if st.button("🗑️ Clear Conversation", help="Clear all chat history"):
            clear_conversation()
        
        
        if st.session_state.conversation_history:
            export_data = export_conversation()
            if export_data:
                st.download_button(
                    label="📥 Export Chat",
                    data=export_data,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        
        if st.session_state.document_processed:
            st.markdown("### 📄 Document Info")
            st.write(f"**File:** {st.session_state.document_name}")
            st.write(f"**Status:** ✅ Processed")
            st.write(f"**Chat Messages:** {len(st.session_state.conversation_history)}")
            
            
            st.markdown("### 🚀 Quick Actions")
            if st.button("📋 Get Summary"):
                with st.spinner('Generating summary...'):
                    summary_prompt = "Provide a brief summary of this document's main topics and key points."
                    summary, _ = get_streaming_response(summary_prompt, st.session_state.chain)
                    if summary:
                        
                        summary_placeholder = st.empty()
                        full_summary = ""
                        for word in stream_text(summary, delay=streaming_speed):
                            full_summary += word
                            summary_placeholder.markdown(full_summary + "▌")
                        summary_placeholder.markdown(full_summary)
            
            if st.button("❓ Sample Questions"):
                sample_questions = [
                    "What is the main topic of this document?",
                    "Can you summarize the key points?",
                    "What are the important details mentioned?",
                    "Are there any specific recommendations or conclusions?"
                ]
                for i, question in enumerate(sample_questions, 1):
                    st.write(f"{i}. {question}")
        
        
        st.markdown("### ⚙️ Settings")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, help="Size of text chunks for processing")
        chunk_overlap = st.slider("Chunk Overlap", 10, 100, 20, help="Overlap between chunks")
        temperature = st.slider("Model Temperature", 0.0, 1.0, 0.0, 0.1, help="Creativity level of responses")
        streaming_speed = st.slider("Streaming Speed", 0.01, 0.2, 0.05, 0.01, help="Speed of word-by-word display")

    
    if not st.session_state.document_processed:
        
        st.markdown("""
        <div class="welcome-container">
            <h2>🤖 Welcome to PDF Chat Assistant</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0; opacity: 0.9;">
                Upload a PDF document in the sidebar to start chatting with it!
            </p>
            <div class="feature-list">
                <h3>✨ Features:</h3>
                <ul style="text-align: left; display: inline-block; list-style: none; padding: 0;">
                    <li style="margin: 0.5rem 0;">📄 Extract text from PDFs and images (OCR)</li>
                    <li style="margin: 0.5rem 0;">💬 Natural conversation with your documents</li>
                    <li style="margin: 0.5rem 0;">🔍 Source citations for every answer</li>
                    <li style="margin: 0.5rem 0;">📥 Export your conversations</li>
                    <li style="margin: 0.5rem 0;">⚙️ Customizable settings</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        
        st.markdown("### 💬 Chat with your document")
        
        
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                
                if message["sources"] and message["role"] == "assistant":
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"][:3]):
                            st.write(f"**Source {i+1}:**")
                            st.write(f"Page: {source.metadata.get('page', 'Unknown')}")
                            st.write(f"Content: {source.page_content[:200]}...")
        
        
        if prompt := st.chat_input("Ask a question about the document..."):
            
            add_to_conversation("user", prompt)
            
            
            with st.chat_message("user"):
                st.write(prompt)
            
            
            with st.chat_message("assistant"):
                try:
                    
                    with st.spinner('🤔 Thinking...'):
                        response, sources = get_streaming_response(prompt, st.session_state.chain)
                    
                    if response:
                        
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        
                        for word in stream_text(response, delay=streaming_speed):
                            full_response += word
                            message_placeholder.markdown(full_response + "▌")
                        
                       
                        message_placeholder.markdown(full_response)
                        
                       
                        add_to_conversation("assistant", full_response, sources)
                        
                        
                        if sources:
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(sources[:3]):
                                    st.write(f"**Source {i+1}:**")
                                    st.write(f"Page: {source.metadata.get('page', 'Unknown')}")
                                    st.write(f"Content: {source.page_content[:200]}...")
                    else:
                        st.error("Failed to generate response. Please try again.")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    
if __name__ == "__main__":
    
    if not os.path.exists("temp_dir"):
        os.makedirs("temp_dir")
        
    main()