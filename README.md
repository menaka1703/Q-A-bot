# PDF Chat Assistant

A Streamlit-based web application that allows users to upload PDF documents, process them with embeddings for semantic search, and chat with the document content using a local LLM (Ollama). The app supports OCR for extracting text from images within PDFs, maintains conversation history, and provides source citations for responses.

## Features

- **PDF Upload and Processing**: Upload PDF files and automatically extract text, including OCR for embedded images using Tesseract.
- **Semantic Search**: Uses FAISS vector store with Hugging Face embeddings for efficient retrieval.
- **Chat Interface**: Interactive chat with the document, powered by Ollama (Llama3 model by default).
- **Conversation History**: Tracks user-assistant interactions with timestamps and source documents.
- **Streaming Responses**: Simulates word-by-word response generation for a natural feel.
- **Source Citations**: Displays relevant page numbers and content snippets for answers.
- **Export Chat**: Download conversation history as JSON.
- **Quick Actions**: Buttons for document summary and sample questions.
- **Customizable Settings**: Adjust chunk size, overlap, model temperature, and streaming speed via sliders.

## Requirements

- Python 3.8+
- Tesseract OCR installed (path hardcoded to `C:\Program Files\Tesseract-OCR\tesseract.exe` – adjust as needed).
- Ollama running locally at `http://localhost:11434` with the `llama3:8b` model pulled.

### Python Dependencies

Install via `pip`:
```
streamlit
langchain
faiss-cpu
huggingface-hub
pymupdf
pytesseract
fitz
ollama
```

Full list from imports:
- streamlit
- langchain (including embeddings, document_loaders, text_splitter, vectorstores, chains, llms, PromptTemplate)
- textwrap
- os
- fitz (PyMuPDF)
- pytesseract
- hashlib (unused in code, but imported)
- json
- datetime
- typing
- time

## Installation

1. Clone or download the repository/script.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Create `requirements.txt` with the list above if needed.)
3. Ensure Tesseract is installed and update the path in the code if necessary.
4. Start Ollama server: Run `ollama serve` and pull the model with `ollama pull llama3:8b`.
5. Create a `temp_dir` folder in the project root (or let the script create it).

## Usage

1. Run the app:
   ```
   streamlit run pdf_chatbot.py
   ```
2. Open the app in your browser (usually `http://localhost:8501`).
3. Upload a PDF via the sidebar.
4. Once processed, ask questions in the chat input.
5. Use sidebar controls to clear chat, export history, get summaries, or adjust settings.

## Configuration

- **Embedding Model**: Defaults to `all-MiniLM-L6-v2`. Change in `load_embedding_model`.
- **LLM**: Uses Ollama with `llama3:8b`. Modify in `main()` under `llm = Ollama(...)`.
- **Chunking**: Adjustable via sliders (default: size=1000, overlap=20).
- **Retrieval**: MMR search with k=5, fetch_k=20.
- **Prompt Template**: Custom system prompt for context-aware, concise responses.

## Limitations

- Runs on CPU for embeddings (change to GPU if available).
- No internet access required, but Ollama must be local.
- OCR may fail on complex images; warnings are shown.
- Conversation history is session-based (clears on refresh unless exported).

## Troubleshooting

- **PDF Loading Errors**: Ensure PyMuPDF is installed correctly.
- **OCR Issues**: Verify Tesseract path and installation.
- **Ollama Connection**: Confirm server is running at the specified URL.
- **Memory Usage**: Large PDFs may require more RAM for embeddings.

## License

MIT License (or specify your own).