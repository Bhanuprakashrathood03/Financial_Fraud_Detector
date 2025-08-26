# Interactive Financial Fraud Detector
![image alt](https://github.com/Bhanuprakashrathood03/Financial_Fraud_Detector/blob/56bec6820794d15d814886976dcca9568bbe431f/app-screenshot.jpg)
An interactive tool that leverages a Large Language Model (LLM) with **Retrieval-Augmented Generation (RAG)** to analyze financial text and classify its potential for fraud. This project uses the `HuggingFaceH4/zephyr-7b-beta` model to provide a risk assessment with justification.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-green.svg)](https://www.langchain.com/)

---
## ## Features

* **LLM-Powered Analysis**: Utilizes the 7-billion parameter Zephyr-7B-Beta model for nuanced text analysis.
* **RAG for Context**: Enhances accuracy by retrieving relevant examples from a vector database before making a prediction, reducing hallucinations.
* **Interactive UI**: Includes an intuitive Streamlit interface for easy text submission and analysis.
* **Command-Line Interface**: Retains the original CLI for backend testing and scripted use.
* **Persistent Knowledge Base**: Uses ChromaDB to create and store a vector database of financial statements on first run.

---
## ## How It Works

The system follows a Retrieval-Augmented Generation (RAG) pipeline to ensure accurate, context-aware analysis:

1.  **Data Ingestion**: On the first run, the system generates a synthetic dataset of fraudulent and non-fraudulent financial statements from `fraud_detector_app.py`.
2.  **Vector Store Creation**: This data is cleaned, processed, and embedded into a **ChromaDB** vector store. This vector store acts as the system's specialized knowledge base.
3.  **User Query**: A user submits a piece of financial text for analysis.
4.  **Document Retrieval**: The system searches the ChromaDB store to find the most semantically similar financial statement (the "context").
5.  **LLM Prompting**: The user's text (as the "question") and the retrieved document (as the "context") are passed to the Zephyr-7B LLM with a specialized prompt, asking it to classify the fraud risk and provide justification.

---
## ## Technology Stack

* **LLM & NLP**: LangChain, Hugging Face Transformers, PyTorch, NLTK
* **Vector Database**: ChromaDB
* **Web Framework**: Streamlit
* **Core Libraries**: Pandas, NumPy, Scikit-learn

---
## ## Getting Started

### ### Prerequisites

* Python 3.9 or higher
* Git
* An NVIDIA GPU with CUDA support is **highly recommended** for running the model locally.
* A Hugging Face account and an [Access Token](https://huggingface.co/settings/tokens) with at least `read` permissions.

### ### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Set up your Hugging Face Token:**
    For seamless authentication, create a file named `.env` in the project's root directory and add your token:
    ```
    HUGGING_FACE_HUB_TOKEN="hf_YourAccessTokenHere"
    ```
    The `transformers` library will automatically detect and use this token.

3.  **Create a virtual environment and install dependencies:**
    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate it (Linux/macOS)
    source venv/bin/activate
    # Or on Windows
    # venv\Scripts\activate

    # Install all required packages
    pip install -r requirements.txt
    ```

4.  **First Run**: The first time you run the application, it will automatically download the necessary NLTK data and create the ChromaDB vector store in the `docs/chroma_rag/` directory. This may take a few minutes.

### ### Usage

You can run the application in two ways:

1.  **With the Streamlit Web Interface (Recommended):**
    ```bash
    streamlit run your_streamlit_app_name.py
    ```
    Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

2.  **With the Command-Line Interface:**
    ```bash
    python fraud_detector_app.py
    ```
    Follow the prompts in your terminal to enter text for analysis.

---
## ## File Structure

```
.
├── 📂 docs/chroma_rag/      # Persisted ChromaDB vector store
├── 🐍 fraud_detector_app.py # Core logic for RAG pipeline and CLI
├──  streamlit_app.py      # (Your Streamlit UI file)
├── 📋 requirements.txt      # Python dependencies
├── 📄 .env                  # For storing your Hugging Face token
└──  README.md               # This file
```

---
## ## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
