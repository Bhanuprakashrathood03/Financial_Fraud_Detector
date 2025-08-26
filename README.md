# Financial_Fraud_Detector
This AI tool uses a powerful Zephyr-7B language model and Retrieval-Augmented Generation (RAG) to detect potential financial fraud. Paste any financial statement, and it will instantly classify the fraud risk as High, Low, or Indeterminate.
# Interactive Financial Fraud Detector

## **This is how it looks initially in the Web Browser.**
<img width="1440" height="819" alt="Screenshot 2025-08-17 at 3 22 24 PM" src="https://github.com/user-attachments/assets/4983fb9c-e4c8-4acc-aa7f-e2236e364a3c" />


An interactive tool that leverages a Large Language Model (LLM) with Retrieval-Augmented Generation (RAG) to analyze financial text and classify its potential for fraud. This project uses the `HuggingFaceH4/zephyr-7b-beta` model to provide risk assessment.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)

---

## 📋 Features

* **LLM-Powered Analysis**: Uses a 7-billion parameter language model for nuanced financial text analysis.
* **RAG for Context**: Enhances accuracy by retrieving relevant examples from a vector database before making a prediction.
* **Interactive UI**: A user-friendly Streamlit interface for easy text submission and analysis.
* **Example Loading**: Quickly load examples of suspicious and non-suspicious text to see the tool in action.
* **Command-Line Interface**: Retains the original CLI for backend testing and scripted use.

---

## ⚙️ How It Works

The system follows a Retrieval-Augmented Generation (RAG) pipeline:

1.  **Synthetic Data**: On first run, a synthetic dataset of fraudulent and non-fraudulent financial statements is generated.
2.  **Vector Store Creation**: This data is cleaned, processed, and stored in a `ChromaDB` vector store using `HuggingFaceEmbeddings`. This vector store acts as the long-term memory for the RAG system.
3.  **User Query**: A user submits a piece of financial text through the UI.
4.  **Document Retrieval**: The system searches the ChromaDB store to find the most semantically similar financial statement (the "context").
5.  **LLM Prompting**: The user's text (as the "question") and the retrieved document (as the "context") are passed to the Zephyr-7B LLM using a specialized prompt.
6.  **Classification & Justification**: The LLM analyzes the information and classifies the text as "High Risk," "Low Risk," or "Indeterminate Risk," providing a brief justification for its decision based on the provided context.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9 or higher
* An NVIDIA GPU with CUDA support is highly recommended for running the model locally.
* A Hugging Face account and an [Access Token](https://huggingface.co/settings/tokens) with at least `read` permissions.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Set up your Hugging Face Token:**
    Create a file named `.env` in the root of the project directory and add your token to it:
    ```
    HUGGING_FACE_HUB_TOKEN="hf_YourAccessTokenHere"
    ```
    The application will load this token automatically.

3.  **Run the setup script:**
    This script will create a virtual environment, install all required dependencies, and download necessary NLTK data.
    ```bash
    bash setup.sh
    ```

### Usage

Once the installation is complete, you can run the application in two ways:

1.  **With the Streamlit Web Interface (Recommended):**
    ```bash
    streamlit run streamlit_app.py
    ```
    Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.

2.  **With the Command-Line Interface:**
    ```bash
    python fraud_detector_app.py
    ```

---

## 📂 File Structure

```
.
├── docs/chroma_rag/      # Directory for the persisted ChromaDB vector store
├── .env                  # For storing your Hugging Face API token
├── fraud_detector_app.py # Core logic for data processing, RAG pipeline, and CLI
├── streamlit_app.py      # The Streamlit user interface
├── requirements.txt      # Python dependencies
├── setup.sh              # Bash script for easy setup
└── README.md             # This file
```

---

## 🛠️ Technology Stack

* **LLM & NLP**: `LangChain`, `Hugging Face Transformers`, `PyTorch`
* **Vector Database**: `ChromaDB`, `FAISS`
* **Web Framework**: `Streamlit`
* **Data Handling**: `Pandas`, `NumPy`

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
