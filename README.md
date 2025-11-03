# 🎬 Movie Plot RAG System

A **production-ready Retrieval-Augmented Generation (RAG)** system for answering questions about movie plots. This project combines **vector search** with **large language models** to generate accurate, context-aware answers about movies.

---

## ✨ Features

* 🧠 **Smart Retrieval** – FAISS vector store with semantic search
* 🤖 **AI-Powered Answers** – Gemini 2.0 Flash integration
* 💾 **Persistent Storage** – Saves vector database for faster re-runs
* 💬 **Interactive Queries** – Command-line interface for natural language input
* 📦 **Structured Output** – JSON results with answers, contexts, and reasoning
* 🎥 **Modern Movie Data** – 400+ curated movies with rich plot descriptions

---

## 🚀 Quick Start

### Prerequisites

* Python **3.11.9+**
* **Gemini API key** (get one free at [Google AI Studio](https://aistudio.google.com/app/apikey))

---

### Installation

1. **Clone and setup**

   ```bash
   git clone <repository-url>
   cd movie-rag-system
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment (optional)**

   ```bash
   cp .env.template .env
   # Edit .env and add your Gemini API key
   ```

---

### Usage

Run the interactive system:

```bash
python run_movie_rag.py
```

The system will:

1. Prompt for your **Gemini API key** (if not found)
2. Load or build the **vector database**
3. Ask for your **movie-related questions**
4. Generate detailed answers with reasoning
5. Save results as **JSON files**

---

### Example Queries

* “Which movies feature artificial intelligence?”
* “Find sci-fi movies from the 1990s.”
* “Movies with prison escape plots.”
* “Romantic comedies from the 2000s.”

---

## 📁 Project Structure

```
movie-rag-system/
├── run_movie_rag.py          # Main entry point
├── requirements.txt          # Python dependencies
├── .env.template             # Environment template
├── data/
│   ├── sample_movies.json    # Filtered movie dataset
│   └── vector_store/         # Persistent FAISS database
├── results/                  # Query results (JSON)
├── src/                      # Core system modules
│   ├── config.py             # Configuration management
│   ├── data_loader.py        # Data loading & preprocessing
│   ├── chunker.py            # Smart text chunking
│   ├── vector_store.py       # FAISS vector operations
│   └── rag_system.py         # Main RAG pipeline
└── examples/
    └── test_real_queries.py
```

---

## 🔧 Configuration

Key settings in `src/config.py`:

| Setting         | Description                  | Default                |
| --------------- | ---------------------------- | ---------------------- |
| `MAX_DOCUMENTS` | Number of movies to process  | 400                    |
| `CHUNK_SIZE`    | Words per text chunk         | 300                    |
| `TOP_K`         | Number of chunks to retrieve | 3                      |
| `GEMINI_MODEL`  | Gemini model version         | `gemini-2.0-flash-exp` |

---

## 📊 How It Works

1. **Data Loading** – Loads 400 modern movies with detailed plots
2. **Smart Chunking** – Splits plots into semantically meaningful text chunks
3. **Vector Embedding** – Uses Sentence Transformers to generate embeddings
4. **Similarity Search** – Retrieves top chunks via FAISS similarity search
5. **Answer Generation** – Gemini LLM synthesizes final answers
6. **Structured Output** – Returns clean JSON with answer, context, and reasoning

---

## 🎯 Example Output

```json
{
  "answer": "Based on the movie plots, 'The Matrix' and 'Ex Machina' feature artificial intelligence...",
  "contexts": [
    "Movie: The Matrix. Year: 1999. Genres: Action, Sci-Fi. Plot: A computer hacker discovers...",
    "Movie: Ex Machina. Year: 2014. Genres: Drama, Sci-Fi. Plot: A programmer is invited to administer..."
  ],
  "reasoning": "The question was about artificial intelligence. I searched through movie plots and found relevant information from 3 chunks. The most relevant movies were: The Matrix, Ex Machina, Her. I used these plot details to form a comprehensive answer."
}
```

---

## 🔍 Advanced Usage

Run queries programmatically:

```python
from src.rag_system import MovieRAGSystem
from src.config import Config

config = Config()
rag_system = MovieRAGSystem(config)

response = rag_system.query("Which movies won Best Picture Oscar?")
print(response["answer"])
```

Analyze the dataset:

```bash
python analyze_filtered_data.py
```

---

## 🛠️ Development

The system is modular and extensible:

* Add new data sources → `data_loader.py`
* Modify chunking strategy → `chunker.py`
* Change vector DB → `vector_store.py`
* Integrate different LLMs → `rag_system.py`

---

## 🧩 Updated Requirements

**requirements.txt**

```
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
google-generativeai>=0.3.0
python-dotenv>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
```

---

## 🚀 Running the Production System

Once everything is set up, just run:

```bash
python run_movie_rag.py
```

The system will:

* Check for API key (prompt if missing)
* Load existing or build new vector store
* Accept natural language queries
* Generate structured, high-quality answers

This is now a **fully production-ready RAG system**! 🎉

---

## 📝 License

This project is provided **for educational purposes** as part of a take-home assignment.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

**Happy movie exploring! 🍿**
