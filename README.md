<h1 align="center">🍽️ TabenAI - Restaurant Recommendation RAG System</h1>

<p align="center">
A modern, high-performance Restaurant Review RAG system powered by Google Gemini and LangChain LCEL.
<br/>
Provides intelligent dining recommendations by analyzing real Google Maps reviews with advanced semantic search.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-0.100.x-009688?logo=fastapi">
  <img src="https://img.shields.io/badge/LangChain-LCEL-blue?logo=python">
  <img src="https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-orange?logo=google">
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-blueviolet">
  <img src="https://img.shields.io/badge/Embeddings-BAAI%20BGE--M3-red">
  <img src="https://img.shields.io/badge/API%20Docs-SwaggerUI-green?logo=swagger">
</p>

---

## 🚀 Features

⚡ **Lightning-Fast Inference:** Powered by Google's `gemini-2.5-flash` with massive 2M token context window for instant responses.

💎 **SOTA Embeddings:** Uses **`BAAI/bge-m3`** - state-of-the-art multilingual embeddings with superior Turkish language support and 8192 token context.

🎯 **Modern RAG Architecture:** Built with **LangChain Expression Language (LCEL)** for clean, maintainable pipelines. Context Stuffing strategy leverages Gemini's huge context window for single-pass processing.

🔍 **Smart Retrieval:** MMR (Maximal Marginal Relevance) algorithm ensures diverse, relevant results. Retrieves top 50 reviews with intelligent restaurant-level grouping.

📝 **Contextual Chunking:** Each review is prefixed with restaurant metadata (`[Restoran adı: X]`) for enhanced semantic understanding.

💾 **Persistent Vector Storage:** ChromaDB-powered vector store with automatic deduplication and efficient batch processing.

🔄 **Dynamic Updates:** Add new reviews instantly via API - immediately available in RAG queries.

📖 **Interactive API:** Full Swagger UI documentation at `http://127.0.0.1:8000/docs` for easy testing.

---

## 🎯 Purpose

TabenAI revolutionizes restaurant discovery by understanding **nuanced dining preferences** through advanced RAG technology. 

Instead of simple keyword matching, it comprehends complex queries like *"Beşiktaş'ta sessiz sakin tavuğu güzel bir yer"* by:

1. **Semantic Search:** `BAAI/bge-m3` embeddings capture deep meaning in Turkish
2. **Context-Aware Retrieval:** MMR algorithm selects 50 most relevant, diverse reviews
3. **Intelligent Analysis:** Gemini 2.5 Flash synthesizes insights with direct quote attribution
4. **Personalized Recommendations:** Compares multiple restaurants with evidence-based reasoning

This hybrid approach combines **local SOTA embeddings** (privacy + precision) with **cloud LLM** (speed + intelligence) for unmatched recommendation quality.

---

## 🏗️ Architecture

### Pipeline Flow
```
User Query
    ↓
[Embedding: BAAI/bge-m3]
    ↓
[Vector Search: ChromaDB + MMR]
    ↓
[Retrieve: Top 50 Reviews]
    ↓
[Group by Restaurant]
    ↓
[Format Context String]
    ↓
[LCEL Chain: Prompt → Gemini 2.5 Flash]
    ↓
Personalized Recommendation
```

### Key Components

- **`vector_store.py`**: Database initialization, contextual chunking, batch embedding
- **`retriever.py`**: MMR-based retrieval, restaurant grouping, context formatting
- **`rag_pipeline.py`**: LCEL chain definition, Gemini integration, streaming support
- **`server.py`**: FastAPI endpoints for query and review addition
- **`main.py`**: Test suite with detailed logging

---

## 🚧 Setup and Running

### 1️⃣ Requirements

- Python (>=3.10)
- CUDA Toolkit (Optional - for GPU-accelerated embeddings)
- Google API Key (for Gemini)

### 2️⃣ Installation

Clone the repository:
```bash
git clone https://github.com/fgunestas/tabenAI.git
cd tabenAI
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Set up environment variables:
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### 3️⃣ Initialize Database

Build vector database from CSV data:
```bash
python -c "from components.vector_store import main; main()"
```

Or simply run the test suite (auto-creates DB if missing):
```bash
python main.py
```

### 4️⃣ Start API Server

Launch FastAPI server:
```bash
python server.py
```

Server will be available at: `http://127.0.0.1:8000`

### 5️⃣ Interactive Testing

Open browser and navigate to:
```
http://127.0.0.1:8000/docs
```

Test the `/query` and `/add_review` endpoints via Swagger UI.

---

## 📡 API Usage

The API server runs locally at `http://127.0.0.1:8000`. All endpoints accept `application/json` format.

### 1. Query Endpoint - Get Restaurant Recommendations

**Endpoint:** `POST /query/`

Send intelligent queries to get personalized restaurant recommendations based on review analysis.

#### Example Request
```python
import requests

url = 'http://127.0.0.1:8000/query/'

data = {
    'query': "Beşiktaş'ta sessiz sakin tavuğu güzel bir yer var mı?"
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

#### Input Parameters

| Parameter | Type     | Description                                              |
|:----------|:---------|:---------------------------------------------------------|
| `query`   | `string` | Natural language question about restaurants (Turkish/English) |

#### Response Format

```json
{
  "output_text": "Beşiktaş'ta tavuk menüleri konusunda güzel seçenekler var..."
}
```

**Response includes:**
- Personalized restaurant recommendations
- Direct quotes from reviews
- Comparative analysis when multiple options exist
- Evidence-based reasoning

---

### 2. Add Review Endpoint - Expand the Database

**Endpoint:** `POST /add_review/`

Dynamically add new reviews to the system. Updates are immediately available in RAG queries.

#### Example Request
```python
import requests
import json

url = 'http://127.0.0.1:8000/add_review/'

data = {
    "restaurant_name": "Fıccın Restoran",
    "review_text": "Çerkez tavuğu muhteşemdi ve porsiyonu çok büyüktü. Sessiz bir ortamda rahat yemek yedik.",
    "location": "41.0432,29.0096"  # Optional: latitude,longitude
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

#### Input Parameters

| Parameter         | Type                | Required | Description                                              |
|:------------------|:--------------------|:---------|:---------------------------------------------------------|
| `restaurant_name` | `string`            | ✅ Yes   | Name of the restaurant                                    |
| `review_text`     | `string`            | ✅ Yes   | Review content to be added                                |
| `location`        | `string`            | ⬜ No    | Coordinates in "lat,lon" format (e.g., "41.0432,29.0096") |

#### Response Format

```json
{
  "status": "success",
  "message": "Review added successfully"
}
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
python main.py
```

**Test output includes:**
1. ✓ Database initialization status
2. ✓ Number of retrieved reviews
3. ✓ Preview of top 3 reviews
4. ✓ Full Gemini response with recommendations

**Example test query:** *"Beşiktaş'ta sessiz sakin tavuğu güzel bir yer var mı?"*

---

## 📊 Technical Details

### Embedding Model
- **Model:** `BAAI/bge-m3`
- **Dimensions:** 1024
- **Max Context:** 8192 tokens
- **Features:** Multilingual, dense/sparse hybrid retrieval

### LLM Configuration
- **Model:** `gemini-2.5-flash`
- **Temperature:** 0.1 (focused, deterministic responses)
- **Context Window:** 2M tokens
- **Provider:** Google AI

### Retrieval Strategy
- **Algorithm:** MMR (Maximal Marginal Relevance)
- **Top-K:** 50 reviews
- **Fetch-K:** 150 candidates (for diversity)
- **Grouping:** Restaurant-level aggregation

### Data Format
- **Chunk Prefix:** `[Restoran adı: {name}] {review_text}`
- **Metadata:** `{restaurant, lat, lon}`
- **Deduplication:** SHA-1 hash per review

---

## 🔧 Configuration

Key settings in respective files:

**`components/vector_store.py`:**
```python
EMBED_MODEL = "BAAI/bge-m3"
DB_PERSIST_DIRECTORY = "./chroma_store"
COLLECTION_NAME = "reviews"
```

**`components/rag_pipeline.py`:**
```python
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.1
RETRIEVAL_K = 50
```

**`components/retriever.py`:**
```python
search_type = "mmr"
k = 50  # Final results
fetch_k = 150  # MMR candidates
```

---

## 🗂️ Project Structure

```
tabenAI/
├── components/
│   ├── vector_store.py      # DB initialization & contextual chunking
│   ├── retriever.py          # MMR retrieval & restaurant grouping
│   ├── rag_pipeline.py       # LCEL chain & Gemini integration
│   └── add_review.py         # Dynamic review addition
├── data/
│   └── besiktas_reviews_serpapi_part_full.csv
├── chroma_store/             # Persistent vector database
├── main.py                   # Test suite
├── server.py                 # FastAPI application
├── requirements.txt
├── .env                      # GOOGLE_API_KEY
└── README.md
```

---

## 📝 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

<p align="center">Made with ❤️ using LangChain, Gemini, and ChromaDB</p>


