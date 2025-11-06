<h1 align="center">🍽️ Restaurant Recommendation RAG API</h1> <p align="center"> A dynamically updatable Restaurant Review RAG system powered by a local Llama 3 8B and FastAPI.

It answers user questions and allows new reviews to be added to the database instantly. </p>

<p align="center"> <img src="https://img.shields.io/badge/FastAPI-0.100.x-009688?logo=fastapi"> <img src="https://img.shields.io/badge/LangChain-MapReduce-blue?logo=python"> <img src="https://img.shields.io/badge/LLM-Llama%203%208B%20(4--bit)-success?logo=meta"> <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-blueviolet"> <img src="https://img.shields.io/badge/Embeddings-SBERT%20-yellow"> <img src="https://img.shields.io/badge/API%20Docs-SwaggerUI-green?logo=swagger"> </p>

## 🚀 Features

🧠 Local LLM: Runs the `meta-llama/Meta-Llama-3-8B-Instruct` model on VRAM with 4-bit quantization via `BitsAndBytesConfig`.

⚙️ Smart RAG Strategy: Uses `MapReduceDocumentsChain` to process a large number of reviews. It utilizes two separate pipelines for the `Map` (short token limit) and `Reduce` (long token limit) steps to prevent VRAM overflows and "broken record" (repetition) issues.

💾 Persistent Vector Storage: Uses `ChromaDB` to permanently store review vectors in the `./chroma_store` directory.

🔄 Dynamic Database: Supports adding new user reviews to the system instantly via the `POST /add_review` endpoint. These reviews are immediately included in RAG queries.

📖 Automatic API Interface: Can be interactively tested and used via the Swagger UI provided by FastAPI at `http://127.0.0.1:8000/docs`.

🎯 Purpose

To create an end-to-end API service that can generate intelligent answers from a dynamically growing dataset, such as user reviews, using a powerful local language model (Llama 3). This project aims to solve real-world challenges like VRAM management (CPU offloading), model caching, and persistent data storage.

🚧 Setup and Running

### 1️⃣ Requirements

    Python (>=3.10)

    NVIDIA GPU: Sufficient VRAM to run the Llama 3 8B 4-bit model (approx. 10-12 GB recommended).

    CUDA Toolkit

### 2️⃣ Project Setup

Clone the project into the target directory.
```bash
git clone https://github.com/fgunestas/RAG_Project.git
```
Install all required libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```
This command will read the CSV file in your `data/` folder and create the `./chroma_store` directory.

### 3️⃣ Start the API Server

Run the API server directly with the python command to load the heavy models and "warm up" the system.
```bash
python server.py
```

You will see logs in the terminal indicating that the models are loading `(Loading checkpoint shards...)` and the server is starting. The server will be ready for requests once the "warm-up" step is complete:

(Note: The first launch may take a few minutes, depending on the download and loading of the models.)

### 4️⃣ Using the API Interface

While the server is running, open your browser and go to the following address to access the interactive API documentation (Swagger UI):

http://127.0.0.1:8000/docs

You can easily test the `/query` and `/add_review` endpoints via this interface.

## API Usage

The API server runs locally at http://127.0.0.1:8000. It accepts data in `application/json` format.

### 1. RAG Sorgusu Yapma (Soru Sorma)
```python
url = 'http://127.0.0.1:8000/query/'
```

```python
data = {
    'query': "Beyoğlunda çerkez tavuğu porsiyonu büyük olan yerler"
}
headers = {
    'Content-Type': 'application/json'
}
```

#### Sending the API Request
```Python
response = requests.post(url, files=sample,data=data)
```

#### Input parameters

| Parametre   | Tip      | Açıklama                                              |
|:------------|:---------|:------------------------------------------------------|
| `query`     | `string` | The prompt entered to get information about restaurants. |

#### API Output

| Parametre   | Tip      | Açıklama                      |
|:------------|:---------|:------------------------------|
| `query`     | `string` | The response generated for the input prompt. |

### Adding a New Review to the System
```python
url = http://127.0.0.1:8000/add_review/
```
```python
data = {
  "restaurant_name": "Fıccın Restoran",
  "review_text": "Bu yepyeni bir test yorumu. Çerkez tavuğu müthişti ve porsiyonu büyüktü.",
  "location": "41.032,41.032"
}

headers = {
    'Content-Type': 'application/json'
}
```
#### Sending the API Request
```Python
requests.post(url, data=json.dumps(data), headers=headers)
```

#### Input parameters

| Parametre         | Tip                  | Açıklama                                              |
|:------------------|:---------------------|:------------------------------------------------------|
| `restaurant_name` | `string`             | Name of the restaurant.                                       |
| `review_text`     | `string`             | The review written about the restaurant.                      |
| `location`        | `string (Opsiyonel)` | Latitude and longitude information indicating the restaurant's location. |


