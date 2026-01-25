# 1. Hafif Python İmajı
FROM python:3.11-slim

# 2. Çalışma Dizini
WORKDIR /app

# 3. Gerekli Sistem Kütüphaneleri (Chromadb ve Tokenizers için gerekli)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Kütüphanelerin Yüklenmesi
COPY requirements.txt .
# --no-cache-dir: İmajı küçültmek için önbelleği temizle
RUN pip install --no-cache-dir -r requirements.txt

# 5. Kodların Kopyalanması
COPY . .

# 6. Port Tanımı
EXPOSE 8000

# 7. Başlatma Komutu
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]