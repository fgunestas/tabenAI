import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# 1. API Anahtarınızı buraya yapıştırın
GOOGLE_API_KEY = GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

print("Erişilebilir Modeller Listeleniyor...\n")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Hata oluştu: {e}")
    print("\nOlası Neden: API anahtarı hatalı veya 'Generative Language API' projede etkin değil.")