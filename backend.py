# backend.py
"""
Backend du projet "Synthétiseur de rêve"
- Transcription avec Groq Whisper
- Analyse d'émotion avec Mistral
- Génération d'image avec Clipdrop
- Enregistrement dans SQLite (historique)
"""

import os
import json
import math
import sqlite3
from typing import Dict, List

import requests
from dotenv import load_dotenv
from groq import Groq
from mistralai import Mistral

load_dotenv()

# ---------------------------------------------------------------------------
# Transcription audio (Whisper via Groq)
# ---------------------------------------------------------------------------
def speech_to_text(audio_path: str, language: str = "fr") -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            prompt="Transcris le rêve le plus fidèlement possible.",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language=language,
            temperature=0.0
        )
    return transcription.text

# ---------------------------------------------------------------------------
# Analyse des émotions (Mistral LLM JSON output)
# ---------------------------------------------------------------------------
def softmax(preds: Dict[str, float]) -> Dict[str, float]:
    exp_vals = {k: math.exp(v * 10) for k, v in preds.items()}
    total = sum(exp_vals.values())
    return {k: round(v / total, 4) for k, v in exp_vals.items()}

def classify_emotion(text: str) -> Dict[str, float]:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    context = "Tu es un assistant qui détecte les émotions exprimées dans un rêve. Ton résultat est un dictionnaire JSON: { 'heureux': float, 'stressant': float, 'neutre': float }"

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": f"Analyse ce rêve et retourne les probabilités : {text}"},
        ],
        response_format={"type": "json_object"}
    )

    raw = json.loads(response.choices[0].message.content)
    return softmax(raw)

# ---------------------------------------------------------------------------
# Génération d'image (Clipdrop API - Stable Diffusion)
# ---------------------------------------------------------------------------
def generate_image(prompt: str) -> str:
    api_key = os.getenv("CLIPDROP_API_KEY")
    url = "https://clipdrop-api.co/text-to-image/v1"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {"prompt": prompt, "style": "dream"}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["image_url"]

# ---------------------------------------------------------------------------
# Persistence locale (SQLite3)
# ---------------------------------------------------------------------------
DB_FILE = "dreams.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS dreams (
            id INTEGER PRIMARY KEY,
            transcription TEXT,
            emotion TEXT,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

init_db()

def save_dream(transcription: str, emotion: Dict[str, float], image_url: str) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT INTO dreams (transcription, emotion, image_url) VALUES (?, ?, ?)",
            (transcription, json.dumps(emotion), image_url)
        )

def list_dreams() -> List[Dict]:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute("SELECT transcription, emotion, image_url, created_at FROM dreams ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [
            {
                "transcription": row[0],
                "emotion": json.loads(row[1]),
                "image_url": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]

# ---------------------------------------------------------------------------
# Exécution en mode test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    print("\n--- Synthèse du rêve ---")
    txt = speech_to_text(audio_path)
    print(f"\nTexte transcrit :\n{txt}")
    emo = classify_emotion(txt)
    print(f"\nDistribution des émotions : {emo}")
    img = generate_image(txt)
    print(f"\nImage générée : {img}")
    save_dream(txt, emo, img)
    print("\nRêve enregistré dans la base de données.")
