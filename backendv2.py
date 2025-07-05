"""
Backend pour le Synthétiseur de rêves
Gère la transcription audio, l'analyse émotionnelle et la génération d'images
"""

from groq import Groq
from mistralai import Mistral
from dotenv import load_dotenv
import os
import json
import math
import requests
from typing import Dict, Tuple, Optional
import base64
from io import BytesIO

load_dotenv()


def read_file(text_file_path: str) -> str:
    """
    Lit le contenu d'un fichier texte.
    
    Args:
        text_file_path (str): Chemin vers le fichier à lire
        
    Returns:
        str: Contenu du fichier
    """
    with open(text_file_path, "r", encoding="utf-8") as file:
        return file.read()


def softmax(predictions: Dict[str, float]) -> Dict[str, float]:
    """
    Applique la fonction softmax aux prédictions pour normaliser les probabilités.
    
    Args:
        predictions (Dict[str, float]): Dictionnaire des prédictions brutes
        
    Returns:
        Dict[str, float]: Probabilités normalisées
    """
    output = {}
    for sentiment, predicted_value in predictions.items():
        output[sentiment] = math.exp(predicted_value * 10) / sum([
            math.exp(value * 10) for value in predictions.values()
        ])
    return output


def speech_to_text(audio_path: str, language: str = "fr") -> str:
    """
    Transcrit un fichier audio en texte en utilisant l'API Groq.
    
    Args:
        audio_path (str): Chemin vers le fichier audio
        language (str): Langue de transcription (défaut: "fr")
        
    Returns:
        str: Texte transcrit
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            prompt="Extrait le texte de l'audio de la manière la plus factuelle possible, en gardant tous les détails du rêve raconté",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language=language,
            temperature=0.0
        )
        
    return transcription.text


def analyze_dream_emotion(text: str) -> Dict[str, float]:
    """
    Analyse l'émotion dominante d'un rêve à partir de sa transcription.
    
    Args:
        text (str): Texte du rêve transcrit
        
    Returns:
        Dict[str, float]: Probabilités des différentes émotions
    """
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    
    system_prompt = """
    Tu es un expert en analyse psychologique des rêves. 
    Analyse l'émotion dominante du rêve décrit et retourne tes prédictions sous format JSON.
    
    Les catégories d'émotions possibles sont :
    - heureux: rêve positif, joyeux, paisible
    - stressant: rêve anxieux, angoissant, cauchemardesque
    - neutre: rêve sans émotion particulière, descriptif
    - mystérieux: rêve étrange, surréaliste, inexpliqué
    - nostalgique: rêve mélancolique, lié au passé
    
    Retourne un JSON avec ces clés et des valeurs entre -1 et 1 représentant l'intensité de chaque émotion.
    """
    
    chat_response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Analyse l'émotion de ce rêve (ta réponse doit être en format JSON) : {text}",
            },
        ],
        response_format={"type": "json_object"}
    )
    
    predictions = json.loads(chat_response.choices[0].message.content)
    return softmax(predictions)


def generate_dream_image_prompt(dream_text: str) -> str:
    """
    Génère un prompt optimisé pour la création d'image basé sur le rêve.
    
    Args:
        dream_text (str): Texte du rêve transcrit
        
    Returns:
        str: Prompt optimisé pour la génération d'image
    """
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    
    system_prompt = """
    Tu es un expert en génération de prompts pour l'art numérique.
    À partir d'un rêve décrit, créé un prompt en anglais optimisé pour générer une image artistique.
    
    Le prompt doit :
    - Capturer l'essence visuelle du rêve
    - Inclure des détails sur l'atmosphère et l'ambiance
    - Utiliser des termes artistiques (style, éclairage, composition)
    - Être concis mais évocateur (maximum 200 mots)
    - Éviter les éléments impossibles à représenter visuellement
    
    Réponds uniquement avec le prompt en anglais, sans explication.
    """
    
    chat_response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Créé un prompt artistique pour ce rêve : {dream_text}",
            },
        ]
    )
    
    return chat_response.choices[0].message.content.strip()


def generate_dream_image(prompt: str) -> Optional[bytes]:
    """
    Génère une image à partir d'un prompt en utilisant l'API Clipdrop.
    
    Args:
        prompt (str): Description textuelle pour la génération d'image
        
    Returns:
        Optional[bytes]: Données de l'image générée ou None si erreur
    """
    try:
        response = requests.post(
            "https://clipdrop-api.co/text-to-image/v1",
            files={
                "prompt": (None, prompt),
            },
            headers={
                "x-api-key": os.getenv("CLIPDROP_API_KEY"),
            },
        )
        
        if response.status_code == 200:
            return response.content
        else:
            print(f"Erreur API Clipdrop: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Erreur lors de la génération d'image: {e}")
        return None


def process_dream(audio_path: str, language: str = "fr") -> Dict:
    """
    Pipeline complet de traitement d'un rêve : transcription, analyse émotionnelle et génération d'image.
    
    Args:
        audio_path (str): Chemin vers le fichier audio du rêve
        language (str): Langue de transcription
        
    Returns:
        Dict: Résultats complets du traitement
    """
    result = {
        "transcription": "",
        "emotions": {},
        "image_prompt": "",
        "image_data": None,
        "dominant_emotion": "",
        "success": False
    }
    
    try:
        # Étape 1: Transcription
        print("🎙️ Transcription audio en cours...")
        result["transcription"] = speech_to_text(audio_path, language)
        
        # Étape 2: Analyse émotionnelle
        print("🎭 Analyse émotionnelle en cours...")
        result["emotions"] = analyze_dream_emotion(result["transcription"])
        result["dominant_emotion"] = max(result["emotions"], key=result["emotions"].get)
        
        # Étape 3: Génération du prompt image
        print("🖼️ Génération du prompt image en cours...")
        result["image_prompt"] = generate_dream_image_prompt(result["transcription"])
        
        # Étape 4: Génération de l'image
        print("🎨 Génération de l'image en cours...")
        result["image_data"] = generate_dream_image(result["image_prompt"])
        
        result["success"] = True
        print("✅ Traitement du rêve terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {e}")
        result["error"] = str(e)
    
    return result


def save_dream_to_history(dream_data: Dict, user_id: str = "default") -> bool:
    """
    Sauvegarde un rêve dans l'historique utilisateur.
    
    Args:
        dream_data (Dict): Données du rêve traité
        user_id (str): Identifiant utilisateur
        
    Returns:
        bool: True si sauvegarde réussie
    """
    try:
        history_file = f"dream_history_{user_id}.json"
        
        # Charger l'historique existant
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
        
        # Ajouter le nouveau rêve avec timestamp
        from datetime import datetime
        dream_entry = {
            "timestamp": datetime.now().isoformat(),
            "transcription": dream_data["transcription"],
            "emotions": dream_data["emotions"],
            "dominant_emotion": dream_data["dominant_emotion"],
            "image_prompt": dream_data["image_prompt"]
        }
        
        history.append(dream_entry)
        
        # Sauvegarder
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        return False


def load_dream_history(user_id: str = "default") -> list:
    """
    Charge l'historique des rêves d'un utilisateur.
    
    Args:
        user_id (str): Identifiant utilisateur
        
    Returns:
        list: Liste des rêves sauvegardés
    """
    try:
        history_file = f"dream_history_{user_id}.json"
        
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return []
            
    except Exception as e:
        print(f"Erreur lors du chargement de l'historique: {e}")
        return []


if __name__ == "__main__":
    # Test du pipeline
    test_audio_path = "test_dream.wav"  # Remplacer par un vrai fichier audio
    
    if os.path.exists(test_audio_path):
        result = process_dream(test_audio_path)
        
        if result["success"]:
            print(f"Transcription: {result['transcription']}")
            print(f"Émotion dominante: {result['dominant_emotion']}")
            print(f"Prompt image: {result['image_prompt']}")
            print(f"Image générée: {'Oui' if result['image_data'] else 'Non'}")
            
            # Sauvegarder dans l'historique
            save_dream_to_history(result)
        else:
            print(f"Erreur: {result.get('error', 'Erreur inconnue')}")
    else:
        print(f"Fichier audio de test non trouvé: {test_audio_path}")