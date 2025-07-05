# app.py
"""Streamlit front‑end for le **Synthétiseur de rêve**.

* Upload (ou enregistrement) d'un fichier audio `.wav` / `.mp3`.
* Transcription avec Whisper (Groq) ➜ affichage.
* Détection de l'émotion ➜ affichage + distribution.
* Génération d'une image (Clipdrop) ➜ affichage.
* Enregistrement du rêve dans une base SQLite minimale.
* Tableau de bord historique.
"""

from __future__ import annotations

from tempfile import NamedTemporaryFile
from typing import Dict

import streamlit as st

from backend import (
    classify_emotion,
    generate_image,
    list_dreams,
    save_dream,
    speech_to_text,
)

# ---------------------------------------------------------------------------
# Page config & header
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Synthétiseur de rêve", page_icon="🌙", layout="centered")

st.title("🌙 Synthétiseur de rêve")
st.write(
    "Racontez un rêve à voix haute ; l'application le transcrit, détecte l'émotion, "
    "et génère une illustration onirique. Enfin, tous vos rêves sont conservés "
    "dans un historique personnel."
)

# ---------------------------------------------------------------------------
# Upload / record section
# ---------------------------------------------------------------------------
uploaded_audio = st.file_uploader(
    "📤 Uploader un fichier audio (.wav ou .mp3)", type=["wav", "mp3"], label_visibility="visible"
)

process_btn = st.button("✨ Synthétiser mon rêve", disabled=uploaded_audio is None, use_container_width=True)

if process_btn and uploaded_audio is not None:
    # 1. Sauvegarde du fichier dans un fichier temporaire
    suffix = ".mp3" if uploaded_audio.name.endswith("mp3") else ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_audio.getbuffer())
        tmp_path = tmp.name

    # 2. Pipeline complet
    with st.spinner("📝 Transcription du rêve en cours…"):
        transcription = speech_to_text(tmp_path)

    st.subheader("📝 Transcription")
    st.write(transcription)

    with st.spinner("🔍 Analyse des émotions…"):
        emotion_scores: Dict[str, float] = classify_emotion(transcription)
        main_emotion = max(emotion_scores, key=emotion_scores.get)

    st.subheader("😊 Émotion détectée")
    st.write(f"Le rêve semble **{main_emotion}**.")
    st.caption(f"Distribution: {emotion_scores}")

    with st.spinner("🎨 Génération de l'image…"):
        image_url = generate_image(transcription)

    st.subheader("🎨 Illustration du rêve")
    st.image(image_url)

    # 3. Persist
    save_dream(transcription, emotion_scores, image_url)
    st.success("Rêve enregistré dans votre historique !")

# ---------------------------------------------------------------------------
# Historical dashboard
# ---------------------------------------------------------------------------

st.divider()
st.header("📜 Historique de vos rêves")
all_dreams = list_dreams()

if not all_dreams:
    st.info("Aucun rêve enregistré pour l'instant.")
else:
    for dream in all_dreams:
        label = max(dream["emotion"], key=dream["emotion"].get)
        with st.expander(f"{dream['created_at']} — {label}"):
            st.write(dream["transcription"])
            st.image(dream["image_url"])
            st.caption(f"Distribution émotions: {dream['emotion']}")
