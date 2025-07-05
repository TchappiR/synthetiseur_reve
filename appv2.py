"""
Application Streamlit pour le Synthétiseur de rêves
Interface utilisateur pour enregistrer, transcrire et visualiser les rêves
"""

import streamlit as st
import tempfile
import os
from datetime import datetime
import json
from PIL import Image
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backendv2 import process_dream, save_dream_to_history, load_dream_history
import base64
import uuid

# Configuration de la page
st.set_page_config(
    page_title="🌙 Synthétiseur de rêves",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #4A90E2;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .dream-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .emotion-heureux { background-color: #4CAF50; }
    .emotion-stressant { background-color: #F44336; }
    .emotion-neutre { background-color: #9E9E9E; }
    .emotion-mystérieux { background-color: #9C27B0; }
    .emotion-nostalgique { background-color: #FF9800; }
</style>
""", unsafe_allow_html=True)

# Initialisation de l'état de session
if 'dream_history' not in st.session_state:
    st.session_state.dream_history = load_dream_history()

if 'current_dream' not in st.session_state:
    st.session_state.current_dream = None

def get_emotion_color(emotion: str) -> str:
    """Retourne la couleur CSS associée à une émotion."""
    colors = {
        'heureux': '#4CAF50',
        'stressant': '#F44336',
        'neutre': '#9E9E9E',
        'mystérieux': '#9C27B0',
        'nostalgique': '#FF9800'
    }
    return colors.get(emotion, '#6C757D')

def display_emotion_badges(emotions: dict):
    """Affiche les badges d'émotions avec leurs probabilités."""
    cols = st.columns(len(emotions))
    
    for i, (emotion, probability) in enumerate(emotions.items()):
        with cols[i]:
            color = get_emotion_color(emotion)
            st.markdown(f"""
                <div style="
                    background-color: {color};
                    color: white;
                    padding: 0.5rem;
                    border-radius: 10px;
                    text-align: center;
                    margin: 0.25rem;
                ">
                    <strong>{emotion.capitalize()}</strong><br>
                    {probability:.1%}
                </div>
            """, unsafe_allow_html=True)

def create_emotion_chart(emotions: dict):
    """Crée un graphique en barres des émotions."""
    df = pd.DataFrame(list(emotions.items()), columns=['Émotion', 'Probabilité'])
    
    fig = px.bar(
        df, 
        x='Émotion', 
        y='Probabilité',
        title="Analyse émotionnelle du rêve",
        color='Émotion',
        color_discrete_map={
            'heureux': '#4CAF50',
            'stressant': '#F44336',
            'neutre': '#9E9E9E',
            'mystérieux': '#9C27B0',
            'nostalgique': '#FF9800'
        }
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Émotions",
        yaxis_title="Probabilité",
        yaxis=dict(tickformat='.1%')
    )
    
    return fig

def create_emotion_timeline():
    """Crée un graphique temporel des émotions."""
    history = st.session_state.dream_history
    
    if not history:
        return None
    
    # Préparer les données
    dates = []
    emotions = []
    probabilities = []
    
    for dream in history:
        date = datetime.fromisoformat(dream['timestamp']).date()
        dominant_emotion = dream['dominant_emotion']
        probability = dream['emotions'].get(dominant_emotion, 0)
        
        dates.append(date)
        emotions.append(dominant_emotion)
        probabilities.append(probability)
    
    df = pd.DataFrame({
        'Date': dates,
        'Émotion': emotions,
        'Probabilité': probabilities
    })
    
    fig = px.scatter(
        df,
        x='Date',
        y='Émotion',
        size='Probabilité',
        color='Émotion',
        title="Évolution des émotions dans vos rêves",
        color_discrete_map={
            'heureux': '#4CAF50',
            'stressant': '#F44336',
            'neutre': '#9E9E9E',
            'mystérieux': '#9C27B0',
            'nostalgique': '#FF9800'
        }
    )
    
    return fig

# Interface principale
def main():
    # En-tête
    st.markdown('<h1 class="main-header">🌙 Synthétiseur de rêves</h1>', unsafe_allow_html=True)
    st.markdown("*Transformez vos rêves en images et découvrez leurs émotions*")
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choisissez une page", ["🎙️ Nouveau rêve", "📊 Tableau de bord", "📚 Historique"])
    
    if page == "🎙️ Nouveau rêve":
        new_dream_page()
    elif page == "📊 Tableau de bord":
        dashboard_page()
    elif page == "📚 Historique":
        history_page()

def new_dream_page():
    """Page pour enregistrer et analyser un nouveau rêve."""
    st.header("🎙️ Racontez votre rêve")
    
    # Options d'upload
    upload_method = st.radio("Comment souhaitez-vous partager votre rêve ?",
                           ["📁 Uploader un fichier audio", "🎤 Enregistrer maintenant"])
    
    audio_file = None
    
    if upload_method == "📁 Uploader un fichier audio":
        audio_file = st.file_uploader(
            "Choisissez un fichier audio", 
            type=['wav', 'mp3', 'm4a', 'ogg'],
            help="Formats supportés: WAV, MP3, M4A, OGG"
        )
    
    else:  # Enregistrement direct
        st.info("🎤 Fonctionnalité d'enregistrement à implémenter avec st-audio-recorder")
        st.markdown("En attendant, utilisez l'option 'Uploader un fichier audio'")
    
    # Paramètres avancés
    with st.expander("Paramètres avancés"):
        language = st.selectbox("Langue de transcription", ["fr", "en", "es", "de"], index=0)
        save_to_history = st.checkbox("Sauvegarder dans l'historique", value=True)
    
    # Bouton de traitement
    if st.button("🔮 Synthétiser le rêve", disabled=audio_file is None):
        if audio_file is not None:
            with st.spinner("🌟 Magie en cours... Analysing your dream..."):
                # Sauvegarder le fichier temporairement
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Traiter le rêve
                    result = process_dream(tmp_file_path, language)
                    
                    if result["success"]:
                        st.session_state.current_dream = result
                        
                        # Sauvegarder dans l'historique si demandé
                        if save_to_history:
                            save_dream_to_history(result)
                            st.session_state.dream_history = load_dream_history()
                        
                        st.success("✨ Rêve synthétisé avec succès!")
                        display_dream_result(result)
                    else:
                        st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")
                
                finally:
                    # Nettoyer le fichier temporaire
                    os.unlink(tmp_file_path)
    
    # Afficher le résultat actuel s'il existe
    if st.session_state.current_dream:
        st.markdown("---")
        st.subheader("🎭 Résultat de votre rêve")
        display_dream_result(st.session_state.current_dream)

def display_dream_result(result: dict):
    """Affiche le résultat complet d'un rêve analysé."""
    
    # Transcription
    with st.expander("📝 Transcription", expanded=True):
        st.write(result["transcription"])
    
    # Analyse émotionnelle
    st.subheader("🎭 Analyse émotionnelle")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        display_emotion_badges(result["emotions"])
        
        # Émotion dominante
        dominant = result["dominant_emotion"]
        confidence = result["emotions"][dominant]
        
        st.markdown(f"""
        <div style="
            background-color: {get_emotion_color(dominant)};
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        ">
            <h3>Émotion dominante: {dominant.capitalize()}</h3>
            <p>Confiance: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Graphique des émotions
        fig = create_emotion_chart(result["emotions"])
        st.plotly_chart(fig, use_container_width=True, key=f"emotion-chart-{uuid.uuid4()}")
    
    # Image générée
    st.subheader("🎨 Visualisation de votre rêve")
    
    if result["image_data"]:
        # Afficher l'image
        image = Image.open(io.BytesIO(result["image_data"]))
        st.image(image, caption="Image générée à partir de votre rêve", use_container_width=True)
        
        # Bouton de téléchargement
        st.download_button(
            label="💾 Télécharger l'image",
            data=result["image_data"],
            file_name=f"dream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png"
        )
    else:
        st.warning("❌ Impossible de générer l'image")
    
    # Prompt utilisé
    with st.expander("🖼️ Prompt artistique utilisé"):
        st.code(result["image_prompt"])

def dashboard_page():
    """Page tableau de bord avec statistiques."""
    st.header("📊 Tableau de bord de vos rêves")
    
    history = st.session_state.dream_history
    
    if not history:
        st.info("🌙 Aucun rêve enregistré pour le moment. Commencez par raconter votre premier rêve !")
        return
    
    # Statistiques générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total des rêves", len(history))
    
    with col2:
        emotions = [dream['dominant_emotion'] for dream in history]
        most_common = max(set(emotions), key=emotions.count)
        st.metric("Émotion dominante", most_common.capitalize())
    
    with col3:
        # Rêves cette semaine
        this_week = sum(1 for dream in history 
            if (datetime.now() - datetime.fromisoformat(dream['timestamp'])).days <= 7)
        st.metric("Rêves cette semaine", this_week)
    
    with col4:
        # Moyenne des mots par rêve
        avg_words = sum(len(dream['transcription'].split()) for dream in history) / len(history)
        st.metric("Mots par rêve", f"{avg_words:.0f}")
    
    # Graphiques
    st.subheader("📈 Analyse temporelle")
    
    # Timeline des émotions
    timeline_fig = create_emotion_timeline()
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True, key=f"timeline-{uuid.uuid4()}")
    
    # Distribution des émotions
    col1, col2 = st.columns(2)
    
    with col1:
        emotions_count = {}
        for dream in history:
            emotion = dream['dominant_emotion']
            emotions_count[emotion] = emotions_count.get(emotion, 0) + 1
        
        fig_pie = px.pie(
            values=list(emotions_count.values()),
            names=list(emotions_count.keys()),
            title="Distribution des émotions",
            color_discrete_map={
                'heureux': '#4CAF50',
                'stressant': '#F44336',
                'neutre': '#9E9E9E',
                'mystérieux': '#9C27B0',
                'nostalgique': '#FF9800'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True, key=f"emotion-pie-{uuid.uuid4()}")
    
    with col2:
        # Longueur des rêves
        lengths = [len(dream['transcription'].split()) for dream in history]
        dates = [datetime.fromisoformat(dream['timestamp']).date() for dream in history]
        
        fig_length = px.scatter(
            x=dates,
            y=lengths,
            title="Longueur des rêves dans le temps",
            labels={'x': 'Date', 'y': 'Nombre de mots'}
        )
        st.plotly_chart(fig_length, use_container_width=True, key=f"length-scatter-{uuid.uuid4()}")

def history_page():
    """Page historique des rêves."""
    st.header("📚 Historique de vos rêves")
    
    history = st.session_state.dream_history
    
    if not history:
        st.info("🌙 Aucun rêve enregistré. Commencez par raconter votre premier rêve !")
        return
    
    # Filtres
    col1, col2 = st.columns(2)
    
    with col1:
        emotions = list(set(dream['dominant_emotion'] for dream in history))
        emotion_filter = st.selectbox("Filtrer par émotion", ["Toutes"] + emotions)
    
    with col2:
        sort_order = st.selectbox("Trier par", ["Plus récent", "Plus ancien"])
    
    # Appliquer les filtres
    filtered_history = history.copy()
    
    if emotion_filter != "Toutes":
        filtered_history = [dream for dream in filtered_history 
                if dream['dominant_emotion'] == emotion_filter]
    
    # Tri
    reverse = sort_order == "Plus récent"
    filtered_history.sort(key=lambda x: x['timestamp'], reverse=reverse)
    
    # Affichage des rêves
    st.write(f"**{len(filtered_history)}** rêve(s) trouvé(s)")
    
    for i, dream in enumerate(filtered_history):
        with st.expander(f"🌙 Rêve du {datetime.fromisoformat(dream['timestamp']).strftime('%d/%m/%Y à %H:%M')}"):
            
            # Métadonnées
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Transcription:**")
                st.write(dream['transcription'])
            
            with col2:
                st.write("**Émotion dominante:**")
                emotion = dream['dominant_emotion']
                confidence = dream['emotions'][emotion]
                
                st.markdown(f"""
                <div style="
                    background-color: {get_emotion_color(emotion)};
                    color: white;
                    padding: 0.5rem;
                    border-radius: 5px;
                    text-align: center;
                ">
                    {emotion.capitalize()}<br>
                    {confidence:.1%}
                </div>
                """, unsafe_allow_html=True)
            
            # Toutes les émotions
            st.write("**Analyse complète:**")
            display_emotion_badges(dream['emotions'])
    
    # Bouton d'export
    if st.button("📥 Exporter l'historique (JSON)"):
        json_data = json.dumps(filtered_history, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Télécharger",
            data=json_data,
            file_name=f"dream_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()