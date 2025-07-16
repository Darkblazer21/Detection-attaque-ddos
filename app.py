import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import io

# Configuration de la page
st.set_page_config(
    page_title="NSL-KDD Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    
    .status-info {
        background: #cce7ff;
        color: #004085;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #007bff;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour générer des données de test
@st.cache_data
def generate_test_data():
    """Génère des données de test basées sur le dataset NSL-KDD"""
    np.random.seed(42)
    
    # Définition des features typiques du NSL-KDD (version simplifiée)
    features = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate'
    ]
    
    attack_types = ['normal', 'neptune']

    levels = [7, 18, 19, 20, 21]
    
    # Génération de 1000 échantillons
    n_samples = 1000
    data = {}
    
    for feature in features:
        if feature in ['protocol_type', 'service', 'flag']:
            # Variables catégorielles
            if feature == 'protocol_type':
                data[feature] = np.random.choice(['tcp', 'udp', 'icmp'], n_samples)
            elif feature == 'service':
                data[feature] = np.random.choice(['http', 'ftp', 'smtp', 'telnet', 'ssh'], n_samples)
            else:  # flag
                data[feature] = np.random.choice(['SF', 'S0', 'REJ', 'RSTR', 'RSTO'], n_samples)
        elif feature in ['land', 'wrong_fragment', 'urgent', 'logged_in', 'root_shell', 'su_attempted',
                        'is_host_login', 'is_guest_login']:
            # Variables binaires
            data[feature] = np.random.choice([0, 1], n_samples)
        else:
            # Variables numériques
            if 'rate' in feature:
                data[feature] = np.random.uniform(0, 1, n_samples)
            elif 'count' in feature:
                data[feature] = np.random.randint(0, 511, n_samples)
            elif feature == 'duration':
                data[feature] = np.random.exponential(10, n_samples)
            elif 'bytes' in feature:
                data[feature] = np.random.exponential(100, n_samples)
            else:
                data[feature] = np.random.randint(0, 100, n_samples)
    
    # Génération des labels d'attaque
    data['attack'] = np.random.choice(attack_types, n_samples, p=[0.7, 0.3])

    data['level'] = np.random.choice(levels, n_samples, p=[0.01, 0.08, 0.16, 0.17, 0.58])
    
    return pd.DataFrame(data)

# Fonction pour créer un formulaire de saisie manuelle
def create_manual_input_form():
    """Crée un formulaire pour la saisie manuelle des données"""
    st.markdown("### ✍️ Saisie Manuelle des Données")
    
    with st.form("manual_input_form"):
        st.markdown("Remplissez les champs ci-dessous pour tester une prédiction individuelle:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Caractéristiques de Base**")
            duration = st.number_input("Duration", min_value=0.0, value=0.0)
            protocol_type = st.selectbox("Protocol Type", ['tcp', 'udp', 'icmp'])
            service = st.selectbox("Service", ['http', 'ftp', 'smtp', 'telnet', 'ssh'])
            flag = st.selectbox("Flag", ['SF', 'S0', 'REJ', 'RSTR', 'RSTO'])
            src_bytes = st.number_input("Source Bytes", min_value=0, value=0)
            dst_bytes = st.number_input("Destination Bytes", min_value=0, value=0)
            
        with col2:
            st.markdown("**Caractéristiques de Sécurité**")
            land = st.selectbox("Land", [0, 1])
            wrong_fragment = st.selectbox("Wrong Fragment", [0, 1])
            urgent = st.selectbox("Urgent", [0, 1])
            hot = st.number_input("Hot", min_value=0, value=0)
            num_failed_logins = st.number_input("Failed Logins", min_value=0, value=0)
            logged_in = st.selectbox("Logged In", [0, 1])
            
        with col3:
            st.markdown("**Caractéristiques Réseau**")
            count = st.number_input("Count", min_value=0, value=1)
            srv_count = st.number_input("Service Count", min_value=0, value=1)
            serror_rate = st.slider("Service Error Rate", 0.0, 1.0, 0.0)
            srv_serror_rate = st.slider("Service Service Error Rate", 0.0, 1.0, 0.0)
            rerror_rate = st.slider("REJ Error Rate", 0.0, 1.0, 0.0)
            srv_rerror_rate = st.slider("Service REJ Error Rate", 0.0, 1.0, 0.0)
        
        # Ajout des autres champs avec des valeurs par défaut
        st.markdown("**Vraie Étiquette (pour évaluation)**")
        true_attack = st.selectbox("Type d'attaque réel", ['normal', 'dos', 'probe', 'r2l', 'u2r'])
        
        submitted = st.form_submit_button("🔍 Prédire cette Instance")
        
        if submitted:
            # Création d'un DataFrame avec toutes les features requises
            manual_data = {
                'duration': duration,
                'protocol_type': protocol_type,
                'service': service,
                'flag': flag,
                'src_bytes': src_bytes,
                'dst_bytes': dst_bytes,
                'land': land,
                'wrong_fragment': wrong_fragment,
                'urgent': urgent,
                'hot': hot,
                'num_failed_logins': num_failed_logins,
                'logged_in': logged_in,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': count,
                'srv_count': srv_count,
                'serror_rate': serror_rate,
                'srv_serror_rate': srv_serror_rate,
                'rerror_rate': rerror_rate,
                'srv_rerror_rate': srv_rerror_rate,
                'same_srv_rate': 1.0,
                'diff_srv_rate': 0.0,
                'srv_diff_host_rate': 0.0,
                'dst_host_count': 255,
                'dst_host_srv_count': 255,
                'dst_host_same_srv_rate': 1.0,
                'dst_host_diff_srv_rate': 0.0,
                'dst_host_same_src_port_rate': 1.0,
                'dst_host_srv_diff_host_rate': 0.0,
                'dst_host_serror_rate': 0.0,
                'dst_host_srv_serror_rate': 0.0,
                'dst_host_rerror_rate': 0.0,
                'dst_host_srv_rerror_rate': 0.0,
                'attack': true_attack
            }
            
            df = pd.DataFrame([manual_data])
            return df
    
    return None
@st.cache_resource
def load_model():
    try:
        return TabularPredictor.load('agModels-predictClass-deployment-clone-opt')
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None

# Fonction pour créer des métriques visuelles
def create_metrics_display(accuracy, num_samples, num_classes):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">Précision</h3>
            <h2 style="margin: 0.5rem 0;">{accuracy:.2%}</h2>
            <p style="margin: 0; color: #6c757d;">Exactitude du modèle</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">Échantillons</h3>
            <h2 style="margin: 0.5rem 0;">{num_samples:,}</h2>
            <p style="margin: 0; color: #6c757d;">Nombre de prédictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">Classes</h3>
            <h2 style="margin: 0.5rem 0;">{num_classes}</h2>
            <p style="margin: 0; color: #6c757d;">Types d'attaques détectées</p>
        </div>
        """, unsafe_allow_html=True)

# Fonction pour créer une matrice de confusion interactive
def create_interactive_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_labels,
        y=class_labels,
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Matrice de Confusion Interactive",
        xaxis_title="Prédictions",
        yaxis_title="Vraies Étiquettes",
        font=dict(size=12),
        height=600
    )
    
    return fig

# Fonction pour créer un graphique de distribution des classes
def create_class_distribution_chart(y_true, y_pred):
    true_counts = pd.Series(y_true).value_counts()
    pred_counts = pd.Series(y_pred).value_counts()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribution Vraies Étiquettes', 'Distribution Prédictions'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=true_counts.index, y=true_counts.values, name="Vraies", marker_color="#667eea"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=pred_counts.index, y=pred_counts.values, name="Prédictions", marker_color="#764ba2"),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Distribution des Classes",
        showlegend=False,
        height=500
    )
    
    return fig

# Interface principale
def main():
    # En-tête principal
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Système de Détection d'Intrusion NSL-KDD</h1>
        <p>Analyse avancée des menaces cybernétiques avec AutoGluon</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar avec informations
    with st.sidebar:
        st.markdown("### 📊 Informations du Modèle")
        st.info("""
        **Modèle:** AutoGluon TabularPredictor
        **Dataset:** NSL-KDD
        **Objectif:** Classification multi-classe
        **Types d'attaques:** Normal, DoS, Probe, R2L, U2R
        """)
        
        st.markdown("### 🔧 Configuration")
        show_probabilities = st.checkbox("Afficher les probabilités", value=True)
        show_detailed_report = st.checkbox("Rapport détaillé", value=True)
        max_samples_display = st.slider("Échantillons à afficher", 5, 50, 10)
    
    # Chargement du modèle
    predictor = load_model()
    
    if predictor is None:
        st.error("Impossible de charger le modèle. Vérifiez le chemin du modèle.")
        return
    
    # Section de chargement de fichier
    st.markdown("## 📁 Chargement des Données")
    
    # Onglets pour différentes options de données
    tab1, tab2, tab3 = st.tabs(["📂 Fichier CSV", "🧪 Données de Test", "✍️ Saisie Manuelle"])
    
    data = None
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="upload-section">
                <h3>📂 Chargez votre fichier CSV</h3>
                <p>Le fichier doit contenir une colonne 'attack' avec les vraies étiquettes</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Sélectionnez un fichier CSV",
                type=['csv'],
                help="Fichier CSV contenant les données à analyser"
            )
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Fichier chargé avec succès ! ({len(data)} lignes, {len(data.columns)} colonnes)")
                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement : {str(e)}")
        
        with col2:
            if uploaded_file is not None:
                st.markdown("""
                <div class="status-success">
                    ✅ <strong>Fichier chargé avec succès</strong><br>
                    Prêt pour l'analyse
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-info">
                    ℹ️ <strong>En attente</strong><br>
                    Chargez un fichier CSV pour commencer
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="upload-section">
            <h3>🧪 Données de Test Préchargées</h3>
            <p>Utilisez des données de test générées automatiquement basées sur le dataset NSL-KDD</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📂 Utiliser KDDTest+.csv", key="use_preloaded_data"):
            try:
                data = pd.read_csv('KDDTest+.csv')
                st.session_state.test_data = data
                st.success(f"✅ Fichier préchargé KDDTest+.csv chargé avec succès ! ({len(data)} échantillons)")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du fichier préchargé : {str(e)}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🎲 Générer des Données de Test", key="generate_test_data"):
                with st.spinner("Génération des données de test..."):
                    data = generate_test_data()
                    st.session_state.test_data = data
                    st.success(f"✅ Données de test générées ! ({len(data)} échantillons)")
        
        with col2:
            if st.button("📊 Utiliser les Données Existantes", key="use_existing_test_data"):
                if 'test_data' in st.session_state:
                    data = st.session_state.test_data
                    st.success("✅ Données de test chargées depuis la session !")
                else:
                    st.warning("⚠️ Aucune donnée de test dans la session. Générez-en d'abord.")
        
        # Affichage des informations sur les données de test
        if 'test_data' in st.session_state:
            test_data = st.session_state.test_data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Échantillons", len(test_data))
            with col2:
                st.metric("Features", len(test_data.columns) - 1)
            with col3:
                st.metric("Classes", test_data['attack'].nunique())
            
            # Distribution des classes dans les données de test
            st.markdown("**Distribution des Classes dans les Données de Test:**")
            class_dist = test_data['attack'].value_counts()
            fig_test = px.bar(
                x=class_dist.index,
                y=class_dist.values,
                title="Distribution des Attaques",
                labels={'x': 'Type d\'attaque', 'y': 'Nombre d\'échantillons'},
                color=class_dist.values,
                color_continuous_scale='viridis'
            )
            fig_test.update_layout(height=300)
            st.plotly_chart(fig_test, use_container_width=True)
    
    with tab3:
        manual_data = create_manual_input_form()
        if manual_data is not None:
            data = manual_data
            st.success("✅ Données saisies manuellement prêtes pour la prédiction !")
    
    # Utilisation des données de test si elles existent et qu'aucune autre source n'est sélectionnée
    if data is None and 'test_data' in st.session_state:
        data = st.session_state.test_data
    
    # Traitement des données
    if data is not None:
        try:
            # Aperçu des données
            st.markdown("## 📋 Aperçu des Données")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nombre de lignes", len(data))
            with col2:
                st.metric("Nombre de colonnes", len(data.columns))
            with col3:
                if 'attack' in data.columns:
                    st.metric("Classes d'attaque", data['attack'].nunique())
            
            # Affichage des premières lignes avec style
            st.markdown("### Échantillon des données")
            st.dataframe(
                data.head(max_samples_display),
                use_container_width=True,
                height=300
            )
            
            # Vérification de la colonne 'attack'
            if 'attack' not in data.columns:
                st.markdown("""
                <div class="status-error">
                    ❌ <strong>Erreur:</strong> Le fichier doit contenir la colonne 'attack' avec les vraies étiquettes.
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Informations sur les données
            st.markdown("### 📊 Informations sur les Données")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Distribution des Classes:**")
                class_counts = data['attack'].value_counts()
                for attack_type, count in class_counts.items():
                    percentage = (count / len(data)) * 100
                    st.write(f"• **{attack_type}**: {count} ({percentage:.1f}%)")
            
            with col2:
                st.markdown("**Statistiques Descriptives:**")
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:  # Exclure la colonne 'attack' si elle est numérique
                    stats = data[numeric_cols].describe()
                    st.dataframe(stats.round(2), height=200)
            
            # Bouton de prédiction avec style
            st.markdown("## 🚀 Analyse et Prédiction")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                predict_button = st.button("🔍 Lancer l'Analyse Complète", key="predict_button")
            with col2:
                if len(data) == 1:
                    st.info("💡 Prédiction sur une instance unique - Idéal pour tester des cas spécifiques")
                else:
                    st.info(f"📊 Prédiction sur {len(data)} instances - Analyse complète du dataset")
            
            if predict_button:
                # Barre de progression
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Préparation des données
                status_text.text("Préparation des données...")
                progress_bar.progress(20)
                
                X = data.drop(columns=['attack'])
                y_true = data['attack']
                
                # Prédiction
                status_text.text("Génération des prédictions...")
                progress_bar.progress(50)
                
                y_pred = predictor.predict(X)
                
                if show_probabilities:
                    status_text.text("Calcul des probabilités...")
                    progress_bar.progress(70)
                    y_proba = predictor.predict_proba(X)
                
                # Calcul des métriques
                status_text.text("Calcul des métriques...")
                progress_bar.progress(90)
                
                accuracy = accuracy_score(y_true, y_pred)
                
                progress_bar.progress(100)
                status_text.text("Analyse terminée!")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                # Stockage des résultats dans le session state
                st.session_state.results = {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'y_proba': y_proba if show_probabilities else None,
                    'accuracy': accuracy,
                    'data_length': len(data),
                    'class_labels': predictor.class_labels
                }
            
            # Affichage des résultats si ils existent
            if 'results' in st.session_state:
                results = st.session_state.results
                
                # Affichage des métriques principales
                st.markdown("## 📊 Résultats de l'Analyse")
                create_metrics_display(results['accuracy'], results['data_length'], len(results['class_labels']))
                
                # Comparaison des résultats
                st.markdown("### 🔄 Comparaison Détaillée")
                
                comparison_df = pd.DataFrame({
                    'Index': range(len(results['y_true'])),
                    'Vraie Étiquette': results['y_true'],
                    'Prédiction': results['y_pred'],
                    'Correct': results['y_true'] == results['y_pred']
                })
                
                # Filtres pour la comparaison (maintenant en temps réel)
                col1, col2 = st.columns(2)
                with col1:
                    filter_correct = st.selectbox(
                        "Filtrer par exactitude",
                        options=["Tous", "Corrects", "Incorrects"],
                        key="filter_correct"
                    )
                with col2:
                    filter_class = st.selectbox(
                        "Filtrer par classe",
                        options=["Toutes"] + list(results['class_labels']),
                        key="filter_class"
                    )
                
                # Application des filtres en temps réel
                filtered_df = comparison_df.copy()
                if filter_correct == "Corrects":
                    filtered_df = filtered_df[filtered_df['Correct'] == True]
                elif filter_correct == "Incorrects":
                    filtered_df = filtered_df[filtered_df['Correct'] == False]
                
                if filter_class != "Toutes":
                    filtered_df = filtered_df[filtered_df['Vraie Étiquette'] == filter_class]
                
                # Affichage du nombre d'éléments filtrés
                st.info(f"📊 {len(filtered_df)} éléments correspondent aux filtres sélectionnés")
                
                st.dataframe(
                    filtered_df.head(max_samples_display),
                    use_container_width=True,
                    height=400
                )
                
                # Visualisations
                st.markdown("## 📈 Visualisations")
                
                # Graphiques côte à côte
                col1, col2 = st.columns(2)
                
                with col1:
                    # Matrice de confusion interactive
                    fig_cm = create_interactive_confusion_matrix(results['y_true'], results['y_pred'], results['class_labels'])
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with col2:
                    # Distribution des classes
                    fig_dist = create_class_distribution_chart(results['y_true'], results['y_pred'])
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # Rapport de classification
                if show_detailed_report:
                    st.markdown("### 📋 Rapport de Classification Détaillé")
                    
                    report = classification_report(results['y_true'], results['y_pred'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    
                    # Formatage du rapport
                    styled_report = report_df.round(4).style.highlight_max(axis=0)
                    st.dataframe(styled_report, use_container_width=True)
                
                # Probabilités
                if show_probabilities and results['y_proba'] is not None:
                    st.markdown("### 🎯 Probabilités par Classe")
                    
                    # Graphique des probabilités moyennes
                    prob_means = results['y_proba'].mean().sort_values(ascending=False)
                    
                    fig_prob = px.bar(
                        x=prob_means.index,
                        y=prob_means.values,
                        title="Probabilités Moyennes par Classe",
                        labels={'x': 'Classes', 'y': 'Probabilité Moyenne'},
                        color=prob_means.values,
                        color_continuous_scale='viridis'
                    )
                    fig_prob.update_layout(height=400)
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Tableau des probabilités
                    st.markdown("#### Échantillon des Probabilités")
                    st.dataframe(
                        results['y_proba'].head(max_samples_display),
                        use_container_width=True,
                        height=300
                    )
                
                # Résumé final
                st.markdown("## 📝 Résumé de l'Analyse")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prédictions Correctes",
                        f"{sum(results['y_true'] == results['y_pred']):,}",
                        f"{results['accuracy']:.2%}"
                    )
                
                with col2:
                    st.metric(
                        "Prédictions Incorrectes",
                        f"{sum(results['y_true'] != results['y_pred']):,}",
                        f"{(1-results['accuracy']):.2%}"
                    )
                
                with col3:
                    most_predicted = pd.Series(results['y_pred']).mode()[0]
                    st.metric(
                        "Classe la Plus Prédite",
                        most_predicted,
                        f"{(pd.Series(results['y_pred']) == most_predicted).sum()} fois"
                    )
        
        except Exception as e:
            st.markdown(f"""
            <div class="status-error">
                ❌ <strong>Erreur lors du traitement:</strong> {str(e)}
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Message d'accueil quand aucune donnée n'est chargée
        st.markdown("## 🚀 Commencer l'Analyse")
        st.markdown("""
        <div class="status-info">
            <h4>👋 Bienvenue dans le système de détection d'intrusion NSL-KDD</h4>
            <p>Pour commencer votre analyse, choisissez l'une des options suivantes:</p>
            <ol>
                <li><strong>📂 Fichier CSV</strong> - Chargez vos propres données</li>
                <li><strong>🧪 Données de Test</strong> - Utilisez des données préchargées</li>
                <li><strong>✍️ Saisie Manuelle</strong> - Testez une instance spécifique</li>
            </ol>
            <p>Assurez-vous que vos données contiennent une colonne 'attack' avec les vraies étiquettes.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()