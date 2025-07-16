# Détection d'Intrusion NSL-KDD avec AutoGluon et Streamlit

Cette application Streamlit permet de prédire les types d'attaques DDOS, plus précisément 'Neptune', dans le dataset NSL-KDD grâce à un modèle AutoGluon.

## Fonctionnalités
- Prédiction sur fichiers CSV contenant la colonne `attack` (label réel)
- Comparaison avec les labels réels
- Affichage des métriques (précision, rappel, F1-score)
- Matrice de confusion
- Probabilités par classe

## Lancement local
```bash
streamlit run app.py
```

## Démo
Déployé sur Streamlit Community Cloud : [Lien vers l'app](https://detection-attaque-ddos.streamlit.app/)
