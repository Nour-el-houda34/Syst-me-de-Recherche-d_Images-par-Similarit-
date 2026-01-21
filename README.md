## 📌 Description

Ce projet implémente un **système complet de recherche et de classification d’images par similarité** en utilisant plusieurs méthodes d’extraction d’embeddings :

* 🟥 **Méthode traditionnelle** : Histogrammes de couleurs HSV
* 🟦 **CNN** : ResNet50 (pré‑entraîné sur ImageNet)
* 🟩 **Transformer** : Vision Transformer (ViT‑Base‑Patch16‑224)

Le système permet :

* la recherche d’images similaires (CBIR – Content Based Image Retrieval)
* la comparaison des performances entre méthodes
* l’évaluation de la précision (Top‑K)
* l’exécution en **mode console** ou via une **API Flask**

Le dataset utilisé est **CIFAR‑100**.

---

## 🧠 Architecture du Projet

```
project/
│
├── README.md # Documentation
├── main_system.py # Système principal (embeddings, recherche, évaluation)
├── flask_api.py # API Flask (classification & similarité)
├── app.js # Logique frontend
├── index.html # Interface web
├── style.css # Styles CSS
├── requirements.txt # Dépendances Python
├── embeddings_vit.pkl # Embeddings ViT sauvegardés
└── embeddings_resnet.pkl # Embeddings ResNet50 sauvegardés
```

---

## ⚙️ Technologies Utilisées

* **Python 3.8+**
* **TensorFlow / Keras** (ResNet50)
* **PyTorch** (Vision Transformer)
* **HuggingFace Transformers**
* **Scikit‑learn** (similarité cosinus, métriques)
* **OpenCV** (traitement d’images)
* **Flask + Flask‑CORS** (API)
* **Matplotlib / Seaborn** (visualisation)

---

## 📦 Installation

Aucune installation manuelle n’est requise.
Le script installe automatiquement les dépendances manquantes :

```bash
pip install transformers torch torchvision scikit-image scikit-learn seaborn tqdm pillow tensorflow flask flask-cors
```

---

## 🚀 Exécution du Système Principal

Lancer le programme principal :

```bash
python main_system.py
```

Au démarrage, trois options sont proposées :

1. **Lancer l’API Flask**
2. **Menu interactif en console**
3. **Test rapide du système**

---

## 🖥️ Menu Console Interactif

Fonctionnalités disponibles :

* 🔍 Recherche d’images similaires
* 📈 Évaluation systématique de la précision
* 🖼️ Recherche avec image personnalisée
* 📊 Statistiques des embeddings

Les résultats incluent :

* scores de similarité
* précision Top‑10
* temps de recherche
* visualisations comparatives

---

## 🌐 API Flask

### ▶️ Lancement

```bash
python flask_api.py
```

Serveur disponible sur :

```
http://localhost:5000
```

---

### 📌 Endpoints Disponibles

#### 🔹 POST `/classify`

Classifie une image et retourne les images similaires.

* **Entrée** : image (form‑data)
* **Sortie** :

  * label prédit
  * confiance
  * top‑5 prédictions
  * images similaires (ViT + Histogram)
  * précision par méthode

## 📊 Méthodes de Similarité

* **Cosine Similarity** entre embeddings
* Normalisation automatique
* Recherche Top‑K

---

## 🧪 Évaluation des Performances

Le système permet une évaluation automatique sur plusieurs requêtes :

* précision moyenne
* écart‑type
* temps moyen d’inférence
* boxplots et barplots
* radar comparatif (Similarité / Précision / Vitesse)

---

## 💾 Checkpoints

Les embeddings sont sauvegardés automatiquement pour éviter les recalculs :

```
embedding_checkpoints/
├── cifar100_VIT_*.pkl
├── cifar100_RESNET50_*.pkl
├── cifar100_HISTOGRAM_*.pkl
└── evaluation_*.pkl
```

---

## 🏆 Résultats Attendus (Indicatifs)

| Méthode   | Précision | Vitesse | Qualité |
| --------- | --------- | ------- | ------- |
| Histogram | ⭐         | ⭐⭐⭐⭐    | ⭐       |
| ResNet50  | ⭐⭐⭐       | ⭐⭐      | ⭐⭐⭐     |
| ViT       | ⭐⭐⭐⭐      | ⭐⭐      | ⭐⭐⭐⭐    |

---

## 📌 Remarques

* Le modèle **ViT** offre la meilleure précision globale
* Les histogrammes sont rapides mais peu discriminants
* Le système est extensible à d’autres datasets

---


## ✅ Auteur

Développé par **Nour el houda HAMIDI**

---
