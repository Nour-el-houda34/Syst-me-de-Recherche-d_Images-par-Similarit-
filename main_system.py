# -*- coding: utf-8 -*-
"""
SYSTÈME COMPLET DE RECHERCHE D'IMAGES PAR SIMILARITÉ
Version avancée avec checkpoints, ViT Transformers, et évaluation de précision
Adapté pour exécution locale
"""

# ============================================================================
# 1. INSTALLATION ET IMPORTS
# ============================================================================

import subprocess
import sys

def install_packages():
    """Installe les packages nécessaires si non déjà installés"""
    packages = [
        'transformers', 'torch', 'torchvision', 'scikit-image', 
        'scikit-learn', 'seaborn', 'tqdm', 'pillow', 'tensorflow'
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installation de {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# Installer les packages si nécessaire
install_packages()

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
import pickle
import warnings
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import io

warnings.filterwarnings('ignore')

# Scikit-learn pour similarité et métriques
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import seaborn as sns

# TensorFlow/Keras pour modèles CNN
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.models import Model

# PyTorch pour ViT Transformers
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel

# Configuration
plt.style.use('default')
sns.set_palette("husl")
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

print("✅ Tous les imports réussis!")

# ============================================================================
# 2. CONFIGURATION ET GESTION DES CHECKPOINTS
# ============================================================================

class EmbeddingCheckpointManager:
    """Gère la sauvegarde et le chargement des checkpoints d'embeddings"""

    def __init__(self, embedding_checkpoint_dir="./embedding_checkpoints"):
        self.embedding_checkpoint_dir = Path(embedding_checkpoint_dir)
        self.embedding_checkpoint_dir.mkdir(exist_ok=True)

    def save_embeddings(self, embeddings, labels, image_db, embedding_method_name, dataset_name="cifar100"):
        """Sauvegarde les embeddings extraits"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{embedding_method_name}_{timestamp}.pkl"
        filepath = self.embedding_checkpoint_dir / filename

        checkpoint_data = {
            'embeddings': embeddings,
            'labels': labels,
            'image_indices': list(range(len(image_db))),
            'embedding_method_name': embedding_method_name,
            'dataset_name': dataset_name,
            'embedding_shape': embeddings.shape,
            'timestamp': timestamp,
            'num_images': len(image_db)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        return filepath

    def load_embeddings(self, embedding_method_name, dataset_name="cifar100"):
        """Charge les embeddings depuis le dernier checkpoint"""
        pattern = f"{dataset_name}_{embedding_method_name}_*.pkl"
        embedding_checkpoint_files = list(self.embedding_checkpoint_dir.glob(pattern))

        if not embedding_checkpoint_files:
            return None

        latest_file = max(embedding_checkpoint_files, key=lambda x: x.stat().st_mtime)

        with open(latest_file, 'rb') as f:
            checkpoint_data = pickle.load(f)

        return checkpoint_data

    def save_evaluation_results(self, results, test_name="precision_evaluation"):
        """Sauvegarde les résultats d'évaluation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{test_name}_{timestamp}.pkl"
        filepath = self.embedding_checkpoint_dir / filename

        with open(filepath, 'wb') as f:
            pickle.dump(results, f)

        return filepath

# Initialiser le gestionnaire de checkpoints d'embeddings
embedding_checkpoint_manager = EmbeddingCheckpointManager()

# ============================================================================
# 3. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

def load_and_prepare_data(dataset_size=2000, use_cifar100=True):
    """
    Charge et prépare le dataset CIFAR-100
    """
    if use_cifar100:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        if dataset_size > len(x_train):
            dataset_size = len(x_train)

        indices = np.random.choice(len(x_train), dataset_size, replace=False)
        image_db = x_train[indices]
        labels_db = y_train[indices]

        test_indices = np.random.choice(len(x_test), min(100, len(x_test)), replace=False)
        test_images = x_test[test_indices]
        test_labels = y_test[test_indices]

        class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]

        return image_db, labels_db, test_images, test_labels, class_names

    else:
        return None, None, None, None, None

# Charger les données
DATASET_SIZE = 2000
image_db, labels_db, test_images, test_labels, class_names = load_and_prepare_data(
    dataset_size=DATASET_SIZE,
    use_cifar100=True
)

# ============================================================================
# 4. INITIALISATION DES MODÈLES D'EMBEDDING
# ============================================================================

def initialize_embedding_models():
    """Initialise les modèles pour l'extraction d'embeddings"""

    embedding_models = {}

    # 1. MÉTHODE ANCIENNE: Histogramme de couleurs
    embedding_models['HISTOGRAM'] = {
        'type': 'traditional',
        'description': 'Histogramme HSV (8 bins par canal)'
    }

    # 2. MÉTHODE CNN: ResNet50
    try:
        resnet = ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        resnet_model = Model(inputs=resnet.input, outputs=resnet.output)
        embedding_models['RESNET50'] = {
            'model': resnet_model,
            'type': 'cnn',
            'preprocess_fn': preprocess_resnet,
            'target_size': (224, 224)
        }
    except Exception as e:
        print(f"Erreur ResNet50: {e}")
        embedding_models['RESNET50'] = None

    # 3. MÉTHODE TRANSFORMER: Vision Transformer (ViT)
    try:
        vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        vit_model.eval()

        embedding_models['VIT'] = {
            'model': vit_model,
            'processor': vit_processor,
            'type': 'transformer',
            'target_size': (224, 224)
        }
    except Exception as e:
        print(f"Erreur ViT: {e}")
        embedding_models['VIT'] = None

    return embedding_models

# Initialiser tous les modèles d'embedding
embedding_models = initialize_embedding_models()

# ============================================================================
# 5. FONCTIONS D'EXTRACTION D'EMBEDDINGS
# ============================================================================

def extract_histogram_embeddings(images, bins=8):
    """Extrait des embeddings utilisant un histogramme de couleurs HSV"""
    embeddings = []

    for img in tqdm(images, desc="Extraction histogramme", leave=False):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()

        embedding = np.concatenate([hist_h, hist_s, hist_v])
        embeddings.append(embedding)

    return np.array(embeddings)

def extract_cnn_embeddings(images, model_info):
    """Extrait des embeddings avec un modèle CNN (ResNet50)"""
    if model_info is None:
        raise ValueError("Modèle ResNet50 non initialisé")

    model = model_info['model']
    preprocess_fn = model_info['preprocess_fn']
    target_size = model_info['target_size']

    if images[0].shape[:2] != target_size:
        resized_images = []
        for img in tqdm(images, desc="Redimensionnement", leave=False):
            resized_img = cv2.resize(img, target_size)
            resized_images.append(resized_img)
        images = np.array(resized_images)

    processed_images = preprocess_fn(images.copy())
    embeddings = model.predict(processed_images, verbose=0)

    return embeddings

def extract_vit_embeddings(images, model_info):
    """Extrait des embeddings avec Vision Transformer"""
    if model_info is None:
        raise ValueError("Modèle ViT non initialisé")

    model = model_info['model']
    processor = model_info['processor']
    target_size = model_info['target_size']

    embeddings = []

    for img in tqdm(images, desc="ViT processing"):
        img_resized = cv2.resize(img, target_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        inputs = processor(images=img_pil, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embedding.flatten())

    return np.array(embeddings)

# ============================================================================
# 6. EXTRACTION ET SAUVEGARDE DES EMBEDDINGS
# ============================================================================

def extract_and_save_all_embeddings(image_db, labels_db, embedding_models, force_recompute=False):
    """Extrait et sauvegarde les embeddings pour toutes les méthodes"""

    all_embeddings = {}

    # 1. HISTOGRAMME (Méthode ancienne)
    if not force_recompute:
        checkpoint_data = embedding_checkpoint_manager.load_embeddings('HISTOGRAM')
        if checkpoint_data is not None:
            all_embeddings['HISTOGRAM'] = checkpoint_data['embeddings']
        else:
            embeddings = extract_histogram_embeddings(image_db)
            all_embeddings['HISTOGRAM'] = embeddings
            embedding_checkpoint_manager.save_embeddings(embeddings, labels_db, image_db, 'HISTOGRAM')
    else:
        embeddings = extract_histogram_embeddings(image_db)
        all_embeddings['HISTOGRAM'] = embeddings
        embedding_checkpoint_manager.save_embeddings(embeddings, labels_db, image_db, 'HISTOGRAM')

    # 2. RESNET50 (CNN)
    if embedding_models['RESNET50'] is not None:
        if not force_recompute:
            checkpoint_data = embedding_checkpoint_manager.load_embeddings('RESNET50')
            if checkpoint_data is not None:
                all_embeddings['RESNET50'] = checkpoint_data['embeddings']
            else:
                embeddings = extract_cnn_embeddings(image_db, embedding_models['RESNET50'])
                all_embeddings['RESNET50'] = embeddings
                embedding_checkpoint_manager.save_embeddings(embeddings, labels_db, image_db, 'RESNET50')
        else:
            embeddings = extract_cnn_embeddings(image_db, embedding_models['RESNET50'])
            all_embeddings['RESNET50'] = embeddings
            embedding_checkpoint_manager.save_embeddings(embeddings, labels_db, image_db, 'RESNET50')

    # 3. VIT (Transformer)
    if embedding_models['VIT'] is not None:
        if not force_recompute:
            checkpoint_data = embedding_checkpoint_manager.load_embeddings('VIT')
            if checkpoint_data is not None:
                all_embeddings['VIT'] = checkpoint_data['embeddings']
            else:
                embeddings = extract_vit_embeddings(image_db, embedding_models['VIT'])
                all_embeddings['VIT'] = embeddings
                embedding_checkpoint_manager.save_embeddings(embeddings, labels_db, image_db, 'VIT')
        else:
            embeddings = extract_vit_embeddings(image_db, embedding_models['VIT'])
            all_embeddings['VIT'] = embeddings
            embedding_checkpoint_manager.save_embeddings(embeddings, labels_db, image_db, 'VIT')

    return all_embeddings

# Extraire toutes les embeddings
FORCE_RECOMPUTE = False
all_embeddings = extract_and_save_all_embeddings(image_db, labels_db, embedding_models, force_recompute=FORCE_RECOMPUTE)

# ============================================================================
# 7. SYSTÈME DE RECHERCHE PAR SIMILARITÉ D'EMBEDDINGS
# ============================================================================

def search_similar_images(query_image, query_label=None, top_k=10):
    """Recherche les images similaires avec toutes les méthodes d'embedding"""

    query_images = [query_image]
    all_results = {}
    search_times = {}

    for embedding_method_name, embeddings_db in all_embeddings.items():
        start_time = time.time()

        if embedding_method_name == 'HISTOGRAM':
            query_embeddings = extract_histogram_embeddings(query_images)
        elif embedding_method_name == 'RESNET50':
            query_embeddings = extract_cnn_embeddings(query_images, embedding_models['RESNET50'])
        elif embedding_method_name == 'VIT':
            query_embeddings = extract_vit_embeddings(query_images, embedding_models['VIT'])
        else:
            continue

        similarities = cosine_similarity(embeddings_db, query_embeddings).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]

        search_time = time.time() - start_time
        search_times[embedding_method_name] = search_time

        precision = None
        if query_label is not None:
            correct_matches = sum(1 for idx in top_indices if labels_db[idx] == query_label)
            precision = correct_matches / top_k

        all_results[embedding_method_name] = {
            'indices': top_indices,
            'scores': top_scores,
            'avg_score': np.mean(top_scores),
            'precision': precision,
            'search_time': search_time
        }

    return all_results, search_times

# ============================================================================
# 8. VISUALISATION DES RÉSULTATS
# ============================================================================

def display_search_results(query_image, query_label, all_results, class_names):
    """Affiche les résultats de recherche de manière comparative"""
    num_methods = len(all_results)

    fig = plt.figure(figsize=(18, 4 * num_methods))

    ax_query = plt.subplot(num_methods + 1, 11, 1)
    ax_query.imshow(query_image)
    if query_label is not None and class_names is not None:
        ax_query.set_title(f"Requête\n{class_names[query_label]}",
                          fontweight='bold', color='red', fontsize=10)
    else:
        ax_query.set_title("Image Requête", fontweight='bold', color='red', fontsize=10)
    ax_query.axis('off')

    for spine in ax_query.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(3)

    for i, (embedding_method_name, results) in enumerate(all_results.items()):
        row_start = i * 11 + 12

        ax_title = plt.subplot(num_methods + 1, 11, row_start)
        ax_title.text(0.5, 0.5, embedding_method_name,
                     fontsize=12, fontweight='bold',
                     horizontalalignment='center',
                     verticalalignment='center')

        if 'HISTOGRAM' in embedding_method_name:
            ax_title.set_facecolor('#FFCCCC')
        elif 'VIT' in embedding_method_name:
            ax_title.set_facecolor('#CCFFCC')
        else:
            ax_title.set_facecolor('#CCCCFF')

        ax_title.axis('off')

        for j in range(10):
            if j < len(results['indices']):
                idx = results['indices'][j]
                score = results['scores'][j]

                ax = plt.subplot(num_methods + 1, 11, row_start + j + 1)
                ax.imshow(image_db[idx])

                is_correct = (labels_db[idx] == query_label) if query_label is not None else False
                title_color = 'green' if is_correct else 'black'

                if class_names is not None:
                    class_name = class_names[labels_db[idx]][:10] + "..." if len(class_names[labels_db[idx]]) > 10 else class_names[labels_db[idx]]
                    ax.set_title(f"{score:.3f}\n{class_name}",
                                fontsize=8, color=title_color)
                else:
                    ax.set_title(f"{score:.3f}", fontsize=8, color=title_color)

                ax.axis('off')

                spine_color = 'green' if is_correct else 'red'
                for spine in ax.spines.values():
                    spine.set_edgecolor(spine_color)
                    spine.set_linewidth(2)
            else:
                ax = plt.subplot(num_methods + 1, 11, row_start + j + 1)
                ax.axis('off')

    plt.suptitle("COMPARAISON DES MÉTHODES DE RECHERCHE D'IMAGES",
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_comparative_analysis(all_results):
    """Crée des visualisations comparatives des méthodes d'embedding"""
    methods = list(all_results.keys())
    avg_scores = [all_results[m]['avg_score'] for m in methods]
    precisions = [all_results[m]['precision'] if all_results[m]['precision'] is not None else 0 for m in methods]
    search_times = [all_results[m]['search_time'] for m in methods]

    colors = []
    for method in methods:
        if 'HISTOGRAM' in method:
            colors.append('#FF6B6B')
        elif 'VIT' in method:
            colors.append('#4ECDC4')
        else:
            colors.append('#3498DB')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(methods, avg_scores, color=colors, edgecolor='black', alpha=0.8)
    axes[0, 0].set_title('Score de Similarité Moyen', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Score moyen', fontsize=12)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    for i, (method, score) in enumerate(zip(methods, avg_scores)):
        axes[0, 0].text(i, score + 0.02, f'{score:.3f}',
                       ha='center', va='bottom', fontweight='bold')

    axes[0, 1].bar(methods, precisions, color=colors, edgecolor='black', alpha=0.8)
    axes[0, 1].set_title('Précision (Top-10)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Précision', fontsize=12)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    for i, (method, precision) in enumerate(zip(methods, precisions)):
        axes[0, 1].text(i, precision + 0.02, f'{precision:.1%}',
                       ha='center', va='bottom', fontweight='bold')

    axes[1, 0].bar(methods, search_times, color=colors, edgecolor='black', alpha=0.8)
    axes[1, 0].set_title('Temps de Recherche', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Temps (secondes)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    for i, (method, time_val) in enumerate(zip(methods, search_times)):
        axes[1, 0].text(i, time_val + 0.02, f'{time_val:.3f}s',
                       ha='center', va='bottom', fontweight='bold')

    axes[1, 1].axis('off')

    fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    norm_scores = np.array(avg_scores) / max(avg_scores) if max(avg_scores) > 0 else avg_scores
    norm_precisions = np.array(precisions) / max(precisions) if max(precisions) > 0 else precisions
    norm_times = 1 - (np.array(search_times) / max(search_times)) if max(search_times) > 0 else [1]*len(search_times)

    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]

    for i, method in enumerate(methods):
        values = [norm_scores[i], norm_precisions[i], norm_times[i]]
        values += values[:1]

        ax_radar.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
        ax_radar.fill(angles, values, alpha=0.1, color=colors[i])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(['Similarité', 'Précision', 'Vitesse'], fontsize=12)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Profil Comparatif des Méthodes', fontsize=14, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax_radar.grid(True)

    plt.suptitle('ANALYSE COMPARATIVE DES PERFORMANCES', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    return axes

# ============================================================================
# 9. ÉVALUATION SYSTÉMATIQUE DE LA PRÉCISION
# ============================================================================

def evaluate_precision_systematically(test_images, test_labels, n_tests=50):
    """Évalue systématiquement la précision de toutes les méthodes d'embedding"""

    precision_results = {method: [] for method in all_embeddings.keys()}
    time_results = {method: [] for method in all_embeddings.keys()}

    for test_num in tqdm(range(min(n_tests, len(test_images))), desc="Tests de précision"):
        query_idx = test_num
        query_image = test_images[query_idx]
        query_label = test_labels[query_idx]

        try:
            search_results, search_times = search_similar_images(
                query_image,
                query_label,
                top_k=10
            )

            for embedding_method_name, results in search_results.items():
                if results['precision'] is not None:
                    precision_results[embedding_method_name].append(results['precision'])
                    time_results[embedding_method_name].append(results['search_time'])
        except Exception as e:
            continue

    stats = {}

    for embedding_method_name in all_embeddings.keys():
        if precision_results[embedding_method_name]:
            precisions = precision_results[embedding_method_name]
            times = time_results[embedding_method_name]

            stats[embedding_method_name] = {
                'mean_precision': np.mean(precisions),
                'std_precision': np.std(precisions),
                'max_precision': np.max(precisions),
                'min_precision': np.min(precisions),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'num_tests': len(precisions)
            }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    methods = list(stats.keys())
    mean_precisions = [stats[m]['mean_precision'] for m in methods]
    std_precisions = [stats[m]['std_precision'] for m in methods]

    colors = ['#FF6B6B' if 'HISTOGRAM' in m else '#4ECDC4' if 'VIT' in m else '#3498DB'
              for m in methods]

    axes[0].bar(methods, mean_precisions, yerr=std_precisions,
                color=colors, edgecolor='black', alpha=0.8, capsize=5)
    axes[0].set_title('Précision Moyenne avec Intervalle de Confiance',
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Précision moyenne', fontsize=12)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='y')

    precision_data = [precision_results[m] for m in methods]
    bp = axes[1].boxplot(precision_data, labels=methods, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_title('Distribution des Précisions', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Précision', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'ÉVALUATION DE PRÉCISION SUR {n_tests} TESTS',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    evaluation_data = {
        'precision_results': precision_results,
        'time_results': time_results,
        'stats': stats,
        'num_tests': n_tests,
        'timestamp': datetime.now().isoformat()
    }

    embedding_checkpoint_manager.save_evaluation_results(evaluation_data, "precision_evaluation")

    return stats

# ============================================================================
# 10. INTERFACE UTILISATEUR SIMPLIFIÉE
# ============================================================================

def run_demo_test():
    """Exécute un test de démonstration complet"""
    query_idx = np.random.randint(0, len(test_images))
    query_image = test_images[query_idx]
    query_label = test_labels[query_idx]

    search_results, search_times = search_similar_images(
        query_image,
        query_label,
        top_k=10
    )

    display_search_results(query_image, query_label, search_results, class_names)
    plot_comparative_analysis(search_results)

    return search_results, search_times

def run_custom_image_search(image_path=None):
    """Permet de rechercher avec une image personnalisée"""
    
    if image_path is None:
        # Demander le chemin de l'image à l'utilisateur
        image_path = input("Entrez le chemin vers votre image (ou appuyez sur Entrée pour utiliser une image par défaut): ").strip()
        
        if not image_path:
            # Créer une image rouge simple par défaut
            image_array = np.zeros((32, 32, 3), dtype=np.uint8)
            image_array[8:24, 8:24, :] = [255, 0, 0]
            query_image = image_array
            query_label = None
        else:
            # Charger l'image depuis le chemin spécifié
            try:
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((32, 32))
                query_image = np.array(img)
                query_label = None
            except Exception as e:
                print(f"❌ Erreur lors du chargement de l'image: {e}")
                return None
    else:
        # Charger l'image depuis le chemin fourni
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((32, 32))
            query_image = np.array(img)
            query_label = None
        except Exception as e:
            print(f"❌ Erreur lors du chargement de l'image: {e}")
            return None

    search_results, search_times = search_similar_images(
        query_image,
        query_label,
        top_k=10
    )

    display_search_results(query_image, query_label, search_results, class_names)

    return search_results, search_times

def get_embedding_statistics():
    """Affiche les statistiques du système d'embedding"""

    print(f"\n📦 Dataset CIFAR-100:")
    print(f"   • Images dans la base: {len(image_db)}")
    print(f"   • Classes disponibles: {len(np.unique(labels_db))}")
    print(f"   • Images de test: {len(test_images)}")

    print(f"\n🔧 Embeddings extraits:")
    for embedding_method, embeddings in all_embeddings.items():
        print(f"   • {embedding_method}: {embeddings.shape}")

    print(f"\n🧠 Modèles d'embedding chargés:")
    for embedding_method_name, model_info in embedding_models.items():
        if isinstance(model_info, dict) and 'type' in model_info:
            print(f"   • {embedding_method_name}: {model_info['type'].upper()}")
        else:
            print(f"   • {embedding_method_name}: {type(model_info).__name__}")

# ============================================================================
# 11. FONCTION PRINCIPALE SIMPLIFIÉE
# ============================================================================

def main():
    """Fonction principale exécutant le système complet"""

    while True:
        print("\n" + "="*50)
        print("📋 MENU PRINCIPAL - RECHERCHE D'IMAGES PAR SIMILARITÉ")
        print("="*50)
        print("1. 🚀 Exécuter une démonstration complète")
        print("2. 📈 Évaluer la précision systématiquement")
        print("3. 🖼️  Rechercher avec une image personnalisée")
        print("4. 📊 Afficher les statistiques du système")
        print("5. 🏁 Quitter")

        choice = input("\n👉 Votre choix (1-5): ").strip()

        if choice == '1':
            print("\n🚀 Lancement de la démonstration...")
            search_results, search_times = run_demo_test()

            best_embedding_method = max(search_results.items(),
                            key=lambda x: x[1]['precision'] if x[1]['precision'] else 0)

            print(f"\n🏆 Meilleure méthode d'embedding: {best_embedding_method[0]}")
            if best_embedding_method[1]['precision']:
                print(f"   Précision: {best_embedding_method[1]['precision']:.2%}")
            print(f"   Score moyen: {best_embedding_method[1]['avg_score']:.4f}")

        elif choice == '2':
            n_tests = input("Nombre de tests à exécuter (défaut: 50): ").strip()
            n_tests = int(n_tests) if n_tests.isdigit() else 50

            print(f"\n📈 Évaluation sur {n_tests} tests...")
            stats = evaluate_precision_systematically(test_images, test_labels, n_tests=n_tests)

            best_embedding_method = max(stats.items(),
                            key=lambda x: x[1]['mean_precision'])

            print(f"\n✅ Pour une précision optimale, utilisez: {best_embedding_method[0]}")
            print(f"   Précision moyenne: {best_embedding_method[1]['mean_precision']:.2%}")
            print(f"   Consistance: {1-best_embedding_method[1]['std_precision']:.3f} (1 = très consistant)")

        elif choice == '3':
            print("\n🖼️  Recherche avec image personnalisée")
            run_custom_image_search()

        elif choice == '4':
            get_embedding_statistics()

        elif choice == '5':
            print("\n👋 Fin du programme")
            break

        else:
            print("❌ Choix invalide. Veuillez entrer un nombre entre 1 et 5.")

        input("\nAppuyez sur Entrée pour continuer...")
# ============================================================================
# 12. EXÉCUTION DU PROGRAMME
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("SYSTÈME DE RECHERCHE D'IMAGES PAR SIMILARITÉ")
    print("Version adaptée pour exécution locale")
    print("="*60)
    
    # Initialiser les embeddings
    print("\n⏳ Initialisation du système...")
    all_embeddings = extract_and_save_all_embeddings(
        image_db, labels_db, embedding_models, force_recompute=False
    )
    print("✅ Initialisation terminée!")
    
    print("\n🔧 Options disponibles:")
    print("1. 🚀 Lancer l'API Flask pour l'interface web")
    print("2. 🖥️  Lancer le menu interactif en console")
    print("3. 📊 Tester rapidement le système")
    
    choice = input("\n👉 Votre choix (1-3): ").strip()
    
    if choice == '1':
        print("\n✅ Le système est prêt pour l'API Flask.")
        print("📁 Pour lancer l'API, exécutez dans un nouveau terminal:")
        print("   python flask_api.py")
        print("\n🌐 Puis ouvrez index.html dans votre navigateur")
        print("\n⚡ Les embeddings ViT sont chargés et prêts!")
        print(f"   • Embeddings: {all_embeddings['VIT'].shape}")
        print(f"   • Base de données: {len(image_db)} images")
        print(f"   • Classes: {len(class_names)}")
    
    elif choice == '2':
        print("\n🚀 Lancement du menu interactif...")
        main()
    
    elif choice == '3':
        print("\n🧪 Test rapide du système...")
        
        # Tester avec une image aléatoire
        test_idx = np.random.randint(0, len(test_images))
        test_image = test_images[test_idx]
        test_label = test_labels[test_idx]
        
        print(f"📸 Image de test: {class_names[test_label]}")
        print("🔍 Recherche des images similaires...")
        
        results, times = search_similar_images(test_image, test_label, top_k=3)
        
        print(f"\n✅ Test terminé!")
        
        for method, result in results.items():
            if result['precision']:
                print(f"   • {method}: Précision = {result['precision']:.1%}, Temps = {result['search_time']:.3f}s")
    
    else:
        print("❌ Choix invalide.")

print("\n✨ Programme terminé")
print("Embeddings sauvegardés dans: ./embedding_checkpoints/")