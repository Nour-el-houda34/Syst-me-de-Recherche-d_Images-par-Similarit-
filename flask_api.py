"""
Serveur Flask pour l'API de classification d'images avec VIT et HISTOGRAM
Version simplifiée - RESNET50 ignoré
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
from PIL import Image
import io
import sys
import os
import time
import traceback
import pickle
from datetime import datetime
from pathlib import Path

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("⚙️  Chargement du système principal...")

# Importer les fonctions du système principal
try:
    from main_system import (
        extract_vit_embeddings, 
        extract_histogram_embeddings,
        embedding_models, 
        all_embeddings,
        image_db, 
        labels_db, 
        test_images,
        test_labels,
        class_names,
        cosine_similarity
    )
    SYSTEM_LOADED = True
    print("✅ Système principal chargé avec succès")
    
except ImportError as e:
    print(f"❌ Erreur lors du chargement du système principal: {e}")
    print(traceback.format_exc())
    SYSTEM_LOADED = False
except Exception as e:
    print(f"❌ Erreur inattendue: {e}")
    print(traceback.format_exc())
    SYSTEM_LOADED = False

app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes

# Variables globales pour le cache
cached_embeddings = None
vit_model_info = None
system_ready = False

def compute_cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre deux vecteurs"""
    vec1 = vec1.flatten()
    vec2 = vec2.reshape(vec2.shape[0], -1)
    
    # Normaliser
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2, axis=1)
    
    if norm1 == 0 or np.any(norm2 == 0):
        return np.zeros(len(vec2))
    
    # Produit scalaire et similarité
    dot_product = np.dot(vec2, vec1)
    similarity = dot_product / (norm1 * norm2)
    
    return similarity

def initialize_system():
    """Initialise le système au démarrage"""
    global cached_embeddings, vit_model_info, system_ready
    
    if not SYSTEM_LOADED:
        print("❌ SYSTEM_LOADED est False")
        return False
    
    try:
        print("\n🔍 Vérification des composants du système...")
        
        # Vérifier VIT
        if 'VIT' not in embedding_models or embedding_models['VIT'] is None:
            print("❌ Modèle ViT non disponible")
            return False
        
        vit_model_info = embedding_models['VIT']
        
        # Vérifier les embeddings
        print("📊 Vérification des embeddings disponibles:")
        for method in ['VIT', 'HISTOGRAM']:
            if method in all_embeddings:
                embeddings = all_embeddings[method]
                print(f"   ✅ {method}: {embeddings.shape[0]} images, {embeddings.shape[1]} dimensions")
            else:
                print(f"   ❌ {method}: Non disponible")
        
        if 'VIT' not in all_embeddings:
            print("❌ Embeddings ViT non disponibles")
            return False
        
        cached_embeddings = all_embeddings['VIT']
        
        system_ready = True
        print("\n✅ Système initialisé avec succès")
        print(f"   📊 Base de données: {cached_embeddings.shape[0]} images")
        print(f"   🔢 Embeddings ViT: {cached_embeddings.shape[1]} dimensions")
        print(f"   🏷️  Classes: {len(class_names)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        traceback.print_exc()
        return False

def classify_with_vit(query_image):
    """Classifie une image avec ViT"""
    try:
        start_time = time.time()
        
        # Extraire l'embedding
        query_embeddings = extract_vit_embeddings([query_image], vit_model_info)
        query_embedding = query_embeddings[0].flatten()
        
        # Calculer les similarités
        similarities = compute_cosine_similarity(query_embedding, cached_embeddings)
        
        # Trouver le meilleur match
        most_similar_idx = np.argmax(similarities)
        confidence = similarities[most_similar_idx]
        
        # Obtenir la classe
        predicted_label_idx = labels_db[most_similar_idx]
        predicted_label = class_names[predicted_label_idx]
        
        inference_time = time.time() - start_time
        
        return {
            'predicted_label': predicted_label,
            'confidence': float(confidence),
            'inference_time': inference_time
        }
        
    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Classification error: {str(e)}")

def get_top_predictions(query_embedding, top_k=5):
    """Obtient les top K prédictions"""
    query_embedding = query_embedding.flatten()
    similarities = compute_cosine_similarity(query_embedding, cached_embeddings)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    predictions = []
    for idx in top_indices:
        label_idx = labels_db[idx]
        predictions.append({
            'label': class_names[label_idx],
            'confidence': float(similarities[idx])
        })
    
    return predictions

def get_similar_images_all_methods(query_image, top_k=10):
    """Obtient les images similaires pour VIT et HISTOGRAM seulement"""
    try:
        all_similar_images = {}
        
        # Méthodes à utiliser (VIT et HISTOGRAM seulement)
        methods_to_use = ['VIT', 'HISTOGRAM']
        
        for method_name in methods_to_use:
            try:
                if method_name not in all_embeddings:
                    print(f"⚠️  Méthode {method_name} non disponible dans les embeddings")
                    continue
                    
                print(f"🔍 Recherche avec méthode: {method_name}")
                embeddings_db = all_embeddings[method_name]
                
                # Extraire l'embedding de la requête
                if method_name == 'HISTOGRAM':
                    query_embeddings = extract_histogram_embeddings([query_image])
                elif method_name == 'VIT':
                    query_embeddings = extract_vit_embeddings([query_image], embedding_models['VIT'])
                else:
                    continue
                
                if len(query_embeddings) == 0:
                    print(f"   ⚠️  Aucun embedding extrait pour {method_name}")
                    continue
                    
                query_embedding = query_embeddings[0].flatten()
                
                # Calculer les similarités
                similarities = compute_cosine_similarity(query_embedding, embeddings_db)
                
                # Obtenir les indices des top images similaires
                similar_indices = np.argsort(similarities)[::-1][:top_k]
                
                similar_images = []
                for idx in similar_indices:
                    similar_img = image_db[idx]
                    
                    # Convertir en base64
                    img_pil = Image.fromarray(similar_img)
                    buffered = io.BytesIO()
                    img_pil.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    label_idx = labels_db[idx]
                    
                    similar_images.append({
                        'image_data': img_str,
                        'label': class_names[label_idx],
                        'similarity': float(similarities[idx]),
                        'method': method_name
                    })
                
                all_similar_images[method_name] = similar_images
                print(f"✅ {method_name}: {len(similar_images)} images similaires trouvées")
                
            except Exception as e:
                print(f"⚠️  Erreur avec la méthode {method_name}: {e}")
                continue
        
        print(f"📊 Méthodes disponibles pour l'affichage: {list(all_similar_images.keys())}")
        return all_similar_images
        
    except Exception as e:
        print(f"❌ Erreur dans get_similar_images_all_methods: {e}")
        return {}

def calculate_method_precisions(all_similar_images, query_label=None):
    """Calcule la précision pour chaque méthode"""
    method_precisions = {}
    
    for method_name, images in all_similar_images.items():
        if not images:
            continue
            
        # 1. Précision basée sur la similarité moyenne
        similarities = [img['similarity'] for img in images]
        avg_similarity = np.mean(similarities)
        
        # 2. Précision basée sur les labels si query_label est fourni
        label_precision = None
        if query_label is not None:
            correct_matches = sum(1 for img in images if img['label'] == query_label)
            label_precision = correct_matches / len(images)
        
        method_precisions[method_name] = {
            'avg_similarity': float(avg_similarity),
            'similarity_precision': float(avg_similarity),
            'label_precision': float(label_precision) if label_precision is not None else None,
            'num_images': len(images),
            'max_similarity': float(max(similarities)),
            'min_similarity': float(min(similarities))
        }
    
    return method_precisions

@app.route('/classify', methods=['POST'])
def classify_image():
    """Endpoint pour classifier une image"""
    try:
        if not system_ready:
            return jsonify({'success': False, 'error': 'Système non initialisé'}), 500
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Aucune image fournie'}), 400
        
        # Traiter l'image
        image_file = request.files['image']
        img = Image.open(image_file.stream)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_resized = img.resize((224, 224))
        query_image = np.array(img_resized)
        
        # Classifier avec ViT
        result = classify_with_vit(query_image)
        
        # Obtenir l'embedding ViT pour les prédictions
        query_embeddings_vit = extract_vit_embeddings([query_image], vit_model_info)
        query_embedding_vit = query_embeddings_vit[0]
        
        # Obtenir les prédictions ViT
        top_predictions = get_top_predictions(query_embedding_vit, top_k=5)
        
        # Obtenir les images similaires pour VIT et HISTOGRAM
        similar_images_all_methods = get_similar_images_all_methods(query_image, top_k=10)
        
        # Calculer les précisions par méthode
        method_precisions = calculate_method_precisions(similar_images_all_methods, result['predicted_label'])
        
        # Obtenir les images similaires ViT (pour compatibilité)
        similar_images_vit = []
        if 'VIT' in similar_images_all_methods:
            similar_images_vit = similar_images_all_methods['VIT'][:6]
        
        return jsonify({
            'success': True,
            'predicted_label': result['predicted_label'],
            'confidence': result['confidence'],
            'top_predictions': top_predictions,
            'similar_images': similar_images_vit,
            'similar_images_all_methods': similar_images_all_methods,
            'method_precisions': method_precisions,
            'inference_time': result['inference_time'],
            'embedding_size': cached_embeddings.shape[1],
            'database_size': cached_embeddings.shape[0],
            'total_classes': len(class_names),
            'available_methods': list(similar_images_all_methods.keys())
        })
        
    except Exception as e:
        print(f"❌ Erreur /classify: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Endpoint pour obtenir les statistiques"""
    try:
        if not system_ready:
            return jsonify({'success': False, 'error': 'Système non initialisé'}), 500
        
        return jsonify({
            'success': True,
            'database_size': cached_embeddings.shape[0],
            'embedding_dimension': cached_embeddings.shape[1],
            'total_classes': len(class_names),
            'model': 'ViT-Base-16-224',
            'dataset': 'CIFAR-100',
            'available_methods': ['VIT', 'HISTOGRAM']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Vérification de santé"""
    return jsonify({
        'status': 'healthy' if system_ready else 'unhealthy',
        'system_ready': system_ready,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("="*60)
    print("🚀 Serveur Flask API pour Classification d'Images")
    print("Méthodes disponibles: VIT et HISTOGRAM")
    print("="*60)
    
    if initialize_system():
        print("\n✅ API prête sur http://localhost:5000")
        print("📌 Points de terminaison:")
        print("   • POST /classify - Classifier une image")
        print("   • GET /stats - Obtenir les statistiques")
        print("   • GET /health - Vérifier l'état du système")
        app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False)
    else:
        print("❌ Impossible d'initialiser le système")