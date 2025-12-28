# app.py - Serveur Flask pour l'API
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
from PIL import Image
import io
import sys
import os

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les fonctions du système principal
from main_system import (
    extract_vit_embeddings, 
    embedding_models, 
    all_embeddings,
    image_db, 
    labels_db, 
    class_names,
    search_similar_images,
    cosine_similarity,
    labels_db
)

app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes

# Statistiques système
def get_system_stats():
    """Retourne les statistiques du système"""
    if 'VIT' in all_embeddings:
        embeddings = all_embeddings['VIT']
        embedding_size = embeddings.shape[1]
        database_size = embeddings.shape[0]
    else:
        embedding_size = 0
        database_size = 0
    
    # Calculer la précision moyenne (simulée)
    average_precision = 0.85  # À remplacer par calcul réel
    
    return {
        'average_precision': average_precision,
        'embedding_size': embedding_size,
        'database_size': database_size,
        'num_classes': len(np.unique(labels_db)) if labels_db is not None else 100
    }

@app.route('/classify', methods=['POST'])
def classify_image():
    """Classifie une image envoyée par l'utilisateur"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Lire l'image
        file = request.files['image']
        img = Image.open(file.stream)
        
        # Convertir en RGB si nécessaire
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionner à 32x32 (taille CIFAR-100)
        img_resized = img.resize((32, 32))
        img_array = np.array(img_resized)
        
        # Extraire les embeddings ViT
        if 'VIT' not in embedding_models or embedding_models['VIT'] is None:
            return jsonify({'error': 'ViT model not initialized'}), 500
        
        # Extraire l'embedding de l'image
        start_time = time.time()
        query_embedding = extract_vit_embeddings([img_array], embedding_models['VIT'])[0]
        
        # Vérifier si nous avons des embeddings de base
        if 'VIT' not in all_embeddings:
            return jsonify({'error': 'Database embeddings not found'}), 500
        
        # Calculer la similarité avec toutes les images
        database_embeddings = all_embeddings['VIT']
        similarities = cosine_similarity([query_embedding], database_embeddings)[0]
        
        # Trouver l'image la plus similaire
        most_similar_idx = np.argmax(similarities)
        confidence = similarities[most_similar_idx]
        
        # Obtenir le label
        if labels_db is not None and class_names is not None:
            predicted_label_idx = labels_db[most_similar_idx]
            predicted_label = class_names[predicted_label_idx]
        else:
            predicted_label = f"Class_{most_similar_idx}"
        
        # Obtenir les top 5 prédictions
        top_k = 5
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        top_predictions = []
        for idx in top_indices:
            label_idx = labels_db[idx] if labels_db is not None else idx
            label_name = class_names[label_idx] if class_names is not None else f"Class_{label_idx}"
            top_predictions.append({
                'label': label_name,
                'probability': float(similarities[idx])
            })
        
        # Obtenir des images similaires (top 6)
        similar_indices = np.argsort(similarities)[::-1][:6]
        similar_images = []
        
        for idx in similar_indices[1:]:  # Exclure la première (identique à la requête)
            similar_img = image_db[idx]
            
            # Convertir en base64 pour l'affichage
            img_pil = Image.fromarray(similar_img)
            buffered = io.BytesIO()
            img_pil.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            label_idx = labels_db[idx] if labels_db is not None else idx
            label_name = class_names[label_idx] if class_names is not None else f"Class_{label_idx}"
            
            similar_images.append({
                'image_data': img_str,
                'label': label_name,
                'similarity': float(similarities[idx])
            })
        
        inference_time = time.time() - start_time
        
        # Statistiques ViT
        vit_precision = 0.85  # À remplacer par calcul réel
        
        return jsonify({
            'success': True,
            'predicted_label': predicted_label,
            'confidence': float(confidence),
            'top_predictions': top_predictions,
            'vit_precision': vit_precision,
            'inference_time': inference_time,
            'embedding_size': f"{query_embedding.shape[0]} dimensions",
            'database_size': f"{database_embeddings.shape[0]} images",
            'similar_images': similar_images
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Retourne les statistiques du système"""
    stats = get_system_stats()
    return jsonify(stats)

@app.route('/health', methods=['GET'])
def health_check():
    """Vérifie l'état du serveur"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(embedding_models.keys()),
        'embeddings_available': list(all_embeddings.keys())
    })

if __name__ == '__main__':
    import time
    app.run(debug=True, port=5000, host='0.0.0.0')