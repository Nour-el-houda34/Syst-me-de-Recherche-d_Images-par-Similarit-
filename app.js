// Configuration
const API_BASE_URL = 'http://localhost:5000';
let selectedFile = null;

// Initialisation
document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ Application initialisée');
    
    // Masquer la section des résultats initialement
    document.getElementById('resultsSection').style.display = 'none';
    
    // Initialiser les éléments
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const browseBtn = document.getElementById('browseBtn');
    const classifyBtn = document.getElementById('classifyBtn');
    const resetBtn = document.getElementById('resetBtn');
    
    // Vérifier la connexion API
    checkApiConnection();
    
    // Événements
    browseBtn.addEventListener('click', () => {
        console.log('📁 Ouverture du sélecteur de fichiers');
        imageInput.click();
    });
    
    imageInput.addEventListener('change', handleFileSelect);
    
    // Glisser-déposer
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Boutons
    classifyBtn.addEventListener('click', classifyImage);
    resetBtn.addEventListener('click', resetAll);
});

// Vérifier la connexion API
async function checkApiConnection() {
    try {
        console.log('🔌 Vérification de la connexion API...');
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.system_ready) {
            console.log('✅ API connectée et prête');
            showNotification('✅ Système prêt à classifier les images', 'success');
        } else {
            console.warn('⚠️ API en cours d\'initialisation');
            showNotification('⚠️ Initialisation du système en cours...', 'warning');
            setTimeout(checkApiConnection, 3000);
        }
    } catch (error) {
        console.error('❌ Impossible de se connecter à l\'API:', error);
        showNotification('❌ Impossible de se connecter au serveur. Vérifiez que flask_api.py est en cours d\'exécution.', 'error');
    }
}

// Gestion des fichiers
function handleFileSelect(event) {
    console.log('📄 Fichier sélectionné');
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
        selectedFile = file;
        updatePreview();
        document.getElementById('classifyBtn').disabled = false;
        showNotification(`✅ Image "${file.name}" sélectionnée`, 'success');
    } else if (file) {
        showNotification('❌ Format de fichier non supporté', 'error');
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        selectedFile = file;
        updatePreview();
        document.getElementById('classifyBtn').disabled = false;
        showNotification(`✅ Image "${file.name}" déposée`, 'success');
    } else {
        showNotification('❌ Format de fichier non supporté', 'error');
    }
}

// Mettre à jour la prévisualisation
function updatePreview() {
    const preview = document.getElementById('imagePreview');
    const placeholder = preview.querySelector('.preview-placeholder');
    const previewImage = document.getElementById('previewImage');
    
    if (!selectedFile) {
        placeholder.style.display = 'block';
        previewImage.style.display = 'none';
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        placeholder.style.display = 'none';
        previewImage.style.display = 'block';
        previewImage.style.animation = 'fadeIn 0.5s ease-out';
    };
    reader.readAsDataURL(selectedFile);
}

// Classifier l'image
async function classifyImage() {
    if (!selectedFile) {
        showNotification('⚠️ Veuillez sélectionner une image d\'abord', 'warning');
        return;
    }
    
    console.log('🔍 Début de la classification...');
    showLoading(true, "Extraction des caractéristiques...");
    
    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        console.log('📤 Envoi de l\'image à l\'API...');
        const response = await fetch(`${API_BASE_URL}/classify`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Erreur inconnue lors de la classification');
        }
        
        console.log('✅ Classification réussie:', data.predicted_label);
        console.log('📊 Données reçues:', {
            methods: data.available_methods,
            confidence: data.confidence
        });
        
        // Afficher les résultats
        displayResults(data);
        
        // Afficher la section des résultats
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'block';
        resultsSection.style.animation = 'fadeIn 0.5s ease-out';
        
        showNotification(`✅ Classification réussie: ${data.predicted_label}`, 'success');
        
    } catch (error) {
        console.error('❌ Erreur:', error);
        showNotification(`❌ Erreur: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Afficher les résultats par méthode d'embedding
function displayEmbeddingMethodsResults(similarImagesAllMethods, methodPrecisions) {
    console.log('🎨 Affichage des méthodes...');
    
    const container = document.getElementById('methodsContainer');
    if (!container) {
        console.error('❌ Container methodsContainer non trouvé!');
        return;
    }
    
    if (!similarImagesAllMethods || Object.keys(similarImagesAllMethods).length === 0) {
        container.innerHTML = `
            <div class="method-section">
                <div class="method-header">
                    <div class="method-title">
                        <div class="method-icon">
                            <i class="fas fa-exclamation-triangle"></i>
                        </div>
                        <span class="method-name">Aucune donnée disponible</span>
                    </div>
                </div>
                <p>Aucune image similaire trouvée.</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    
    // Pour chaque méthode
    for (const [methodName, images] of Object.entries(similarImagesAllMethods)) {
        if (images.length === 0) continue;
        
        // Obtenir la précision
        let precision = 'N/A';
        if (methodPrecisions && methodPrecisions[methodName]) {
            precision = (methodPrecisions[methodName].avg_similarity * 100).toFixed(1);
        } else {
            // Calculer la moyenne
            const avg = images.reduce((sum, img) => sum + img.similarity, 0) / images.length;
            precision = (avg * 100).toFixed(1);
        }
        
        // Icône selon la méthode
        let icon = 'fa-question';
        let color = '#4ECDC4';
        
        if (methodName === 'VIT') {
            icon = 'fa-robot';
            color = '#4ECDC4';
        } else if (methodName === 'HISTOGRAM') {
            icon = 'fa-chart-bar';
            color = '#FF6B6B';
        }
        
        html += `
            <div class="method-section">
                <div class="method-header">
                    <div class="method-title">
                        <div class="method-icon" style="background: linear-gradient(135deg, ${color}, ${color}88)">
                            <i class="fas ${icon}"></i>
                        </div>
                        <div>
                            <span class="method-name">${methodName}</span>
                            <div class="method-description">
                                ${methodName === 'VIT' ? 'Vision Transformer - Modèle d\'attention' : 'Histogramme de couleurs - Méthode traditionnelle'}
                            </div>
                        </div>
                    </div>
                    <div class="method-precision">${precision}%</div>
                </div>
                
                <div class="method-images-grid">
        `;
        
        // Ajouter les images
        images.forEach((img, index) => {
            // Couleur de l'overlay selon la similarité
            let qualityClass = 'low';
            if (img.similarity > 0.7) qualityClass = 'high';
            else if (img.similarity > 0.4) qualityClass = 'medium';
            
            html += `
                <div class="method-image-item">
                    <div class="image-container">
                        <img src="data:image/jpeg;base64,${img.image_data}" alt="${img.label}">
                        <div class="image-overlay ${qualityClass}">${(img.similarity * 100).toFixed(0)}%</div>
                    </div>
                    <div class="method-similarity">${(img.similarity * 100).toFixed(1)}%</div>
                    <div class="method-label" title="${img.label}">${img.label}</div>
                </div>
            `;
        });
        
        html += `
                </div>
                
                <div class="method-stats">
                    <div class="method-stat">
                        <span class="stat-value">${images.length}</span>
                        <span class="stat-label">Images</span>
                    </div>
                    <div class="method-stat">
                        <span class="stat-value">${precision}%</span>
                        <span class="stat-label">Précision</span>
                    </div>
                    <div class="method-stat">
                        <span class="stat-value">${(Math.max(...images.map(img => img.similarity)) * 100).toFixed(1)}%</span>
                        <span class="stat-label">Meilleure</span>
                    </div>
                    <div class="method-stat">
                        <span class="stat-value">${(Math.min(...images.map(img => img.similarity)) * 100).toFixed(1)}%</span>
                        <span class="stat-label">Moins bonne</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    container.innerHTML = html;
    console.log('✅ Méthodes affichées avec succès');
}

// Afficher les résultats
function displayResults(data) {
    console.log('🖥️  Affichage des résultats...');
    
    // 1. Classe prédite et confiance
    document.getElementById('predictedClass').textContent = data.predicted_label;
    document.getElementById('confidenceValue').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    
    // 2. Top 5 prédictions
    const predictionsList = document.getElementById('predictionsList');
    predictionsList.innerHTML = '';
    
    if (data.top_predictions && data.top_predictions.length > 0) {
        data.top_predictions.forEach((pred, index) => {
            const predictionItem = document.createElement('div');
            predictionItem.className = 'prediction-item';
            predictionItem.style.animationDelay = `${index * 0.1}s`;
            predictionItem.innerHTML = `
                <span class="class-name">${pred.label}</span>
                <span class="class-prob">${(pred.confidence * 100).toFixed(1)}%</span>
            `;
            predictionsList.appendChild(predictionItem);
        });
    }
    
    // 3. Métriques de base
    document.getElementById('vitPrecision').textContent = '85.0%'; // Valeur par défaut
    document.getElementById('inferenceTime').textContent = `${data.inference_time.toFixed(3)}s`;
    document.getElementById('embeddingSize').textContent = `${data.embedding_size} dimensions`;
    document.getElementById('databaseSize').textContent = `${data.database_size} images`;
    
    // 4. Afficher les résultats par méthode d'embedding
    if (data.similar_images_all_methods) {
        displayEmbeddingMethodsResults(data.similar_images_all_methods, data.method_precisions);
        document.getElementById('embeddingMethodsResults').style.display = 'block';
    } else {
        document.getElementById('embeddingMethodsResults').style.display = 'none';
    }
    
    // 5. Images similaires ViT (section originale)
    const similarImagesGrid = document.getElementById('similarImagesGrid');
    similarImagesGrid.innerHTML = '';
    
    if (data.similar_images && data.similar_images.length > 0) {
        document.getElementById('similarImagesSection').style.display = 'block';
        
        data.similar_images.forEach((img, index) => {
            const imgDiv = document.createElement('div');
            imgDiv.className = 'similar-image';
            imgDiv.style.animationDelay = `${index * 0.1}s`;
            imgDiv.innerHTML = `
                <img src="data:image/jpeg;base64,${img.image_data}" alt="${img.label}">
                <div class="similarity">${(img.similarity * 100).toFixed(1)}%</div>
                <div class="image-label">${img.label}</div>
            `;
            similarImagesGrid.appendChild(imgDiv);
        });
    }
}

// Afficher/masquer le chargement
function showLoading(show, message = '') {
    const overlay = document.getElementById('loadingOverlay');
    const messageElement = document.getElementById('loadingMessage');
    
    if (show) {
        if (message) messageElement.textContent = message;
        overlay.style.display = 'flex';
    } else {
        overlay.style.display = 'none';
    }
}

// Afficher une notification
function showNotification(message, type = 'info') {
    // Supprimer les notifications existantes
    document.querySelectorAll('.notification').forEach(n => n.remove());
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    let icon = 'info-circle';
    if (type === 'success') icon = 'check-circle';
    if (type === 'error') icon = 'exclamation-circle';
    if (type === 'warning') icon = 'exclamation-triangle';
    
    notification.innerHTML = `
        <i class="fas fa-${icon}"></i>
        <span>${message}</span>
    `;
    
    // Styles inline pour la notification
    notification.style.cssText = `
        position: fixed;
        top: 25px;
        right: 25px;
        background: ${type === 'success' ? '#2ECC71' : 
                     type === 'error' ? '#E74C3C' : 
                     type === 'warning' ? '#F39C12' : '#3498DB'};
        color: white;
        padding: 18px 24px;
        border-radius: 10px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        z-index: 10000;
        display: flex;
        align-items: center;
        gap: 15px;
        animation: slideIn 0.3s ease-out;
        max-width: 450px;
        font-size: 1.05rem;
        font-weight: 500;
        border-left: 5px solid ${type === 'success' ? '#27ae60' : 
                              type === 'error' ? '#c0392b' : 
                              type === 'warning' ? '#e67e22' : '#2980b9'};
    `;
    
    document.body.appendChild(notification);
    
    // Supprimer après 5 secondes
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// Réinitialiser
function resetAll() {
    console.log('🔄 Réinitialisation...');
    selectedFile = null;
    document.getElementById('imageInput').value = '';
    document.getElementById('classifyBtn').disabled = true;
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('embeddingMethodsResults').style.display = 'none';
    updatePreview();
    showNotification('Système réinitialisé. Prêt pour une nouvelle classification.', 'info');
}