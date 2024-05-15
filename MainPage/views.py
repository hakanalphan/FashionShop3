# views.py
import pickle

from django.shortcuts import render, redirect
from .models import UploadedImage
from PIL import Image
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

# Load feature embeddings and filenames
feature_list = np.array(pickle.load(open(r'MainPage/featurevector.pkl', 'rb')))
# Assuming `filenames` is defined somewhere else in your code
filenames = [...]

# Load the pre-trained ResNet50 model with pre-trained weights
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

def save_uploaded_file(uploaded_file):
    try:
        image = UploadedImage(image=uploaded_file)
        image.save()
        return image.image.url
    except:
        return None

def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except:
        print(f"Error processing file: {img_path}")
        return None

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

def index(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('image')
        if uploaded_file:
            # Yüklenen dosyayı kaydet
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                # Özellikleri çıkar
                features = feature_extraction(file_path, model)
                if features is not None:
                    # Önerileri al
                    indices = recommend(features, feature_list)
                    # İlgili dosya adlarını al
                    recommended_files = [filenames[idx] for idx in indices[0]]
                    # Önerilen fotoğrafları görüntüleme şablonuyla birlikte gönder
                    return render(request, 'recommendations.html', {'recommended_files': recommended_files})
                else:
                    return render(request, 'index.html', {'message': 'Dosya işlenirken bir hata oluştu.'})
            else:
                return render(request, 'index.html', {'message': 'Dosya yüklenirken bir hata oluştu.'})
        else:
            return render(request, 'index.html', {'message': 'Lütfen bir fotoğraf seçin.'})
    return render(request, 'index.html')