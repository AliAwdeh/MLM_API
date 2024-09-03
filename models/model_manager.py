import os
import numpy as np
import faiss
import pandas as pd
import pickle
from utils.feature_extraction import extract_image_features
import logging

class ModelManager:
    def __init__(self):
        self.version_dir = 'model_versions'
        self.inventory_dir = 'inventory'
        self.index = None
        self.combined_features = None
        self.encoder = None

    def list_versions(self):
        versions = [f for f in os.listdir(self.version_dir) if f.startswith('mlmv')]
        return sorted(versions, key=lambda x: int(x.replace('mlmv', '')))

    def load_model(self, version):
        version_path = os.path.join(self.version_dir, version)
        
        self.index = faiss.read_index(os.path.join(version_path, f'{version}_index.bin'))
        self.combined_features = np.load(os.path.join(version_path, f'{version}_features.npy'))

        with open(os.path.join(version_path, f'{version}_encoder.pkl'), 'rb') as f:
            self.encoder = pickle.load(f)

        logging.info(f"Model version {version} loaded successfully from local storage!")

    def find_similar_items(self, query_vector, k=5):
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        return indices[0]

    def test_model(self, image_path, version):
        self.load_model(version)
        
        # Extract features from the query image
        query_features = extract_image_features(image_path)
        query_vector = np.hstack((np.zeros(self.encoder.transform([['gender_placeholder', 'category_placeholder', 'subCategory_placeholder', 'articleType_placeholder', 'baseColour_placeholder', 'season_placeholder', 'year_placeholder', 'usage_placeholder']]).toarray().shape[1]), query_features))
        
        # Find similar items in the inventory
        similar_indices = self.find_similar_items(query_vector, k=5)
        similar_ids = [os.path.basename(os.path.join(self.inventory_dir, f)).split('.')[0] for f in similar_indices]

        logging.info(f"Similar items found for image {image_path}: {similar_ids}")

        return similar_ids

    def is_valid_version(self, version):
        return version in self.list_versions()
