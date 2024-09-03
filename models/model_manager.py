import os
import numpy as np
import faiss
import pandas as pd
import pickle
from utils.feature_extraction import extract_image_features
from sklearn.preprocessing import OneHotEncoder
import logging

class ModelManager:
    def __init__(self):
        self.version_dir = 'model_versions'
        self.image_dir = 'images'
        self.styles_df = pd.read_csv('styles.csv')
        self.styles_df = self.styles_df.drop(['id', 'productDisplayName'], axis=1)
        self.encoder = OneHotEncoder()
        self.encoded_features = self.encoder.fit_transform(self.styles_df[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage']]).toarray()
        self.index = None
        self.combined_features = None

        # Ensure the version directory exists
        if not os.path.exists(self.version_dir):
            os.makedirs(self.version_dir)

    def get_latest_version(self):
        existing_versions = [int(f.replace('mlmv', '')) for f in os.listdir(self.version_dir) if f.startswith('mlmv')]
        if not existing_versions:
            return None
        latest_version = max(existing_versions)
        return f"mlmv{latest_version}"

    def get_next_version(self):
        existing_versions = [int(f.replace('mlmv', '')) for f in os.listdir(self.version_dir) if f.startswith('mlmv')]
        next_version = max(existing_versions, default=-1) + 1
        return f"mlmv{next_version}"

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

    def train_model(self):
        latest_version = self.get_latest_version()
        
        if latest_version:
            logging.info(f"Loading the latest model version: {latest_version}")
            self.load_model(latest_version)
        else:
            logging.info("No previous model found. Initializing a new model.")
            dimension = self.encoded_features.shape[1] + 2048
            self.index = faiss.IndexFlatL2(dimension)
            self.combined_features = np.empty((0, dimension))

        image_features = []
        num_images = len(self.styles_df)
        
        for i, image_name in enumerate(self.styles_df.index):
            image_path = os.path.join(self.image_dir, str(image_name) + '.jpg')
            if os.path.exists(image_path):
                features = extract_image_features(image_path)
                image_features.append(features)
            else:
                image_features.append(np.zeros(2048))  # Assuming 2048-dimensional feature vectors from ResNet
            
            if (i + 1) % 100 == 0 or (i + 1) == num_images:
                logging.info(f"Processed {(i + 1) / num_images * 100:.2f}% of images")

        new_combined_features = np.hstack((self.encoded_features, np.array(image_features)))

        self.index.add(new_combined_features.astype('float32'))

        self.combined_features = np.vstack((self.combined_features, new_combined_features))

        version = self.get_next_version()
        version_path = os.path.join(self.version_dir, version)
        os.makedirs(version_path)
        
        faiss.write_index(self.index, os.path.join(version_path, f'{version}_index.bin'))
        np.save(os.path.join(version_path, f'{version}_features.npy'), self.combined_features)
        
        with open(os.path.join(version_path, f'{version}_encoder.pkl'), 'wb') as f:
            pickle.dump(self.encoder, f)

        logging.info(f"Model trained and saved successfully as version {version} on this device!")

    def find_similar_items(self, query_vector, k=5):
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        return indices[0]

    def test_model(self, image_path, version):
        self.load_model(version)
        
        # Extract features from the query image
        query_features = extract_image_features(image_path)
        query_vector = np.hstack((np.zeros(self.encoded_features.shape[1]), query_features))
        
        # Find similar items
        similar_indices = self.find_similar_items(query_vector, k=5)
        similar_ids = [self.styles_df.index[idx] for idx in similar_indices]

        # Log the IDs of the similar items found
        logging.info(f"Similar items found for image {image_path}: {similar_ids}")

        return similar_ids

    def is_valid_version(self, version):
        return version in self.list_versions()
