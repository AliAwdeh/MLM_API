import os
import numpy as np
import faiss
import pandas as pd
import pickle
from utils.feature_extraction import extract_image_features
from sklearn.preprocessing import OneHotEncoder

# Directories
training_dir = 'images'
inventory_dir = 'inventory'
version_dir = 'model_versions'

# Ensure the version directory exists
if not os.path.exists(version_dir):
    os.makedirs(version_dir)

# Load and preprocess CSV data for training
styles_df = pd.read_csv('styles.csv')
styles_df = styles_df.drop(['id', 'productDisplayName'], axis=1)

encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(styles_df[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage']]).toarray()

# Initialize FAISS index for similarity search
dimension = encoded_features.shape[1] + 2048  # Metadata + image feature dimensions
index = faiss.IndexFlatL2(dimension)
combined_features = np.empty((0, dimension))

# Function to get the next model version
def get_next_version():
    existing_versions = [int(f.replace('mlmv', '')) for f in os.listdir(version_dir) if f.startswith('mlmv')]
    next_version = max(existing_versions, default=-1) + 1
    return f"mlmv{next_version}"

# Train the model on the training dataset
print("Training the model on the training dataset...")

for image_name in os.listdir(training_dir):
    image_path = os.path.join(training_dir, image_name)
    features = extract_image_features(image_path)
    combined_features = np.vstack((combined_features, np.hstack((encoded_features[0], features))))

# Add inventory images to the FAISS index
print("Indexing inventory images...")

for image_name in os.listdir(inventory_dir):
    image_path = os.path.join(inventory_dir, image_name)
    features = extract_image_features(image_path)
    combined_features = np.vstack((combined_features, np.hstack((encoded_features[0], features))))

index.add(combined_features.astype('float32'))

# Save the new model version
version = get_next_version()
version_path = os.path.join(version_dir, version)
os.makedirs(version_path)

faiss.write_index(index, os.path.join(version_path, f'{version}_index.bin'))
np.save(os.path.join(version_path, f'{version}_features.npy'), combined_features)

with open(os.path.join(version_path, f'{version}_encoder.pkl'), 'wb') as f:
    pickle.dump(encoder, f)

print(f"Model trained and saved successfully as version {version} on this device!")
