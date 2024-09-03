from flask import Flask, request, jsonify
from models.model_manager import ModelManager
from utils.logging_setup import setup_logging

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logger = setup_logging()

# Initialize model manager
model_manager = ModelManager()

@app.route('/train', methods=['POST'])
def train_model():
    logger.info("Received request to train the model.")
    model_manager.train_model()
    return jsonify({"message": "Model trained successfully!"})

@app.route('/test', methods=['POST'])
def test_model():
    image_path = request.json.get('image_path')
    version = request.json.get('version')

    if not image_path or not version:
        logger.error("Missing image path or version in the request.")
        return jsonify({"error": "Please provide both image path and version."}), 400

    if not model_manager.is_valid_version(version):
        logger.error(f"Invalid version: {version}")
        return jsonify({"error": "Invalid model version provided."}), 400

    logger.info(f"Testing model version {version} with image {image_path}.")
    similar_ids = model_manager.test_model(image_path, version)
    return jsonify({"similar_ids": similar_ids})

if __name__ == "__main__":
    app.run(debug=True)
