from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from CNN import CNN
from torchvision import transforms
from PIL import Image
import io
import os

app = Flask(__name__)

# allow who to access
if os.environ.get('FLASK_ENV') == 'dev':
    CORS(app)
else:
    CORS(app, resources={r"/predict": {"origins": "https://somedomain.com"}})

# Load model
model = CNN(output_classes=15)
model.load_state_dict(torch.load("traffic_model.pth", weights_only=True))
model.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/predict", methods=["POST"])
def predict():
    # Check if file is in the request
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        # Read the image
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Perform prediction
        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()

        return jsonify({"class_id": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()