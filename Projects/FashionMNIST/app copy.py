# app.py

import os
import io
import random
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, send_file
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import FashionMNIST
from FashionMNIST import loaded_cnn_model, class_names
app = Flask(__name__)

def load_model():
    model = loaded_cnn_model
    model.load_state_dict(torch.load("CNN_model.pth"))
    model.eval()
    return model

def random_image():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    idx = random.randint(0, len(dataset) - 1)
    img, label = dataset[idx]

    return img, label

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/random_image')
def get_random_image():
    img, label = random_image()
    img = (img * 255).byte().squeeze().numpy()
    img_pil = Image.fromarray(img, mode='L')
    
    # Resize the image to 512x512 pixels
    img_pil = img_pil.resize((512, 512), Image.ANTIALIAS)
    
    # Load the model
    model = load_model()

      # Convert img back to tensor and perform the prediction
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pred = model(img_tensor)
    
    # Get the predicted class name
    pred_class = class_names[torch.argmax(pred, 1).item()]

    # Get the ground truth label
    truth_label = class_names[label]

    # Add text to the image
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Predicted: {pred_class} | Truth: {truth_label}", font=font, fill=255)

    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
