from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
MODEL_PATH = "E:/project4/brain_hemorrhage_model1.h5"  # Update with your model path
LAST_CONV_LAYER = "conv2d_2"  # Update this with the correct last convolutional layer

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the model
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

IMG_SIZE = (224, 224)

def is_valid_ct_scan(image_path):
    """
    Validates that the uploaded image is a brain CT scan.
    Checks include:
      - Verifying that the image is essentially grayscale by comparing color channels.
      - Ensuring the grayscale image has a sufficient intensity variation.
    """
    # Read the image in color to check for color differences.
    color_img = cv2.imread(image_path)
    if color_img is None:
        print("❌ Could not read the image for CT scan validation.")
        return False

    # Split into channels.
    b, g, r = cv2.split(color_img)
    diff1 = cv2.absdiff(b, g)
    diff2 = cv2.absdiff(b, r)
    diff_mean1 = np.mean(diff1)
    diff_mean2 = np.mean(diff2)
    
    # If differences between channels are significant, the image is not grayscale.
    if diff_mean1 > 10 or diff_mean2 > 10:
        print("❌ Color differences too high; image appears to be in color.")
        return False

    # Load image in grayscale for further checks.
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("❌ Could not read grayscale image for CT scan validation.")
        return False

    std_dev = np.std(gray)
    print(f"✅ CT scan validation: std_dev = {std_dev:.2f}")

    # If the standard deviation is too low, the image might be a scanned document or non-CT image.
    if std_dev < 20:
        print("❌ Standard deviation too low; likely not a CT scan image.")
        return False

    return True

# Load and preprocess image (convert to grayscale)
def load_and_preprocess_image(img_path):
    print(f"🔍 Loading image from: {img_path}")
    
    # Read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Error: Failed to read the image. Check the file format.")
    
    print(f"✅ Original Image Shape: {img.shape}")

    # Resize and normalize
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    img = img.astype("float32") / 255.0  # Normalize
    
    print(f"✅ Processed Image Shape: {img.shape}")
    return img

# Function to generate Grad-CAM heatmap
def get_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output.numpy()[0]
    heatmap = np.mean(conv_output * pooled_grads.numpy(), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Function to overlay heatmap on image
def overlay_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + original_img
    return np.uint8(superimposed_img)

# Process uploaded image and generate heatmap
def process_image(image_path):
    try:
        img_array = load_and_preprocess_image(image_path)
        heatmap = get_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
        original_img = cv2.imread(image_path)
        result = overlay_heatmap(original_img, heatmap)
        result_path = os.path.join(RESULT_FOLDER, "heatmap_output.jpg")
        cv2.imwrite(result_path, result)
        print(f"✅ Heatmap saved at: {result_path}")
        return result_path
    except Exception as e:
        print(f"❌ Error processing heatmap: {e}")
        return None

# Route to serve the home page
@app.route('/')
def home():
    return render_template("home.html")

# Route to serve the index page
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Save the file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print(f"✅ Image saved at: {filepath}")

        # Validate that the uploaded image is a CT scan
        if not is_valid_ct_scan(filepath):
            return jsonify({'error': 'Invalid image. Please upload a valid brain CT scan image.'})
        
        # Load and preprocess the image for prediction
        image = load_and_preprocess_image(filepath)
        prediction = model.predict(image)
        result = "Hemorrhagic" if prediction[0][0] > 0.5 else "Normal"
        print(f"🧠 Prediction: {result} (Confidence: {prediction[0][0]:.2f})")

        # Generate heatmap if prediction indicates hemorrhage
        if result == "Hemorrhagic":
            heatmap_path = process_image(filepath)
            return jsonify({'result': result, 'heatmap': heatmap_path})
        else:
            return jsonify({'result': result})

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return jsonify({'error': f'Prediction failed: {e}'})

if __name__ == '__main__':
    app.run(debug=True)
