# Brain-Hemorrhage-Detection-using-Hybrid-Deep-Learning


````markdown
# 🧠 Brain Hemorrhage Detection using Hybrid Deep Learning

This project aims to detect brain hemorrhage from CT scan images using a hybrid deep learning model that combines CNN, LSTM, and GRU architectures. It also uses Grad-CAM for visualizing the regions of the brain contributing to the prediction, aiding in interpretability and decision support.

## 📌 Features

- Detects brain hemorrhage from CT scan images
- Hybrid deep learning architecture: CNN + LSTM + GRU
- Uses Grad-CAM for heatmap visualization of critical regions
- High accuracy and interpretability for clinical relevance

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Grad-CAM for visual explanation

## 🧠 Model Architecture

- **CNN**: For feature extraction from CT images
- **LSTM + GRU**: For capturing spatial and sequential patterns in features
- Final dense layers for binary classification (hemorrhage / no hemorrhage)

## 📂 Dataset

- CT scan dataset sourced from a public medical imaging dataset
- Preprocessing includes:
  - Image resizing
  - Normalization
  - Data augmentation (rotation, flipping, etc.)

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com//Mhari2410//Brain-Hemorrhage-Detection-using-Hybrid-Deep-Learning.git
   cd Brain-Hemorrhage-Detection
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**

   ```bash
   python train_model.py
   ```

4. **Predict and visualize**

   ```bash
   python predict_with_gradcam.py --image path_to_ct_image.jpg
   ```

## 📊 Output

* Classification Report
* Grad-CAM heatmaps overlaid on CT scan images

## 🖼️ Sample Grad-CAM Output

> Shows the brain region influencing the model’s prediction
> 

## 📄 Project Report

Includes:

* Detailed explanation of hybrid architecture
* Dataset overview
* Model evaluation and metrics
* Grad-CAM interpretation
* Future scope and improvements

Refer to `Project_Report.pdf` for full documentation.

## 👩‍💻 Author

**Haripriya**
B.E. Computer Science and Engineering
P.A. College of Engineering and Technology

---

⭐ *If you found this project helpful, please star the repository and share it!*

```


