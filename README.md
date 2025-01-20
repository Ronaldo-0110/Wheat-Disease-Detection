# Wheat Disease Detection and Fertilizer Recommendation

### 🌾 Overview
This project focuses on leveraging machine learning for accurate classification of wheat diseases through image recognition. By enabling early detection, the model helps reduce crop losses and enhances agricultural productivity. Additionally, it integrates diverse datasets to analyze production trends, offering actionable insights for agricultural optimization and yield prediction.

---

### 📌 Key Features
- **Disease Detection:** Uses image recognition techniques to classify wheat diseases with high accuracy.
- **Fertilizer Recommendation:** Provides tailored fertilizer suggestions to mitigate the impact of detected diseases.
- **Trend Analysis:** Combines production and environmental datasets to predict yield and optimize agricultural practices.

---

### 🛠️ Technologies Used
- **Programming Language:** Python
- **Libraries and Frameworks:**
  - TensorFlow/Keras: For building and training the image recognition model
  - OpenCV: For image preprocessing
  - Pandas & NumPy: For data manipulation and analysis
  - Matplotlib & Seaborn: For visualizations
- **Machine Learning Techniques:**
  - Convolutional Neural Networks (CNNs) for image classification
  - Regression models for yield prediction
- **Data Sources:** Public agricultural datasets and field data

---

### 🚀 How It Works
1. **Image Processing:**
   - Input wheat leaf images are preprocessed using OpenCV to enhance feature extraction.
   - Augmentation techniques are applied to improve model robustness.

2. **Disease Classification:**
   - A CNN model is trained to identify diseases such as rust, mildew, and blight from the images.
   - The model predicts the probability of each disease category for a given input image.

3. **Fertilizer Recommendation:**
   - Based on the identified disease, the system suggests suitable fertilizers and pest control measures.

4. **Yield Prediction:**
   - Historical production data is analyzed alongside disease occurrence to forecast yield trends.

---

### 📂 Project Structure
```
├── data
│   ├── raw_data          # Raw datasets
│   ├── processed_data    # Processed and cleaned datasets
├── models
│   ├── cnn_model.h5      # Trained CNN model for disease detection
├── notebooks
│   ├── data_analysis.ipynb   # Data exploration and trend analysis
│   ├── model_training.ipynb  # Model development and training
├── src
│   ├── preprocess.py     # Image preprocessing functions
│   ├── predict.py        # Disease detection and fertilizer recommendation
├── README.md
```

---

### 📈 Results
- Achieved **X% accuracy** in classifying wheat diseases on a test dataset.
- Reduced misclassification rates by incorporating advanced augmentation techniques.
- Enabled actionable fertilizer recommendations, improving crop recovery rates by **Y%**.
- Analyzed production trends to predict yield with an error margin of **Z%**.

---

### 🛡️ Future Scope
- Expand the model to support other crop types.
- Integrate real-time data from IoT sensors for enhanced predictions.
- Deploy the model as a web-based application for easy farmer access.

---

### 🤝 Contributing
Contributions are welcome! If you'd like to improve the project, feel free to fork the repository and submit a pull request.

---

### 📬 Contact
For any questions or suggestions, reach out via:
- **Email:** ronaldosjr07@gmail.com

- **GitHub Issues:** [Open an Issue](https://github.com/yourusername/wheat-disease-detection/issues)

---

### 📝 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

