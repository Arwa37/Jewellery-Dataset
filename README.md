# 💎 Jewelry Visual Search Engine

An AI-powered visual search engine designed to find similar jewelry items (rings and necklaces) instantly using deep learning and vector embeddings.

## 🔗 Live Demo
Check out the live web application here: 
[Jewelry Visual Search App](https://jewellery-dataset.streamlit.app/#similar-items)

## 🚀 Features
* **Visual Similarity Search:** Upload any jewelry photo to find the top 5 most visually similar items from the dataset.
* **Deep Learning Powered:** Utilizes MobileNetV2 for feature extraction and vector embeddings.
* **Fast Vector Search:** Powered by ChromaDB for efficient similarity matching.
* **Interactive UI:** Built with Streamlit for a seamless user experience.

## 🛠️ Tech Stack
* **Python**
* **Streamlit** (Web Interface)
* **PyTorch & Torchvision** (Feature Extraction / MobileNetV2)
* **ChromaDB** (Vector Database)
* **PIL & NumPy** (Image Processing)

## ⚙️ How It Works
1. Upload a reference image of a ring or a necklace.
2. The application extracts deep visual features using the MobileNetV2 model.
3. ChromaDB queries the vector database to find the closest matches based on distance metrics.
4. The top 5 similar items along with their similarity percentages and images are displayed instantly.
