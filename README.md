# VisionAI: Vehicle Detection

A real-time web application that uses computer vision to detect vehicles in user-uploaded images. 

Powered by **Streamlit** and **TensorFlow's MobileNetV2** (pre-trained on ImageNet), this application processes images locally and provides a confidence metric along with the specific vehicle classification.

## Features
* **Pre-trained AI:** Utilizes MobileNetV2 for fast, highly accurate predictions without needing local model weights.
* **Live Web Interface:** Built with Streamlit for a responsive, interactive user experience.
* **Smart Confidence Metrics:** Displays the primary classification and top 3 raw neural network telemetry results.

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Mohamad-101/Car_classifier.git
   cd Car_classifier


2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Launch the app:
    ```bash
   streamlit run app.py