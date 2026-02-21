# Oil Spill Detection & Marine Monitoring System

AI-Powered Environmental Surveillance Platform

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![DeepLearning](https://img.shields.io/badge/Deep%20Learning-3D%20CNN%20%2B%201D%20Model-red)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)
![Deployment](https://img.shields.io/badge/Deployment-Vercel-black)

---

## About The Project

Marine oil spills pose severe threats to ocean ecosystems, marine biodiversity, coastal economies, and public health. Traditional detection systems rely heavily on manual satellite interpretation and optical imagery, which:

* Fail during cloudy or nighttime conditions
* Cannot accurately detect thin oil sheens
* Require expensive proprietary tools
* Depend on expert-level analysis
* Delay emergency response

This project introduces an automated AI-powered monitoring system that evolves across three progressive versions:

* Oil Spill Detection using SAR + Hyperspectral Imagery
* Marine Life Detection and Ecosystem Monitoring
* Intelligent Chatbot with Real-Time Alert System

The final system achieves:

* 96% Classification Accuracy
* Automated Spill Area and Volume Estimation
* All-weather, Day/Night Detection
* Reduced False Positives
* Near Real-Time Alert System
* Web Deployment (Vercel)

Developed under HACK YOUR PATH 7.0 – HITAM Hackathon Club
Team Name: ODA

---

## Built With

### Deep Learning and AI

* PyTorch
* Keras
* DeepHyperX
* Hybrid 3D CNN Architecture
* 1D Hamida Spectral Model
* YOLOv8 

### Data Processing and Analysis

* NumPy
* OpenCV
* scikit-learn (PCA for dimensionality reduction)
* Geospatial libraries

### Visualization

* Matplotlib

### Deployment

* GitHub (Version Control and Final Hosting)
* Vercel (Website Deployment – Version 3)

---

## Getting Started

To get a local copy up and running follow these steps.

### Prerequisites

* Python 3.8+
* pip or conda
* CUDA-enabled GPU (recommended for training)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/oil-spill-detection.git
cd oil-spill-detection
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train.py
```

### 5. Run inference

```bash
python predict.py
```

---

## Project Structure

```
oil-spill-detection/
│
├── version-1/
├── version-2/
├── version-3/
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## Project Versions

### Version 1 – Core Oil Spill Detection System

Objective: Detect oil spills using fused SAR and Hyperspectral imagery.

Key Features:

* Fusion of Hyperspectral and Sentinel-1 SAR data
* PCA reduction (224 → 40 spectral bands)
* Hybrid 3D CNN + 1D Hamida Model
* Automated oil spill segmentation
* Spill area and volume estimation
* 96% classification accuracy

Version 1 establishes the core AI detection engine.

---

### Version 2 – Marine Ecosystem Monitoring

Objective: Extend Version 1 by integrating Marine Life Detection.

Improvements:

* Multi-class segmentation (Oil / Water / Marine Life)
* Detection of ecological impact zones
* Environmental risk analysis
* Broader ocean monitoring capability

Version 2 expands the system into a marine ecosystem monitoring platform.

---

### Version 3 – Intelligent Alert and Deployment System (Final)

Objective: Transform the detection system into a real-time intelligent platform.

Enhancements:

* AI Chatbot for spill-related queries
* Voice-enabled chatbot interaction
* Near real-time Email alerts
* SMS notifications to higher authorities
* Interactive monitoring dashboard
* Web deployment via Vercel

Deployment:

* Final production version hosted on GitHub
* Web interface deployed on Vercel

Version 3 delivers a complete intelligent environmental monitoring solution.

---

## Roadmap

### Completed

* Hyperspectral preprocessing (PCA)
* SAR preprocessing (Lee filtering, calibration)
* Hybrid deep learning model
* Area and volume estimation
* 96% detection accuracy

### In Progress

* Attention mechanisms
* Multi-scale feature fusion
* Real-time inference optimization

### Future Enhancements

* Spill trajectory prediction
* Multi-pollutant detection
* Edge deployment
* Cloud-native scaling

---

## Performance Metrics

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 96%   |
| Precision | 94.5% |
| Recall    | 97.2% |
| F1 Score  | 95.8% |
| IoU       | 89.3% |

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create your feature branch

   ```
   git checkout -b feature/NewFeature
   ```
3. Commit your changes

   ```
   git commit -m "Added new feature"
   ```
4. Push to branch

   ```
   git push origin feature/NewFeature
   ```
5. Open a Pull Request

Please ensure:

* Code is well-documented
* Follows Python standards
* Includes testing where applicable

---

## License

This project is licensed under the MIT License.
You are free to use, modify, and distribute with proper attribution.

---

## Contact

Team Name: ODA
Team Leader: Kushal Manikonda

Team Members:

* I Aravind
* K Sai Revanth
* D Priyansh

College: Vasavi College of Engineering
Hackathon: HACK YOUR PATH 7.0

Mobile: +91 9959583328

---

## Acknowledgements

* HITAM Hackathon Club
* Vasavi College of Engineering
* IEEE DataPort
* Sentinel-1 Satellite Mission
* Open-source Deep Learning Community

