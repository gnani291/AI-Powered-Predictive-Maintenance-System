# 🤖 AI-Powered Predictive Maintenance System

> An end-to-end Machine Learning system that predicts equipment failures, estimates Remaining Useful Life (RUL), detects anomalies, and provides an interactive analytics dashboard for proactive maintenance.

<p align="center">
  <a href="https://predictivemaintenancedashboard.vercel.app/">🌐 Live Demo</a> •
  <a href="https://github.com/gnani291/AI-Powered-Predictive-Maintenance-System">💻 Source Code</a>
</p>

---

## 📌 Overview

Traditional maintenance strategies often lead to unexpected equipment failures and increased operational costs.

This project leverages **Machine Learning** and **Explainable AI** to analyze industrial sensor data, predict machine failures before they occur, estimate Remaining Useful Life (RUL), and visualize insights through an interactive dashboard. The dashboard presents machine health metrics, sensor trends, and maintenance recommendations in an intuitive interface.

**🌐 Live Demo:** https://predictivemaintenancedashboard.vercel.app/

---

## ✨ Features

* 🔧 Predict equipment failures
* 📈 Remaining Useful Life (RUL) prediction
* 🚨 Anomaly detection
* 📊 Interactive machine health dashboard
* 📉 Sensor trend visualization
* 🧠 Explainable AI (SHAP)
* ⚡ FastAPI backend for predictions
* 🏭 Multi-machine monitoring
* 📂 Modular and scalable project architecture

---

# 🏗 Workflow

```text
Industrial Sensor Data
          │
          ▼
 Data Preprocessing
          │
          ▼
 Feature Engineering
          │
          ▼
 Machine Learning Models
          │
 ┌────────┼────────┐
 ▼        ▼        ▼
Failure   RUL   Anomaly
Prediction Prediction Detection
          │
          ▼
 Explainable AI (SHAP)
          │
          ▼
 FastAPI Backend
          │
          ▼
 Interactive Dashboard
```

---

# 📂 Project Structure

```text
AI-Powered-Predictive-Maintenance-System/

├── api/
├── dashboard/
├── data/
├── models/
├── outputs/
├── src/
├── requirements.txt
└── README.md
```

---

# 🛠 Tech Stack

### Programming

* Python

### Machine Learning

* Scikit-learn
* Random Forest
* Gradient Boosting

### Data Processing

* Pandas
* NumPy

### Visualization

* Plotly
* Matplotlib

### Explainability

* SHAP

### Backend

* FastAPI

### Frontend

* HTML
* CSS
* JavaScript

---

# 📸 Dashboard Preview

# 🚀 Getting Started

Clone the repository

```bash
git clone https://github.com/gnani291/AI-Powered-Predictive-Maintenance-System.git

cd AI-Powered-Predictive-Maintenance-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the FastAPI server

```bash
uvicorn api.main:app --reload
```

Open the dashboard

```text
dashboard/index.html
```

---

# 🌐 Live Demo

**Interactive Dashboard**

https://predictivemaintenancedashboard.vercel.app/

---

# 📈 Future Enhancements

* Docker support
* MLflow experiment tracking
* GitHub Actions (CI/CD)
* Automated testing
* Cloud deployment
* Real-time IoT sensor integration
* Model monitoring

---

# 👨‍💻 Author

**Tadiparthi Gnaneswar**

* GitHub: https://github.com/gnani291

---

⭐ If you found this project useful, consider giving it a **Star**.
