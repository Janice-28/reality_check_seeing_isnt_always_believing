
# 🔍 Reality Check – Deepfake Detection Platform

Reality Check is an AI-powered web application that detects deepfake images and videos using deep learning. The goal of this project is to help users verify the authenticity of digital media and combat misinformation.

---

## 🚀 Features

* 🖼️ Deepfake detection for images
* 🎥 Video deepfake detection using frame extraction
* 📊 Frame-by-frame analysis with overall prediction
* 🤖 TensorFlow-based deep learning model
* 🌐 Django backend for API handling
* 💻 User-friendly frontend for uploading and viewing results

---

## 🧠 How It Works

1. User uploads an image or video
2. For videos:

   * Frames are extracted using video processing
3. Each frame/image is passed through the trained deep learning model
4. The model predicts whether the content is **Real or Fake**
5. Results are displayed:

   * Frame-level predictions (for videos)
   * Overall authenticity score

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript (or Flutter if applicable)
* **Backend:** Django (Python)
* **Machine Learning:** TensorFlow, OpenCV
* **Other Tools:** NumPy, Pandas

---

## 📂 Project Structure

```
Reality-Check/
│
├── frontend/              # UI for uploading media
├── backend/              # Django backend
│   ├── api/              # API endpoints
│   ├── models/           # ML model integration
│   └── utils/            # Frame extraction & processing
│
├── model/                # Trained deepfake detection model
├── media/                # Uploaded files
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/reality-check.git
cd reality-check
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Django server

```bash
python manage.py runserver
```

---

## 📌 Future Enhancements

* 🔎 Reverse image search integration (Google, Reddit, Twitter)
* 🔔 Real-time deepfake alerts
* 🔐 User authentication (Firebase)
* 📈 Improved model accuracy with advanced architectures

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Janice Mascarenhas**
BSc IT Graduate | Aspiring Data Engineer / AI Engineer

---

If you want, I can next:

* Make it **more “resume-level impressive” with metrics (like accuracy, dataset, model type)**
* Or tailor it specifically for **placements / recruiters (very high impact version)**
