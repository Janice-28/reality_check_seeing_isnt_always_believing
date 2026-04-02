
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


<img width="1440" height="672" alt="Screenshot 2026-04-02 at 4 31 19 PM" src="https://github.com/user-attachments/assets/cc2f5e07-b840-49c2-8f05-f19eaec32b73" />
<img width="1440" height="672" alt="Screenshot 2026-04-02 at 4 31 10 PM" src="https://github.com/user-attachments/assets/454ebbf2-1d17-4a10-8001-f9fe6b514bcc" />
<img width="1440" height="672" alt="Screenshot 2026-04-02 at 4 31 03 PM" src="https://github.com/user-attachments/assets/28c87a57-fb56-4fe4-9f24-8320eb84c83d" />
<img width="1440" height="672" alt="Screenshot 2026-04-02 at 4 30 39 PM" src="https://github.com/user-attachments/assets/18753936-dc3b-41a4-86b8-f6bbe6bfd870" />
<img width="1440" height="672" alt="Screenshot 2026-04-02 at 4 30 25 PM" src="https://github.com/user-attachments/assets/8b2f12e4-e718-4e56-b3d4-caaae66c113d" />
<img width="1440" height="672" alt="Screenshot 2026-04-02 at 4 30 19 PM" src="https://github.com/user-attachments/assets/6413876b-9247-4010-9485-cdd14eb311fa" />
<img width="1440" height="672" alt="Screenshot 2026-04-02 at 4 30 10 PM" src="https://github.com/user-attachments/assets/3534649b-e5d3-404f-8d88-0d6583a75833" />
<img width="1440" height="672" alt="Screenshot 2026-04-02 at 4 29 55 PM" src="https://github.com/user-attachments/assets/814eab7c-136f-455d-a266-00ac417fb5fc" />
<img width="1440" height="672" alt="Screenshot 2026-04-02 at 4 29 47 PM" src="https://github.com/user-attachments/assets/42f7e095-1ea3-468b-af24-8ea2c9e4d2f0" />

## 👩‍💻 Author

*Janice Mascarenhas*
BSc IT Graduate | Aspiring Data Engineer / AI Engineer
