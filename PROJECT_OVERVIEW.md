# BioLock AI: Panduan Lengkap Proyek Keamanan Biometrik (Project Overview)

Dokumen ini menyediakan breakdown lengkap untuk proyek BioLock AI, mencakup arsitektur, fitur, dan implementasi teknisnya. Gunakan ini sebagai panduan/skrip untuk video penjelasan kamu.

## 🎯 1. Visi Proyek
BioLock AI adalah sistem keamanan biometrik real-time yang dirancang untuk memberikan otentikasi user yang mulus namun tetap kuat (robust). Proyek ini memanfaatkan computer vision dan AI untuk memastikan hanya user yang terdaftar yang memiliki akses, sambil terus memonitor ancaman keamanan secara terus-menerus.

## 🛠 2. Technical Stack
| Kategori | Teknologi |
| :--- | :--- |
| **Language** | Python 3.x |
| **Backend** | Flask (Utama), FastAPI (Alternatif/Secondary) |
| **Computer Vision** | OpenCV (cv2) |
| **Face AI** | MediaPipe Face Mesh |
| **Algoritma** | ORB (Oriented FAST and Rotated BRIEF) untuk ekstraksi fitur |
| **Frontend** | HTML5, CSS3, JavaScript (Real-time Video Streaming) |
| **Data Storage** | Pickle (Binary DB), CSV (Event Logs) |

## ✨ 3. Fitur Utama & Cara Kerjanya

### 🆕 A. User Enrollment (Berbasis ORB)
- **Apa itu**: Sistem "mempelajari" user baru dengan mengambil beberapa snapshot wajah.
- **Detail Teknis**: Menggunakan algoritma ORB untuk mengekstrak titik-titik fitur (feature points) unik dari ROI (Region of Interest) wajah. Deskriptor ini disimpan dalam dictionary biner (`faces_db.pkl`).
- **Tips Video**: Tunjukkan layar enrollment dan overlay "RECORDING..." saat proses pengambilan sampel.

### 🔍 B. Face Recognition Real-time
- **Apa itu**: Mengidentifikasi user yang sudah terdaftar secara langsung melalui feed kamera.
- **Detail Teknis**: Menggunakan `BFMatcher` (Brute-Force Matcher) dengan Hamming Distance untuk membandingkan deskriptor live dengan yang ada di database.
- **Async Processing**: Menggunakan `RecognitionWorker` (threaded) untuk menangani logika pencocokan yang berat di luar main thread video, sehingga performa video tetap smooth (30+ FPS).

### 🛡 C. Liveness Detection (Challenge-Response)
- **Apa itu**: Mencegah "spoofing" (misalnya, orang yang menodongkan foto ke kamera) dengan mewajibkan interaksi fisik.
- **Tantangan**: Secara acak sistem akan meminta kamu untuk **Blink** (kedip), **Smile** (senyum), **Frown** (merengut), atau **Surprised** (kaget).
- **Matematika**: Menggunakan "EAR" (Eye Aspect Ratio) untuk kedipan dan kalkulasi curvatue pada landmark bibir untuk emosi.

### 👁 D. Attention Guard (Privacy Mode)
- **Apa itu**: Secara otomatis mendeteksi jika user sedang tidak melihat ke layar atau jika ada orang lain yang mengintip.
- **Detail Teknis**: Menghitung "Head Pose" (Pitch, Yaw, Roll) menggunakan `solvePnP`. Jika user menoleh (Yaw > 50°), sistem akan memicu status privasi.

### 🚨 E. Intruder Alert & Logging
- **Apa itu**: Mendeteksi dan mencatat percobaan akses yang tidak sah.
- **Tindakan**: Jika wajah "Unknown" terdeteksi lebih dari 150 frame, sistem akan mengambil snapshot resolusi tinggi, menyimpannya ke `data/security_log/`, dan mengirim alert ke UI.

## 🏗 4. Arsitektur Kode (The "Guts")

### 📂 Struktur File
- `MAIN.py`: Jantung dari aplikasi. Berisi server Flask, loop generator video, dan semua logika AI.
- `server.py`: Server sekunder berbasis FastAPI untuk menunjukkan arsitektur yang berbeda.
- `templates/`: Komponen UI yang modern dan responsif.
- `known/`: Menyimpan thumbnail wajah untuk identifikasi visual.

### 🔄 Data Flow
1. **Input**: Kamera menangkap frame melalui OpenCV.
2. **Analysis**: MediaPipe mendeteksi landmark wajah.
3. **Threading**: ROI dikirim ke thread `RecognitionWorker`.
4. **Validation**: Pengecekan Liveness dan Attention berjalan secara bersamaan (concurrently).
5. **Output**: Frame yang sudah diproses di-encode ke JPEG dan di-stream ke browser melalui route `/video_feed`.

## 📸 5. Bagian Menarik untuk Ditampilkan di Video
1. **Verifikasi Liveness**: Rekam diri kamu saat melakukan tantangan "Action: SMILE".
2. **Multi-Face Support**: Tunjukkan sistem melacak beberapa orang sekaligus tetapi hanya mengizinkan "Face 0" (user utama).
3. **Intruder Alert**: Minta teman masuk ke frame dan tunjukkan "Privacy Mode" aktif atau snapshot intruder yang tersimpan di folder log.
4. **Settings Panel**: Tunjukkan perubahan real-time pada sensitivitas dan smoothing.

---
> [!TIP]
> Fokuskan video pada fitur **Liveness Detection** dan **Attention Guard**; ini adalah bagian yang paling menarik secara visual dan unik secara teknis di proyek ini!
