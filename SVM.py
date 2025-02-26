import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset from CSV
color_data = pd.read_csv('colors.csv')
X = color_data[['R', 'G', 'B']].values
y = color_data['color_name'].values

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Pilih model SVM
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Akurasi pada data latih: {train_acc * 100:.2f}%")
print(f"Akurasi pada data uji: {test_acc * 100:.2f}%")

# Simpan model dan scaler
joblib.dump(model, 'color_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model dan scaler berhasil disimpan!")

# Muat kembali model dan scaler untuk deteksi warna
model = joblib.load('color_model.pkl')
scaler = joblib.load('scaler.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

    # Konversi gambar ke HSV untuk deteksi warna
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Gunakan inRange untuk deteksi warna utama
    lower_bound = np.array([0, 50, 50])  # Nilai HSV bawah
    upper_bound = np.array([180, 255, 255])  # Nilai HSV atas
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Temukan kontur dari area yang dideteksi
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue  # Abaikan area kecil untuk meningkatkan akurasi

        x, y, w, h = cv2.boundingRect(cnt)
        avg_color = frame[y:y + h, x:x + w].mean(axis=(0, 1))
        avg_color_scaled = scaler.transform(avg_color.reshape(1, -1))
        color_pred = model.predict(avg_color_scaled)[0]
        prob = model.predict_proba(avg_color_scaled).max() * 100
        box_color = tuple(map(int, avg_color))

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(frame, f'{color_pred} ({prob:.2f}%)', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Konversi kembali ke BGR untuk tampilan OpenCV
    cv2.imshow('Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()