import cv2
import joblib
import numpy as np
from sklearn import svm

# Muat model SVM dan scaler
svm_model = joblib.load('intelligent-control-week2/knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Pastikan model SVM dilatih dengan probability=True
if not hasattr(svm_model, 'predict_proba'):
    raise ValueError("Model SVM tidak mendukung probabilitas. Silakan latih ulang dengan SVC(probability=True)")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ambil dua area sejajar horizontal untuk deteksi warna
    height, width, _ = frame.shape
    left_x, left_y = width // 4, height // 2
    right_x, right_y = 3 * width // 4, height // 2

    pixel_left = frame[left_y, left_x]
    pixel_right = frame[right_y, right_x]

    # Konversi BGR ke RGB agar warna sesuai
    pixel_left = pixel_left[::-1]
    pixel_right = pixel_right[::-1]

    # Normalisasi pixel sebelum prediksi
    pixel_left_scaled = scaler.transform([pixel_left])
    pixel_right_scaled = scaler.transform([pixel_right])

    # Prediksi warna secara real-time dengan probabilitas
    color_pred_left = svm_model.predict(pixel_left_scaled)[0]
    color_pred_right = svm_model.predict(pixel_right_scaled)[0]

    prob_left = np.max(svm_model.predict_proba(pixel_left_scaled)) * 100
    prob_right = np.max(svm_model.predict_proba(pixel_right_scaled)) * 100

    # Gambar bounding box sejajar horizontal
    box_size = 40
    cv2.rectangle(frame, (left_x - box_size, left_y - box_size), (left_x + box_size, left_y + box_size), (0, 255, 0), 2)
    cv2.rectangle(frame, (right_x - box_size, right_y - box_size), (right_x + box_size, right_y + box_size), (255, 0, 0), 2)

    # Tampilkan warna dan akurasi pada frame
    cv2.putText(frame, f'Left Color: {color_pred_left} ({prob_left:.2f}%)', (left_x - 100, left_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Right Color: {color_pred_right} ({prob_right:.2f}%)', (right_x - 100, right_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Tampilkan hasil secara real-time
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()