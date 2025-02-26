import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset dari file CSV
try:
    color_data = pd.read_csv('colors.csv')
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")
    exit()

# Verifikasi data untuk memastikan warna merah tidak tertukar dengan warna biru
def verify_and_correct_data(data):
    for index, row in data.iterrows():
        r, g, b = row['R'], row['G'], row['B']
        if r > b and row['color_name'] == 'blue':
            data.at[index, 'color_name'] = 'red'
        elif b > r and row['color_name'] == 'red':
            data.at[index, 'color_name'] = 'blue'
    return data

color_data = verify_and_correct_data(color_data)

X = color_data[['R', 'G', 'B']].values
y = color_data['color_name'].values

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training Model ML
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prediksi data test
y_pred = knn.predict(X_test)

# Menghitung akurasi model awal
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model Awal: {accuracy * 100:.2f}%")

# Muat model dan scaler terbaru
try:
    knn = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

detected_colors = []
detected_true_labels = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height // 2, width // 2]
    pixel_center_reshaped = pixel_center.reshape(1, -1)
    pixel_center_scaled = scaler.transform(pixel_center_reshaped)

    # Prediksi warna
    color_pred = knn.predict(pixel_center_scaled)[0]

    # Temukan warna asli terdekat (untuk perhitungan akurasi real-time)
    distances = np.linalg.norm(X_scaled - pixel_center_scaled, axis=1)
    nearest_idx = np.argmin(distances)
    true_color = y[nearest_idx]

    # Simpan prediksi dan warna asli
    detected_colors.append(color_pred)
    detected_true_labels.append(true_color)

    # Batasi jumlah data untuk perhitungan akurasi real-time (misalnya, 50 frame terakhir)
    if len(detected_colors) > 50:
        detected_colors.pop(0)
        detected_true_labels.pop(0)

    # Hitung akurasi real-time dengan menghitung akurasi berdasarkan deteksi sebelumnya
    if len(detected_colors) > 0:
        realtime_accuracy = accuracy_score(detected_true_labels, detected_colors) * 100
    else:
        realtime_accuracy = 0.0

    # Tampilkan prediksi dan akurasi real-time
    cv2.putText(frame, f'Color: {color_pred} | Accuracy: {realtime_accuracy:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250,235,215), 2)
    print(f'Color: {color_pred} | Accuracy: {realtime_accuracy:.2f}%')

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()