# ==========================================================
# PROGRAM KLASIFIKASI DATA UMKM MENGGUNAKAN
# ALGORITMA K-NEAREST NEIGHBORS (KNN) DAN DECISION TREE
# ==========================================================

# ==========================================================
# IMPORT LIBRARY
# Mengimpor library untuk pengolahan data, visualisasi,
# preprocessing, klasifikasi, dan evaluasi model.
# ==========================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# MEMBACA DAN MENGENALI DATASET
# Membaca dataset serta menampilkan informasi awal.
# ==========================================================
df = pd.read_csv("synthetic_umkm_data.csv")

print("="*60)
print("5 DATA PERTAMA")
print("="*60)
print(df.head())

print("\n" + "="*60)
print("UKURAN DATASET")
print("="*60)
print(df.shape)

print("\n" + "="*60)
print("INFO DATASET")
print("="*60)
df.info()

print("\n" + "="*60)
print("STATISTIK DESKRIPTIF")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("MISSING VALUE")
print("="*60)
print(df.isnull().sum())

# ==========================================================
# VISUALISASI DATA
# Menampilkan distribusi kelas sebelum preprocessing.
# ==========================================================
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Class")
plt.title("Distribusi Class")
plt.xlabel("Class")
plt.ylabel("Jumlah")
plt.show()

# ==========================================================
# PREPROCESSING DATA
# Menghapus atribut yang tidak digunakan dan melakukan
# encoding terhadap data kategorikal.
# ==========================================================
df.drop(columns=["ID", "Review_Text"], inplace=True)

encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

print("\n" + "="*60)
print("ENCODING")
print("="*60)
print(df.head())

# ==========================================================
# VISUALISASI KORELASI
# Menampilkan hubungan antar atribut numerik.
# ==========================================================
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.show()

# ==========================================================
# PEMBAGIAN DATA
# Memisahkan feature dan target kemudian membagi data
# menjadi data training dan testing.
# ==========================================================
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "="*60)
print("DATA TRAINING DAN TESTING")
print("="*60)
print("Training :", X_train.shape)
print("Testing  :", X_test.shape)

# ==========================================================
# STANDARDISASI DATA
# Menyamakan skala seluruh feature numerik.
# ==========================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n" + "="*60)
print("HASIL STANDARDISASI")
print("="*60)
print(X_train[:5])

# ==========================================================
# IMPLEMENTASI ALGORITMA KNN
# Melatih model dan melakukan prediksi.
# ==========================================================
print("\n" + "="*60)
print("IMPLEMENTASI KNN")
print("="*60)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average="weighted")
recall_knn = recall_score(y_test, y_pred_knn, average="weighted")
f1_knn = f1_score(y_test, y_pred_knn, average="weighted")

print("Accuracy :", accuracy_knn)
print("Precision:", precision_knn)
print("Recall   :", recall_knn)
print("F1-Score :", f1_knn)

print("\nClassification Report KNN")
print(classification_report(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn)

print("\nConfusion Matrix KNN")
print(cm_knn)

plt.figure(figsize=(6,5))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix KNN")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

# ==========================================================
# IMPLEMENTASI ALGORITMA DECISION TREE
# Melatih model dan melakukan prediksi.
# ==========================================================
print("\n" + "="*60)
print("IMPLEMENTASI DECISION TREE")
print("="*60)

dt = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average="weighted")
recall_dt = recall_score(y_test, y_pred_dt, average="weighted")
f1_dt = f1_score(y_test, y_pred_dt, average="weighted")

print("Accuracy :", accuracy_dt)
print("Precision:", precision_dt)
print("Recall   :", recall_dt)
print("F1-Score :", f1_dt)

print("\nClassification Report Decision Tree")
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)

print("\nConfusion Matrix Decision Tree")
print(cm_dt)

plt.figure(figsize=(6,5))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix Decision Tree")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

# ==========================================================
# PERBANDINGAN MODEL
# Membandingkan hasil evaluasi kedua algoritma.
# ==========================================================
hasil = pd.DataFrame({
    "Algoritma": ["KNN", "Decision Tree"],
    "Accuracy": [accuracy_knn, accuracy_dt],
    "Precision": [precision_knn, precision_dt],
    "Recall": [recall_knn, recall_dt],
    "F1-Score": [f1_knn, f1_dt]
})

print("\n" + "="*60)
print("HASIL PERBANDINGAN MODEL")
print("="*60)
print(hasil)

# ==========================================================
# VISUALISASI HASIL
# Menampilkan grafik perbandingan performa model.
# ==========================================================
plt.figure(figsize=(7,5))
plt.bar(hasil["Algoritma"], hasil["Accuracy"])

plt.title("Perbandingan Accuracy Model")
plt.xlabel("Algoritma")
plt.ylabel("Accuracy")

for i, value in enumerate(hasil["Accuracy"]):
    plt.text(i, value, f"{value:.4f}", ha="center", va="bottom")

plt.show()

hasil.set_index("Algoritma").plot(kind="bar", figsize=(9,6))
plt.title("Perbandingan Accuracy, Precision, Recall dan F1-Score")
plt.ylabel("Nilai")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.show()

# ==========================================================
# ANALISIS HASIL
# Menentukan model terbaik berdasarkan nilai Accuracy.
# ==========================================================
print("\n" + "="*60)
print("ANALISIS HASIL")
print("="*60)

if accuracy_dt > accuracy_knn:
    print("Model terbaik berdasarkan Accuracy adalah Decision Tree.")
elif accuracy_knn > accuracy_dt:
    print("Model terbaik berdasarkan Accuracy adalah KNN.")
else:
    print("Kedua model memiliki nilai Accuracy yang sama.")

print("\nRingkasan Hasil Evaluasi")
print(hasil)

print("\nProgram selesai dijalankan.")
