import pandas as pd
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import os

# Load dataset
file_path = 'D:/project/Predict_Obesity/Obesity_Dataset/Obesity_Dataset.csv'
df = pd.read_csv(file_path)

# Xử lý tối thiểu: Tách đặc trưng và nhãn
X = df.drop('Class', axis=1)  # Giả sử 'Class' là cột đích
y = df['Class']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xử lý dữ liệu không cân bằng với SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dummy Classifier
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train_scaled, y_train)

# SVC Classifier
svc_clf = SVC(kernel='linear')
svc_clf.fit(X_train_scaled, y_train)

# Lưu các mô hình vào file pickle
os.makedirs('data_pkl', exist_ok=True)

# Lưu Dummy Classifier
with open('data_pkl/dummy_classifier.pkl', 'wb') as file:
    pickle.dump(dummy_clf, file)
print("Dummy Classifier đã được lưu thành công")

# Lưu SVC Classifier
with open('data_pkl/svc_classifier.pkl', 'wb') as file:
    pickle.dump(svc_clf, file)
print("SVC đã được lưu thành công")
