import pandas as pd
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Loading the dataset
file_path = 'D:/project/Predict_Obesity/Obesity_Dataset/Obesity_Dataset.csv'
df = pd.read_csv(file_path)
print("---------- Tiền xử lý dữ liệu -----------")

# Xử lý các cột có dữ liệu phân loại bằng cách mã hóa (One-hot encoding)
df = pd.get_dummies(df, drop_first=True)

# Tách dữ liệu thành đặc trưng và nhãn mục tiêu
X = df.drop(columns='Class')  # Giả sử 'Class' là cột đích phân loại
y = df['Class']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Dummy Classifier
print("---------- Dummy Classifier -----------")
dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(X_train, y_train)
y_dummy_pred = dummy_clf.predict(X_test)
print("Đánh giá Dummy Classifier")
print(classification_report(y_test, y_dummy_pred))
print(confusion_matrix(y_test, y_dummy_pred))

# SVC Classifier
# Chuẩn hóa dữ liệu cho SVC
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("---------- Kết thúc tiền xử lý dữ liệu -----------")
print("---------- SVC (Support Vector Classifier) -----------")
svc_clf = SVC()
svc_clf.fit(X_train_scaled, y_train)
y_svc_pred = svc_clf.predict(X_test_scaled)
print("Đánh giá SVC")
print(classification_report(y_test, y_svc_pred))
print(confusion_matrix(y_test, y_svc_pred))

# Tạo các file pickle 
# Tạo thư mục lưu trữ các file pickle nếu chưa tồn tại
os.makedirs('data_pkl', exist_ok=True)

# Lưu Dummy Classifier vào file pickle
with open('data_pkl/dummy_classifier.pkl', 'wb') as file:
    pickle.dump(dummy_clf, file)
print("Dummy Classifier đã được lưu thành công")

# Lưu SVC vào file pickle
with open('data_pkl/svc_classifier.pkl', 'wb') as file:
    pickle.dump(svc_clf, file)
print("SVC đã được lưu thành công")