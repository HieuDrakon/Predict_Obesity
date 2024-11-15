import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# Đọc dữ liệu từ file Excel
df = pd.read_excel('D:/project/Predict_Obesity/Obesity_Dataset/Obesity_Dataset.xlsx')

# Hiển thị một vài hàng đầu tiên của dữ liệu để hiểu cấu trúc
print("Dữ liệu gốc:")
print(df.head())

# Bước làm sạch dữ liệu

# 1. Xử lý các giá trị bị thiếu
# Kiểm tra các giá trị bị thiếu
missing_values = df.isnull().sum()
print("\nSố lượng giá trị bị thiếu trong mỗi cột:")
print(missing_values)

# Xóa các hàng có bất kỳ giá trị nào bị thiếu
df = df.dropna()  # Xóa các hàng có giá trị bị thiếu

# 2. Xử lý các hàng trùng lặp
# Kiểm tra các hàng trùng lặp
duplicate_rows = df.duplicated().sum()
print("\nSố lượng hàng trùng lặp:")
print(duplicate_rows)

# Xóa các hàng trùng lặp
df = df.drop_duplicates()

# 3. Chuyển đổi kiểu dữ liệu nếu cần thiết
# Chuyển các cột số về kiểu dữ liệu số nguyên
df['Age'] = df['Age'].astype(int)
df['Height'] = df['Height'].astype(int)

# 4. Chuẩn hóa các cột phân loại nếu cần (ví dụ: chuyển đổi các cột phân loại thành chữ thường)
categorical_columns = ['Sex', 'Overweight_Obese_Family', 'Consumption_of_Fast_Food', 'Frequency_of_Consuming_Vegetables',
                       'Number_of_Main_Meals_Daily', 'Food_Intake_Between_Meals', 'Smoking', 'Liquid_Intake_Daily',
                       'Calculation_of_Calorie_Intake', 'Physical_Excercise', 'Schedule_Dedicated_to_Technology',
                       'Type_of_Transportation_Used', 'Class']

for col in categorical_columns:
    df[col] = df[col].astype(str).str.lower()

# Hiển thị dữ liệu đã làm sạch
print("\nDữ liệu đã làm sạch:")
print(df.head())

# Lưu dữ liệu đã làm sạch vào file csv mới
directory = 'D:/project/Predict_Obesity/Obesity_Dataset'
file_path = os.path.join(directory, 'Obesity_Dataset.csv')
df.to_csv(file_path, index=False)

# Trực quan hóa dữ liệu
sns.set(style="whitegrid")

# Biểu đồ: Phân phối độ tuổi
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], kde=True, color='skyblue')
plt.title('Phân Phối Độ Tuổi')
plt.xlabel('Tuổi')
plt.ylabel('Tần suất')
plt.show()

# Biểu đồ: Phân phối chiều cao
plt.figure(figsize=(10, 5))
sns.histplot(df['Height'], kde=True, color='salmon')
plt.title('Phân Phối Chiều Cao')
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Tần suất')
plt.show()

# Biểu đồ: Tần suất tiêu thụ thức ăn nhanh theo phân loại
plt.figure(figsize=(10, 5))
sns.countplot(x='Consumption_of_Fast_Food', hue='Class', data=df, palette='viridis')
plt.title('Tiêu Thụ Thức Ăn Nhanh Theo Phân Loại')
plt.xlabel('Mức tiêu thụ thức ăn nhanh')
plt.ylabel('Số lượng')
plt.legend(title='Phân loại', loc='upper right')
plt.show()

# Biểu đồ: Mức độ béo phì trong gia đình so với phân loại
plt.figure(figsize=(10, 5))
sns.countplot(x='Overweight_Obese_Family', hue='Class', data=df, palette='magma')
plt.title('Mức Độ Béo Phì Trong Gia Đình So Với Phân Loại')
plt.xlabel('Béo phì trong gia đình')
plt.ylabel('Số lượng')
plt.legend(title='Phân loại', loc='upper right')
plt.show()
