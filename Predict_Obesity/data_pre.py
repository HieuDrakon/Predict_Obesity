import os
import pandas as pd

# Load the dataset from the Excel file
df = pd.read_excel('D:/project/Predict_Obesity/Obesity_Dataset/Obesity_Dataset.xlsx')

# Display the first few rows of the dataset to understand its structure
print("Original Dataset:")
print(df.head())

# Data Cleaning Steps

# 1. Xử lý các giá trị bị thiếu
# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Fill missing values with appropriate values or drop rows/columns with missing values
df = df.dropna()  # Dropping rows with any missing values

# 2. Xử lý các hàng trùng lặp
# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
print("\nNumber of duplicate rows:")
print(duplicate_rows)

# Drop duplicate rows
df = df.drop_duplicates()

# 3. Chuyển đổi kiểu dữ liệu nếu cần thiết
# Display data types of each column
print("\nData types of each column:")
print(df.dtypes)

# Convert data types if necessary (e.g., convert numerical columns to integers)
df['Age'] = df['Age'].astype(int)
df['Height'] = df['Height'].astype(int)

# 4. Chuẩn hóa các cột phân loại nếu cần (ví dụ: chuyển đổi các cột phân loại thành chữ thường)
categorical_columns = ['Sex', 'Overweight_Obese_Family', 'Consumption_of_Fast_Food', 'Frequency_of_Consuming_Vegetables',
                       'Number_of_Main_Meals_Daily', 'Food_Intake_Between_Meals', 'Smoking', 'Liquid_Intake_Daily',
                       'Calculation_of_Calorie_Intake', 'Physical_Excercise', 'Schedule_Dedicated_to_Technology',
                       'Type_of_Transportation_Used', 'Class']

for col in categorical_columns:
    df[col] = df[col].astype(str).str.lower()

# Display the cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

# Save the cleaned dataset to a new csv file
directory = 'D:/project/Predict_Obesity/Obesity_Dataset'
file_path = os.path.join(directory, 'Obesity_Dataset.csv')
df.to_csv(file_path, index=False)
