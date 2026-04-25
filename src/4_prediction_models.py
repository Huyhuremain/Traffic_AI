import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import os

# 1. ĐỌC DỮ LIỆU TỪ FILE CSV
csv_path = 'results/traffic_data.csv'

# Kiểm tra xem file đã tồn tại chưa
if not os.path.exists(csv_path):
    print(f"Lỗi: Không tìm thấy file {csv_path}!")
    print("Vui lòng chạy file nhận diện để xuất dữ liệu ra CSV trước.")
    exit()

print("Đang đọc dữ liệu và huấn luyện mô hình...")
df = pd.read_csv(csv_path)

# CHÚ Ý: Cập nhật tên cột dưới đây cho khớp với file CSV thật của bạn
# Giả sử file có cột 'Minute' (Thời gian) và 'Total_Vehicles' (Số lượng xe)
try:
    X = df[['Minute']].values 
    y_thuc_te = df['Total_Vehicles'].values
except KeyError:
    print("Lỗi: Không tìm thấy cột 'Minute' hoặc 'Total_Vehicles' trong file CSV.")
    print(f"Các cột hiện có trong file của bạn là: {df.columns.tolist()}")
    exit()

# 2. KHỞI TẠO VÀ HUẤN LUYỆN 3 MÔ HÌNH
# Mô hình 1: Hồi quy tuyến tính (Baseline)
model_linear = LinearRegression()
model_linear.fit(X, y_thuc_te)
y_pred_linear = model_linear.predict(X)

# Mô hình 2: Hồi quy đa thức (Mức trung bình - Bậc 3)
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y_thuc_te)
y_pred_poly = model_poly.predict(X_poly)

# Mô hình 3: Random Forest (Mức cao)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X, y_thuc_te)
y_pred_rf = model_rf.predict(X)

# 3. VẼ BIỂU ĐỒ SO SÁNH
plt.figure(figsize=(12, 6))

# Vẽ điểm dữ liệu thực tế
plt.scatter(X, y_thuc_te, color='black', label='Dữ liệu thực tế', alpha=0.6)

# Vẽ đường dự đoán
plt.plot(X, y_pred_linear, color='blue', linestyle='--', linewidth=2, label='Hồi quy Tuyến tính')
plt.plot(X, y_pred_poly, color='orange', linewidth=2, label='Hồi quy Đa thức (Bậc 3)')
plt.plot(X, y_pred_rf, color='green', linewidth=2, label='Random Forest')

# Trang trí
plt.title('Dự đoán Lưu lượng Giao thông từ Dữ liệu Thực tế', fontsize=16, fontweight='bold')
plt.xlabel('Thời gian (Phút)', fontsize=12)
plt.ylabel('Số lượng xe', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)

# Hiển thị
print("Hoàn tất! Đang mở biểu đồ...")
plt.tight_layout()
plt.show()