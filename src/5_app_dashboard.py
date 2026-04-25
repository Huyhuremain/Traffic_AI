import streamlit as st
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(page_title="Traffic AI Dashboard", layout="wide")
st.title("🚦 Hệ Thống Phân Tích Giao Thông Thông Minh")
st.markdown("Bảng điều khiển giám sát Camera và Dữ liệu theo thời gian thực (Real-time).")

# ==========================================
# 2. KHỞI TẠO BIẾN VÀ AI (Dùng cache để không load lại nhiều lần)
# ==========================================
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Tạo 2 cột: Cột trái (Video) chiếm 60%, Cột phải (Biểu đồ) chiếm 40%
col_video, col_charts = st.columns([6, 4])

# Các vị trí trống (placeholder) để Streamlit cập nhật hình ảnh liên tục
with col_video:
    st.subheader("📷 Live Camera Feed")
    video_placeholder = st.empty()
    kpi_placeholder = st.empty()

with col_charts:
    st.subheader("📊 Phân Tích Dữ Liệu")
    chart_placeholder = st.empty()
    pie_placeholder = st.empty()

# Nút Bật/Tắt hệ thống
run_system = st.sidebar.checkbox("Bật Hệ Thống Nhận Diện", value=False)

# ==========================================
# 3. VÒNG LẶP XỬ LÝ CHÍNH
# ==========================================
if run_system:
    cap = cv2.VideoCapture(0) # Đổi thành đường dẫn video nếu muốn
    
    # Biến lưu trữ dữ liệu tạm thời để vẽ biểu đồ trực tiếp
    data_history = pd.DataFrame(columns=['Minute', 'Total_Vehicles', 'Bicycles', 'Motorcycles', 'Cars', 'Buses', 'Trucks'])
    
    prev_frame_time = 0
    start_log_time = time.time()
    minute_counter = 1
    LOG_INTERVAL = 5 # Chu kỳ cập nhật dữ liệu (5 giây)

    while cap.isOpened() and run_system:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể đọc camera/video.")
            break
        
        # --- NHẬN DIỆN YOLO ---
        results = model(frame, classes=[0, 1, 2, 3, 5, 7], stream=True)
        
        person_count = bicycle_count = car_count = motorcycle_count = bus_count = truck_count = 0
        
        for r in results:
            frame = r.plot() # YOLO tự vẽ Bounding Box
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0: person_count += 1
                elif cls == 1: bicycle_count += 1
                elif cls == 2: car_count += 1
                elif cls == 3: motorcycle_count += 1
                elif cls == 5: bus_count += 1
                elif cls == 7: truck_count += 1

        total_vehicles = bicycle_count + car_count + motorcycle_count + bus_count + truck_count

        # --- CẬP NHẬT GIAO DIỆN VIDEO VÀ KPI ---
        # OpenCV dùng hệ màu BGR, Streamlit dùng RGB nên phải chuyển đổi
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Hiển thị các chỉ số nhanh (KPIs)
        with kpi_placeholder.container():
            k1, k2, k3 = st.columns(3)
            k1.metric("Tổng Phương Tiện Hiện Tại", total_vehicles)
            k2.metric("Số Người Đi Bộ", person_count)
            k3.metric("Xe Máy Đang Di Chuyển", motorcycle_count)

        # --- GHI NHẬN DỮ LIỆU & VẼ BIỂU ĐỒ (Mỗi 5 giây) ---
        current_time = time.time()
        if current_time - start_log_time >= LOG_INTERVAL:
            # Thêm dòng dữ liệu mới vào lịch sử
            new_row = {
                'Minute': minute_counter, 'Total_Vehicles': total_vehicles, 
                'Bicycles': bicycle_count, 'Motorcycles': motorcycle_count, 
                'Cars': car_count, 'Buses': bus_count, 'Trucks': truck_count
            }
            data_history = pd.concat([data_history, pd.DataFrame([new_row])], ignore_index=True)
            
            # Cập nhật Biểu đồ đường (Line Chart) theo thời gian
            chart_placeholder.line_chart(data_history.set_index('Minute')['Total_Vehicles'])
            
            # Cập nhật Biểu đồ tròn (Pie Chart) tỷ trọng
            if total_vehicles > 0:
                fig, ax = plt.subplots(figsize=(4, 4))
                vehicle_types = ['Bicycles', 'Motorcycles', 'Cars', 'Buses', 'Trucks']
                totals = data_history[vehicle_types].sum()
                totals = totals[totals > 0] # Lọc bỏ số 0
                
                ax.pie(totals, labels=totals.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                pie_placeholder.pyplot(fig)
            
            minute_counter += 1
            start_log_time = current_time
            
    cap.release()
else:
    st.info("👈 Hãy đánh dấu vào ô 'Bật Hệ Thống Nhận Diện' ở thanh công cụ bên trái để bắt đầu!")