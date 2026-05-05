import streamlit as st
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import subprocess
import zipfile

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(page_title="Hệ Sinh Thái Giao Thông AI", layout="wide")
st.title("🚦 Hệ Thống Trí Tuệ Nhân Tạo: Quản Lý Giao Thông Toàn Diện")
st.markdown("Từ thu thập dữ liệu, huấn luyện mô hình đến nhận diện thời gian thực.")

# Tạo các thư mục cần thiết nếu chưa có
os.makedirs("data/videos", exist_ok=True)
os.makedirs("data/dataset", exist_ok=True)
os.makedirs("data/images", exist_ok=True)

# ==========================================
# 2. THANH ĐIỀU KHIỂN (SIDEBAR) - LOGIC BẮT LỖI CHẶT CHẼ
# ==========================================
st.sidebar.title("🛠️ BẢNG ĐIỀU KHIỂN TỔNG")

# --- BƯỚC 1: XỬ LÝ VIDEO & CẮT ẢNH ---
with st.sidebar.expander("1️⃣ Nhập Video & Cắt Ảnh", expanded=False):
    uploaded_video = st.file_uploader("Kéo thả Video vào đây (.mp4)", type=["mp4", "avi"])
    video_path = os.path.join("data/videos", "traffic_01.mp4")
    
    if uploaded_video is not None:
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.success("✅ Đã tải video lên thành công!")

    if st.button("✂️ Trích xuất Ảnh (Chạy File 1)"):
        # KIỂM TRA: Có video chưa?
        if not os.path.exists(video_path):
            st.error("❌ LỖI: Chưa có video! Vui lòng tải video ở ô bên trên trước.")
        else:
            with st.spinner("Đang cắt ảnh từ video..."):
                subprocess.run(["python", "src/1_extract_frames.py"])
            st.success("✅ Đã cắt ảnh xong! (Kiểm tra thư mục data/images/)")

# --- BƯỚC 2: GÁN NHÃN TỰ ĐỘNG ---
with st.sidebar.expander("2️⃣ Gán Nhãn Dữ Liệu", expanded=False):
    st.info("Sử dụng AI để tự động khoanh vùng xe cộ.")
    if st.button("🏷️ Chạy Auto Label (Chạy File 6)"):
        # KIỂM TRA: Đã có ảnh trong thư mục chưa?
        if len(os.listdir("data/images")) == 0:
            st.error("❌ LỖI: Thư mục ảnh đang trống! Hãy chạy Bước 1 trước để cắt ảnh.")
        else:
            with st.spinner("AI đang tự động gán nhãn... Quá trình này có thể mất vài phút."):
                subprocess.run(["python", "src/6_auto_label.py"])
            st.success("✅ Gán nhãn hoàn tất! (Kiểm tra data/dataset/labels/)")

# --- BƯỚC 3: HUẤN LUYỆN (TRAINING) ---
with st.sidebar.expander("3️⃣ Nhập Dataset & Huấn Luyện AI", expanded=False):
    uploaded_zip = st.file_uploader("Kéo thả Dataset (từ Roboflow) dạng .zip", type=["zip"])
    if uploaded_zip is not None:
        zip_path = os.path.join("data/dataset", "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/dataset/")
        st.success("✅ Đã giải nén Dataset! Sẵn sàng huấn luyện.")

    if st.button("🚀 Bắt đầu Huấn luyện (Chạy File 2)"):
        # KIỂM TRA: Đã có file data.yaml chưa?
        if not os.path.exists("data/dataset/data.yaml"):
            st.error("❌ LỖI: Không tìm thấy file data.yaml! Hãy tải và kéo thả file Dataset .zip vào trước.")
        else:
            st.warning("⏳ Đang huấn luyện... Hãy theo dõi Terminal trên VS Code để xem tiến độ (Epochs).")
            with st.spinner("Đang chạy Huấn luyện..."):
                subprocess.run(["python", "src/2_train_yolo.py"])
            st.success("✅ Huấn luyện xong! Trọng số đã lưu ở results/traffic_model/weights/best.pt")

# --- BƯỚC 4: DỰ ĐOÁN TƯƠNG LAI ---
with st.sidebar.expander("4️⃣ Máy Học Dự Đoán", expanded=False):
    st.info("Chạy 3 thuật toán ML để dự đoán lưu lượng.")
    if st.button("📈 Mở Biểu đồ Dự đoán (Chạy File 4)"):
        if not os.path.exists('results/traffic_data.csv'):
            st.error("❌ LỖI: Chưa có dữ liệu CSV! Hãy bật hệ thống nhận diện thực tế (công tắc bên dưới) để thu thập dữ liệu trước.")
        else:
            st.success("Đang mở biểu đồ phân tích...")
            subprocess.Popen(["python", "src/4_prediction_models.py"])

st.sidebar.markdown("---")
# --- BƯỚC 5: BẬT/TẮT NHẬN DIỆN THỰC TẾ ---
run_system = st.sidebar.checkbox("🟢 BẬT CAMERA / NHẬN DIỆN THỰC TẾ", value=False)


# ==========================================
# 3. KHU VỰC HIỂN THỊ CHÍNH (CAMERA & BIỂU ĐỒ)
# ==========================================
@st.cache_resource
def load_model():
    if os.path.exists("best.pt"):
        return YOLO('best.pt')
    return YOLO('yolov8n.pt')

model = load_model()

col_video, col_charts = st.columns([6, 4])

with col_video:
    st.subheader("📷 Live Camera Feed")
    video_placeholder = st.empty()
    kpi_placeholder = st.empty()

with col_charts:
    st.subheader("📊 Phân Tích Dữ Liệu")
    chart_placeholder = st.empty()
    pie_placeholder = st.empty()

if run_system:
    # Ở đây đang dùng Camera (số 0). Nếu muốn web đọc video thì sửa số 0 thành "data/videos/traffic_01.mp4"
    cap = cv2.VideoCapture(0) 
    
    data_history = pd.DataFrame(columns=['Minute', 'Total_Vehicles', 'Bicycles', 'Motorcycles', 'Cars', 'Buses', 'Trucks'])
    prev_frame_time = 0
    start_log_time = time.time()
    minute_counter = 1
    LOG_INTERVAL = 5 

    while cap.isOpened() and run_system:
        ret, frame = cap.read()
        if not ret:
            st.error("Đã kết thúc video hoặc không thể đọc camera.")
            break
        
        # --- NHẬN DIỆN YOLO ---
        results = model(frame, classes=[0, 1, 2, 3, 5, 7], stream=True)
        person_count = bicycle_count = car_count = motorcycle_count = bus_count = truck_count = 0
        
        for r in results:
            frame = r.plot()
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        with kpi_placeholder.container():
            k1, k2, k3 = st.columns(3)
            k1.metric("Tổng Phương Tiện Hiện Tại", total_vehicles)
            k2.metric("Số Người Đi Bộ", person_count)
            k3.metric("Xe Máy Đang Di Chuyển", motorcycle_count)

        # --- GHI NHẬN DỮ LIỆU & VẼ BIỂU ĐỒ ---
        current_time = time.time()
        if current_time - start_log_time >= LOG_INTERVAL:
            new_row = {
                'Minute': minute_counter, 'Total_Vehicles': total_vehicles, 
                'Bicycles': bicycle_count, 'Motorcycles': motorcycle_count, 
                'Cars': car_count, 'Buses': bus_count, 'Trucks': truck_count
            }
            data_history = pd.concat([data_history, pd.DataFrame([new_row])], ignore_index=True)
            
            chart_placeholder.line_chart(data_history.set_index('Minute')['Total_Vehicles'])
            
            if total_vehicles > 0:
                fig, ax = plt.subplots(figsize=(4, 4))
                vehicle_types = ['Bicycles', 'Motorcycles', 'Cars', 'Buses', 'Trucks']
                totals = data_history[vehicle_types].sum()
                totals = totals[totals > 0]
                if not totals.empty:
                    ax.pie(totals, labels=totals.index, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    pie_placeholder.pyplot(fig)
            
            minute_counter += 1
            start_log_time = current_time
            
    cap.release()
else:
    st.info("👈 Mở bảng điều khiển bên trái để thao tác dữ liệu, hoặc bật công tắc 'BẬT CAMERA' để xem hệ thống hoạt động!")