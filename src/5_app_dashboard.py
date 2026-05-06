import streamlit as st
import cv2
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import subprocess
import zipfile
import sys

# ==========================================
# 1. CẤU HÌNH TRANG & ĐƯỜNG DẪN GỐC
# ==========================================
st.set_page_config(page_title="Hệ Sinh Thái Giao Thông AI", layout="wide")
st.title("🚦 Hệ Thống Trí Tuệ Nhân Tạo: Quản Lý Giao Thông Toàn Diện")
st.markdown("Từ thu thập dữ liệu, huấn luyện mô hình đến nhận diện thời gian thực.")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable

DATA_VIDEOS  = os.path.join(BASE_DIR, "data", "videos")
DATA_IMAGES  = os.path.join(BASE_DIR, "data", "images")
DATA_DATASET = os.path.join(BASE_DIR, "data", "dataset")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
CSV_FILE     = os.path.join(RESULTS_DIR, "traffic_data.csv")
VIDEO_PATH   = os.path.join(DATA_VIDEOS, "traffic_01.mp4")

for d in [DATA_VIDEOS, DATA_IMAGES, DATA_DATASET, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# HÀM TIỆN ÍCH
# ==========================================
def run_script(script_name):
    script_path = os.path.join(BASE_DIR, "src", script_name)
    return subprocess.run(
        [PYTHON_EXE, script_path],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

# ==========================================
# 2. SIDEBAR - BẢNG ĐIỀU KHIỂN
# ==========================================
st.sidebar.title("🛠️ BẢNG ĐIỀU KHIỂN TỔNG")

# --- BƯỚC 1: VIDEO & CẮT ẢNH ---
with st.sidebar.expander("1️⃣ Nhập Video & Cắt Ảnh", expanded=False):
    uploaded_video = st.file_uploader("Kéo thả Video (.mp4)", type=["mp4", "avi"])

    if uploaded_video is not None:
        with open(VIDEO_PATH, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.success("✅ Đã lưu video thành công!")

    if os.path.exists(VIDEO_PATH):
        size_mb = os.path.getsize(VIDEO_PATH) / (1024 * 1024)
        st.info(f"📹 traffic_01.mp4 ({size_mb:.1f} MB)")
    else:
        st.warning("⚠️ Chưa có video. Hãy tải lên trước.")

    if st.button("✂️ Trích xuất Ảnh (Chạy File 1)"):
        if not os.path.exists(VIDEO_PATH):
            st.error("❌ Chưa có video!")
        else:
            with st.spinner("Đang cắt ảnh từ video..."):
                result = run_script("1_extract_frames.py")
            if result.returncode == 0:
                img_count = len([f for f in os.listdir(DATA_IMAGES) if f.endswith(('.jpg', '.png'))])
                st.success(f"✅ Xong! Đã trích xuất {img_count} ảnh.")
            else:
                st.error(f"❌ Lỗi:\n{result.stderr}")

# --- BƯỚC 2: GÁN NHÃN ---
with st.sidebar.expander("2️⃣ Gán Nhãn Dữ Liệu", expanded=False):
    st.info("Dùng AI để tự động gán nhãn xe cộ.")
    img_count = len([f for f in os.listdir(DATA_IMAGES) if f.endswith(('.jpg', '.png'))]) if os.path.exists(DATA_IMAGES) else 0
    st.caption(f"Số ảnh hiện có: {img_count}")

    if st.button("🏷️ Chạy Auto Label (Chạy File 6)"):
        if img_count == 0:
            st.error("❌ Thư mục ảnh trống! Hãy chạy Bước 1 trước.")
        else:
            with st.spinner("AI đang gán nhãn... (có thể mất vài phút)"):
                result = run_script("6_auto_label.py")
            if result.returncode == 0:
                st.success("✅ Gán nhãn xong! Kiểm tra data/dataset/labels/")
            else:
                st.error(f"❌ Lỗi:\n{result.stderr}")

# --- BƯỚC 2.5: CHUẨN BỊ DATASET ---
with st.sidebar.expander("2.5️⃣ Chuẩn Bị Dataset (Train/Val)", expanded=False):
    st.info("Tự động chia ảnh 80/20 và tạo data.yaml. Chạy sau Bước 2.")

    label_dir   = os.path.join(DATA_DATASET, "labels")
    label_count = len([f for f in os.listdir(label_dir) if f.endswith('.txt')]) if os.path.exists(label_dir) else 0
    st.caption(f"Số nhãn hiện có: {label_count}")

    yaml_path = os.path.join(DATA_DATASET, "data.yaml")
    if os.path.exists(yaml_path):
        st.success("✅ data.yaml đã tồn tại — sẵn sàng huấn luyện!")

    if st.button("⚙️ Tạo Dataset & data.yaml"):
        if label_count == 0:
            st.error("❌ Chưa có nhãn! Hãy chạy Bước 2 trước.")
        else:
            with st.spinner("Đang chuẩn bị dataset..."):
                result = run_script("7_prepare_dataset.py")
            if result.returncode == 0:
                st.success("✅ Xong! Dataset đã sẵn sàng để huấn luyện.")
                st.code(result.stdout)
            else:
                st.error(f"❌ Lỗi:\n{result.stderr}")

# --- BƯỚC 3: HUẤN LUYỆN ---
with st.sidebar.expander("3️⃣ Huấn Luyện AI", expanded=False):
    yaml_path = os.path.join(DATA_DATASET, "data.yaml")

    uploaded_zip = st.file_uploader("(Tuỳ chọn) Kéo thả Dataset Roboflow (.zip)", type=["zip"])
    if uploaded_zip is not None:
        zip_path = os.path.join(DATA_DATASET, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DATASET)
        st.success("✅ Đã giải nén Dataset!")

    if os.path.exists(yaml_path):
        st.info("✅ Đã có data.yaml — sẵn sàng huấn luyện.")
    else:
        st.warning("⚠️ Chưa có data.yaml. Hãy chạy Bước 2.5 trước.")

    if st.button("🚀 Bắt đầu Huấn luyện (Chạy File 2)"):
        if not os.path.exists(yaml_path):
            st.error("❌ Không tìm thấy data.yaml!")
        else:
            st.warning("⏳ Đang huấn luyện... Xem tiến độ ở Terminal VS Code.")
            with st.spinner("Đang chạy huấn luyện..."):
                result = run_script("2_train_yolo.py")
            if result.returncode == 0:
                st.success("✅ Xong! Trọng số lưu tại results/traffic_model/weights/best.pt")
            else:
                st.error(f"❌ Lỗi:\n{result.stderr}")

# --- BƯỚC 4: DỰ ĐOÁN ---
with st.sidebar.expander("4️⃣ Máy Học Dự Đoán", expanded=False):
    st.info("Chạy 3 thuật toán ML để dự đoán lưu lượng.")
    if st.button("📈 Mở Biểu đồ Dự đoán (Chạy File 4)"):
        if not os.path.exists(CSV_FILE):
            st.error("❌ Chưa có CSV! Hãy bật camera để thu thập dữ liệu trước.")
        else:
            script_path = os.path.join(BASE_DIR, "src", "4_prediction_models.py")
            subprocess.Popen([PYTHON_EXE, script_path], cwd=BASE_DIR)
            st.success("✅ Đã mở biểu đồ!")

st.sidebar.markdown("---")
run_system = st.sidebar.checkbox("🟢 BẬT CAMERA / NHẬN DIỆN THỰC TẾ", value=False)

# ==========================================
# 3. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    best_pt = os.path.join(BASE_DIR, "best.pt")
    if os.path.exists(best_pt):
        return YOLO(best_pt)
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

# ==========================================
# 4. VÒNG LẶP VIDEO & NHẬN DIỆN (ĐÃ SỬA)
# ==========================================
if run_system:
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Minute', 'Total_Vehicles', 'People',
                             'Bicycles', 'Motorcycles', 'Cars', 'Buses', 'Trucks'])

    # --- SỬA TẠI ĐÂY: Ưu tiên chạy video từ Bước 1, nếu không có mới dùng webcam ---
    if os.path.exists(VIDEO_PATH):
        cap = cv2.VideoCapture(VIDEO_PATH)
        st.sidebar.success(f"🎬 Đang chạy nhận diện trên video: traffic_01.mp4")
    else:
        cap = cv2.VideoCapture(0)
        st.sidebar.warning("⚠️ Không tìm thấy video, đang sử dụng Webcam.")
    
    if not cap.isOpened():
        st.error("❌ Không thể mở nguồn video!")
    else:
        # (Giữ nguyên các phần khởi tạo data_history bên dưới...)
        data_history = pd.DataFrame(columns=['Minute', 'Total_Vehicles', 'Bicycles',
                                              'Motorcycles', 'Cars', 'Buses', 'Trucks'])
        start_log_time = time.time()
        minute_counter = 1
        LOG_INTERVAL = 3 # Giảm xuống 3 giây để biểu đồ nhảy nhanh hơn cho bạn dễ quan sát

        while cap.isOpened() and run_system:
            ret, frame = cap.read()
            if not ret:
                # Nếu chạy hết video thì quay lại từ đầu để đếm tiếp (Loop)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Sử dụng mô hình bạn đã train (best.pt) với ngưỡng conf thấp do ít ảnh
            results = model(frame, classes=[0, 1, 2, 3, 5, 7], conf=0.15)
            
            # (Giữ nguyên phần code đếm xe và vẽ frame của bạn...)
            person_count = bicycle_count = car_count = motorcycle_count = bus_count = truck_count = 0
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == 0: person_count += 1
                    elif cls == 1: bicycle_count += 1
                    elif cls == 2: car_count += 1
                    elif cls == 3: motorcycle_count += 1
                    elif cls == 5: bus_count += 1
                    elif cls == 7: truck_count += 1
                frame = r.plot()

            total_vehicles = bicycle_count + car_count + motorcycle_count + bus_count + truck_count

            # Hiển thị lên giao diện
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Lưu vào CSV để file 4_prediction_models.py có dữ liệu vẽ biểu đồ
            current_time = time.time()
            if current_time - start_log_time >= LOG_INTERVAL:
                with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([minute_counter, total_vehicles, person_count,
                                     bicycle_count, motorcycle_count, car_count,
                                     bus_count, truck_count])
                
                minute_counter += 1
                start_log_time = current_time
        cap.release()