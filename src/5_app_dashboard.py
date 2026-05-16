import streamlit as st
import cv2
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from ultralytics import YOLO
import os
import subprocess
import zipfile
import sys

st.set_page_config(page_title="Traffic AI Dashboard", layout="wide")
st.title("He Thong Quan Ly Giao Thong AI")
st.markdown("Thu thap du lieu, huan luyen mo hinh, nhan dien thoi gian thuc.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable

DATA_VIDEOS     = os.path.join(BASE_DIR, "data", "videos")
DATA_IMAGES     = os.path.join(BASE_DIR, "data", "images")
DATA_DATASET    = os.path.join(BASE_DIR, "data", "dataset")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
CSV_FILE        = os.path.join(RESULTS_DIR, "traffic_data.csv")
VIDEO_PATH      = os.path.join(DATA_VIDEOS, "traffic_01.mp4")
TEST_VIDEO_PATH = os.path.join(DATA_VIDEOS, "test_video.mp4")

for d in [DATA_VIDEOS, DATA_IMAGES, DATA_DATASET, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

def run_script(script_name):
    script_path = os.path.join(BASE_DIR, "src", script_name)
    return subprocess.run(
        [PYTHON_EXE, script_path],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

def draw_prediction_chart(df_log):
    if len(df_log) < 3:
        return None
    X = df_log[["Minute"]].values
    y = df_log["Total_Vehicles"].values
    m1 = LinearRegression()
    m1.fit(X, y)
    y1 = m1.predict(X)
    poly = PolynomialFeatures(degree=min(3, len(df_log) - 1))
    Xp = poly.fit_transform(X)
    m2 = LinearRegression()
    m2.fit(Xp, y)
    y2 = m2.predict(Xp)
    m3 = RandomForestRegressor(n_estimators=100, random_state=42)
    m3.fit(X, y)
    y3 = m3.predict(X)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.scatter(X, y, color="white", label="Thuc te", alpha=0.8, zorder=5, s=30)
    ax.plot(X, y1, color="#4FC3F7", linestyle="--", linewidth=2, label="Tuyen tinh")
    ax.plot(X, y2, color="#FFB74D", linewidth=2, label="Da thuc (Bac 3)")
    ax.plot(X, y3, color="#81C784", linewidth=2, label="Random Forest")
    ax.set_title("Du doan Luu luong Giao thong", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Chu ky", color="#aaa")
    ax.set_ylabel("So phuong tien", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3, color="#555")
    plt.tight_layout()
    return fig

def draw_pie_chart(car, moto, bus, truck, bicycle, person):
    labels = ["Xe hoi", "Mo to", "Xe buyt", "Xe tai", "Xe dap", "Nguoi"]
    sizes  = [car, moto, bus, truck, bicycle, person]
    colors = ["#4FC3F7", "#FFB74D", "#CE93D8", "#EF9A9A", "#80CBC4", "#A5D6A7"]
    if sum(sizes) == 0:
        return None
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.pie(sizes, labels=labels, colors=colors,
           autopct=lambda p: f"{p:.0f}%" if p > 0 else "",
           startangle=90, textprops={"color": "white", "fontsize": 9})
    ax.set_title("Phan loai phuong tien", color="white", fontsize=10)
    plt.tight_layout()
    return fig

def kpi_table(car, moto, bus, truck, bicycle, person, current_total,
              total_car, total_moto, total_bus, total_truck, total_bicycle, total_person, grand_total):
    return (
        f"**Hien tai (trong khung hinh):** {current_total} phuong tien\n\n"
        f"| Xe hoi | Mo to | Xe buyt | Xe tai | Xe dap | Nguoi |\n"
        f"|:------:|:-----:|:-------:|:------:|:------:|:-----:|\n"
        f"| {car} | {moto} | {bus} | {truck} | {bicycle} | {person} |\n\n"
        f"---\n\n"
        f"**Tong da di qua (tich luy):** {grand_total} phuong tien\n\n"
        f"| Xe hoi | Mo to | Xe buyt | Xe tai | Xe dap | Nguoi |\n"
        f"|:------:|:-----:|:-------:|:------:|:------:|:-----:|\n"
        f"| {total_car} | {total_moto} | {total_bus} | {total_truck} | {total_bicycle} | {total_person} |"
    )

def process_tracking(results, seen_ids,
                     pc, bc, cc, mc, bsc, tc,
                     total_pc, total_bc, total_cc, total_mc, total_bsc, total_tc):
    """
    Xu ly ket qua tracking: dem xe trong khung hinh hien tai (tuc thoi)
    va tong xe da di qua (tich luy theo track ID).
    Tra ve: (frame da ve, cac bien dem tuc thoi, cac bien dem tich luy)
    """
    frame = None
    pc = bc = cc = mc = bsc = tc = 0
    for r in results:
        # Dem tuc thoi: so xe trong khung hinh hien tai
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:   pc  += 1
            elif cls == 1: bc  += 1
            elif cls == 2: cc  += 1
            elif cls == 3: mc  += 1
            elif cls == 5: bsc += 1
            elif cls == 7: tc  += 1
        # Dem tich luy: chi tinh xe co track_id chua tung thay
        if r.boxes.id is not None:
            for box, track_id in zip(r.boxes, r.boxes.id.int().tolist()):
                if track_id not in seen_ids:
                    seen_ids.add(track_id)
                    cls = int(box.cls[0])
                    if cls == 0:   total_pc  += 1
                    elif cls == 1: total_bc  += 1
                    elif cls == 2: total_cc  += 1
                    elif cls == 3: total_mc  += 1
                    elif cls == 5: total_bsc += 1
                    elif cls == 7: total_tc  += 1
        frame = r.plot()
    return (frame, pc, bc, cc, mc, bsc, tc,
            total_pc, total_bc, total_cc, total_mc, total_bsc, total_tc)

st.sidebar.title("BANG DIEU KHIEN")

with st.sidebar.expander("1. Nhap Video & Cat Anh", expanded=False):
    uploaded_video = st.file_uploader("Keo tha Video (.mp4 / .avi)", type=["mp4", "avi"])
    if uploaded_video is not None:
        save_path = os.path.join(DATA_VIDEOS, uploaded_video.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.success(f"Da luu: {uploaded_video.name}")
    video_files = sorted([f for f in os.listdir(DATA_VIDEOS) if f.endswith((".mp4", ".avi"))]) if os.path.exists(DATA_VIDEOS) else []
    if video_files:
        st.markdown("**Video hien co trong data/videos/:**")
        total_video_mb = 0
        for vf in video_files:
            vpath = os.path.join(DATA_VIDEOS, vf)
            vmb = os.path.getsize(vpath) / (1024 * 1024)
            total_video_mb += vmb
            vcap = cv2.VideoCapture(vpath)
            vfps = vcap.get(cv2.CAP_PROP_FPS)
            vframes = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            vsec = int(vframes / vfps) if vfps > 0 else 0
            vcap.release()
            st.info(f"{vf} | {vmb:.1f} MB | {vsec//60}p{vsec%60}s | {int(vfps)} fps")
        st.caption(f"Tong: {len(video_files)} video | {total_video_mb:.1f} MB")
    else:
        st.warning("Chua co video. Hay tai len truoc.")
    st.markdown("---")
    img_files = [f for f in os.listdir(DATA_IMAGES) if f.endswith((".jpg", ".png"))] if os.path.exists(DATA_IMAGES) else []
    img_count = len(img_files)
    if img_count > 0:
        st.markdown("**Anh da trich xuat:**")
        label_dir_tmp = os.path.join(DATA_DATASET, "labels")
        labeled_count = 0
        if os.path.exists(label_dir_tmp):
            labeled_names = {f.replace(".txt", "") for f in os.listdir(label_dir_tmp) if f.endswith(".txt")}
            labeled_count = sum(1 for f in img_files if f.rsplit(".", 1)[0] in labeled_names)
        st.success(f"Tong anh: {img_count}")
        st.info(f"Da gan nhan: {labeled_count} / {img_count}")
        st.caption(f"Chua gan nhan: {img_count - labeled_count}")
    else:
        st.caption("Chua co anh. Hay chay Trich xuat Anh.")
    if st.button("Trich xuat Anh (Chay File 1)"):
        if not video_files:
            st.error("Chua co video!")
        else:
            with st.spinner("Dang cat anh tu tat ca video..."):
                result = run_script("1_extract_frames.py")
            if result.returncode == 0:
                new_count = len([f for f in os.listdir(DATA_IMAGES) if f.endswith((".jpg", ".png"))])
                st.success(f"Xong! Tong {new_count} anh da duoc trich xuat.")
                st.code(result.stdout)
            else:
                st.error(f"Loi:\n{result.stderr}")

with st.sidebar.expander("2. Gan Nhan Du Lieu", expanded=False):
    st.info("Dung AI de tu dong gan nhan xe co.")
    img_count = len([f for f in os.listdir(DATA_IMAGES) if f.endswith((".jpg", ".png"))]) if os.path.exists(DATA_IMAGES) else 0
    st.caption(f"So anh hien co: {img_count}")
    if st.button("Chay Auto Label (Chay File 6)"):
        if img_count == 0:
            st.error("Thu muc anh trong! Hay chay Buoc 1 truoc.")
        else:
            with st.spinner("AI dang gan nhan..."):
                result = run_script("6_auto_label.py")
            if result.returncode == 0:
                st.success("Gan nhan xong! Kiem tra data/dataset/labels/")
            else:
                st.error(f"Loi:\n{result.stderr}")

with st.sidebar.expander("2.5. Chuan Bi Dataset", expanded=False):
    st.info("Tu dong chia anh 80/20 va tao data.yaml. Chay sau Buoc 2.")
    label_dir   = os.path.join(DATA_DATASET, "labels")
    label_count = len([f for f in os.listdir(label_dir) if f.endswith(".txt")]) if os.path.exists(label_dir) else 0
    st.caption(f"So nhan hien co: {label_count}")
    yaml_path = os.path.join(DATA_DATASET, "data.yaml")
    if os.path.exists(yaml_path):
        st.success("data.yaml da ton tai - san sang huan luyen!")
    if st.button("Tao Dataset & data.yaml"):
        if label_count == 0:
            st.error("Chua co nhan! Hay chay Buoc 2 truoc.")
        else:
            with st.spinner("Dang chuan bi dataset..."):
                result = run_script("7_prepare_dataset.py")
            if result.returncode == 0:
                st.success("Xong! Dataset da san sang de huan luyen.")
                st.code(result.stdout)
            else:
                st.error(f"Loi:\n{result.stderr}")

with st.sidebar.expander("3. Huan Luyen AI", expanded=False):
    yaml_path    = os.path.join(DATA_DATASET, "data.yaml")
    uploaded_zip = st.file_uploader("(Tuy chon) Keo tha Dataset Roboflow (.zip)", type=["zip"])
    if uploaded_zip is not None:
        zip_path = os.path.join(DATA_DATASET, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DATASET)
        st.success("Da giai nen Dataset!")
    if os.path.exists(yaml_path):
        st.info("Da co data.yaml - san sang huan luyen.")
    else:
        st.warning("Chua co data.yaml. Hay chay Buoc 2.5 truoc.")
    if st.button("Bat dau Huan luyen (Chay File 2)"):
        if not os.path.exists(yaml_path):
            st.error("Khong tim thay data.yaml!")
        else:
            st.warning("Dang huan luyen... Xem tien do o Terminal VS Code.")
            with st.spinner("Dang chay huan luyen..."):
                result = run_script("2_train_yolo.py")
            if result.returncode == 0:
                st.success("Xong! App se tu dong tim va dung best.pt moi nhat.")
            else:
                st.error(f"Loi:\n{result.stderr}")

with st.sidebar.expander("4. May Hoc Du Doan", expanded=False):
    st.success("Bieu do du doan da duoc tich hop vao man hinh chinh - hien thi realtime khi bat Camera hoac chay Test Video.")

with st.sidebar.expander("5. Test Video & Du Doan", expanded=True):
    st.info("Tai video len de nhan dien va xem du doan luu luong ngay tren dashboard.")
    uploaded_test = st.file_uploader("Keo tha Video Test (.mp4 / .avi)", type=["mp4", "avi"], key="test_video_uploader")
    if uploaded_test is not None:
        with open(TEST_VIDEO_PATH, "wb") as f:
            f.write(uploaded_test.getbuffer())
        st.success(f"Da luu: {uploaded_test.name}")
    if os.path.exists(TEST_VIDEO_PATH):
        size_mb = os.path.getsize(TEST_VIDEO_PATH) / (1024 * 1024)
        st.info(f"test_video.mp4 ({size_mb:.1f} MB)")
    test_conf     = st.slider("Nguong tin cay (conf)", 0.10, 0.90, 0.25, 0.05)
    test_interval = st.slider("Ghi log moi N giay", 1, 10, 3)
    if st.button("Bat dau Test & Du doan", type="primary"):
        st.session_state["run_test"] = True
    if st.button("Dung Test"):
        st.session_state["run_test"] = False

st.sidebar.markdown("---")
run_system = st.sidebar.checkbox("BAT CAMERA / NHAN DIEN THUC TE", value=False)

if "run_test" not in st.session_state:
    st.session_state["run_test"] = False

@st.cache_resource
def load_model():
    import glob
    search_dirs = [
        os.path.join(BASE_DIR, "results"),
        os.path.join(BASE_DIR, "runs", "detect", "results"),
    ]
    candidates = []
    for d in search_dirs:
        if os.path.exists(d):
            pattern = os.path.join(d, "traffic_model*", "weights", "best.pt")
            candidates.extend(glob.glob(pattern))
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        best = candidates[0]
        rel = os.path.relpath(best, BASE_DIR)
        st.sidebar.success(f"Model: {rel}")
        return YOLO(best)
    root_pt = os.path.join(BASE_DIR, "best.pt")
    if os.path.exists(root_pt):
        st.sidebar.success("Model: best.pt (thu muc goc)")
        return YOLO(root_pt)
    st.sidebar.warning("Khong tim thay best.pt - dung YOLOv8n mac dinh.")
    return YOLO("yolov8n.pt")

model = load_model()

if st.session_state["run_test"]:
    if not os.path.exists(TEST_VIDEO_PATH):
        st.error("Chua co video test! Hay tai len o Buoc 5 trong sidebar.")
        st.session_state["run_test"] = False
    else:
        st.markdown("---")
        st.subheader("Ket Qua Test Video")
        col_vid, col_chart = st.columns([6, 4])
        with col_vid:
            st.markdown("**Video dang nhan dien**")
            ph_frame = st.empty()
            ph_kpi   = st.empty()
        with col_chart:
            st.markdown("**Bieu do Du doan (cap nhat theo thoi gian)**")
            ph_chart = st.empty()
            ph_pie   = st.empty()

        test_log   = []
        cap_test   = cv2.VideoCapture(TEST_VIDEO_PATH)
        n_frames   = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
        t_log      = time.time()
        minute_ctr = 1
        frame_idx  = 0
        LOG_INTERVAL = test_interval
        progress_bar = st.progress(0, text="Dang xu ly video...")

        # Bien dem tuc thoi (trong khung hinh)
        pc = bc = cc = mc = bsc = tc = 0
        # Bien dem tich luy (tong xe da di qua)
        total_pc = total_bc = total_cc = total_mc = total_bsc = total_tc = 0
        # Tap hop track_id da tung xuat hien
        seen_ids = set()

        while cap_test.isOpened() and st.session_state["run_test"]:
            ret, frame = cap_test.read()
            if not ret:
                break
            frame_idx += 1
            progress_bar.progress(min(frame_idx / max(n_frames, 1), 1.0),
                                   text=f"Dang xu ly... {int(frame_idx/max(n_frames,1)*100)}%")
            if frame_idx % 3 != 0:
                continue

            # Dung model.track thay vi model() de lay track_id
            results = model.track(frame, classes=[0, 1, 2, 3, 5, 7],
                                  conf=test_conf, persist=True, verbose=False)

            frame, pc, bc, cc, mc, bsc, tc, total_pc, total_bc, total_cc, total_mc, total_bsc, total_tc = \
                process_tracking(results, seen_ids,
                                 pc, bc, cc, mc, bsc, tc,
                                 total_pc, total_bc, total_cc, total_mc, total_bsc, total_tc)
            if frame is None:
                continue

            current_total = bc + cc + mc + bsc + tc
            grand_total   = total_bc + total_cc + total_mc + total_bsc + total_tc

            ph_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            ph_kpi.markdown(kpi_table(
                cc, mc, bsc, tc, bc, pc, current_total,
                total_cc, total_mc, total_bsc, total_tc, total_bc, total_pc, grand_total
            ))

            now = time.time()
            if now - t_log >= LOG_INTERVAL:
                # Ghi log voi gia tri tich luy (grand_total) de bieu do phan anh thuc te
                test_log.append({
                    "Minute":         minute_ctr,
                    "Total_Vehicles": grand_total,
                    "People":         total_pc,
                    "Bicycles":       total_bc,
                    "Motorcycles":    total_mc,
                    "Cars":           total_cc,
                    "Buses":          total_bsc,
                    "Trucks":         total_tc
                })
                minute_ctr += 1
                t_log = now
                if len(test_log) >= 3:
                    df_tmp = pd.DataFrame(test_log)
                    fig = draw_prediction_chart(df_tmp)
                    if fig:
                        ph_chart.pyplot(fig)
                        plt.close(fig)
                    fig_pie = draw_pie_chart(total_cc, total_mc, total_bsc,
                                             total_tc, total_bc, total_pc)
                    if fig_pie:
                        ph_pie.pyplot(fig_pie)
                        plt.close(fig_pie)

        cap_test.release()
        st.session_state["run_test"] = False
        progress_bar.progress(1.0, text="Xu ly xong!")

        if len(test_log) >= 2:
            df_final = pd.DataFrame(test_log)
            st.markdown("---")
            st.subheader("Phan Tich & Du Doan Cuoi Cung")
            col_a, col_b = st.columns(2)
            with col_a:
                fig_f = draw_prediction_chart(df_final)
                if fig_f:
                    st.pyplot(fig_f)
                    plt.close(fig_f)
            with col_b:
                st.dataframe(df_final, use_container_width=True)
                df_final.to_csv(os.path.join(RESULTS_DIR, "test_video_result.csv"), index=False, encoding="utf-8")
                st.success("Da luu ket qua: results/test_video_result.csv")
                st.markdown("**Thong ke tong hop**")
                r1a, r1b = st.columns(2)
                r2a, r2b = st.columns(2)
                r1a.metric("Tong chu ky", len(df_final))
                r1b.metric("TB xe / chu ky", f"{df_final['Total_Vehicles'].mean():.1f}")
                r2a.metric("Tong xe da di qua", int(df_final["Total_Vehicles"].max()))
                r2b.metric("Thap nhat / chu ky", int(df_final["Total_Vehicles"].min()))
        else:
            st.warning("Video qua ngan hoac khong phat hien duoc phuong tien nao.")

elif run_system:
    col_video, col_charts = st.columns([6, 4])
    with col_video:
        st.subheader("Live Camera Feed")
        ph_frame = st.empty()
        ph_kpi   = st.empty()
    with col_charts:
        st.subheader("Phan Tich Du Lieu")
        ph_chart = st.empty()
        ph_pie   = st.empty()
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Minute", "Total_Vehicles", "People",
                                     "Bicycles", "Motorcycles", "Cars", "Buses", "Trucks"])
    using_webcam = not os.path.exists(VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH if not using_webcam else 0)
    if using_webcam:
        st.sidebar.warning("Khong tim thay video, dang su dung Webcam.")
    else:
        st.sidebar.success("Dang chay nhan dien tren video: traffic_01.mp4")
    if not cap.isOpened():
        st.error("Khong the mo nguon video!")
    else:
        if os.path.exists(CSV_FILE):
            _df = pd.read_csv(CSV_FILE)
            minute_ctr = int(_df["Minute"].max()) + 1 if len(_df) > 0 else 1
        else:
            minute_ctr = 1
        t_log        = time.time()
        LOG_INTERVAL = test_interval
        frame_idx    = 0
        # Bien dem tuc thoi
        pc = bc = cc = mc = bsc = tc = 0
        # Bien dem tich luy
        total_pc = total_bc = total_cc = total_mc = total_bsc = total_tc = 0
        seen_ids = set()

        while cap.isOpened() and run_system:
            ret, frame = cap.read()
            if not ret:
                if using_webcam:
                    st.warning("Mat ket noi camera. Dang dung...")
                    break
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            frame_idx += 1
            if frame_idx % 3 != 0:
                continue

            results = model.track(frame, classes=[0, 1, 2, 3, 5, 7],
                                  conf=test_conf, persist=True, verbose=False)

            frame, pc, bc, cc, mc, bsc, tc, total_pc, total_bc, total_cc, total_mc, total_bsc, total_tc = \
                process_tracking(results, seen_ids,
                                 pc, bc, cc, mc, bsc, tc,
                                 total_pc, total_bc, total_cc, total_mc, total_bsc, total_tc)
            if frame is None:
                continue

            current_total = bc + cc + mc + bsc + tc
            grand_total   = total_bc + total_cc + total_mc + total_bsc + total_tc

            ph_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            ph_kpi.markdown(kpi_table(
                cc, mc, bsc, tc, bc, pc, current_total,
                total_cc, total_mc, total_bsc, total_tc, total_bc, total_pc, grand_total
            ))

            now = time.time()
            if now - t_log >= LOG_INTERVAL:
                # Ghi log grand_total (tich luy) de bieu do chinh xac
                with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        minute_ctr, grand_total,
                        total_pc, total_bc, total_mc,
                        total_cc, total_bsc, total_tc
                    ])
                df_live = pd.read_csv(CSV_FILE)
                if len(df_live) >= 3:
                    fig_live = draw_prediction_chart(df_live)
                    if fig_live:
                        ph_chart.pyplot(fig_live)
                        plt.close(fig_live)
                fig_pie = draw_pie_chart(total_cc, total_mc, total_bsc,
                                         total_tc, total_bc, total_pc)
                if fig_pie:
                    ph_pie.pyplot(fig_pie)
                    plt.close(fig_pie)
                minute_ctr += 1
                t_log = now
        cap.release()

else:
    col_l, col_r = st.columns([6, 4])
    with col_l:
        st.subheader("Live Camera Feed")
        st.info("Bat BAT CAMERA o sidebar de xem nhan dien thoi gian thuc, hoac dung Buoc 5 de test voi video co san.")
    with col_r:
        st.subheader("Phan Tich Du Lieu")
        if os.path.exists(CSV_FILE):
            df_ex = pd.read_csv(CSV_FILE)
            if len(df_ex) >= 3:
                st.markdown("**Du lieu tu phien truoc:**")
                fig = draw_prediction_chart(df_ex)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Chua co du du lieu de ve bieu do.")
        else:
            st.info("Chua co du lieu CSV. Hay chay nhan dien de thu thap du lieu.")