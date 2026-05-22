import streamlit as st
import cv2
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
import sys
import glob
import re
from ultralytics import YOLO

# Import tu cac file tach rieng
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    BASE_DIR, PYTHON_EXE,
    DATA_VIDEOS, DATA_IMAGES, DATA_DATASET, RESULTS_DIR,
    CSV_FILE, VIDEO_PATH, TEST_VIDEO_PATH,
    run_script, find_latest_model_dir,
    draw_prediction_chart, draw_pie_chart,
    kpi_table, process_tracking
)
from eda import render_eda

# ==========================================
# 1. CAU HINH TRANG
# ==========================================
st.set_page_config(page_title="Traffic AI Dashboard", layout="wide")
st.title("He Thong Quan Ly Giao Thong AI")
st.markdown("Thu thap du lieu, huan luyen mo hinh, nhan dien thoi gian thuc.")

for d in [DATA_VIDEOS, DATA_IMAGES, DATA_DATASET, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 2. SIDEBAR
# ==========================================
st.sidebar.title("BANG DIEU KHIEN")

# --- BUOC 1 ---
with st.sidebar.expander("1. Nhap Video & Cat Anh", expanded=False):
    uploaded_video = st.file_uploader("Keo tha Video (.mp4 / .avi)", type=["mp4", "avi"])
    if uploaded_video is not None:
        save_path = os.path.join(DATA_VIDEOS, uploaded_video.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.success(f"Da luu: {uploaded_video.name}")
    video_files = sorted([f for f in os.listdir(DATA_VIDEOS)
                          if f.endswith((".mp4", ".avi"))]) if os.path.exists(DATA_VIDEOS) else []
    if video_files:
        st.markdown("**Video hien co trong data/videos/:**")
        total_video_mb = 0
        for vf in video_files:
            vpath = os.path.join(DATA_VIDEOS, vf)
            vmb = os.path.getsize(vpath) / (1024 * 1024)
            total_video_mb += vmb
            vcap = cv2.VideoCapture(vpath)
            vfps    = vcap.get(cv2.CAP_PROP_FPS)
            vframes = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            vsec    = int(vframes / vfps) if vfps > 0 else 0
            vcap.release()
            st.info(f"{vf} | {vmb:.1f} MB | {vsec//60}p{vsec%60}s | {int(vfps)} fps")
        st.caption(f"Tong: {len(video_files)} video | {total_video_mb:.1f} MB")
    else:
        st.warning("Chua co video. Hay tai len truoc.")
    st.markdown("---")
    img_files = ([f for f in os.listdir(DATA_IMAGES)
                  if f.endswith((".jpg", ".png"))]
                 if os.path.exists(DATA_IMAGES) else [])
    img_count = len(img_files)
    if img_count > 0:
        st.markdown("**Anh da trich xuat:**")
        label_dir_tmp = os.path.join(DATA_DATASET, "labels")
        labeled_count = 0
        if os.path.exists(label_dir_tmp):
            labeled_names = {f.replace(".txt", "")
                             for f in os.listdir(label_dir_tmp) if f.endswith(".txt")}
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
                new_count = len([f for f in os.listdir(DATA_IMAGES)
                                 if f.endswith((".jpg", ".png"))])
                st.success(f"Xong! Tong {new_count} anh da duoc trich xuat.")
                st.code(result.stdout)
            else:
                st.error(f"Loi:\n{result.stderr}")

# --- BUOC 2 ---
with st.sidebar.expander("2. Gan Nhan Du Lieu", expanded=False):
    st.info("Dung AI de tu dong gan nhan xe co.")
    img_count = (len([f for f in os.listdir(DATA_IMAGES)
                      if f.endswith((".jpg", ".png"))])
                 if os.path.exists(DATA_IMAGES) else 0)
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

# --- BUOC 2.5 ---
with st.sidebar.expander("2.5. Chuan Bi Dataset", expanded=False):
    st.info("Tu dong chia anh 80/20 va tao data.yaml. Chay sau Buoc 2.")
    label_dir   = os.path.join(DATA_DATASET, "labels")
    label_count = (len([f for f in os.listdir(label_dir) if f.endswith(".txt")])
                   if os.path.exists(label_dir) else 0)
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

# --- BUOC 3 ---
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

# --- BUOC 4 ---
with st.sidebar.expander("4. May Hoc Du Doan", expanded=False):
    st.success("Bieu do du doan da duoc tich hop vao man hinh chinh.")

# --- BUOC 5 ---
with st.sidebar.expander("5. Test Video & Du Doan", expanded=True):
    st.info("Tai video len de nhan dien va xem du doan luu luong.")
    uploaded_test = st.file_uploader("Keo tha Video Test (.mp4 / .avi)",
                                     type=["mp4", "avi"], key="test_video_uploader")
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

# --- BUOC 6 ---
with st.sidebar.expander("6. Phan Tich Du Lieu & Mo Hinh (EDA)", expanded=False):
    st.info("Hien thi phan bo nhan dataset, Loss chart va Confusion Matrix.")
    if st.button("Hien thi Phan Tich"):
        st.session_state["show_eda"] = True
    if st.button("Dong Phan Tich"):
        st.session_state["show_eda"] = False

st.sidebar.markdown("---")
run_system = st.sidebar.checkbox("BAT CAMERA / NHAN DIEN THUC TE", value=False)

# Khoi tao session_state
for key in ["run_test", "show_eda"]:
    if key not in st.session_state:
        st.session_state[key] = False

# ==========================================
# 3. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    search_dirs = [
        os.path.join(BASE_DIR, "runs", "detect", "results"),
        os.path.join(BASE_DIR, "results"),
    ]
    candidates = []
    for d in search_dirs:
        if os.path.exists(d):
            pattern = os.path.join(d, "traffic_model*", "weights", "best.pt")
            candidates.extend(glob.glob(pattern))
    if candidates:
        def _num(p):
            m = re.search(r"traffic_model-?(\d*)", p)
            return int(m.group(1)) if m and m.group(1) else 0
        candidates.sort(key=_num, reverse=True)
        best = candidates[0]
        rel  = os.path.relpath(best, BASE_DIR)
        st.sidebar.success(f"Model: {rel}")
        return YOLO(best)
    root_pt = os.path.join(BASE_DIR, "best.pt")
    if os.path.exists(root_pt):
        st.sidebar.success("Model: best.pt (thu muc goc)")
        return YOLO(root_pt)
    st.sidebar.warning("Khong tim thay best.pt - dung YOLOv8n mac dinh.")
    return YOLO("yolov8n.pt")

model = load_model()

# ==========================================
# 4. VONG LAP NHAN DIEN - TEST VIDEO
# ==========================================
if st.session_state["run_test"]:
    if not os.path.exists(TEST_VIDEO_PATH):
        st.error("Chua co video test! Hay tai len o Buoc 5.")
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

        test_log = []
        cap_test = cv2.VideoCapture(TEST_VIDEO_PATH)
        n_frames = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
        t_log    = time.time()
        minute_ctr = 1
        frame_idx  = 0
        LOG_INTERVAL = test_interval
        progress_bar = st.progress(0, text="Dang xu ly video...")

        pc = bc = cc = mc = bsc = tc = 0
        total_pc = total_bc = total_cc = total_mc = total_bsc = total_tc = 0
        seen_ids  = set()
        fps_list  = []          # Luu FPS tung frame de tinh trung binh
        t_frame   = time.time() # Thoi gian frame truoc

        while cap_test.isOpened() and st.session_state["run_test"]:
            ret, frame = cap_test.read()
            if not ret:
                break
            frame_idx += 1
            progress_bar.progress(
                min(frame_idx / max(n_frames, 1), 1.0),
                text=f"Dang xu ly... {int(frame_idx/max(n_frames,1)*100)}%"
            )
            # Tinh FPS thuc te
            now_f   = time.time()
            fps     = int(1 / (now_f - t_frame)) if (now_f - t_frame) > 0 else 0
            t_frame = now_f
            fps_list.append(fps)
            avg_fps = sum(fps_list) / len(fps_list)

            # Luon track moi frame de tracker khong mat dau xe (persist=True)
            results = model.track(frame, classes=[0, 1, 2, 3, 5, 7],
                                  conf=test_conf, persist=True, verbose=False)
            (frame, pc, bc, cc, mc, bsc, tc,
             total_pc, total_bc, total_cc, total_mc, total_bsc, total_tc) = process_tracking(
                results, seen_ids, pc, bc, cc, mc, bsc, tc,
                total_pc, total_bc, total_cc, total_mc, total_bsc, total_tc,
                fps=fps, avg_fps=avg_fps
            )
            if frame is None:
                continue

            current_total = bc + cc + mc + bsc + tc
            grand_total   = total_bc + total_cc + total_mc + total_bsc + total_tc

            # Chi cap nhat UI moi 3 frame de giam lag Streamlit
            # Tracker van nhan du frame nen khong mat dau xe
            if frame_idx % 3 == 0:
                ph_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)
                ph_kpi.markdown(kpi_table(
                    cc, mc, bsc, tc, bc, pc, current_total,
                    total_cc, total_mc, total_bsc, total_tc, total_bc, total_pc,
                    grand_total, fps=fps, avg_fps=avg_fps
                ))

            now = time.time()
            if now - t_log >= LOG_INTERVAL:
                test_log.append({
                    "Minute": minute_ctr, "Total_Vehicles": grand_total,
                    "People": total_pc, "Bicycles": total_bc,
                    "Motorcycles": total_mc, "Cars": total_cc,
                    "Buses": total_bsc, "Trucks": total_tc
                })
                minute_ctr += 1
                t_log = now
                if len(test_log) >= 3:
                    df_tmp = pd.DataFrame(test_log)
                    fig, _ = draw_prediction_chart(df_tmp)
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
                fig_f, metrics_f = draw_prediction_chart(df_final)
                if fig_f:
                    st.pyplot(fig_f)
                    plt.close(fig_f)
                if metrics_f:
                    st.markdown("**So sanh hieu suat 3 mo hinh hoi quy:**")
                    rows = [{"Mo hinh": k,
                             "MSE":  round(v["MSE"],  3),
                             "RMSE": round(v["RMSE"], 3),
                             "MAE":  round(v["MAE"],  3),
                             "R2":   round(v["R2"],   4)}
                            for k, v in metrics_f.items()]
                    df_metrics = pd.DataFrame(rows).set_index("Mo hinh")
                    st.dataframe(df_metrics, use_container_width=True)
                    best_r2 = df_metrics["R2"].idxmax()
                    st.caption(f"Mo hinh tot nhat theo R2: {best_r2}")
            with col_b:
                st.dataframe(df_final, use_container_width=True)
                df_final.to_csv(
                    os.path.join(RESULTS_DIR, "test_video_result.csv"),
                    index=False, encoding="utf-8"
                )
                st.success("Da luu ket qua: results/test_video_result.csv")
                st.markdown("**Thong ke tong hop**")
                r1a, r1b = st.columns(2)
                r2a, r2b = st.columns(2)
                r1a.metric("Tong chu ky", len(df_final))
                r1b.metric("TB xe / chu ky",
                           f"{df_final['Total_Vehicles'].mean():.1f}")
                r2a.metric("Tong xe da di qua",
                           int(df_final["Total_Vehicles"].max()))
                r2b.metric("Thap nhat / chu ky",
                           int(df_final["Total_Vehicles"].min()))
                if fps_list:
                    st.markdown("**Hieu nang xu ly (FPS):**")
                    f1, f2, f3 = st.columns(3)
                    f1.metric("FPS trung binh", f"{sum(fps_list)/len(fps_list):.1f}")
                    f2.metric("FPS cao nhat",   max(fps_list))
                    f3.metric("FPS thap nhat",  min(fps_list))
                    perf = "Real-time" if sum(fps_list)/len(fps_list) >= 20 else                            "Chap nhan duoc" if sum(fps_list)/len(fps_list) >= 10 else                            "Can toi uu them"
                    st.caption(f"Danh gia hieu nang: {perf} (>= 20 FPS: Real-time | 10-20: OK | < 10: Cham)")
        else:
            st.warning("Video qua ngan hoac khong phat hien duoc phuong tien nao.")

# ==========================================
# 5. VONG LAP NHAN DIEN - LIVE CAMERA
# ==========================================
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
        pc = bc = cc = mc = bsc = tc = 0
        total_pc = total_bc = total_cc = total_mc = total_bsc = total_tc = 0
        seen_ids  = set()
        fps_list  = []
        t_frame   = time.time()

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
            # Tinh FPS thuc te
            now_f   = time.time()
            fps     = int(1 / (now_f - t_frame)) if (now_f - t_frame) > 0 else 0
            t_frame = now_f
            fps_list.append(fps)
            avg_fps = sum(fps_list) / len(fps_list)

            # Luon track moi frame de tracker khong mat dau xe (persist=True)
            results = model.track(frame, classes=[0, 1, 2, 3, 5, 7],
                                  conf=test_conf, persist=True, verbose=False)
            (frame, pc, bc, cc, mc, bsc, tc,
             total_pc, total_bc, total_cc, total_mc, total_bsc, total_tc) = process_tracking(
                results, seen_ids, pc, bc, cc, mc, bsc, tc,
                total_pc, total_bc, total_cc, total_mc, total_bsc, total_tc,
                fps=fps, avg_fps=avg_fps
            )
            if frame is None:
                continue

            current_total = bc + cc + mc + bsc + tc
            grand_total   = total_bc + total_cc + total_mc + total_bsc + total_tc

            # Chi cap nhat UI moi 3 frame de giam lag Streamlit
            # Tracker van nhan du frame nen khong mat dau xe
            if frame_idx % 3 == 0:
                ph_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)
                ph_kpi.markdown(kpi_table(
                    cc, mc, bsc, tc, bc, pc, current_total,
                    total_cc, total_mc, total_bsc, total_tc, total_bc, total_pc,
                    grand_total, fps=fps, avg_fps=avg_fps
                ))

            now = time.time()
            if now - t_log >= LOG_INTERVAL:
                with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        minute_ctr, grand_total,
                        total_pc, total_bc, total_mc,
                        total_cc, total_bsc, total_tc
                    ])
                df_live = pd.read_csv(CSV_FILE)
                if len(df_live) >= 3:
                    fig_live, _ = draw_prediction_chart(df_live)
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

# ==========================================
# 6. MAN HINH MAC DINH
# ==========================================
else:
    col_l, col_r = st.columns([6, 4])
    with col_l:
        st.subheader("Live Camera Feed")
        st.info("Bat BAT CAMERA o sidebar de xem nhan dien thoi gian thuc, "
                "hoac dung Buoc 5 de test voi video co san.")
    with col_r:
        st.subheader("Phan Tich Du Lieu")
        if os.path.exists(CSV_FILE):
            df_ex = pd.read_csv(CSV_FILE)
            if len(df_ex) >= 3:
                st.markdown("**Du lieu tu phien truoc:**")
                fig, _ = draw_prediction_chart(df_ex)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("Chua co du du lieu de ve bieu do.")
        else:
            st.info("Chua co du lieu CSV. Hay chay nhan dien de thu thap du lieu.")

# ==========================================
# 7. EDA (Buoc 6 sidebar)
# ==========================================
if st.session_state["show_eda"]:
    render_eda()