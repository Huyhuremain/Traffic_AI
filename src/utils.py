import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import subprocess
import sys


# ==========================================
# DUONG DAN GOC
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable

DATA_VIDEOS     = os.path.join(BASE_DIR, "data", "videos")
DATA_IMAGES     = os.path.join(BASE_DIR, "data", "images")
DATA_DATASET    = os.path.join(BASE_DIR, "data", "dataset")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
CSV_FILE        = os.path.join(RESULTS_DIR, "traffic_data.csv")
VIDEO_PATH      = os.path.join(DATA_VIDEOS, "traffic_01.mp4")
TEST_VIDEO_PATH = os.path.join(DATA_VIDEOS, "test_video.mp4")


# ==========================================
# HAM TIEN ICH CHUNG
# ==========================================
def run_script(script_name):
    """Chay mot file Python trong thu muc src."""
    script_path = os.path.join(BASE_DIR, "src", script_name)
    return subprocess.run(
        [PYTHON_EXE, script_path],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )


def find_latest_model_dir(base_dir):
    """Tim thu muc traffic_model* moi nhat dua tren so trong ten."""
    search_dirs = [
        os.path.join(base_dir, "runs", "detect", "results"),
        os.path.join(base_dir, "results"),
    ]
    candidates = []
    for d in search_dirs:
        if os.path.exists(d):
            pattern = os.path.join(d, "traffic_model*")
            candidates.extend([p for p in glob.glob(pattern) if os.path.isdir(p)])
    if not candidates:
        return None
    def extract_num(path):
        m = re.search(r"traffic_model-?(\d*)", os.path.basename(path))
        return int(m.group(1)) if m and m.group(1) else 0
    candidates.sort(key=extract_num, reverse=True)
    return candidates[0]


def count_labels(label_dir):
    """Dem so luong nhan theo tung class trong mot thu muc labels."""
    class_names = {0:"person", 1:"bicycle", 2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}
    counts = {v: 0 for v in class_names.values()}
    for fpath in glob.glob(os.path.join(label_dir, "*.txt")):
        for line in open(fpath, "r", encoding="utf-8"):
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                if cls in class_names:
                    counts[class_names[cls]] += 1
    return counts


# ==========================================
# HAM VE BIEU DO DU DOAN (REGRESSION)
# ==========================================
def calc_metrics(y_true, y_pred):
    """Tinh MSE, RMSE, MAE, R2 cho 1 model."""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def draw_prediction_chart(df_log):
    """Ve bieu do du doan luu luong va tra ve (fig, dict chi so danh gia)."""
    if len(df_log) < 3:
        return None, None
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

    metrics = {
        "Hoi quy Tuyen tinh": calc_metrics(y, y1),
        "Hoi quy Da thuc":    calc_metrics(y, y2),
        "Random Forest":      calc_metrics(y, y3),
    }
    return fig, metrics


def draw_pie_chart(car, moto, bus, truck, bicycle, person):
    """Ve bieu do tron phan loai phuong tien."""
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


# ==========================================
# HAM HIEN THI KPI
# ==========================================
def kpi_table(car, moto, bus, truck, bicycle, person, current_total,
              total_car, total_moto, total_bus, total_truck, total_bicycle, total_person, grand_total):
    """Tao bang KPI hien thi so xe tuc thoi va tich luy."""
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


# ==========================================
# HAM XU LY TRACKING
# ==========================================
def process_tracking(results, seen_ids,
                     pc, bc, cc, mc, bsc, tc,
                     total_pc, total_bc, total_cc, total_mc, total_bsc, total_tc):
    """
    Xu ly ket qua tracking:
    - Dem tuc thoi: so xe trong khung hinh tai frame hien tai
    - Dem tich luy: chi tinh xe co track_id chua tung xuat hien
    """
    frame = None
    pc = bc = cc = mc = bsc = tc = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:   pc  += 1
            elif cls == 1: bc  += 1
            elif cls == 2: cc  += 1
            elif cls == 3: mc  += 1
            elif cls == 5: bsc += 1
            elif cls == 7: tc  += 1
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