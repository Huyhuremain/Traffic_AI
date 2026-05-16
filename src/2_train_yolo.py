from ultralytics import YOLO
import os
import glob

def find_latest_best_pt(base_dir):
    search_dirs = [
        os.path.join(base_dir, "results"),
        os.path.join(base_dir, "runs", "detect", "results"),
    ]
    candidates = []
    for d in search_dirs:
        if os.path.exists(d):
            pattern = os.path.join(d, "traffic_model*", "weights", "best.pt")
            candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml_path = os.path.join(base_dir, "data", "dataset", "data.yaml")

    # ==========================================
    # 1. KIEM TRA DATA.YAML
    # ==========================================
    if not os.path.exists(data_yaml_path):
        print(f"Loi: Khong tim thay file {data_yaml_path}!")
        print("Vui long chay Buoc 2.5 tren Dashboard truoc.")
        return

    # ==========================================
    # 2. KHOI TAO MO HINH
    # Tu dong chon: dung best.pt cu neu co (fine-tuning),
    # neu khong moi dung yolov8n.pt (train moi)
    # ==========================================
    latest_best = find_latest_best_pt(base_dir)

    if latest_best:
        print("=" * 50)
        print("[Fine-tuning] Tim thay model cu:")
        print(f"  {latest_best}")
        print("AI se hoc them tren nen kien thuc cu, khong train lai tu dau.")
        print("=" * 50)
        model = YOLO(latest_best)
        freeze_layers = 10
    else:
        print("=" * 50)
        print("[Train moi] Khong tim thay model cu, bat dau tu yolov8n.pt")
        print("=" * 50)
        model = YOLO("yolov8n.pt")
        freeze_layers = 0

    # ==========================================
    # 3. TIEN HANH HUAN LUYEN
    # project dung duong dan tuyet doi -> YOLO luu vao DLCK/results/traffic_model*
    # khong bi lap thanh runs/detect/runs/detect/results
    # ==========================================
    print("BAT DAU HUAN LUYEN MO HINH...")

    model.train(
        data=data_yaml_path,
        epochs=30,
        imgsz=640,
        batch=4,
        workers=1,
        project="results",
        name="traffic_model",
        plots=True,
        device="0",
        freeze=freeze_layers,
        patience=15,
    )

    # ==========================================
    # 4. KET THUC
    # ==========================================
    new_best = find_latest_best_pt(base_dir)
    print("=" * 50)
    print("HUAN LUYEN HOAN TAT!")
    if new_best:
        print("Trong so tot nhat luu tai:")
        print(f"  {new_best}")
    print("=" * 50)
    print("Mo Dashboard va bat dau Test Video hoac Live Camera.")
    print("App se tu dong tim va dung best.pt moi nhat - khong can sua gi them.")

if __name__ == "__main__":
    main()