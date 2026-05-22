from ultralytics import YOLO
import os
import glob
import re

def find_latest_best_pt(base_dir):
    search_dirs = [
        os.path.join(base_dir, 'results'),
        os.path.join(base_dir, 'runs', 'detect', 'results'),
    ]
    candidates = []
    for d in search_dirs:
        if os.path.exists(d):
            pattern = os.path.join(d, 'traffic_model*', 'weights', 'best.pt')
            candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    # Sap xep theo SO trong ten thu muc, khong dung getmtime (co the bi nham)
    def extract_num(path):
        m = re.search(r'traffic_model-?(\d*)', path)
        return int(m.group(1)) if m and m.group(1) else 0
    candidates.sort(key=extract_num, reverse=True)
    return candidates[0]

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_yaml_path = os.path.join(base_dir, 'data', 'dataset', 'data.yaml')

    # ==========================================
    # 1. KIEM TRA DATA.YAML
    # ==========================================
    if not os.path.exists(data_yaml_path):
        print(f'Loi: Khong tim thay file {data_yaml_path}!')
        print('Vui long chay Buoc 2.5 tren Dashboard truoc.')
        return

    # ==========================================
    # 2. KHOI TAO MO HINH
    # ==========================================
    latest_best = find_latest_best_pt(base_dir)

    if latest_best:
        print('=' * 50)
        print('[Fine-tuning] Tim thay model cu:')
        print(f'  {latest_best}')
        print('AI se hoc them tren nen kien thuc cu, khong train lai tu dau.')
        print('=' * 50)
        model = YOLO(latest_best)
        freeze_layers = 0
    else:
        print('=' * 50)
        print('[Train moi] Khong tim thay model cu, bat dau tu yolov8n.pt')
        print('=' * 50)
        model = YOLO('yolov8n.pt')
        freeze_layers = 0

    # ==========================================
    # 3. TIEN HANH HUAN LUYEN
    # Dung duong dan tuyet doi cho project de tranh YOLO tu them
    # runs/detect/ vao truoc va gay lap thu muc
    # ==========================================
    print('BAT DAU HUAN LUYEN MO HINH...')

    # Thu muc luu ket qua: DLCK/runs/detect/results/traffic_model*
    project_dir = os.path.join(base_dir, 'runs', 'detect', 'results')
    os.makedirs(project_dir, exist_ok=True)

    model.train(
    data=data_yaml_path,
    epochs=30,          
    imgsz=640,
    batch=8,
    workers=1,
    project=project_dir,
    name="traffic_model",
    plots=True,
    device="0",
    freeze=freeze_layers,
    patience=20,         # Tang patience de khong dung som
    # lr0=0.005,           # Tang LR cao hon de hoc dataset lon
    # lrf=0.01,
    # warmup_epochs=5,     # Warmup lau hon voi dataset lon
)

    # ==========================================
    # 4. KET THUC
    # ==========================================
    new_best = find_latest_best_pt(base_dir)
    print('=' * 50)
    print('HUAN LUYEN HOAN TAT!')
    if new_best:
    print('Trong so tot nhat luu tai:')
    print(f'  {new_best}')
    print('=' * 50)
    print('Mo Dashboard va bat dau Test Video hoac Live Camera.')
    print('App se tu dong tim va dung best.pt moi nhat - khong can sua gi them.')

if __name__ == '__main__':
    main()