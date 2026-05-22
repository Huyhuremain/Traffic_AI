import sys
import os

# ✅ Fix lỗi Unicode tiếng Việt trên Windows
sys.stdout.reconfigure(encoding='utf-8')

from ultralytics import YOLO

# ==========================================
# 1. DUONG DAN TUYET DOI
# ==========================================
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_dir = os.path.join(BASE_DIR, 'data', 'images')
label_dir = os.path.join(BASE_DIR, 'data', 'dataset', 'labels')

os.makedirs(label_dir, exist_ok=True)

print(f"BASE_DIR  : {BASE_DIR}")
print(f"Image dir : {image_dir}")
print(f"Label dir : {label_dir}")

# ==========================================
# 2. KIEM TRA THU MUC ANH
# ==========================================
if not os.path.exists(image_dir):
    print(f"[ERROR] Image directory not found: {image_dir}")
    exit(1)

images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

if len(images) == 0:
    print("[ERROR] No images found! Run Step 1 first.")
    exit(1)

print(f"[INFO] Found {len(images)} images. Starting auto-label 2.0...")
print("-" * 40)

# ==========================================
# 3. LOAD MODEL VA GAN NHAN (NÂNG CẤP)
# ==========================================
# Sử dụng yolov8x.pt (bản Extra Large) để tăng tối đa độ chính xác gán nhãn
print("[INFO] Đang tải mô hình YOLOv8x.pt (Extra Large) - Có thể mất chút thời gian...")
model = YOLO('yolov8x.pt')
success_count = 0
skip_count    = 0
force_truck_count = 0 # Đếm số lượng nhãn được ép thành Truck

for img_name in images:
    img_path   = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + '.txt')

    # Tăng ngưỡng conf=0.45: Chỉ lấy những kết quả mà AI chắc chắn >= 45%
    results = model(img_path, classes=[0, 1, 2, 3, 5, 7], conf=0.45, verbose=False)

    r = results[0]

    # Bỏ qua ảnh không có đối tượng nào
    if len(r.boxes) == 0:
        skip_count += 1
        print(f"  [SKIP] {img_name} - Không tìm thấy đối tượng rõ ràng")
        continue

    # Lưu file txt gốc từ YOLO
    r.save_txt(label_path)


    # ---------------------------------------------------------
    # KỸ THUẬT ÉP NHÃN (RULE-BASED FORCING)
    # ---------------------------------------------------------
    is_forced = False
    if "truck" in img_name.lower():
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    cls_id = int(parts[0])
                    # Nếu tên ảnh có chữ 'truck' mà AI lại đoán là Car(2) hoặc Bus(5)
                    # -> Cưỡng chế đổi thành Truck(7)
                    if cls_id in [2, 5]:
                        parts[0] = '7'
                        is_forced = True
                        force_truck_count += 1
                new_lines.append(" ".join(parts) + "\n")

            # Ghi đè lại file txt bằng nhãn đã sửa
            if is_forced:
                with open(label_path, 'w') as f:
                    f.writelines(new_lines)
    # ---------------------------------------------------------

    success_count += 1
    if is_forced:
        print(f"  [OK-FORCED] {img_name} - Đã ép nhãn thành Truck!")
    else:
        print(f"  [OK] {img_name} - {len(r.boxes)} objects")

# ==========================================
# 4. TONG KET
# ==========================================
print("-" * 40)
print(f"[DONE] Hoàn tất Auto-Label 2.0!")
print(f"     Labeled : {success_count} images")
print(f"     Skipped : {skip_count} images (no objects)")
print(f"     Forced  : Đã can thiệp ép {force_truck_count} nhãn thành Truck")
print(f"     Saved to: {label_dir}")