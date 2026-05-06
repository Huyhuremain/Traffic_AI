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

print(f"[INFO] Found {len(images)} images. Starting auto-label...")
print("-" * 40)

# ==========================================
# 3. LOAD MODEL VA GAN NHAN
# ==========================================
# Dùng model lớn nhất để gán nhãn chính xác nhất
model = YOLO('yolov8m.pt')

success_count = 0
skip_count    = 0

for img_name in images:
    img_path   = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + '.txt')

    results = model(img_path, classes=[0, 1, 2, 3, 5, 7])

    # ✅ Chỉ lấy kết quả đầu tiên, tránh ghi đè
    r = results[0]

    # ✅ Bỏ qua ảnh không có đối tượng nào
    if len(r.boxes) == 0:
        skip_count += 1
        print(f"  [SKIP] {img_name} - no objects detected")
        continue

    r.save_txt(label_path)
    success_count += 1
    print(f"  [OK] {img_name} - {len(r.boxes)} objects")

# ==========================================
# 4. TONG KET
# ==========================================
print("-" * 40)
print(f"[OK] Done!")
print(f"     Labeled : {success_count} images")
print(f"     Skipped : {skip_count} images (no objects)")
print(f"     Saved to: {label_dir}")