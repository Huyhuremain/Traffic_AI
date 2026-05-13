# src/7_prepare_dataset.py
import os
import sys
import shutil
import random

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_dir   = os.path.join(BASE_DIR, 'data', 'images')
label_dir   = os.path.join(BASE_DIR, 'data', 'dataset', 'labels')
dataset_dir = os.path.join(BASE_DIR, 'data', 'dataset')

# Tạo thư mục train/val
for split in ['train', 'val']:
    os.makedirs(os.path.join(dataset_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', split), exist_ok=True)

# Chỉ lấy ảnh có file nhãn tương ứng
images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
paired = []
for img in images:
    label_name = img.rsplit('.', 1)[0] + '.txt'
    if os.path.exists(os.path.join(label_dir, label_name)):
        paired.append(img)
    else:
        print(f"  [SKIP] No label for: {img}")

print(f"[INFO] Paired: {len(paired)}/{len(images)} images")

if len(paired) == 0:
    print("[ERROR] No paired images found!")
    exit(1)

# Chia 80% train / 20% val
random.seed(42)
random.shuffle(paired)
split_idx  = int(len(paired) * 0.8)
train_list = paired[:split_idx]
val_list   = paired[split_idx:]

print(f"[INFO] Train: {len(train_list)} | Val: {len(val_list)}")
print("-" * 40)

# Copy vào đúng thư mục
for img_name in train_list:
    label_name = img_name.rsplit('.', 1)[0] + '.txt'
    shutil.copy(os.path.join(image_dir, img_name),
                os.path.join(dataset_dir, 'images', 'train', img_name))
    shutil.copy(os.path.join(label_dir, label_name),
                os.path.join(dataset_dir, 'labels', 'train', label_name))

for img_name in val_list:
    label_name = img_name.rsplit('.', 1)[0] + '.txt'
    shutil.copy(os.path.join(image_dir, img_name),
                os.path.join(dataset_dir, 'images', 'val', img_name))
    shutil.copy(os.path.join(label_dir, label_name),
                os.path.join(dataset_dir, 'labels', 'val', label_name))

print("[OK] Copied all files.")

# Tạo data.yaml — dùng đường dẫn tuyệt đối
yaml_content = f"""path: {dataset_dir}
train: images/train
val: images/val

nc: 8
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
"""

yaml_path = os.path.join(dataset_dir, 'data.yaml')
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"[OK] Created: {yaml_path}")
print("-" * 40)
print("[OK] Dataset ready! Now press 'Bat dau Huan luyen' on dashboard.")

#.\venv\Scripts\activate
#streamlit run src/5_app_dashboard.py
