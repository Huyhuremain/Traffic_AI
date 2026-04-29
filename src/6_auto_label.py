from ultralytics import YOLO
import os

# 1. Load model "Khủng" để gán nhãn chuẩn nhất
model = YOLO('yolov8x.pt') 

image_dir = 'data/images/'
label_dir = 'data/dataset/labels/' # Nơi lưu file .txt sau khi gán nhãn
os.makedirs(label_dir, exist_ok=True)

# 2. Quét toàn bộ ảnh
images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

print(f"Đang tự động gán nhãn cho {len(images)} ảnh...")

for img_name in images:
    img_path = os.path.join(image_dir, img_name)
    
    # Chạy AI nhận diện (chỉ lấy các lớp: người, xe đạp, ô tô, xe máy, xe bus, xe tải)
    results = model(img_path, classes=[0, 1, 2, 3, 5, 7])
    
    # Xuất kết quả ra định dạng YOLO (.txt)
    for r in results:
        # Tên file nhãn trùng tên ảnh nhưng đuôi .txt
        label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + '.txt')
        r.save_txt(label_path)

print(f"✅ Đã xong! Check ngay thư mục: {label_dir}")