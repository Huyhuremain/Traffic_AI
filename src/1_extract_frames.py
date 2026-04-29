#.\venv\Scripts\activate
import cv2
import os

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THÔNG SỐ
# ==========================================
# Đổi tên file này thành tên video bạn vừa bỏ vào thư mục
video_path = 'data/videos/traffic_01.mp4' 
output_dir = 'data/images/'

# Tùy chỉnh: Bao nhiêu giây cắt 1 bức ảnh? (Khuyên dùng: 1 hoặc 2)
SECONDS_PER_FRAME = 1  

# Tạo thư mục lưu ảnh nếu chưa có
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. XỬ LÝ TRÍCH XUẤT ẢNH
# ==========================================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Lỗi: Không thể mở video {video_path}. Vui lòng kiểm tra lại đường dẫn!")
    exit()

# Lấy thông số FPS (số khung hình/giây) của video gốc
fps = cap.get(cv2.CAP_PROP_FPS)
# Tính toán số khung hình cần nhảy qua để đạt đúng số giây mong muốn
frame_interval = int(fps * SECONDS_PER_FRAME)

print(f"Video FPS: {fps}")
print(f"Hệ thống sẽ cắt {SECONDS_PER_FRAME} giây / 1 ảnh (Tương đương cách nhau {frame_interval} frames).")
print("-" * 40)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break # Kết thúc video

    # Chỉ lưu ảnh khi số thứ tự khung hình chia hết cho khoảng thời gian quy định
    if frame_count % frame_interval == 0:
        # Tạo tên file có đánh số thứ tự (vd: frame_0001.jpg)
        filename = f"frame_{saved_count:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # LƯU Ý AN TOÀN: Dùng imwrite lưu thẳng xuống ổ cứng để giải phóng RAM
        cv2.imwrite(filepath, frame)
        saved_count += 1
        print(f"Đã trích xuất: {filename}")

    frame_count += 1

cap.release()
print("-" * 40)
print(f"✅ Hoàn tất! Đã lưu tổng cộng {saved_count} bức ảnh vào thư mục {output_dir}.")