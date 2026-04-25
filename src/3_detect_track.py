import cv2
import time
import os
import csv
from ultralytics import YOLO

# ==========================================
# 1. XỬ LÝ LỖI VÀ CHUẨN BỊ MÔI TRƯỜNG CSV
# ==========================================
# Tự động dọn dẹp file mô hình hỏng (nếu có) để tránh lỗi EOFError
if os.path.exists('yolov8n.pt') and os.path.getsize('yolov8n.pt') < 1000:
    os.remove('yolov8n.pt')
    print("Đã dọn dẹp file mô hình lỗi!")

# Tạo thư mục results nếu chưa có
os.makedirs('results', exist_ok=True)
csv_file = 'results/traffic_data.csv'

# Tạo file CSV và viết tiêu đề (nếu file chưa tồn tại)
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Minute', 'Total_Vehicles', 'People', 'Bicycles', 'Motorcycles', 'Cars', 'Buses', 'Trucks'])
    print("Đã tạo file traffic_data.csv mới để lưu dữ liệu.")

# ==========================================
# 2. KHỞI TẠO AI VÀ CAMERA
# ==========================================
print("Đang tải mô hình YOLOv8...")
model = YOLO('yolov8n.pt') 

# Mở Webcam (Để dùng video thật sau này, thay số 0 bằng: 'data/videos/test.mp4')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Lỗi: Không thể mở camera/video. Hãy kiểm tra lại!")
    exit()

# Các biến phục vụ tính toán thời gian và ghi log
prev_frame_time = 0
start_log_time = time.time()
minute_counter = 1
LOG_INTERVAL = 5 # Thời gian mỗi chu kỳ lưu dữ liệu (giây)

# ==========================================
# 3. VÒNG LẶP NHẬN DIỆN CHÍNH
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Tính FPS (Có thêm điều kiện tránh lỗi chia cho 0 ở frame đầu tiên)
    current_time = time.time()
    if (current_time - prev_frame_time) > 0:
        fps = int(1 / (current_time - prev_frame_time))
    else:
        fps = 0
    prev_frame_time = current_time

    # Nhận diện & Lọc class: 0(Người), 1(Xe đạp), 2(Ô tô), 3(Xe máy), 5(Xe buýt), 7(Xe tải)
    results = model(frame, classes=[0, 1, 2, 3, 5, 7], stream=True)
    
    # Khởi tạo lại bộ đếm cho mỗi khung hình
    person_count = 0
    bicycle_count = 0
    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0
    
    for r in results:
        frame = r.plot() # Vẽ Bounding Box lên hình
        
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0: person_count += 1
            elif cls == 1: bicycle_count += 1
            elif cls == 2: car_count += 1
            elif cls == 3: motorcycle_count += 1
            elif cls == 5: bus_count += 1
            elif cls == 7: truck_count += 1

    # Tính tổng lượng phương tiện (Không cộng số người lùi bộ vào tổng xe)
    total_vehicles = bicycle_count + car_count + motorcycle_count + bus_count + truck_count

    # ==========================================
    # 4. GHI DỮ LIỆU TỰ ĐỘNG RA FILE CSV
    # ==========================================
    if current_time - start_log_time >= LOG_INTERVAL:
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([minute_counter, total_vehicles, person_count, bicycle_count, motorcycle_count, car_count, bus_count, truck_count])
        
        print(f"[LOG] Đã lưu dữ liệu chu kỳ {minute_counter} -> Tổng xe: {total_vehicles} | Người: {person_count}")
        
        minute_counter += 1
        start_log_time = current_time # Đặt lại đồng hồ để tính chu kỳ tiếp theo

    # ==========================================
    # 5. VẼ GIAO DIỆN DASHBOARD
    # ==========================================
    overlay = frame.copy()
    
    # Kéo dài khung chữ nhật xuống y=330 để chứa đủ các dòng
    cv2.rectangle(overlay, (20, 20), (350, 330), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Ghi Text
    cv2.putText(frame, f"TRAFFIC DASHBOARD (Log: {LOG_INTERVAL}s)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"- People (Nguoi): {person_count}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"- Bicycles (Xe dap): {bicycle_count}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"- Motorcycles (Xe may): {motorcycle_count}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"- Cars (O to): {car_count}", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"- Buses (Xe buyt): {bus_count}", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"- Trucks (Xe tai): {truck_count}", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Hiển thị FPS
    cv2.putText(frame, f"FPS: {fps}", (250, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 6. Hiển thị lên màn hình
    cv2.imshow("He Thong Nhan Dien Giao Thong - Tich hop CSV", frame)
    
    # Nhấn 'q' để tắt an toàn
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp bộ nhớ khi kết thúc
cap.release()
cv2.destroyAllWindows()