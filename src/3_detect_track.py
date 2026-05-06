import cv2
import time
import os
import csv
from ultralytics import YOLO

# ==========================================
# 1. XỬ LÝ LỖI VÀ CHUẨN BỊ MÔI TRƯỜNG CSV
# ==========================================
if os.path.exists('yolov8n.pt') and os.path.getsize('yolov8n.pt') < 1000:
    os.remove('yolov8n.pt')
    print("Đã dọn dẹp file mô hình lỗi!")

current_dir = os.getcwd()
results_dir = os.path.join(current_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

csv_file = os.path.join(results_dir, 'traffic_data.csv')

print("=" * 60)
print(f"👉 CHÚ Ý: File CSV của bạn sẽ được lưu chính xác tại:")
print(f"📁 {csv_file}")
print("=" * 60)

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Minute', 'Total_Vehicles', 'People', 'Bicycles', 'Motorcycles', 'Cars', 'Buses', 'Trucks'])
    print("Đã tạo file traffic_data.csv mới để lưu dữ liệu.")

# ==========================================
# 2. KHỞI TẠO AI VÀ CAMERA
# ==========================================
print("\nĐang tải mô hình YOLOv8...")
model = YOLO('yolov8n.pt') 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Lỗi: Không thể mở camera/video. Hãy kiểm tra lại!")
    exit()

prev_frame_time = 0
start_log_time = time.time()
minute_counter = 1
LOG_INTERVAL = 5 

print("\n🟢 HỆ THỐNG ĐANG CHẠY...")
print("⚠️ HƯỚNG DẪN TẮT: CLICK CHUỘT VÀO CỬA SỔ CAMERA, TẮT UNIKEY, VÀ NHẤN GIỮ PHÍM 'q' 1 GIÂY!\n")

# ==========================================
# 3. VÒNG LẶP NHẬN DIỆN CHÍNH
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = time.time()
    if (current_time - prev_frame_time) > 0:
        fps = int(1 / (current_time - prev_frame_time))
    else:
        fps = 0
    prev_frame_time = current_time

    # ✅ BỎ stream=True
    results = model(frame, classes=[0, 1, 2, 3, 5, 7])
    
    person_count = bicycle_count = car_count = motorcycle_count = bus_count = truck_count = 0
    
    for r in results:
        # ✅ XỬ LÝ BOXES TRƯỚC
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0: person_count += 1
            elif cls == 1: bicycle_count += 1
            elif cls == 2: car_count += 1
            elif cls == 3: motorcycle_count += 1
            elif cls == 5: bus_count += 1
            elif cls == 7: truck_count += 1
            
        # ✅ VẼ PLOT SAU
        frame = r.plot()

    total_vehicles = bicycle_count + car_count + motorcycle_count + bus_count + truck_count

    # ==========================================
    # 4. GHI DỮ LIỆU TỰ ĐỘNG RA FILE CSV
    # ==========================================
    if current_time - start_log_time >= LOG_INTERVAL:
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([minute_counter, total_vehicles, person_count, bicycle_count, motorcycle_count, car_count, bus_count, truck_count])
        
        print(f"[LOG] Đã lưu dữ liệu chu kỳ {minute_counter} -> Tổng xe: {total_vehicles} | Người: {person_count}")
        
        minute_counter += 1
        start_log_time = current_time

    # ==========================================
    # 5. VẼ GIAO DIỆN DASHBOARD
    # ==========================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (350, 330), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"TRAFFIC DASHBOARD (Log: {LOG_INTERVAL}s)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"- People: {person_count}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"- Bicycles: {bicycle_count}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"- Motorcycles: {motorcycle_count}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"- Cars: {car_count}", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"- Buses: {bus_count}", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"- Trucks: {truck_count}", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, f"FPS: {fps}", (250, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("He Thong Nhan Dien Giao Thong - Tich hop CSV", frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("\n🛑 Nhận lệnh tắt từ người dùng. Đang đóng hệ thống...")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Đã đóng camera an toàn!")