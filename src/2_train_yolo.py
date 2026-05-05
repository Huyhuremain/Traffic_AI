from ultralytics import YOLO
import os

def main():
    # ==========================================
    # 1. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU
    # ==========================================
    # File data.yaml này sẽ có được sau khi bạn tải dataset từ Roboflow về
    # Đảm bảo bạn giải nén dataset vào đúng thư mục 'data/dataset/'
    data_yaml_path = 'data/dataset/data.yaml'

    if not os.path.exists(data_yaml_path):
        print(f"❌ Lỗi: Không tìm thấy file {data_yaml_path}!")
        print("Vui lòng tải Dataset từ Roboflow về, giải nén và đặt file data.yaml đúng vị trí.")
        return

    # ==========================================
    # 2. KHỞI TẠO MÔ HÌNH
    # ==========================================
    # LƯU Ý: Lúc gán nhãn ta dùng bản X (rất nặng). 
    # Nhưng lúc Train để chạy web real-time, ta dùng bản N (Nano) cho nhẹ và mượt.
    print("⏳ Đang tải mô hình YOLOv8 Nano cơ bản...")
    model = YOLO('yolov8n.pt')

    # ==========================================
    # 3. TIẾN HÀNH HUẤN LUYỆN (TRAINING)
    # ==========================================
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH...")
    
    # Bắt đầu quá trình học
    results = model.train(
        data=data_yaml_path,
        epochs=50,              # Số vòng học (Đồ án sinh viên để 50 - 100 là đẹp)
        imgsz=640,              # Kích thước ảnh chuẩn của YOLO
        batch=8,                # Số ảnh đưa vào RAM 1 lúc (Nếu máy yếu/báo lỗi Out of Memory thì giảm xuống 4)
        project='results',      # Thư mục chính lưu kết quả
        name='traffic_model',   # Tên thư mục con lưu lần chạy này
        plots=True,             # Tự động vẽ các biểu đồ đánh giá (Loss, mAP) để cho vào báo cáo
        device=''               # Để trống: Hệ thống tự động chọn Card màn hình (GPU) nếu có, không thì chạy CPU
    )

    # ==========================================
    # 4. KẾT THÚC VÀ HƯỚNG DẪN
    # ==========================================
    print("\n" + "="*50)
    print("✅ HUẤN LUYỆN HOÀN TẤT!")
    print("👉 Trọng số AI tốt nhất của bạn đã được lưu tại:")
    print("   results/traffic_model/weights/best.pt")
    print("="*50)
    print("HƯỚNG DẪN BƯỚC TIẾP THEO:")
    print("1. Copy file 'best.pt' trong thư mục trên ra ngoài thư mục gốc của dự án (nằm chung với thư mục src).")
    print("2. Mở file '3_detect_track.py' và '5_app_dashboard.py', tìm dòng:")
    print("   model = YOLO('yolov8n.pt')")
    print("   -> Sửa thành: model = YOLO('best.pt')")

# Bắt buộc phải có cấu trúc này trên Windows khi chạy thư viện có multiprocessing như YOLO
if __name__ == '__main__':
    main()