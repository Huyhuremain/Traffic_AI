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
        epochs=50,
        imgsz=640,
        batch=4,                # ✅ GIẢM TỪ 8 XUỐNG 4 (Để tiết kiệm RAM)
        workers=1,              # ✅ THÊM DÒNG NÀY: Chặn YOLO tạo ra hàng chục luồng ngốn RAM
        project='results',
        name='traffic_model',
        plots=True,
        device=''
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