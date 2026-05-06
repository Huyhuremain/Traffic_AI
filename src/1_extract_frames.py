import cv2
import os
import sys

# ✅ Fix lỗi Unicode tiếng Việt trên Windows
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 1. DUONG DAN TUYET DOI
# ==========================================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
video_path = os.path.join(BASE_DIR, 'data', 'videos', 'traffic_01.mp4')
output_dir = os.path.join(BASE_DIR, 'data', 'images')

SECONDS_PER_FRAME = 1

os.makedirs(output_dir, exist_ok=True)

print(f"BASE_DIR : {BASE_DIR}")
print(f"Video    : {video_path}")
print(f"Output   : {output_dir}")

# ==========================================
# 2. KIEM TRA VIDEO
# ==========================================
if not os.path.exists(video_path):
    print(f"[ERROR] Video not found: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {video_path}")
    exit(1)

# ==========================================
# 3. TRICH XUAT ANH
# ==========================================
fps            = cap.get(cv2.CAP_PROP_FPS)
total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = max(1, int(fps * SECONDS_PER_FRAME))

print(f"FPS: {fps:.1f} | Total frames: {total_frames} | Interval: {frame_interval} frames")
print("-" * 40)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = f"frame_{saved_count:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        saved_count += 1
        print(f"  Saved: {filename}")

    frame_count += 1

cap.release()
print("-" * 40)
print(f"[OK] Done! Saved {saved_count} images to: {output_dir}")