import cv2
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
video_dir  = os.path.join(BASE_DIR, 'data', 'videos')
output_dir = os.path.join(BASE_DIR, 'data', 'images')

SECONDS_PER_FRAME = 0.5

os.makedirs(output_dir, exist_ok=True)

print(f'BASE_DIR   : {BASE_DIR}')
print(f'Video dir  : {video_dir}')
print(f'Output dir : {output_dir}')
print('-' * 40)

# Lay danh sach tat ca video trong thu muc
video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]

if len(video_files) == 0:
    print('[ERROR] Khong tim thay video nao trong data/videos/')
    exit(1)

print(f'[INFO] Tim thay {len(video_files)} video: {video_files}')
print('-' * 40)

# Lay danh sach anh da co de tranh cat trung
existing_imgs = set(os.listdir(output_dir))
total_saved = 0

for video_file in sorted(video_files):
    video_path = os.path.join(video_dir, video_file)
    # Lay ten goc cua video (bo duoi .mp4/.avi) de dat ten anh
    video_name = os.path.splitext(video_file)[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[ERROR] Khong mo duoc: {video_file}')
        continue

    fps            = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * SECONDS_PER_FRAME))
    duration_sec   = int(total_frames / fps) if fps > 0 else 0

    print(f'[VIDEO] {video_file}')
    print(f'        FPS: {fps:.1f} | Frames: {total_frames} | Duration: {duration_sec}s | Interval: {frame_interval}')

    frame_count  = 0
    saved_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Dat ten theo video_name de tranh trung giua cac video
            filename = f'{video_name}_f{saved_count:04d}.jpg'
            # Bo qua neu anh da ton tai
            if filename not in existing_imgs:
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame)
                existing_imgs.add(filename)
                print(f'  Saved: {filename}')
            saved_count += 1

        frame_count += 1

    cap.release()
    total_saved += saved_count
    print(f'  [OK] {video_file}: {saved_count} anh')
    print('-' * 40)

print(f'[DONE] Tong cong da trich xuat {total_saved} anh tu {len(video_files)} video')
print(f'       Luu tai: {output_dir}')