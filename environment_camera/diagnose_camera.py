"""Find USB/environment camera source index on a Windows laptop."""
import os
import cv2

backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
print('Available camera indices (0-4):')
for index in range(5):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f'  {index}: unavailable')
    else:
        ok, frame = cap.read()
        print(f'  {index}: {"OK" if ok else "opened, no frames"}' +
              (f'  {frame.shape[1]}x{frame.shape[0]}' if ok else ''))
    cap.release()
print('Set CAMERA_SOURCE in .env to the correct index, then restart server.py.')
