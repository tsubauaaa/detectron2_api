import argparse
import base64

import cv2
import numpy as np
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0", help="detectron2 server host address")
parser.add_argument("--deviceid", default=0, help="detectron2 server host address")
parser.add_argument("--width", default=800, help="detectron2 server host address")
parser.add_argument("--height", default=600, help="detectron2 server host address")
args = parser.parse_args()

deviceId = args.deviceid  # Webカメラのデバイスインデックス
width = args.width
height = args.height
linewidth = 2
colors = [(0, 0, 175), (175, 0, 0), (0, 175, 0)]

cap = cv2.VideoCapture(deviceId)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("FPS:{}　WIDTH:{}　HEIGHT:{}".format(fps, width, height))


while True:

    # カメラ画像取得
    ret, frame = cap.read()
    if frame is None:
        continue

    _, jpg = cv2.imencode(".jpg", frame)
    encimg = base64.b64encode(jpg)
    encimg_str = encimg.decode("utf-8")
    payload = {"image": encimg_str}

    detections = requests.post(
        f"http://{args.host}:8001/predict",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    nparr = np.frombuffer(detections.content, np.uint8)

    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    cv2.imshow("frame", img_np)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
