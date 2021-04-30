import base64

import cv2
import numpy as np
import requests

profile = "developer"
endPoint = "sampleEndPoint"
categories = ["Bisco", "BlackThunder", "Alfort"]

deviceId = 0  # Webカメラのデバイスインデックス
height = 600
width = 800
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
        # "http://52.192.43.214:8001/predict",
        "http://0.0.0.0:8001/predict",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    nparr = np.frombuffer(detections.content, np.uint8)

    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # for detection in detections["prediction"]:
    #     clsId = int(detection[0])
    #     confidence = detection[1]
    #     x1 = int(detection[2] * width)
    #     y1 = int(detection[3] * height)
    #     x2 = int(detection[4] * width)
    #     y2 = int(detection[5] * height)
    #     label = "{} {:.2f}".format(categories[clsId], confidence)
    #     if confidence > 0.6:  # 信頼度
    #         frame = cv2.rectangle(frame, (x1, y1), (x2, y2), colors[clsId], linewidth)
    #         frame = cv2.rectangle(
    #             frame, (x1, y1), (x1 + 150, y1 - 20), colors[clsId], -1
    #         )
    #         cv2.putText(
    #             frame,
    #             label,
    #             (x1 + 2, y1 - 2),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             (255, 255, 255),
    #             1,
    #             cv2.LINE_AA,
    #         )

    cv2.imshow("frame", img_np)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
