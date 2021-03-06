import base64
from io import BytesIO

import cv2
import numpy as np
import torch
import uvicorn
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from starlette.responses import StreamingResponse

# from d_profile import profile
from visualizer_custom import VisImage, Visualizer

app = FastAPI()

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg.MODEL.DEVICE = device

predictor = DefaultPredictor(cfg)


class Data(BaseModel):
    image: bytes


# @profile
def predict(img):
    outputs: dict = predictor(img)
    return outputs


# @profile
def visualize(img, outputs):
    v = Visualizer(img, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out: VisImage = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    _, out_jpg = cv2.imencode(".jpg", out.get_image())

    return out_jpg


@app.post("/predict")
async def index(data: Data):
    # decode to image
    decimg = base64.b64decode(data.image, validate=True)
    decimg = Image.open(BytesIO(decimg))
    decimg = np.array(decimg, dtype=np.uint8)
    decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)

    outputs = predict(decimg)
    out_jpg = visualize(decimg, outputs)

    return StreamingResponse(BytesIO(out_jpg.tobytes()), media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
