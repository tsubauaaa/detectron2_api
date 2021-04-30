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
from detectron2.utils.visualizer import VisImage, Visualizer
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from starlette.responses import StreamingResponse

app = FastAPI()

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg.MODEL.DEVICE = device


class Data(BaseModel):
    image: bytes


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/predict")
async def predict(data: Data):
    # decode to image
    # decimg = base64.b64decode(img_str.split(",")[1], validate=True)
    decimg = base64.b64decode(data.image, validate=True)
    decimg = Image.open(BytesIO(decimg))
    decimg = np.array(decimg, dtype=np.uint8)
    decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
    resize_decimg = cv2.resize(decimg, (300, 224))
    predictor = DefaultPredictor(cfg)
    outputs: dict = predictor(resize_decimg)

    v = Visualizer(resize_decimg, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out: VisImage = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite("out.jpg", out.get_image())

    # content = jsonable_encoder({"out": "out"})
    # return JSONResponse(content=content)
    _, out_jpg = cv2.imencode(".jpg", out.get_image())
    return StreamingResponse(BytesIO(out_jpg.tobytes()), media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
