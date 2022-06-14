from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime
import torch
from app.recognition import load_face_db, mtcnn_get_embeddings, mtcnn_recognition

from app.model import predict_from_image_byte

from PIL import Image 
import numpy as np 
import io 
from app.util import DrawRectImg

app = FastAPI()

orders = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds = [0.6, 0.7, 0.7], factor=0.709, post_process=True, 
        device=device, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
face_db = load_face_db("../data/test_images", "./face_db", device)


@app.get("/")
def hello_world():
    return {"hello": "world"}

class DetectionResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    result: str

class ResultList(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[DetectionResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    

#########################

@app.post("/order", description="주문을 요청합니다")
async def make_order(files: List[UploadFile] = File(...),
                     whichmodel: str = 'MTCNN'):
    products = []
    for file in files:
        image_bytes = await file.read()
        model.eval()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        bboxes, probs = model.detect(image_array)
        print(len(bboxes)) # detected faces number
        
        faces, unknown_embeddings = mtcnn_get_embeddings(model, resnet, image, bboxes, device)
        print(unknown_embeddings)
        face_ids, result_probs = mtcnn_recognition(image, face_db, unknown_embeddings, 0.8, device)
        processed_img = DrawRectImg(image, bboxes, face_ids, isPIL= True)
        product = DetectionResult(result="Success")
        products.append(product)
    all_result = ResultList(products=products)
    return all_result