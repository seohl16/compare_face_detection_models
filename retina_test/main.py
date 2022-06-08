from time import time
import cv2
from PIL import Image
import torch
import torchvision

from util import Mosaic, DrawRectImg
from args import Args

import ml_part as ML
from detection import load_face_db
from facenet_pytorch import MTCNN, InceptionResnetV1

from retinaface_utils.utils.model_utils import load_model
from retinaface_utils.models.retinaface import RetinaFace
from retinaface_utils.data.config import cfg_mnet
from detection import mtcnn_get_embeddings, mtcnn_recognition, load_face_db
from retinaface_utils.util import retinaface_detection



def init(args):
    model_args = {}
    # 초기에 불러올 모델을 설정하는 공간입니다.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_args['Device'] = device
    if args['DEBUG_MODE']:
        print('Running on device : {}'.format(device))

    # Load mtcnn
    # 이걸로 image crop하는데 이 것도 나중에 자체 기능으로 빼야함. util.py의 CropRoiImg를 좀 쓰면 될 듯.
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds = [0.6, 0.7, 0.7], factor=0.709, post_process=True, 
        device=device, keep_all=True
    )
    model_args['Mtcnn'] = mtcnn
    
    # Load Detection Model
    if args['DETECTOR'] == 'mtcnn':
        # model_detection = MTCNN(keep_all=True)
        # 나중에 image crop 자체 기능 구현되면 위 코드를 여기로
        model_detection = mtcnn
    else:
        model_path = 'retinaface_utils/weights/mobilenet0.25_Final.pth'
        backbone_path = './retinaface_utils/weights/mobilenetV1X0.25_pretrain.tar'
        model_detection = RetinaFace(cfg=cfg_mnet, backbone_path=backbone_path, phase = 'test')
        model_detection = load_model(model_detection, model_path, device)
        model_detection.to(device)
        model_detection.eval()
    model_args['Detection'] = model_detection

    # Load Recognition Models
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    model_args['Recognition'] = resnet

    # Load Face DB
    face_db_path = "./database/face_db"
    if args['DETECTOR'] == 'retinaface':
        face_db_path += "_BGR"

    face_db = load_face_db("../data/test_images",
                            face_db_path,
                            "./database/img_db",
                            device, args, model_args)

    model_args['Face_db'] = face_db

    return model_args



def ProcessImage(img, args, model_args):
    process_target = args['PROCESS_TARGET']

    # Object Detection
    # bboxes = ML.Detection(img, args, model_args)
    device = model_args['Device']
    bboxes = retinaface_detection(model_args['Detection'], img, device, args['IMAGE_DIR'])
    if bboxes is None:
        return img

    # Object Recognition
    # face_ids = ML.Recognition(img, bboxes, args, model_args)
    

    faces, unknown_embeddings = mtcnn_get_embeddings(model_args['Mtcnn'],
                                                     model_args['Recognition'],
                                                     img, bboxes, device)
    if args['DEBUG_MODE']:
        print(faces.shape)
        print(bboxes.shape)

    face_ids, result_probs = mtcnn_recognition(model_args['Face_db'],
                                            unknown_embeddings,
                                            args['RECOG_THRESHOLD'])
    # Mosaic
    # img = Mosaic(img, bboxes, face_ids, n=10)

    # 특정인에 bbox와 name을 보여주고 싶으면
    processed_img = DrawRectImg(img, bboxes, face_ids)

    return processed_img


def main(args):
    model_args = init(args)

    # =================== Image =======================
    image_dir = args['IMAGE_DIR']
    if args['PROCESS_TARGET'] == 'Image':
        # Color channel: BGR
        img = cv2.imread(image_dir)
        start = time()
        img = ProcessImage(img, args, model_args)
        print('image done.', time() - start)
        cv2.imwrite(args['SAVE_DIR'] + '/output.jpg', img)
        print('image process complete!')
    # =================== Image =======================

    # =================== Video =======================
    elif args['PROCESS_TARGET'] == 'Video':
        video_path = args['VIDEO_DIR']

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(args['SAVE_DIR'] + '/output.mp4', fourcc, fps, (width, height))
        start = time()
        while True:
            ret, img = cap.read()
            # Color channel: BGR
            if ret:
                img = ProcessImage(img, args, model_args)
                out.write(img)
            else:
                break

        cap.release()
        out.release()
        print('video done.', time() - start)
    # ====================== Video ===========================


if __name__ == "__main__":
    args = Args().params
    main(args)
