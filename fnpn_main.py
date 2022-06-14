import cv2
import torch

from time import time
from util import Mosaic, DrawRectImg
from args import Args

import ml_part as ML
from database import load_face_db
from facenet_pytorch import InceptionResnetV1, MTCNN

from deep_sort.deep_sort_face import DeepSortFace

from retinaface_utils.utils.model_utils import load_model
from retinaface_utils.models.retinaface import RetinaFace
from retinaface_utils.data.config import cfg_mnet

from detect_face import load_models
import os
from glob import glob


def init(args):
    model_args = {}
    # 초기에 불러올 모델을 설정하는 공간입니다.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_args['Device'] = device

    if args['DEBUG_MODE']:
        print('Running on device : {}'.format(device))

    # Load Detection Model
    if args['DETECTOR'] == 'retinaface':
        model_path = 'retinaface_utils/weights/mobilenet0.25_Final.pth'
        backbone_path = './retinaface_utils/weights/mobilenetV1X0.25_pretrain.tar'
        model_detection = RetinaFace(cfg=cfg_mnet, backbone_path=backbone_path, phase = 'test')
        model_detection = load_model(model_detection, model_path, device)
        model_detection.to(device)
        model_detection.eval()
    elif args['DETECTOR'] == 'mtcnn':
        model_detection = MTCNN(keep_all=True, device=device).eval()
    else: # yolo
        model_detection = load_models("./weights/yolov5n-face.pt", device)
        model_detection.to(device)
        print('yolo model on device...')
        model_detection.eval()

    model_args['Detection'] = model_detection
    print(args['DETECTOR'],  'loaded')
    # Load Recognition Models
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    model_args['Recognition'] = resnet

    model_args['Deepsort'] = DeepSortFace(device=device)

    # Load Face DB
    db_path = "./database"
    faces_path = "../data/test_images"

    face_db = load_face_db(faces_path,
                            db_path,
                            device, args, model_args)

    model_args['Face_db'] = face_db

    return model_args

def ProcessImage(img, args, model_args):
    process_target = args['PROCESS_TARGET']

    # Object Detection
    bboxes, probs = ML.Detection(img, args, model_args)
    if bboxes is None: return img, 0

    # Object Recognition
    # face_ids, probs = ML.Recognition(img, bboxes, args, model_args)
    # if args['DEBUG_MODE']:
    #     print(len(bboxes))

    # Mosaic
    face_ids = ['kno'] * len(bboxes)
    img = Mosaic(img, bboxes, face_ids, n=10)

    # 특정인에 bbox와 name을 보여주고 싶으면
    # img = DrawRectImg(img, bboxes, face_ids)

    return img, len(bboxes)


def ProcessVideo(img, args, model_args, id_name):
    # global id_name
    # Object Detection
    bboxes, probs = ML.Detection(img, args, model_args)
    
    # ML.DeepsortRecognition
    
    outputs = ML.Deepsort(img, bboxes, probs, model_args['Deepsort'])
    # last_out = outputs 

    # if boxes is None:
    #     return img, outputs
    known, unknown = 0,0
    if len(outputs) > 0:
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        if identities[-1] not in id_name.keys(): # Update가 생기면
            id_name, probs = ML.Recognition(img, bbox_xyxy, args, model_args, id_name, identities)                                       

        processed_img, known, unknown  = Mosaic(img, bbox_xyxy, identities, 10, id_name)
    
        # 특정인에 bbox와 name을 보여주고 싶으면
        processed_img = DrawRectImg(processed_img, bbox_xyxy, identities, id_name)
    else:
        processed_img = img
    
    return processed_img, id_name, known, unknown


def main(args):
    model_args = init(args)

    # =================== Image =======================
    image_dir = args['IMAGE_DIR']
    if not os.path.exists(args['SAVE_DIR']):
        os.mkdir(args['SAVE_DIR'])
    if args['PROCESS_TARGET'] == 'Image':
        # Color channel: BGR
        img = cv2.imread(image_dir)
        start = time()
        img = ProcessImage(img, args, model_args)
        print('image done', time() - start)
        cv2.imwrite(args['SAVE_DIR'] + '/output.jpg', img)
        print('image process complete!')
    # =================== Image =======================

    # =================== Video =======================
    elif args['PROCESS_TARGET'] == 'Video' and False:
        video_path = '../data/99_YTN_program-0000_done/video-0030/clip-0000.mp4'
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(cap.get(3))
        height = int(cap.get(4))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(args['SAVE_DIR'] + '/output.mp4', fourcc, fps, (width, height))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(frame_count)
        id_name = {}
        start = time()
        frame_count = 0
        while True:
            ret, img = cap.read()
            # Color channel: BGR
            if ret:
                frame_count += 1
                if args['TRACKING']:
                    img, id_name = ProcessVideo(img, args, model_args, id_name)
                else:
                    img = ProcessImage(img, args, model_args)
                # out.write(img)
            else:
                break

        cap.release()
        out.release()
        totaltime = time()-start
        print(f'time: {totaltime}')
        fps = frame_count / totaltime
        print('{:s} {:f} {:f} {:f}\n'.format(video_path, fps, frame_count, totaltime))
        print('done.')
    elif args['PROCESS_TARGET'] == 'Video':
        # video_path = '../data/program-0000/'
        # videos = glob(video_path + '*/*.mp4')
        videosf = open('./mgvl.txt', 'r')
        videos = videosf.readlines()
        print(videos)
        if not os.path.exists('./saved'):
            os.makedirs('./saved')
        fw = open(os.path.join('./saved/', 'vdgen_OBS_programm-0000' + '.txt'), 'a')
        # manybboxes = open(os.path.join('./saved/', 'manybboxes' + '_YTN_programm-0000' + '.txt'), 'w')
        # model_args['manb'] = manybboxes
        cnt = 0
        for single_video_path in videos : 
            print(single_video_path)
            # cnt += 1
            # if cnt < 50: 
            #     continue 
            # if cnt == 100: 
            #     break
            cap = cv2.VideoCapture(single_video_path[:-1])
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            width = int(cap.get(3))
            height = int(cap.get(4))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            fps = cap.get(cv2.CAP_PROP_FPS)
            img_name = (single_video_path).split('/')
            img_name = (img_name[-2] + img_name[-1]).split('.')[0]
            # img_name = img_name[-1]
            out = cv2.VideoWriter(args['SAVE_DIR'] + '/' + img_name + '.mp4', fourcc, fps, (width, height))
            # out = cv2.VideoWriter(args['SAVE_DIR'] + '/output.mp4', fourcc, fps, (width, height))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(frame_count)
            id_name = {}
            start = time()
            frame_count = 0
            global stopp
            stopp = 0
            knowntotal = 0
            unknowntotal = 0
            while True:
                ret, img = cap.read()
                # Color channel: BGR
                if ret:
                    frame_count += 1
                    if args['TRACKING']:
                        img, id_name, known, unknown = ProcessVideo(img, args, model_args, id_name)
                        knowntotal += known
                        unknowntotal += unknown
                        font_scale = 3 # 위와 동일
                        font_color = (0, 0, 255) # BGR
                        font_thickness = 2 # 위와 동일

                        cv2.putText(img, '{:d} = known: {:d}, unknown: {:d}'.format(frame_count, knowntotal, unknowntotal), \
                            (100, 100), 1, font_scale, font_color, font_thickness)
                    else:
                        img, lenn = ProcessImage(img, args, model_args)
                        # if lenn > 7: 
                        #     stopp = 1
                    out.write(img)
                else:
                    break
            print(knowntotal, unknowntotal)
            cap.release()
            out.release()
            totaltime = time()-start
            print(f'time: {totaltime}')
            # if stopp == 1: 
            #     print('[[[[', img_name, 'has many bboxes]]]]]')
            #     manybboxes.write('{:s} {:f}\n'.format(img_name, lenn))
            fps = frame_count / totaltime
            fw.write('{:s} {:f} {:f} {:f}\n'.format(img_name, fps, frame_count, totaltime))
            print('{:s} {:f} {:f} {:f}\n'.format(img_name, fps, frame_count, totaltime))

            print('done.')
    # ====================== Video ===========================

    else: # WebCam
        return
        webcam = cv2.VideoCapture(0)
        print('width :%d, height : %d' % (cap.get(3), cap.get(4)))
        
        if not webcam.isOpened():
            print("Could not open webcam")
            exit()

        while webcam.isOpened():
            # read frame from webcam 
            status, frame = webcam.read()

            if not status:
                break

            frame = ProcessImage(frame, args, model_args)
            # display output
            cv2.imshow("Real-time object detection", frame)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # release resources
        webcam.release()
        cv2.destroyAllWindows()   

if __name__ == "__main__":
    args = Args().params
    main(args)
