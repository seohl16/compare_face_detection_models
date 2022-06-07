# detection_comp_exp
detection models comparative experiment 


### MTCNN 
(bc) root@baa303cde5d2:~/input/detection_comp_exp/mtcnn_test# python main.py 
finished load face_data!
image done. *0.4900543689727783*
image process complete!


### RetinaFace 
(bc) root@baa303cde5d2:~/input/detection_comp_exp/retina_test# python main.py 
Loading pretrained model from retinaface_utils/weights/mobilenet0.25_Final.pth
remove prefix 'module.'
Missing keys:0
Unused checkpoint keys:0
Used keys:300
face_data_path not exist!,try to get face Database transform!
finished faceDatabase transform! 6
image done. *0.14011192321777344*
image process complete!

### Yolov5
(bc) root@baa303cde5d2:~/input/detection_comp_exp/yolo_test# python main.py 
Fusing layers... 
face_data_path not exist!,try to get face Database transform!
finished faceDatabase transform! 6
original video fps: 0.0
time: 0.0005924701690673828
done.

