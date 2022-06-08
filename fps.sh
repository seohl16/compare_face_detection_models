cd yolo_test 
# python main.py --detector yolo 
# python main.py --detector mtcnn 
python main.py --detector retinaface 
cd ../fps
python calculate_avg.py 