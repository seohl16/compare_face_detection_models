# cp util.py mtcnn_test/util.py
# cp util.py retina_test/util.py
# cp util.py yolo_test/util.py
IDIR="../data/dest_images/japanstst.jpeg" #FIXME
NUM=stst3 #FIXME

cd mtcnn_test
python main.py --process_target 'Image' --image_dir $IDIR --debug_mode True
mv ./saved/output.jpg ../results/$NUM-mtcnn.jpg

cd ../retina_test 
python main.py  --process_target 'Image' --image_dir $IDIR --debug_mode True 
mv ./saved/output.jpg ../results/$NUM-retina.jpg

cd ../yolo_test 
python main.py  --process_target 'Image' --image_dir $IDIR --debug_mode True
mv ./saved/output.jpg ../results/$NUM-yolo.jpg

cd ../opencv_test 
python main.py --image_dir $IDIR 
mv ./saved/output.jpg ../results/$NUM-opencv.jpg

cd ../faceboxes_test 
python main.py --image_dir $IDIR 
mv ./saved/output.jpg ../results/$NUM-faceboxes.jpg

cd ../