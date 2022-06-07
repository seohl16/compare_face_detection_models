cp util.py mtcnn_test/util.py
cp util.py retina_test/util.py
cp util.py yolo_test/util.py
IDIR="../data/dest_images/kakao/kakao2.jpeg" #FIXME
NUM=kakao2 #FIXME

cd mtcnn_test
python main.py --process_target 'Image' --image_dir $IDIR --debug_mode True

cd ../retina_test 
python main.py  --process_target 'Image' --image_dir $IDIR --debug_mode True 

cd ../yolo_test 
python main.py  --process_target 'Image' --image_dir $IDIR --debug_mode True

cd ../opencv_test 
python opencvtest.py --image_dir $IDIR 

cd ../
mv mtcnn_test/saved/output.jpg ./results/$NUM-mtcnn.jpg
mv retina_test/saved/output.jpg ./results/$NUM-retina.jpg
mv yolo_test/saved/output.jpg ./results/$NUM-yolo.jpg
mv opencv_test/output.jpg ./results/$NUM-opencv.jpg