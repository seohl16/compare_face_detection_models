IDIR="../data/dest_images/stock.jpeg" #FIXME
NUM=stock #FIXME

# cd mtcnn_test
# python main.py --process_target 'Image' --image_dir $IDIR --debug_mode True
# mv ./saved/output.jpg ../results/$NUM-mtcnn.jpg


# cd ../retina_test 
# python main.py  --process_target 'Image' --image_dir $IDIR --debug_mode True 
# mv ./saved/output.jpg ../results/$NUM-retina.jpg

# cd ../yolo_test 
# python main.py  --process_target 'Image' --image_dir $IDIR --debug_mode True
# mv ./saved/output.jpg ../results/$NUM-yolo.jpg

# cd ../opencv_test 
# python opencvtest.py --image_dir $IDIR 
# mv ./saved/output.jpg ../results/$NUM-opencv.jpg

cd faceboxes_test 
python main.py --image_dir $IDIR 
mv ./saved/output.jpg ../results/$NUM-faceboxes.jpg

cd ../