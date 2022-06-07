for VARIABLE in mtcnn retina 
do 
	TESTT=_test
	VARTEST=$VARIABLE$TESTT
	echo $VARTEST
	cd $VARTEST
	IDIR="../data/dest_images/stock.jpeg"
	NUM=stock 
	python main.py --image_dir $IDIR 
	if [ ! -d '../results' ]; then
		mkdir '../results'
	fi
	mv ./saved/output.jpg ../results/$NUM-$VARIABLE.jpg

	IDIR="../data/dest_images/kakao/kakao1.jpeg"
	NUM=kakao1 
	python main.py --image_dir $IDIR 
	mv ./saved/output.jpg ../results/$NUM-$VARIABLE.jpg

	IDIR="../data/dest_images/kakao/kakao2.jpeg"
	NUM=kakao2 
	python main.py --image_dir $IDIR 
	mv ./saved/output.jpg ../results/$NUM-$VARIABLE.jpg

	IDIR="../data/dest_images/kakao/kakao3.jpeg"
	NUM=kakao3 
	python main.py --image_dir $IDIR 
	mv ./saved/output.jpg ../results/$NUM-$VARIABLE.jpg

	cd ../
done

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

# cd faceboxes_test 
# python main.py --image_dir $IDIR 
# mv ./saved/output.jpg ../results/$NUM-faceboxes.jpg

# cd ../