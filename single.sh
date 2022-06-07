MNAME=opencv
MNAME_TEST=opencv_test

cd $MNAME_TEST


IDIR="../data/dest_images/stock.jpeg"
NUM=stock 
python main.py --image_dir $IDIR 
mv ./saved/output.jpg ../results/$NUM-$MNAME.jpg

IDIR="../data/dest_images/kakao/kakao1.jpeg"
NUM=kakao1 
python main.py --image_dir $IDIR 
mv ./saved/output.jpg ../results/$NUM-$MNAME.jpg

IDIR="../data/dest_images/kakao/kakao2.jpeg"
NUM=kakao2 
python main.py --image_dir $IDIR 
mv ./saved/output.jpg ../results/$NUM-$MNAME.jpg

IDIR="../data/dest_images/kakao/kakao3.jpeg"
NUM=kakao3 
python main.py --image_dir $IDIR 
mv ./saved/output.jpg ../results/$NUM-$MNAME.jpg

