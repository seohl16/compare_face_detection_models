from glob import glob 


# texts = glob('../yolo_test/saved/*YTN_programm*txt')
texts = glob('./*txt')
for txt in texts:
	f = open(txt, 'r')
	lines = f.readlines()
	fpss = []
	for line in lines:
		filename, fps, frame, time = line.split()
		fpss.append(float(fps))
	# print(fpss)
	print(txt.split('/')[-1], sum(fpss)/len(fpss))
	f.close()