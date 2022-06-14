import cv2  
import sys 
import argparse
import os
class Args(object):
    parser = argparse.ArgumentParser(description='Arguments for Unknown mosaic')
    parser.add_argument('--image_dir', default="../data/dest_images/findobama/twopeople.jpeg", help='Directory to image')
    parser.add_argument('--save_name', default="output.jpg", help='Directory to image')
    parse = parser.parse_args()
    

def main(args):
    # Load the cascade  
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
    
    # Read the input image  
    img = cv2.imread(args.image_dir)  
    
    # Convert into grayscale  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    
    # Detect faces  
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  
    
    # Draw rectangle around the faces  
    for (x, y, w, h) in faces:  
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
    
    # Display the output  
    # cv2.imshow('image', img)  
    # cv2.waitKey() 

    if not os.path.exists('./saved'):
        os.makedirs('./saved')
    cv2.imwrite('./saved/output.jpg', img)
    # cv2.imwrite(args.save_name, img)

if __name__ == "__main__":
    args = Args().parse
    print(args)
    main(args)