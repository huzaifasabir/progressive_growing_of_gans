import cv2
import os, sys
dir = os.getcwd()
import numpy as np
import cv2

# Load an color image in grayscale
#img = cv2.imread(dir+'/sampleTestPreperation/0002558327.jpg',0)
#print(img)
for item in os.listdir(dir+'/TestPreparation/'):
    print(item)
    print(len(item))
    try:
        im = cv2.imread(dir+'/TestPreparation/' + item)
        print(item)
        f, e = os.path.splitext(item)
        #cv2.imshow("Image", im)
        # exit at closing of window
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        imResize = cv2.resize(im, (256,256))
        cv2.imwrite(dir + '/resized/' + f + '.jpg', imResize)
    except cv2.error:
        print("s")
        continue
    except sys.stderr:
        continue
    



