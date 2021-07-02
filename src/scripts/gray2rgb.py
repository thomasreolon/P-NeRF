import cv2
import os
import os.path as osp

ROOT_PATH = osp.dirname(os.path.abspath(__file__))

# requires opencv-contrib-python

imgs = os.listdir(ROOT_PATH+'/../../input/dataset/mymodel/rgb/')
for f in imgs:
    source_path = ROOT_PATH+'/../../input/dataset/mymodel/rgb/'+f
    gray = cv2.imread(source_path, 0)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(source_path, rgb)

