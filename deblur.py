import cv2 as cv
import numpy as np
from utils import custom_blur_demo
import glob
from tqdm import tqdm


img_list = list(sorted(glob.glob('yourname_shinkai' + '/*.*')))
num=0
for i in tqdm(img_list):
    num+=1
    src = cv.imread(i)
    dst=custom_blur_demo(src)
    cv.imwrite('yourname_shinkai/s'+str(num)+'.png',dst)