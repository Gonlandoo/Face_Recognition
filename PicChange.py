import os

import cv2
if __name__ == '__main__':

    pic = cv2.imread('./Yale/16/s2.bmp')
    pic = cv2.resize(pic, (100, 100), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('', pic)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    os.remove('./Yale/16/s17.bmp')
    cv2.imwrite('./Yale/16/s17.bmp', pic)
    os.remove('./Yale/16/s17.bmp')
