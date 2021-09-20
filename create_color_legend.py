import cv2
import numpy as np
from cityscapes_utiles import LABEL_LEGEND

TOTAL_SLOTS = len(LABEL_LEGEND)
SLOT_SHAPE = (40, 350)
TEXT_ORIGIN = (10, 30)

blank_slot = np.full(shape=(*SLOT_SHAPE, 3), fill_value=(255, 255, 255), dtype=np.uint8)
column1 = np.vstack((blank_slot, blank_slot))
column2 = blank_slot

for class_index in LABEL_LEGEND:
    label = LABEL_LEGEND[class_index]['name']
    rgb_value = np.array(LABEL_LEGEND[class_index]['color'], dtype=np.uint8)
    color_box = np.full(shape=(*SLOT_SHAPE, 3), fill_value=rgb_value, dtype=np.uint8)
    color_box = cv2.putText(color_box, text=label, org=TEXT_ORIGIN,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    if 0 <= class_index < TOTAL_SLOTS//2:
        column1 = np.vstack((column1, color_box))
    else:
        column2 = np.vstack((column2, color_box))

res_image = np.hstack((column1, column2))
res_image = res_image[40:, :, :]
res_image = cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('color_legend.png', res_image)
# cv2.imshow('img', res_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
