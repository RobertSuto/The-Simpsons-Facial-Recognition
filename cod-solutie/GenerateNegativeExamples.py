import os
import numpy as np
import cv2 as cv

root_path = "..\\antrenare\\"
names = ["bart", "homer", "lisa", "marge"]
image_names = []
bboxes = []
characters = []
nb_examples = 0


for name in names:
    filename_annotations = root_path + name + ".txt"
    f = open(filename_annotations)
    for line in f:
        a = line.split(os.sep)[-1]
        b = a.split(" ")

        image_name = root_path + name+ "\\" + b[0]
        bbox = [int(b[1]), int(b[2]), int(b[3]), int(b[4])]
        character = b[5][:-1]

        image_names.append(image_name)
        bboxes.append(bbox)
        characters.append(character)
        nb_examples = nb_examples + 1

width_hog = 36
height_hog = 36


# compute negative examples using 36 X 36 template

for idx, img_name in enumerate(image_names):
    if(idx != len(image_names) - 1 and img_name == image_names[idx + 1]):
        continue
    print(idx, img_name)
    img = cv.imread(img_name)
    print("img shape")
    print(img.shape)
    num_rows = img.shape[0]
    num_cols = img.shape[1]
    # genereaza 10 exemple negative fara sa compari cu nimic, iei ferestre la intamplare 36 x 36
    for i in range(22):
        x = np.random.randint(low=0, high=num_cols - width_hog)
        y = np.random.randint(low=0, high=num_rows - height_hog)

        aux_idx = idx
        mno = 0
        while image_names[aux_idx] == img_name:
            if((bboxes[aux_idx][0] < x < bboxes[aux_idx][2]) and (bboxes[aux_idx][1] < y < bboxes[aux_idx][3])) or ((bboxes[aux_idx][0] < x + width_hog  < bboxes[aux_idx][2]) and (bboxes[aux_idx][1] < y + height_hog  < bboxes[aux_idx][3])):
                mno = 1
                break
            aux_idx -= 1
        if mno == 1:
            continue
        bbox_current = [x, y, x + width_hog, y + height_hog]

        img5 = img[y:y+height_hog,x:x+width_hog]
        # cv.imshow('',img5)
        # cv.waitKey(0)
        low_yellow = (20, 90, 190)
        high_yellow = (62, 255, 255)
        img_hsv = cv.cvtColor(img5, cv.COLOR_BGR2HSV)
        mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
        if mask_yellow_hsv.mean() < 35:
            continue

        x_min = bbox_current[0]
        y_min = bbox_current[1]
        x_max = bbox_current[2]
        y_max = bbox_current[3]
        negative_image = img[y_min: y_max, x_min: x_max]
        filename = "..\\data\\exempleNegative\\" + str(idx) + "_" + str(i) + ".jpg"
        cv.imwrite(filename, negative_image)



