import os
import cv2 as cv

root_path = "..\\antrenare\\"

names = ["bart", "homer", "lisa", "marge"]

image_names = []
folders = []
bboxes = []
characters = []
nb_examples = 0

for name in names:
    filename_annotations = root_path + name + ".txt"
    f = open(filename_annotations)
    for line in f:
        a = line.split(os.sep)[-1]
        b = a.split(" ")

        image_name = root_path + name + "\\" + b[0]
        bbox = [int(b[1]), int(b[2]), int(b[3]), int(b[4])]
        character = b[5][:-1]

        image_names.append(image_name)
        folders.append(b[5].replace('\n', ''))
        bboxes.append(bbox)
        characters.append(character)
        nb_examples = nb_examples + 1

width_hog = 36
height_hog = 36

# compute positive examples using 36 X 36 template
number_roots = 1

folder_bart = 0
folder_homer = 0
folder_lisa = 0
folder_marge = 0
folder_unknown = 0

for idx, img_name in enumerate(image_names):
    print(idx, img_name)
    img = cv.imread(img_name)
    bbox = bboxes[idx]
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
    print(xmin, ymin, xmax, ymax)
    face = img[ymin:ymax, xmin:xmax]
    print("original face shape:", face.shape)
    face_warped = cv.resize(face, (height_hog, width_hog))
    print("warped face shape:", face_warped.shape)
    if folders[idx] == 'bart':
        img_id = folder_bart
        folder_bart += 1
    elif folders[idx] == 'homer':
        img_id = folder_homer
        folder_homer += 1
    elif folders[idx] == 'lisa':
        img_id = folder_lisa
        folder_lisa += 1
    elif folders[idx] == 'marge':
        img_id = folder_marge
        folder_marge += 1
    elif folders[idx] == 'unknown':
        img_id = folder_unknown
        folder_unknown += 1

    filename = "..\\task2_antrenare\\" + str(folders[idx]) + "\\" + str(img_id) + ".jpg"
    cv.imwrite(filename, face_warped)
