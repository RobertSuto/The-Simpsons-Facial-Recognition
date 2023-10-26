from Task2Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pickle
import ntpath
from copy import deepcopy
from skimage.feature import hog
from sklearn.preprocessing import Normalizer


class FacialDetector:
    def __init__(self, params: Task2Parameters):
        self.params = params
        self.best_model = None
        self.best_faces_model = None
        self.sc = Normalizer()

    def get_positive_descriptors(self):
        print("AM GENERAT MAI INTAI EXEMPLELE POZITIVE")

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []

        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):

            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)

            positive_descriptors.append(features)

            if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                               cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_negative_descriptors(self):
        print("AM GENERAT MAI INTAI EXEMPLELE NEGATIVE")

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []

        print('Calculam descriptorii pt %d imagini negative...' % num_images)
        for i in range(num_images):

            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)

            negative_descriptors.append(features)
            if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                               cells_per_block=(2, 2), feature_vector=True)
                negative_descriptors.append(features)

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def get_descriptors(self, example):
        print("AM GENERAT MAI INTAI EXEMPLELE POZITIVE CONTINAND FETE")

        if example == 'bart':
            dir_examples = self.params.dir_bart_examples
        elif example == 'homer':
            dir_examples = self.params.dir_homer_examples
        elif example == 'lisa':
            dir_examples = self.params.dir_lisa_examples
        else:
            # pentru marge
            dir_examples = self.params.dir_marge_examples

        images_path = os.path.join(dir_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        descriptors = []

        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul %s numarul %d...' % (example, i))
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (36, 36))
            img = np.array(img.flatten())
            descriptors.append(img)

        positive_descriptors = np.array(descriptors)
        return positive_descriptors

    def train_classifier(self, training_examples, train_labels, ignore_restore=True):
        """
        Antreneaza un clasificator pe datele din task-ul 1.
        """

        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))

        if os.path.exists(svm_file_name) and ignore_restore:
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]

        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c, max_iter=2000)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = deepcopy(model)
        pickle.dump(best_model, open(svm_file_name, 'wb'))
        self.best_model = best_model

    def train_faces_classifier(self, training_examples, train_labels, ignore_restore=True):
        print('Training a classifier for task 2')
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_faces_model_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))

        if os.path.exists(svm_file_name) and ignore_restore:
            self.best_faces_model = pickle.load(open(svm_file_name, 'rb'))
            return

        # scaling images
        self.sc.fit(training_examples)
        training_examples = self.sc.transform(training_examples)

        best_accuracy = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]

        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = deepcopy(model)

        pickle.dump(best_model, open(svm_file_name, 'wb'))

        self.best_faces_model = best_model

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i]:
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j]:
                        if self.intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False

        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def run(self):
        """
        Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din self.params.dir_test_examples
        Directorul cu numele self.params.dir_test_examples contine imagini ce
        pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
        date MIT+CMU dar si pentru alte imagini
        Functia 'non_maximal_suppression' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
        Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.
        :return:
        detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
        detections[i, :] = [x_min, y_min, x_max, y_max]
        scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
        file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
        (doar numele, nu toata calea).
        """

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        print(test_files)
        detections = None  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le obtinem
        file_names = np.array(
            [])  # array cu fisiele, in aceasta lista fisierele vor aparea de mai multe ori, pentru fiecare
        # detectie din imagine, numele imaginii va aparea in aceasta lista
        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]

        num_test_images = len(test_files)

        ret = {
            'bart': {
                'detections': [],
                'scores': [],
                'img_paths': []
            },
            'homer': {
                'detections': [],
                'scores': [],
                'img_paths': []
            },
            'lisa': {
                'detections': [],
                'scores': [],
                'img_paths': []
            },
            'marge': {
                'detections': [],
                'scores': [],
                'img_paths': []
            }
        }

        for i in range(num_test_images):
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            # TODO: completati codul functiei in continuare
            image_scores = []
            image_detections = []
            original_image = cv.imread(test_files[i], cv.IMREAD_COLOR)
            mask_img = original_image.copy()
            img_hsv = cv.cvtColor(mask_img, cv.COLOR_BGR2HSV)
            low_yellow = (20, 100, 100)
            high_yellow = (30, 255, 255)
            mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
            original_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
            scalare = 0.18
            scalarey = 0.15
            while scalare < 1 and scalarey < 1:
                img = cv.resize(original_image, (0, 0), fx=scalare, fy=scalarey)
                hog_descriptor = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                     cells_per_block=(2, 2), feature_vector=False)
                num_cols = img.shape[1] // self.params.dim_hog_cell - 1
                num_rows = img.shape[0] // self.params.dim_hog_cell - 1
                num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - 1

                for y in range(0, num_rows - num_cell_in_template):
                    for x in range(0, num_cols - num_cell_in_template):
                        x_min = int(x * self.params.dim_hog_cell * 1 // scalare)
                        y_min = int(y * self.params.dim_hog_cell * 1 // scalarey)
                        x_max = int((x * self.params.dim_hog_cell + self.params.dim_window) * 1 // scalare)
                        y_max = int((y * self.params.dim_hog_cell + self.params.dim_window) * 1 // scalarey)
                        if mask_yellow_hsv[y_min:y_max, x_min:x_max].mean() < 71:
                            continue

                        descr = hog_descriptor[y: y + num_cell_in_template, x: x + num_cell_in_template].flatten()
                        score = np.dot(descr, w)[0] + bias
                        if score > self.params.threshold:
                            x_min = int(x * self.params.dim_hog_cell * 1 // scalare)
                            y_min = int(y * self.params.dim_hog_cell * 1 // scalare)
                            x_max = int((x * self.params.dim_hog_cell + self.params.dim_window) * 1 // scalare)
                            y_max = int((y * self.params.dim_hog_cell + self.params.dim_window) * 1 // scalare)
                            image_detections.append([x_min, y_min, x_max, y_max])

                            image_scores.append(score)

                scalare *= 1.03
                scalarey *= 1.03

            if len(image_scores) > 0:
                image_detections, image_scores = self.non_maximal_suppression(np.array(image_detections),
                                                                              np.array(image_scores),
                                                                              original_image.shape)
            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                scores = np.append(scores, image_scores)
                short_file_name = ntpath.basename(test_files[i])
                image_names = [short_file_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

                for j in range(len(image_detections)):
                    x_min = image_detections[j][0]
                    y_min = image_detections[j][1]
                    x_max = image_detections[j][2]
                    y_max = image_detections[j][3]

                    if image_scores[j] > 0:
                        face = img[y_min:y_max, x_min:x_max]

                        face_warped = cv.resize(face, (36, 36))
                        face_warped = [face_warped.flatten()]

                        face_warped = self.sc.transform(face_warped)

                        predictions = self.best_faces_model.predict(face_warped)
                        if int(predictions[0]) == 0:
                            print('bart')
                            ret['bart']['detections'].append(image_detections[j])
                            ret['bart']['scores'].append(image_scores[j])
                            path_nou = ntpath.basename(test_files[i])
                            ret['bart']['img_paths'].append(path_nou)

                        elif int(predictions[0]) == 1:
                            print('homer')
                            ret['homer']['detections'].append(image_detections[j])
                            ret['homer']['scores'].append(image_scores[j])
                            path_nou = ntpath.basename(test_files[i])
                            ret['homer']['img_paths'].append(path_nou)

                        elif int(predictions[0]) == 2:
                            print('lisa')
                            ret['lisa']['detections'].append(image_detections[j])
                            ret['lisa']['scores'].append(image_scores[j])
                            path_nou = ntpath.basename(test_files[i])
                            ret['lisa']['img_paths'].append(path_nou)

                        elif int(predictions[0]) == 3:
                            print('marge')
                            ret['marge']['detections'].append(image_detections[j])
                            ret['marge']['scores'].append(image_scores[j])
                            path_nou = ntpath.basename(test_files[i])
                            ret['marge']['img_paths'].append(path_nou)

        return ret

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names, who):
        if who == 'bart':
            gt_file = self.params.path_annotations_bart
        elif who == 'homer':
            gt_file = self.params.path_annotations_homer
        elif who == 'lisa':
            gt_file = self.params.path_annotations_lisa
        else:
            # marge
            gt_file = self.params.path_annotations_marge

        ground_truth_file = np.loadtxt(gt_file, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        print(detections)
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(who+'Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()
