from FacialDetectorTask2 import FacialDetector
from VisualizeTask2 import *


params: Task2Parameters = Task2Parameters()
params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 6  # dimensiunea celulei
params.overlap = 0.3
params.number_bart_positive_examples = 1146  # numarul exemplelor pozitive
params.number_homer_positive_examples = 1157  # numarul exemplelor pozitive
params.number_lisa_positive_examples = 1132  # numarul exemplelor pozitive
params.number_marge_positive_examples = 1155  # numarul exemplelor pozitive

params.number_positive_examples = 5454  # numarul exemplelor pozitive
params.number_negative_examples = 6000  # numarul exemplelor negative

params.number_unknown_positive_examples = 864  # numarul exemplelor pozitive

params.threshold = 0  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
# params.has_annotations = True
params.bart_has_annotations = True
params.homer_has_annotations = True
params.lisa_has_annotations = True
params.marge_has_annotations = True


params.scaling_ratio = 0.9
params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

if params.use_flip_images:
    params.number_bart_positive_examples *= 2
    params.number_homer_positive_examples *= 2
    params.number_lisa_positive_examples *= 2
    params.number_marge_positive_examples *= 2
    params.number_unknown_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)
# exemple pozitive

""" ----------- BART EXAMPLES ----------- """
bart_features_path = os.path.join(params.dir_save_files, 'bart_' + str(params.dim_hog_cell) + '_' +
                                  str(params.number_bart_positive_examples) + '.npy')
if os.path.exists(bart_features_path):
    bart_features = np.load(bart_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    bart_features = facial_detector.get_descriptors('bart')
    np.save(bart_features_path, bart_features)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % bart_features_path)

""" ----------- HOMER EXAMPLES ----------- """
homer_features_path = os.path.join(params.dir_save_files, 'homer_' + str(params.dim_hog_cell) + '_' +
                                   str(params.number_homer_positive_examples) + '.npy')
if os.path.exists(homer_features_path):
    homer_features = np.load(homer_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    homer_features = facial_detector.get_descriptors('homer')
    np.save(homer_features_path, homer_features)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % homer_features_path)

""" ----------- LISA EXAMPLES ----------- """
lisa_features_path = os.path.join(params.dir_save_files, 'lisa_' + str(params.dim_hog_cell) + '_' +
                                  str(params.number_lisa_positive_examples) + '.npy')
if os.path.exists(lisa_features_path):
    lisa_features = np.load(lisa_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    lisa_features = facial_detector.get_descriptors('lisa')
    np.save(lisa_features_path, lisa_features)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % lisa_features_path)

""" ----------- MARGE EXAMPLES ----------- """
marge_features_path = os.path.join(params.dir_save_files, 'marge_' + str(params.dim_hog_cell) + '_' +
                                   str(params.number_marge_positive_examples) + '.npy')
if os.path.exists(marge_features_path):
    marge_features = np.load(marge_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    marge_features = facial_detector.get_descriptors('marge')
    np.save(marge_features_path, marge_features)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % marge_features_path)

# exemple pozitive
positive_features_path = os.path.join(params.dir_save_files,
                                      'descriptoriExemplePozitive_' + str(params.dim_hog_cell) + '_' +
                                      str(params.number_positive_examples) + '.npy')
if os.path.exists(positive_features_path):
    positive_features = np.load(positive_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features = facial_detector.get_positive_descriptors()
    np.save(positive_features_path, positive_features)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)

# exemple negative
negative_features_path = os.path.join(params.dir_save_files,
                                      'descriptoriExempleNegative_' + str(params.dim_hog_cell) + '_' +
                                      str(params.number_negative_examples) + '.npy')
if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features = facial_detector.get_negative_descriptors()
    np.save(negative_features_path, negative_features)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)

""" ----------- CLASSIFIER ----------- """
# clasificator fata sau nu
training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
facial_detector.train_classifier(training_examples, train_labels)

# clasificator ce fata e
training_faces_examples = np.concatenate((np.squeeze(bart_features), np.squeeze(homer_features),
                                          np.squeeze(lisa_features), np.squeeze(marge_features)),
                                         axis=0)  # concatenez toate personajele

train_faces_labels = np.concatenate((np.zeros(bart_features.shape[0]),
                                     np.ones(homer_features.shape[0]),
                                     np.ones(lisa_features.shape[0]) + 1,
                                     np.ones(marge_features.shape[0]) + 2))
facial_detector.train_faces_classifier(training_faces_examples, train_faces_labels)

info = facial_detector.run()

""" ----------- EVAL DETECTIONS ----------- """

# eval detections for bart
if params.bart_has_annotations:

    facial_detector.eval_detections(np.array(info['bart']['detections']), np.array(info['bart']['scores']),
                                    np.array(info['bart']['img_paths']), 'bart')

    show_detections_with_ground_truth(np.array(info['bart']['detections']), np.array(info['bart']['scores']),
                                      np.array(info['bart']['img_paths']), params)
else:
    show_detections_without_ground_truth(np.array(info['bart']['detections']),
                                         np.array(info['bart']['scores']),
                                         np.array(info['bart']['img_paths']), params)


# eval detections for homer
print('ceva')
if params.homer_has_annotations:
    print('ceva')
    facial_detector.eval_detections(np.array(info['homer']['detections']), np.array(info['homer']['scores']),
                                    np.array(info['homer']['img_paths']), 'homer')
    print('----------------co')
    show_detections_with_ground_truth(np.array(info['homer']['detections']), np.array(info['homer']['scores']),
                                      np.array(info['homer']['img_paths']), params)
else:
    show_detections_without_ground_truth(np.array(info['homer']['detections']), np.array(info['homer']['scores']),
                                         np.array(info['homer']['img_paths']), params)

# eval detections for lisa
if params.lisa_has_annotations:
    facial_detector.eval_detections(np.array(info['lisa']['detections']), np.array(info['lisa']['scores']),
                                    np.array(info['lisa']['img_paths']), 'lisa')
    print('----------------asdasda')
    print(info)
    show_detections_with_ground_truth(np.array(info['lisa']['detections']), np.array(info['lisa']['scores']),
                                      np.array(info['lisa']['img_paths']), params)
else:
    show_detections_without_ground_truth(np.array(info['lisa']['detections']), np.array(info['lisa']['scores']),
                                         np.array(info['lisa']['img_paths']), params)

# eval detections for marge
if params.marge_has_annotations:
    facial_detector.eval_detections(np.array(info['marge']['detections']), np.array(info['marge']['scores']),
                                    np.array(info['marge']['img_paths']), 'marge')
    print(info)
    show_detections_with_ground_truth(np.array(info['marge']['detections']), np.array(info['marge']['scores']),
                                      np.array(info['marge']['img_paths']), params)
else:
    show_detections_without_ground_truth(np.array(info['marge']['detections']), np.array(info['marge']['scores']),
                                         np.array(info['marge']['img_paths']), params)


np.save('..\\evaluare\\fisiere_solutie\\Suto_Robert_311\\task2\\detections_bart.npy', info['bart']['detections'])
np.save('..\\evaluare\\fisiere_solutie\\Suto_Robert_311\\task2\\scores_bart.npy', info['bart']['scores'])
np.save('..\\evaluare\\fisiere_solutie\\Suto_Robert_311\\task2\\file_names_bart.npy', info['bart']['img_paths'])

np.save('..\\evaluare\\fisiere_solutie\\Suto_Robert_311\\task2\\detections_homer.npy', info['homer']['detections'])
np.save('..\\evaluare\\fisiere_solutie\\Suto_Robert_311\\task2\\scores_homer.npy', info['homer']['scores'])
np.save('..\\evaluare\\fisiere_solutie\\Suto_Robert_311\\task2\\file_names_homer.npy', info['homer']['img_paths'])

np.save('..\\evaluare\\fisiere_solutie\\Suto_Robert_311\\task2\\detections_lisa.npy', info['lisa']['detections'])
np.save('..\\evaluare\\fisiere_solutie\\Suto_Robert_311\\task2\\scores_lisa.npy', info['lisa']['scores'])
np.save('..\\evaluare\\fisiere_solutie\\Suto_Robert_311\\task2\\file_names_lisa.npy', info['lisa']['img_paths'])

np.save('..\\evaluare\\fisiere_solutie\\Suto_Robert_311\\task2\\detections_marge.npy', info['marge']['detections'])
np.save('..\\evaluare\\fisiere_solutie\\Suto_Robert_311\\task2\\scores_marge.npy', info['marge']['scores'])
np.save('..\\evaluare\\fisiere_solutie\\Suto_Robert_311\\task2\\file_names_marge.npy', info['marge']['img_paths'])