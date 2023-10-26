import os


class Task2Parameters:
    def __init__(self):
        self.base_dir = '..//task2_antrenare//'
        self.dir_pos_examples = os.path.join(self.base_dir, '..//data/exemplePozitive')
        self.dir_neg_examples = os.path.join(self.base_dir, '..//data/exempleNegative')
        self.dir_bart_examples = os.path.join(self.base_dir, 'bart')
        self.dir_homer_examples = os.path.join(self.base_dir, 'homer')
        self.dir_lisa_examples = os.path.join(self.base_dir, 'lisa')
        self.dir_marge_examples = os.path.join(self.base_dir, 'marge')
        self.dir_test_examples = os.path.join(self.base_dir, '..//validare//simpsons_validare//')
        # 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        self.bart_annotations = os.path.join(self.base_dir, '..//validare//task2_bart_gt.txt')
        self.homer_annotations = os.path.join(self.base_dir, '..//validare//task2_homer_gt.txt')
        self.lisa_annotations = os.path.join(self.base_dir, '..//validare//task2_lisa_gt.txt')
        self.marge_annotations = os.path.join(self.base_dir, '..//validare//task2_marge_gt.txt')
        self.dir_save_files = os.path.join(self.base_dir, '..//task2_data//salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_bart_positive_examples = 1146  # numarul exemplelor pozitive
        self.number_homer_positive_examples = 1157  # numarul exemplelor pozitive
        self.number_lisa_positive_examples = 1132  # numarul exemplelor pozitive
        self.number_marge_positive_examples = 1155  # numarul exemplelor pozitive

        self.number_positive_examples = 5454  # numarul exemplelor pozitive
        self.number_negative_examples = 6000  # numarul exemplelor negative

        self.threshold = 0
        # toate ferestrele cu scorul > threshold si maxime locale devin detectii
        # self.has_annotations = True
        self.bart_has_annotations = True
        self.homer_has_annotations = True
        self.lisa_has_annotations = True
        self.marge_has_annotations = True

        self.scaling_ratio = 0.9
        self.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
        self.use_flip_images = False  # adauga imaginile cu fete oglindite

        self.path_annotations_bart = os.path.join(self.base_dir, '..//validare//task2_bart_gt.txt')
        self.path_annotations_homer = os.path.join(self.base_dir, '..//validare//task2_homer_gt.txt')
        self.path_annotations_lisa = os.path.join(self.base_dir, '..//validare//task2_lisa_gt.txt')
        self.path_annotations_marge = os.path.join(self.base_dir, '..//validare//task2_marge_gt.txt')
        self.path_annotations_unknown = os.path.join(self.base_dir, '..//validare//task2_unknown_gt.txt')
