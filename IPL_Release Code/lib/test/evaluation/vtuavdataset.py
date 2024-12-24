import numpy as np
from lib.test.evaluation.data import Sequence, Sequence_RGBT, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class VTUAVDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vtuav_path
        self.st_path = os.path.join(self.base_path, "test_ST")
        # self.st_path = "/home/zhaojiacong/datasets/ST_val_split.txt"
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        anno_path = sequence_info['anno_path']
        ground_truth_rect = load_text(str(anno_path), delimiter=[' ', '\t', ','], dtype=np.float64)
        img_list_v = sorted([p for p in os.listdir(os.path.join(sequence_path, 'rgb')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_v = [os.path.join(sequence_path, 'rgb', img) for img in img_list_v]
        
        img_list_i = sorted([p for p in os.listdir(os.path.join(sequence_path, 'ir')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_i = [os.path.join(sequence_path, 'ir', img) for img in img_list_i]
        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence_RGBT(sequence_info['name'], frames_v, frames_i, 'vtuav', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)
        
    def _get_sequence_list(self):
        sequence_list= [
            'animal_001', 'bike_003', 'bike_005', 'bike_006', 'bike_008', 'bus_001', 'bus_004', 'bus_006', 'bus_007', 'bus_010', 'bus_012', 'bus_014', 'bus_019', 'bus_021', 'bus_026', 'bus_028', 'bus_029', 'c-vehicle_003', 'cable_002', 'car_004', 'car_005', 'car_006', 'car_007', 'car_012', 'car_020', 'car_022', 'car_027', 'car_042', 'car_049', 'car_053', 'car_056', 'car_059', 'car_060', 'car_061', 'car_063', 'car_064', 'car_065', 'car_067', 'car_072', 'car_075', 'car_077', 'car_079', 'car_096', 'car_097', 'car_101', 'car_106', 'car_109', 'car_110', 'car_112', 'car_123', 'car_128', 'car_129', 'car_132', 'elebike_002', 'elebike_004', 'elebike_005', 'elebike_006', 'elebike_007', 'elebike_008', 'elebike_010', 'elebike_011', 'elebike_018', 'elebike_019', 'elebike_031', 'elebike_032', 'excavator_001', 'pedestrian_001', 'pedestrian_005', 'pedestrian_006', 'pedestrian_007', 'pedestrian_010', 'pedestrian_015', 'pedestrian_016', 'pedestrian_017', 'pedestrian_019', 'pedestrian_020', 'pedestrian_023', 'pedestrian_025', 'pedestrian_026', 'pedestrian_027', 'pedestrian_028', 'pedestrian_033', 'pedestrian_034', 'pedestrian_036', 'pedestrian_038', 'pedestrian_041', 'pedestrian_044', 'pedestrian_046', 'pedestrian_050', 'pedestrian_051', 'pedestrian_052', 'pedestrian_053', 'pedestrian_056', 'pedestrian_058', 'pedestrian_060', 'pedestrian_062', 'pedestrian_064', 'pedestrian_077', 'pedestrian_079', 'pedestrian_080', 'pedestrian_088', 'pedestrian_089', 'pedestrian_093', 'pedestrian_095', 'pedestrian_098', 'pedestrian_109', 'pedestrian_110', 'pedestrian_111', 'pedestrian_112', 'pedestrian_113', 'pedestrian_117', 'pedestrian_119', 'pedestrian_120', 'pedestrian_121', 'pedestrian_122', 'pedestrian_123', 'pedestrian_127', 'pedestrian_130', 'pedestrian_134', 'pedestrian_136', 'pedestrian_138', 'pedestrian_139', 'pedestrian_142', 'pedestrian_143', 'pedestrian_148', 'pedestrian_149', 'pedestrian_150', 'pedestrian_151', 'pedestrian_152', 'pedestrian_153', 'pedestrian_154', 'pedestrian_155', 'pedestrian_156', 'pedestrian_161', 'pedestrian_162', 'pedestrian_163', 'pedestrian_164', 'pedestrian_173', 'pedestrian_179', 'pedestrian_183', 'pedestrian_185', 'pedestrian_192', 'pedestrian_195', 'pedestrian_196', 'pedestrian_209', 'pedestrian_211', 'pedestrian_213', 'pedestrian_215', 'pedestrian_217', 'pedestrian_227', 'pedestrian_229', 'pedestrian_230', 'pedestrian_234', 'ship_001', 'train_003', 'train_004', 'tricycle_003', 'tricycle_004', 'tricycle_005', 'tricycle_006', 'tricycle_007', 'tricycle_008', 'tricycle_009', 'tricycle_010', 'tricycle_011', 'tricycle_016', 'tricycle_017', 'tricycle_019', 'tricycle_023', 'tricycle_027', 'tricycle_032', 'tricycle_035', 'tricycle_037', 'truck_004', 'truck_007', 'truck_008'
        ]

        sequence_info_list = []
        for i in range(len(sequence_list)):
            sequence_info = {}
            sequence_info["name"] = sequence_list[i] 
            sequence_info["path"] = os.path.join(self.st_path, sequence_info["name"])

            sequence_info["anno_path"] = os.path.join(sequence_info["path"], 'rgb.txt')
            sequence_info_list.append(sequence_info)
        return sequence_info_list
    