import numpy as np
from lib.test.evaluation.data import Sequence,Sequence_RGBT,  BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
class GTOTDataset(BaseDataset):
    # GTOT dataset
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.gtot_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        anno_path = sequence_info['anno_path']
        ground_truth_rect = load_text(str(anno_path), delimiter=['', '\t', ','], dtype=np.float64)
        img_list_v = sorted([p for p in os.listdir(os.path.join(sequence_path, 'v')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_v = [os.path.join(sequence_path, 'v', img) for img in img_list_v]
        
        img_list_i = sorted([p for p in os.listdir(os.path.join(sequence_path, 'i')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_i = [os.path.join(sequence_path, 'i', img) for img in img_list_i]
        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence_RGBT(sequence_info['name'], frames_v, frames_i, 'gtot', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)
        
    def _get_sequence_list(self):
        sequence_list= ['Jogging','Otcbvs','Gathering','occBike','LightOcc','Walking','GarageHover','Torabi','Quarreling','BusScale','Minibus','tunnel','RainyMotor1','WalkingOcc','OccCar-1','Otcbvs1','Pool','Cycling','WalkingNig','crowdNig','FastCarNig','WalkingNig1','carNig','OccCar-2','Exposure4','Motorbike','BlackSwan1','GoTogether','RainyCar2','Crossing','Tricycle','BlackCar','BlueCar','DarkNig','Football','MotorNig','MinibusNigOcc','Exposure2','MinibusNig','Motorbike1','RainyPeople','RainyMotor2','Running','Torabi1','FastMotor','BusScale1','Minibus1','FastMotorNig','RainyCar1','fastCar2']
        sequence_info_list = []
        for i in range(len(sequence_list)):
            sequence_info = {}
            sequence_info["name"] = sequence_list[i] 
            sequence_info["path"] = '/data1/andong.lu/data/RGBT_DATA/GTOT/'+sequence_info["name"]
            #sequence_info["startFrame"] = int('1')
            #print(end_frame[i])
            #sequence_info["endFrame"] = end_frame[i]
                
            #sequence_info["nz"] = int('6')
            #sequence_info["ext"] = 'jpg'
            sequence_info["anno_path"] = sequence_info["path"]+'/init.txt'
            #sequence_info["object_class"] = 'person'
            sequence_info_list.append(sequence_info)
        return sequence_info_list
    