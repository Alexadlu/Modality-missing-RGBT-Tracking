import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
from lib.test.utils.load_text import load_text

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import json

def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False, checkpoint_path=None, debug=False, update=[1., 0., 0.]):
        # runid改成要测试的epoch
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.display_name = display_name
        self.checkpoint_path = checkpoint_path

        
        self.update = False if update[0]>=1. else True
        self.params = self.get_parameters(update)
        self.run_id = self.params.cfg.TEST.EPOCH
        # self.run_id = run_id

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            if self.update:
                self.results_dir = '{}/{}/{}_ep{:03d}_{}'.format(env.results_path, self.name, self.parameter_name, self.run_id, \
                                                               str(update)[1:-1].replace(', ','').replace('0',''))
            else:
                self.results_dir = '{}/{}/{}_ep{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """

        params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        print('select dataset:', self.dataset_name)
        
        if self.dataset_name == 'rgbt234_miss':
            print('testing rgbt234 missing')
            with open("../rgbt234_miss/json/missing_results_rgbt234.json",'r') as load_f:
                mask_dict = json.load(load_f)            
            seq_mask_list = mask_dict[seq.name]['data']
            
        elif self.dataset_name == 'lashertestingset_miss':
            print('testing lashertestingset missing')
            with open("../lasher245_miss/json/missing_results_lasher245.json",'r') as load_f:
                mask_dict = json.load(load_f)            
            seq_mask_list = mask_dict[seq.name]['data']

        elif self.dataset_name == 'vtuav_miss':
            print('testing vtuav missing')
            with open("../vtuav_miss/json/missing_results_vtuav176.json",'r') as load_f:
                mask_dict = json.load(load_f)            
            seq_mask_list = mask_dict[seq.name]['data']
            
        else:
            print('testing no missing dataset')
            seq_mask_list = None
            
        

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        # image = self._read_image(seq.frames[0])
        image_v = self._read_image(seq.frames_v[0])
        image_i = self._read_image(seq.frames_i[0])

        start_time = time.time()
        #out = tracker.initialize(image, init_info)
        out = tracker.initialize(image_v, image_i, init_info) # Initialize network

        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        out_prev = out
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)
        frame_num =0 
        for frame_path_v, frame_path_i in zip(seq.frames_v[1:], seq.frames_i[1:]):
            frame_num += 1
            info = seq.frame_info(frame_num)

            # miss_dataset
            if 'miss' in self.dataset_name:
                if seq_mask_list[frame_num][0] == 0.0 and seq_mask_list[frame_num][1] == 1.0: # miss rgb
                    image_i = self._read_image(frame_path_i)
                    image_v = image_i #load each frame image
                    input_state = 'tir'
                elif seq_mask_list[frame_num][1] == 0.0 and seq_mask_list[frame_num][0] == 1.0: # miss tir
                    image_v = self._read_image(frame_path_v) #load each frame image
                    image_i = image_v
                    input_state = 'rgb'
                elif seq_mask_list[frame_num][0] == 0.0 and seq_mask_list[frame_num][1] == 0.0: # all miss
                    image_v = None #self._read_image(frame_path_v)-self._read_image(frame_path_v) #load each frame image
                    image_i = None #self._read_image(frame_path_i)-self._read_image(frame_path_i)
                    input_state = 'skip'                    
                elif seq_mask_list[frame_num][0] == 1.0 and seq_mask_list[frame_num][1] == 1.0: # no miss
                    image_v = self._read_image(frame_path_v) #load each frame image
                    image_i = self._read_image(frame_path_i)
                    input_state = 'rgbtir'
            else:
                #print('testing RGBT normal dataset!!!')
                image_v = self._read_image(frame_path_v) #load each frame image
                image_i = self._read_image(frame_path_i)
                input_state = 'rgbtir'      

            
            
            start_time = time.time()

            
            info['previous_output'] = prev_output
            try:
                info['baseline_rect'] = self.baseline_rect[seq.name][frame_num]
            except:
                info['baseline_rect'] = [0.,0.,0.,0.]

            # if len(seq.ground_truth_rect) > 1:
            #     info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image_v, image_i, info, input_state, out_prev)

            out_prev = out
            prev_output = OrderedDict(out)

            # print(out)
            # print(prev_output)
            
            _store_outputs(out, {'time': time.time() - start_time})
            
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self, update:list):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        if not self.update:
            params = param_module.parameters(self.parameter_name, self.checkpoint_path) # 不采用模板更新
        else:
            params = param_module.parameters(self.parameter_name, self.checkpoint_path, update) # 采用模板更新
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



