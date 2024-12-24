import math

from lib.models.IPL import build_IPL
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
import torch.nn.functional as F
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class IPLTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(IPLTrack, self).__init__(params)
        network = build_IPL(params.cfg, training=False)
        try:
            network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        except:
            network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu'), strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image_v, image_i, info: dict):
        self.temps = []
        self.temps_score = []
        # forward the template once
        z_patch_arr_rgb, resize_factor_rgb, z_amask_arr_rgb = sample_target(image_v, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir = sample_target(image_i, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        # z_patch_arr_tir[:16,:16,:]*=0
        # z_patch_arr_tir[-16:,:16,:]=255.
        # z_patch_arr_rgb[:,-32:,:]=255
        self.z_patch_arr_rgb = z_patch_arr_rgb
        self.z_patch_arr_tir = z_patch_arr_tir

        template_rgb = self.preprocessor.process(z_patch_arr_rgb, z_amask_arr_rgb)
        template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)
        with torch.no_grad():
            self.z_dict1_rgb = template_rgb
            self.z_dict1_tir = template_tir
            self.z_dict = [self.z_dict1_rgb,self.z_dict1_tir]
        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox_rgb = self.transform_bbox_to_crop(info['init_bbox'], resize_factor_rgb,
                                                        template_rgb.tensors.device).squeeze(1)
            self.box_mask_z_rgb = generate_mask_cond(self.cfg, 1, template_rgb.tensors.device, template_bbox_rgb)

            template_bbox_tir = self.transform_bbox_to_crop(info['init_bbox'], resize_factor_tir,
                                                        template_tir.tensors.device).squeeze(1)
            self.box_mask_z_tir = generate_mask_cond(self.cfg, 1, template_tir.tensors.device, template_bbox_tir)

            self.box_mask_z = [self.box_mask_z_rgb,self.box_mask_z_tir]
            self.box_mask_z = self.box_mask_z[0]
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}
    def track(self, image_v,image_i, info: dict = None, input_state=None,prev_output=None):
        if input_state == 'rgbtir':
            H, W, _ = image_v.shape
            self.frame_id += 1
            x_patch_arr_rgb, resize_factor_rgb, x_amask_arr_rgb = sample_target(image_v, self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
            x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(image_i, self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
            search_rgb = self.preprocessor.process(x_patch_arr_rgb, x_amask_arr_rgb)
            search_tir = self.preprocessor.process(x_patch_arr_tir, x_amask_arr_tir)

            with torch.no_grad():
                x_dict_rgb = search_rgb
                x_dict_tir = search_tir
                x_dict = [x_dict_rgb,x_dict_tir]
                out_dict = self.network.forward_test(
                    template=[self.z_dict[0].tensors, self.z_dict[1].tensors], 
                    search=[x_dict[0].tensors, x_dict[1].tensors], 
                    ce_template_mask=self.box_mask_z,
                    input_state = 'rgbtir'
                    )    

            # add hann windows
            pred_score_map = out_dict['score_map']
            response = self.output_window * pred_score_map
            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        
            pred_boxes = pred_boxes.view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor_rgb).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor_rgb), H, W, margin=10)
        

        elif input_state == 'rgb':
            H, W, _ = image_v.shape
            self.frame_id += 1
            x_patch_arr_rgb, resize_factor_rgb, x_amask_arr_rgb = sample_target(image_v, self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
            search_rgb = self.preprocessor.process(x_patch_arr_rgb, x_amask_arr_rgb)

            with torch.no_grad():
                x_dict_rgb = search_rgb
                out_dict = self.network.forward_test(
                    template=[self.z_dict[0].tensors, self.z_dict[1].tensors], 
                    search=[x_dict_rgb.tensors, None], 
                    ce_template_mask=self.box_mask_z,
                    input_state = 'rgb'
                    )    

            # add hann windows
            pred_score_map = out_dict['score_map']
            response = self.output_window * pred_score_map
            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        
            pred_boxes = pred_boxes.view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor_rgb).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor_rgb), H, W, margin=10)

        elif input_state == 'tir':
            H, W, _ = image_i.shape
            self.frame_id += 1
            x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(image_i, self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
            search_tir = self.preprocessor.process(x_patch_arr_tir, x_amask_arr_tir)

            with torch.no_grad():
                x_dict_tir = search_tir
                out_dict = self.network.forward_test(
                    template=[self.z_dict[0].tensors, self.z_dict[1].tensors], 
                    search=[None,x_dict_tir.tensors], 
                    ce_template_mask=self.box_mask_z,
                    input_state = 'tir'
                    )    

            # add hann windows
            pred_score_map = out_dict['score_map']
            response = self.output_window * pred_score_map
            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        
            pred_boxes = pred_boxes.view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor_tir).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor_tir), H, W, margin=10)

        elif input_state == 'skip':
            # get the final box result
            return prev_output #clip_box(self.map_box_back(pred_box, resize_factor_tir), H, W, margin=10)
            
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image_v, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                if 1:
                    # print attn
                    # for i,item in enumerate(self.enc_attn_weights[-24:-12]):
                    #     self.visdom.register(item[0].sum(0), 'heatmap', 1, 'attn-rgb-'+str(i))
                    # for i,item in enumerate(self.enc_attn_weights[-12:]):
                    #     self.visdom.register(item[0].sum(0), 'heatmap', 1, 'attn-tir-'+str(i))
                        
                    # 模板对搜索区域
                    # self.visdom.register(self.enc_attn_weights[-13][0].sum(0)[:64,64:], 'heatmap', 1, 'attn-rgb-qt')
                    # self.visdom.register(self.enc_attn_weights[-1][0].sum(0)[:64,64:], 'heatmap', 1, 'attn-tir-qt')

                    # self.visdom.register(self.enc_attn_weights[-13][0].sum(0)[:64,64:].mean(0), 'lineplot', 1, 'attn-rgb-qt-l')
                    # self.visdom.register(self.enc_attn_weights[-1][0].sum(0)[:64,64:].mean(0), 'lineplot', 1, 'attn-tir-qt-l')
                    # 搜索区域对模板
                    # self.visdom.register(self.enc_attn_weights[-13][0].sum(0)[64:,:64], 'heatmap', 1, 'attn-rgb-qs')
                    # self.visdom.register(self.enc_attn_weights[-1][0].sum(0)[64:,:64], 'heatmap', 1, 'attn-tir-qs')
                    
                    # self.visdom.register(self.enc_attn_weights[-13][0].sum(0)[64:,:64].mean(0), 'lineplot', 1, 'attn-rgb-qs-l')
                    # self.visdom.register(self.enc_attn_weights[-1][0].sum(0)[64:,:64].mean(0), 'lineplot', 1, 'attn-tir-qs-l')
                    # self.visdom.register(self.enc_attn_weights[-13][0].sum(0)[64:,:64].mean(0).reshape(8,8), 'heatmap', 1, 'attn-rgb-qs-h')
                    # self.visdom.register(self.enc_attn_weights[-1][0].sum(0)[64:,:64].mean(0).reshape(8,8), 'heatmap', 1, 'attn-tir-qs-h')

                    st_12_tir=torch.zeros([256]).cuda()
                    st_12_tir[out_dict['aux_dict_tir']['global_index_s'].squeeze().long()] = self.enc_attn_weights[-1][0].sum(0)[64:,:64].max(-1).values
                    # st_12_tir[out_dict['aux_dict_tir']['global_index_s'].squeeze().long()] = self.enc_attn_weights[-1][0].sum(0)[64:,:64].mean(-1)
                    st_12_tir = F.interpolate(st_12_tir.reshape(16,16).unsqueeze(0).unsqueeze(0), scale_factor=8, mode='bilinear')
                    self.visdom.register(st_12_tir, 'heatmap', 1, 'attn-tir-st')

                    st_12_rgb=torch.zeros([256]).cuda()
                    st_12_rgb[out_dict['aux_dict_rgb']['global_index_s'].squeeze().long()] = self.enc_attn_weights[-13][0].sum(0)[64:,:64].max(-1).values
                    # st_12_rgb[out_dict['aux_dict_rgb']['global_index_s'].squeeze().long()] = self.enc_attn_weights[-13][0].sum(0)[64:,:64].mean(-1)
                    st_12_rgb = F.interpolate(st_12_rgb.reshape(16,16).unsqueeze(0).unsqueeze(0), scale_factor=8, mode='bilinear')
                    self.visdom.register(st_12_rgb, 'heatmap', 1, 'attn-rgb-st')
                        
                    while len(self.enc_attn_weights)>24:
                        del self.enc_attn_weights[0]

                self.visdom.register((image_v, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr_rgb).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(x_patch_arr_tir).permute(2, 0, 1), 'image', 1, 'search_region_t')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_rgb).permute(2, 0, 1), 'image', 1, 'template_v')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_tir).permute(2, 0, 1), 'image', 1, 'template_t')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
                
                # enc_opt = out_dict['backbone_feat'][0, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
                # opt = enc_opt.reshape()

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr_rgb, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor_rgb, resize_factor_rgb)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return IPLTrack
