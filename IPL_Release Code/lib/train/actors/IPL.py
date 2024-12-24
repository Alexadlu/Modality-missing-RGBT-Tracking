from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import torch.nn as nn
import torch.nn.functional as F






class IPL(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict_rgbtir, out_dict_rgb, out_dict_tir  = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict_rgbtir, out_dict_rgb, out_dict_tir, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region


        # print(data['template_images'].shape) # torch.Size([2, 32, 3, 128, 128])
        # print(data['search_images'].shape) # torch.Size([2, 32, 3, 128, 128])

        assert len(data['template_images']) == 2
        assert len(data['search_images']) == 2

        #exit()
        # template_list = []
        # for i in range(self.settings.num_template):
        #     template_img_i = data['template_images'][i].view(-1,
        #                                                      *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
        #     # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
        #     template_list.append(template_img_i) # [template_rgb, template_tir]


        template_list = []
        for i in range(len(data['template_images'])):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 320, 320)
            template_list.append(template_img_i)

        search_list = []
        for i in range(len(data['search_images'])):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
            search_list.append(search_img_i)

        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, self.cfg.TRAIN.BATCH_SIZE, template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict_rgbtir, out_dict_rgb, out_dict_tir = self.net(template=torch.stack(template_list),
                            search=torch.stack(search_list),
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict_rgbtir, out_dict_rgb, out_dict_tir


    def compute_losses(self, pred_dict_rgbtir, pred_dict_rgb, pred_dict_tir, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes_rgbtir = pred_dict_rgbtir['pred_boxes']
        if torch.isnan(pred_boxes_rgbtir).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes_rgbtir.size(1)
        pred_boxes_vec_rgbtir = box_cxcywh_to_xyxy(pred_boxes_rgbtir).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        # Get rgb boxes
        pred_boxes_rgb = pred_dict_rgb['pred_boxes']
        if torch.isnan(pred_boxes_rgb).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes_rgb.size(1)
        pred_boxes_vec_rgb = box_cxcywh_to_xyxy(pred_boxes_rgb).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        

        # Get tir boxes
        pred_boxes_tir = pred_dict_tir['pred_boxes']
        if torch.isnan(pred_boxes_tir).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes_tir.size(1)
        pred_boxes_vec_tir = box_cxcywh_to_xyxy(pred_boxes_tir).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)


        # compute giou and iou
        try:
            giou_loss_rgbtir, iou_rgbtir = self.objective['giou'](pred_boxes_vec_rgbtir, gt_boxes_vec)  # (BN,4) (BN,4)
            giou_loss_rgb, iou_rgb = self.objective['giou'](pred_boxes_vec_rgb, gt_boxes_vec)  # (BN,4) (BN,4)
            giou_loss_tir, iou_tir = self.objective['giou'](pred_boxes_vec_tir, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss_rgbtir, iou_rgbtir = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            giou_loss_rgb, iou_rgb = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            giou_loss_tir, iou_tir = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        # compute l1 loss
        l1_loss_rgbtir = self.objective['l1'](pred_boxes_vec_rgbtir, gt_boxes_vec)  # (BN,4) (BN,4)
        l1_loss_rgb = self.objective['l1'](pred_boxes_vec_rgb, gt_boxes_vec)  # (BN,4) (BN,4)
        l1_loss_tir = self.objective['l1'](pred_boxes_vec_tir, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict_rgbtir:
            location_loss_rgbtir = self.objective['focal'](pred_dict_rgbtir['score_map'], gt_gaussian_maps)
            location_loss_tir = self.objective['focal'](pred_dict_tir['score_map'], gt_gaussian_maps)
            location_loss_rgb = self.objective['focal'](pred_dict_rgb['score_map'], gt_gaussian_maps)
        else:
            location_loss_rgbtir = torch.tensor(0.0, device=l1_loss_rgbtir.device)
            location_loss_rgb = torch.tensor(0.0, device=l1_loss_rgb.device)
            location_loss_tir = torch.tensor(0.0, device=l1_loss_tir.device)
        
        # weighted sum
        loss_rgbtir = self.loss_weight['giou'] * giou_loss_rgbtir + self.loss_weight['l1'] * l1_loss_rgbtir + self.loss_weight['focal'] * location_loss_rgbtir

        loss_rgb = self.loss_weight['giou'] * giou_loss_rgb + self.loss_weight['l1'] * l1_loss_rgb + self.loss_weight['focal'] * location_loss_rgb

        loss_tir = self.loss_weight['giou'] * giou_loss_tir + self.loss_weight['l1'] * l1_loss_tir + self.loss_weight['focal'] * location_loss_tir

        loss_miss_task =  loss_rgb + loss_tir

        loss_rgb_pred_kl = nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(pred_boxes_vec_rgb), nn.Softmax(dim=1)(pred_boxes_vec_rgbtir.detach())) + nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(pred_dict_rgb['score_map']), nn.Softmax(dim=1)(pred_dict_rgbtir['score_map'].detach()))
        loss_tir_pred_kl = nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(pred_boxes_vec_tir), nn.Softmax(dim=1)(pred_boxes_vec_rgbtir.detach())) + nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(pred_dict_tir['score_map']), nn.Softmax(dim=1)(pred_dict_rgbtir['score_map'].detach()))

        loss_rgb_feat_MSE = 0 
        loss_tir_feat_MSE = 0 
        for i in range(len(pred_dict_rgb['backbone_feat_realrgb2tir'])):
            loss_rgb_feat_MSE += F.mse_loss(pred_dict_rgb['backbone_feat_realrgb2tir'][i], pred_dict_rgbtir['backbone_feat_tir'][i]) + F.mse_loss(pred_dict_rgb['backbone_feat_faketir2rgb'][i], pred_dict_rgbtir['backbone_feat_rgb'][i]) 
            loss_tir_feat_MSE += F.mse_loss(pred_dict_tir['backbone_feat_realtir2rgb'][i], pred_dict_rgbtir['backbone_feat_rgb'][i]) + F.mse_loss(pred_dict_tir['backbone_feat_fakergb2tir'][i], pred_dict_rgbtir['backbone_feat_tir'][i])
        

        loss = loss_miss_task + loss_rgb_pred_kl + loss_tir_pred_kl + loss_rgb_feat_MSE*0.5 + loss_rgb_feat_MSE*0.5 
        
        
        
        if return_status:
            # status for log
            mean_iou_rgbtir = iou_rgbtir.detach().mean()
            mean_iou_tir = iou_tir.detach().mean()
            mean_iou_rgb = iou_rgb.detach().mean()

            status = {"Loss/total": loss.item(),
                      "Loss/giou_rgbtir": giou_loss_rgbtir.item(),
                      "Loss/giou_rgb": giou_loss_rgb.item(),
                      "Loss/giou_tir": giou_loss_tir.item(),
                      "Loss/l1_rgbtir": l1_loss_rgbtir.item(),
                      "Loss/l1_rgb": l1_loss_rgb.item(),
                      "Loss/l1_tir": l1_loss_tir.item(),
                      "Loss/location_rgbtir": location_loss_rgbtir.item(),
                      "Loss/location_rgb": location_loss_rgb.item(),
                      "Loss/location_tir": location_loss_tir.item(),
                      "Loss/kl_rgb": loss_rgb_pred_kl.item(),
                      "Loss/kl_tir": loss_tir_pred_kl.item(),
                      "Loss/mse_rgb": loss_rgb_feat_MSE.item(),
                      "Loss/mse_tir": loss_tir_feat_MSE.item(),
                      "Loss/task_rgbtir": loss_rgbtir.item(),                      
                      "Loss/task_rgb": loss_rgb.item(),
                      "Loss/task_tir": loss_tir.item(),                      
                      "IoU_rgbtir": mean_iou_rgbtir.item(),
                      "IoU_rgb": mean_iou_rgb.item(),
                      "IoU_tir": mean_iou_tir.item()}

            
            return loss, status
        else:
            return loss


