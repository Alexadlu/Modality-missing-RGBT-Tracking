"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.IPL.vit import vit_base_patch16_224
from lib.models.IPL.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack_twobranch(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        #self.backbone_i = transformer_i
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)



    def forward_test(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                input_state = None
                ):
        if input_state == 'rgbtir':
            ################################### input complete modality##################################
            x_rgb, aux_dict_rgb,_,_ = self.backbone(z=template[0], x=search[0],
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, input_state='rgb')

            x_tir, aux_dict_tir,_,_ = self.backbone(z=template[1], x=search[1],
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, input_state='tir')
            
            
        elif input_state == 'rgb':
        ################################### input rgb modality##################################
            x_rgb, aux_dict_rgb,_,_ = self.backbone(z=template[0], x=search[0],
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, input_state='rgb')

            x_tir, aux_dict_tir,_,_ = self.backbone(z=template[0], x=search[0],
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, input_state='rgb2tir')
        elif input_state == 'tir':
        ################################### input tir modality##################################
            x_tir, aux_dict_tir,_,_ = self.backbone(z=template[1], x=search[1],
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, input_state='tir')

            x_rgb, aux_dict_rgb,_,_ = self.backbone(z=template[1], x=search[1],
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, input_state='tir2rgb')        


        aux_dict = {'aux_dict_rgb':aux_dict_rgb, 'aux_dict_tir':aux_dict_tir}
        
        
        x = torch.cat([x_rgb,x_tir],2)
        
        
        # Forward head
        feat_last = x

        if isinstance(x, list):
            feat_last = x[-1]

            
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat_rgb'] = x_rgb
        out['backbone_feat_tir'] = x_tir
    
        return out

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        ################################### input complete modality##################################
        x_rgb, aux_dict_rgb, feat_layer_list_rgb,_ = self.backbone(z=template[0], x=search[0],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, input_state='rgb')

        x_tir, aux_dict_tir, feat_layer_list_tir,_ = self.backbone(z=template[1], x=search[1],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, input_state='tir')

        ################################### input rgb2tir modality#################################
        x_tir_fake, aux_dict_tir_fake, feat_layer_list_realrgb2tir_fake, feat_layer_list_faketir2rgb_fake = self.backbone(z=template[0], x=search[0],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, input_state='rgb2tir')
        
        ################################### input tir2rgb modality##################################
        x_rgb_fake, aux_dict_rgb_fake, feat_layer_list_realtir2rgb_fake, feat_layer_list_fakergb2tir_fake = self.backbone(z=template[1], x=search[1],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, input_state='tir2rgb')        


        aux_dict = {'aux_dict_rgb':aux_dict_rgb, 'aux_dict_tir':aux_dict_tir}
        
        aux_dict_fakergb = {'aux_dict_rgb_fake':aux_dict_rgb_fake}
        aux_dict_faketir = {'aux_dict_rgb_fake':aux_dict_tir_fake}
        

        feat_last_rgb = x_rgb
        feat_last_tir = x_tir
        feat_last_tirfake = x_tir_fake
        feat_last_rgbfake = x_rgb_fake
        
        feat_last = torch.cat([feat_last_rgb,feat_last_tir],2)
        x_rgb_tirfake = torch.cat([feat_last_rgb,feat_last_tirfake],2)
        x_rgbfake_tir = torch.cat([feat_last_rgbfake,feat_last_tir],2)
    
        out = self.forward_head(feat_last, None)
        out_rgb_tirfake = self.forward_head(x_rgb_tirfake, None)
        out_rgbfake_tir = self.forward_head(x_rgbfake_tir, None)

        out.update(aux_dict)
        out['backbone_feat_rgb'] = feat_layer_list_rgb
        out['backbone_feat_tir'] = feat_layer_list_tir

        out_rgbfake_tir.update(aux_dict_fakergb)
        out_rgbfake_tir['backbone_feat_realtir2rgb'] = feat_layer_list_realtir2rgb_fake
        out_rgbfake_tir['backbone_feat_fakergb2tir'] = feat_layer_list_fakergb2tir_fake
    
        out_rgb_tirfake.update(aux_dict_faketir)
        out_rgb_tirfake['backbone_feat_realrgb2tir'] = feat_layer_list_realrgb2tir_fake
        out_rgb_tirfake['backbone_feat_faketir2rgb'] = feat_layer_list_faketir2rgb_fake

    
        return out, out_rgb_tirfake, out_rgbfake_tir

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_IPL(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = cfg.MODEL.PRETRAIN_FILE

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':  # this is selected
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        # for RGBT hidden_dim
        hidden_dim = backbone.embed_dim + backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack_twobranch(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )


    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        param_dict_rgbt = dict()
        for k,v in checkpoint["net"].items():
            # if k in ['box_head.conv1_ctr.0.weight','box_head.conv1_offset.0.weight','box_head.conv1_size.0.weight']:
            #     param_dict_rgbt[k] = torch.cat([v,v],1)

            if 'patch_embed.' in k:
                param_dict_rgbt[k] = v
                kk = k.replace('patch_embed.','patch_embed_rgb2tir.')
                param_dict_rgbt[kk] = v
                kkk = k.replace('patch_embed.','patch_embed_tir2rgb.')
                param_dict_rgbt[kkk] = v
            else:
                param_dict_rgbt[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(param_dict_rgbt, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('missing_keys, unexpected_keys',missing_keys, unexpected_keys)
    return model
