import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, Block, Block_align, CEBlock_align

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_embed_rgb2tir = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)        
        
        self.patch_embed_tir2rgb = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)   
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock_align(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )


        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)


        # modality specific layer (msa)
        self.num_mst_token = 1
        self.mst_rgb_token = nn.Parameter(torch.zeros(3, 1, self.num_mst_token, embed_dim))
        self.mst_tir_token = nn.Parameter(torch.zeros(3,1, self.num_mst_token, embed_dim))
        
        self.msa_loc = [3, 6, 9]
        #self.msa_loc = [2, 4, 6, 8, 10]
        self.msa_drop_path = [x.item() for x in torch.linspace(0, drop_path_rate, 3)]  # stochastic depth decay rule
        self.rgb_msa_layers = nn.ModuleList()
        self.tir_msa_layers = nn.ModuleList()
        if self.msa_loc is not None and type(self.msa_loc) == list:
            for i in range(len(self.msa_loc)):
                self.rgb_msa_layers.append(Block_align(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=self.msa_drop_path[i], norm_layer=norm_layer, act_layer=act_layer))
                
                self.tir_msa_layers.append(Block_align(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=self.msa_drop_path[i], norm_layer=norm_layer, act_layer=act_layer))                

        nn.init.normal_(self.mst_rgb_token, std=.02)
        nn.init.normal_(self.mst_tir_token, std=.02)
        self.init_weights(weight_init)



    def forward_features_rgbtir_add(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False,input_data=None
                         ):
        B, H, W = x[0].shape[0], x[0].shape[2], x[0].shape[3]

        x_rgb = self.patch_embed(x[0])
        z_rgb = self.patch_embed(z[0])

        x_tir = self.patch_embed(x[1])
        z_tir = self.patch_embed(z[1])        


        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            # 对最后一个维度进行下采样
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z_rgb += self.pos_embed_z
        x_rgb += self.pos_embed_x

        z_tir += self.pos_embed_z
        x_tir += self.pos_embed_x        


        if self.add_sep_seg:
            x_rgb += self.search_segment_pos_embed
            z_rgb += self.template_segment_pos_embed
            x_tir += self.search_segment_pos_embed
            z_tir += self.template_segment_pos_embed
            

        x_rgb = combine_tokens(z_rgb, x_rgb, mode=self.cat_mode)
        x_tir = combine_tokens(z_tir, x_tir, mode=self.cat_mode)
        if self.add_cls_token:
            x_rgb = torch.cat([cls_tokens, x_rgb], dim=1)
            x_tir = torch.cat([cls_tokens, x_tir], dim=1)

        x_rgb = self.pos_drop(x_rgb)
        x_tir = self.pos_drop(x_tir)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x_rgb.device)
        global_index_t_rgb  = global_index_t.repeat(B, 1)
        global_index_t_tir = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x_rgb.device)
        global_index_s_rgb = global_index_s.repeat(B, 1)
        global_index_s_tir = global_index_s.repeat(B, 1)
        


        
        removed_indexes_s_rgb = []
        removed_indexes_s_tir = []
        msa_index = 0
        feat_layer_list=[]
        for i, blk in enumerate(self.blocks):
            x_rgb, x_tir, global_index_t_rgb, global_index_t_tir, global_index_s_rgb, global_index_s_tir, removed_index_s_rgb, removed_index_s_tir, attn_rgb, attn_tir = \
                blk.forward_testadd(x_rgb, global_index_t_rgb, global_index_s_rgb, global_index_t_tir, global_index_s_tir, mask_x, ce_template_mask, ce_keep_rate,x_tir=x_tir)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s_rgb.append(removed_index_s_rgb)
                removed_indexes_s_tir.append(removed_index_s_tir)

            if self.msa_loc is not None and i in self.msa_loc:
                # if input_data=='rgb':
                rgb_mst_token = self.mst_rgb_token[msa_index]
                rgb_mst_tokens = rgb_mst_token.expand(x_rgb.shape[0], -1, -1)
                x_rgb = torch.cat((rgb_mst_tokens,x_rgb),1) 
                x, feat_align, _ = self.rgb_msa_layers[msa_index](x_rgb)
                x_rgb = x_rgb[:, self.num_mst_token:, :]
                #msa_index = msa_index + 1
                #feat_layer_list.append(feat_align)
                # elif input_data=='tir':
                tir_mst_token = self.mst_tir_token[msa_index]
                tir_mst_tokens = tir_mst_token.expand(x_tir.shape[0], -1, -1) 
                x_tir = torch.cat((tir_mst_tokens,x_tir),1)                          
                x_tir, feat_align, _ = self.tir_msa_layers[msa_index](x_tir)
                x_tir = x_tir[:, self.num_mst_token:, :]
                msa_index = msa_index + 1
                #feat_layer_list.append(feat_align)
                
                
                
        x_rgb = self.norm(x_rgb)
        x_tir = self.norm(x_tir)
        lens_x_new_rgb = global_index_s_rgb.shape[1]
        lens_z_new_rgb = global_index_t_rgb.shape[1]
        lens_x_new_tir = global_index_s_tir.shape[1]
        lens_z_new_tir = global_index_t_tir.shape[1]
        

        z_rgb = x_rgb[:, :lens_z_new_rgb]
        x_rgb = x_rgb[:, lens_z_new_rgb:]

        z_tir = x_tir[:, :lens_z_new_tir]
        x_tir = x_tir[:, lens_z_new_tir:]
        

        if removed_indexes_s_rgb and removed_indexes_s_rgb[0] is not None:
            removed_indexes_cat_rgb = torch.cat(removed_indexes_s_rgb, dim=1)

            pruned_lens_x_rgb = lens_x - lens_x_new_rgb
            pad_x_rgb = torch.zeros([B, pruned_lens_x_rgb, x_rgb.shape[2]], device=x_rgb.device)
            x_rgb = torch.cat([x_rgb, pad_x_rgb], dim=1)
            index_all_rgb = torch.cat([global_index_s_rgb, removed_indexes_cat_rgb], dim=1)
            # recover original token order
            C = x_rgb.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x_rgb = torch.zeros_like(x_rgb).scatter_(dim=1, index=index_all_rgb.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x_rgb)


        if removed_indexes_s_tir and removed_indexes_s_tir[0] is not None:
            removed_indexes_cat_tir = torch.cat(removed_indexes_s_tir, dim=1)

            pruned_lens_x_tir = lens_x - lens_x_new_tir
            pad_x_tir = torch.zeros([B, pruned_lens_x_tir, x_tir.shape[2]], device=x_tir.device)
            x_tir = torch.cat([x_tir, pad_x_tir], dim=1)
            index_all_tir = torch.cat([global_index_s_tir, removed_indexes_cat_tir], dim=1)
            # recover original token order
            C = x_tir.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x_tir = torch.zeros_like(x_tir).scatter_(dim=1, index=index_all_tir.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x_tir)

        x_tir = recover_tokens(x_tir, lens_z_new_tir, lens_x, mode=self.cat_mode)
        x_rgb = recover_tokens(x_rgb, lens_z_new_rgb, lens_x, mode=self.cat_mode)
        # re-concatenate with the template, which may be further used by other modules
        x_rgb = torch.cat([z_rgb, x_rgb], dim=1)
        x_tir = torch.cat([z_tir, x_tir], dim=1)

        aux_dict_rgb = {
            "attn": attn_rgb,
            "removed_indexes_s": removed_indexes_s_rgb,  # used for visualization
            "global_index_s": global_index_s_rgb,
        }

        aux_dict_tir = {
            "attn": attn_tir,
            "removed_indexes_s": removed_indexes_s_tir,  # used for visualization
            "global_index_s": global_index_s_tir,
        }

        return x_rgb, aux_dict_rgb, x_tir, aux_dict_tir




    def forward_features_rgbtir(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False,input_data=None
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            # 对最后一个维度进行下采样
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        msa_index = 0
        feat_layer_list=[]
        for i, blk in enumerate(self.blocks):
            x, feat_align, _, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)
            feat_layer_list.append(feat_align)
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

            if self.msa_loc is not None and i in self.msa_loc:
                if input_data=='rgb':
                    rgb_mst_token = self.mst_rgb_token[msa_index]
                    rgb_mst_tokens = rgb_mst_token.expand(x.shape[0], -1, -1)
                    x = torch.cat((rgb_mst_tokens,x),1) 
                    x, feat_align, _ = self.rgb_msa_layers[msa_index](x)
                    x = x[:, self.num_mst_token:, :]
                    msa_index = msa_index + 1
                    feat_layer_list.append(feat_align)
                elif input_data=='tir':
                    tir_mst_token = self.mst_tir_token[msa_index]
                    tir_mst_tokens = tir_mst_token.expand(x.shape[0], -1, -1) 
                    x = torch.cat((tir_mst_tokens,x),1)                          
                    x, feat_align, _ = self.tir_msa_layers[msa_index](x)
                    x = x[:, self.num_mst_token:, :]
                    msa_index = msa_index + 1
                    feat_layer_list.append(feat_align)
                
                
                
        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "global_index_s": global_index_s,
        }

        return x, aux_dict,feat_layer_list,feat_layer_list



    def forward_features_rgb2tir(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False,input_data=None
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed_rgb2tir(x)
        z = self.patch_embed_rgb2tir(z)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            # 对最后一个维度进行下采样
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        msa_index = 0
        feat_layer_list=[]
        feat_layer_reverse_list=[]
        for i, blk in enumerate(self.blocks):
            x, feat_align, feat_align_reverse, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate, input_state='rgb2tir')
            feat_layer_list.append(feat_align)
            feat_layer_reverse_list.append(feat_align_reverse)
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

            if self.msa_loc is not None and i in self.msa_loc:
                tir_mst_token = self.mst_tir_token[msa_index]
                tir_mst_tokens = tir_mst_token.expand(x.shape[0], -1, -1)
                x = torch.cat((tir_mst_tokens,x),1) 
                x, feat_align, feat_align_reverse = self.tir_msa_layers[msa_index](x, input_state='rgb2tir')
                x = x[:, self.num_mst_token:, :]
                msa_index = msa_index + 1
                feat_layer_list.append(feat_align)
                feat_layer_reverse_list.append(feat_align_reverse)
                
                
                
        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "global_index_s": global_index_s,
        }

        return x, aux_dict,feat_layer_list,feat_layer_reverse_list

    def forward_features_tir2rgb(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False,input_data=None
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed_tir2rgb(x)
        z = self.patch_embed_tir2rgb(z)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            # 对最后一个维度进行下采样
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        msa_index = 0
        feat_layer_list=[]
        feat_layer_reverse_list = []
        for i, blk in enumerate(self.blocks):
            x, feat_align, feat_align_reverse, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate, input_state='tir2rgb')
            feat_layer_list.append(feat_align)
            feat_layer_reverse_list.append(feat_align_reverse)
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

            if self.msa_loc is not None and i in self.msa_loc:
                rgb_mst_token = self.mst_rgb_token[msa_index]
                rgb_mst_tokens = rgb_mst_token.expand(x.shape[0], -1, -1)
                x = torch.cat((rgb_mst_tokens,x),1)                 
                x, feat_align, feat_align_reverse = self.rgb_msa_layers[msa_index](x, input_state='tir2rgb')
                x = x[:, self.num_mst_token:, :]
                msa_index = msa_index + 1
                feat_layer_list.append(feat_align)
                feat_layer_reverse_list.append(feat_align_reverse)
                
                
                
        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "global_index_s": global_index_s,
        }

        return x, aux_dict,feat_layer_list, feat_layer_reverse_list

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, return_last_attn=False, input_state=None):
        if input_state == 'rgb' or input_state == 'tir' :
            x, aux_dict, feat_layer_list, feat_layer_reverse_list = self.forward_features_rgbtir(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,input_data=input_state)
            
        elif input_state == 'rgb2tir':
            x, aux_dict, feat_layer_list, feat_layer_reverse_list = self.forward_features_rgb2tir(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        elif input_state == 'tir2rgb':
            x, aux_dict, feat_layer_list, feat_layer_reverse_list = self.forward_features_tir2rgb(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        elif input_state == 'rgbtir': # which for missing-agnostic scenarios
            x_rgb, aux_dict_rgb, x_tir, aux_dict_tir = self.forward_features_rgbtir_add(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,input_data=input_state)
            return x_rgb, aux_dict_rgb, x_tir, aux_dict_tir

        return x, aux_dict, feat_layer_list,feat_layer_reverse_list

def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            
            try:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            except:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
