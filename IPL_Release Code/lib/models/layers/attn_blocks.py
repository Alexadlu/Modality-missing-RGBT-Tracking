import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention, Attention_qkv


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template=None, global_index_search=None, mask=None, ce_template_mask=None, keep_ratio_search=None):
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)

        if global_index_template!=None:
            lens_t = global_index_template.shape[1]

            removed_index_search = None
            if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
                keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
                x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, global_index_template, global_index_search, removed_index_search, attn
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
   
        
class CEBlock_align(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search

        self.align_adapter_rgb2tir = AlignAdapter(dim)
        self.align_adapter_tir2rgb = AlignAdapter(dim)

    def forward(self, x, global_index_template=None, global_index_search=None, mask=None, ce_template_mask=None, keep_ratio_search=None, input_state=None):
        
        if input_state == 'rgb2tir':
            feat_align = self.align_adapter_rgb2tir(x,reverse=True)
            feat_align_reverse = self.align_adapter_rgb2tir(feat_align)

        elif input_state == 'tir2rgb':
            feat_align = self.align_adapter_tir2rgb(x,reverse=True)
            # if self.training:
            feat_align_reverse = self.align_adapter_tir2rgb(feat_align)         
        else:
            
            feat_align = x
            feat_align_reverse = x
            
        
        x = feat_align
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)

        if global_index_template!=None:
            lens_t = global_index_template.shape[1]

            removed_index_search = None
            if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
                keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
                x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

            x = x + self.drop_path(self.mlp(self.norm2(x)))
            
            return x, feat_align, feat_align_reverse,  global_index_template, global_index_search, removed_index_search, attn
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, feat_align, feat_align_reverse
        
    def forward_testadd(self, x, global_index_template_rgb=None, global_index_search_rgb=None, global_index_template_tir=None, global_index_search_tir=None, mask=None, ce_template_mask=None, keep_ratio_search=None, input_state=None,x_tir=None):
        feat_prompt_tir = self.align_adapter_rgb2tir(x,reverse=True)
        
        feat_prompt_rgb = self.align_adapter_tir2rgb(x_tir,reverse=True)
        x = x + feat_prompt_rgb
        x_tir = x_tir + feat_prompt_tir
        
    
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)
        
        x_attn_tir, attn_tir = self.attn(self.norm1(x_tir), mask, True)
        x_tir = x_tir + self.drop_path(x_attn_tir)

        

        if global_index_template_rgb!=None:
            lens_t = global_index_template_rgb.shape[1]

            removed_index_search_rgb = None
            removed_index_search_tir = None

            if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
                keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
                x, global_index_search_rgb, removed_index_search_rgb = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search_rgb, ce_template_mask)
                
                x_tir, global_index_search_tir, removed_index_search_tir = candidate_elimination(attn_tir, x_tir, lens_t, keep_ratio_search, global_index_search_tir, ce_template_mask)

            x = x + self.drop_path(self.mlp(self.norm2(x)))
            x_tir = x_tir + self.drop_path(self.mlp(self.norm2(x_tir)))
            
            return x, x_tir, global_index_template_rgb, global_index_template_tir, global_index_search_rgb, global_index_search_tir, removed_index_search_rgb, removed_index_search_tir, attn, attn_tir
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            x_tir = x_tir + self.drop_path(self.mlp(self.norm2(x_tir)))
            
            return x, x_tir


class IPerBlock(nn.Module):
    # input: x, output: y.
    # forward:
    # y1 = x1 + F(x2)    # y2 = x2 + G(y1)
    # reverse:
    # y2 = x2 - G(x1)    # y1 = x1 - F(y2)
    def __init__(self,in_dim):
        super(IPerBlock, self).__init__()
        assert in_dim%2 == 0
        fdim = int(in_dim/2)

        self.F = nn.Sequential(
            nn.Linear(fdim,fdim), nn.ReLU(inplace=False),
            nn.Linear(fdim,fdim)
            )
        self.G = nn.Sequential(
            nn.Linear(fdim,fdim), nn.ReLU(inplace=False),
            nn.Linear(fdim,fdim)
            )
    
    def forward(self, x, reverse=False):
        if reverse:
            x1, x2 = torch.chunk(x, 2, dim=2)
            # print(x1.shape, x2.shape)
            y2 = x2 - self.G(x1)
            y1 = x1 - self.F(y2)
            y = torch.cat((y1,y2),dim=2)
        else:
            x1, x2 = torch.chunk(x, 2, dim=2)
            y1 = x1 + self.F(x2)
            y2 = x2 + self.G(y1)
            y = torch.cat((y1,y2),dim=2)
        return y


class AlignAdapter(nn.Module):
    # Each AlignAdapter consists of two IPer blocks
    def __init__(self, feature_dim, block_number=2):
        super().__init__()
        self.model = nn.ModuleList([IPerBlock(feature_dim) for _ in range(block_number)])
    
    def forward(self, x, reverse=False):
        if reverse:
            for i in reversed((range(len(self.model)))):   #可以考虑一下交换x1,x2
                x = self.model[i](x, reverse= reverse)
        else:
            for i in range(len(self.model)):
                x = self.model[i](x, reverse= reverse)
        return x


class Block_align(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.align_adapter_rgb2tir = AlignAdapter(dim)
        self.align_adapter_tir2rgb = AlignAdapter(dim)
        
        
    def forward(self, x, mask=None, input_state=None):
        
        if input_state == 'rgb2tir':
            feat_align = self.align_adapter_rgb2tir(x,reverse=True)
            # if self.training:
            feat_align_reverse = self.align_adapter_rgb2tir(feat_align)
            # else:
                # feat_align_reverse = 0
        elif input_state == 'tir2rgb':
            feat_align = self.align_adapter_tir2rgb(x,reverse=True)
            # if self.training:
            feat_align_reverse = self.align_adapter_tir2rgb(feat_align)            
            # else:
                # feat_align_reverse = 0
        else:
            feat_align = x
            feat_align_reverse = x
        
        x = feat_align
        
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, feat_align, feat_align_reverse


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, input_state=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    


