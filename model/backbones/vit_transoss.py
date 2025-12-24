import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# --------------------------------------------------------
# 基础模块: MLP, Attention, Block
# --------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B, N_q, C = x_q.shape
        N_k = x_kv.shape[1]

        q = self.wq(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x_kv).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x_kv).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_q, x_kv):
        x = x_q + self.drop_path(self.cross_attn(self.norm1(x_q), self.norm_kv(x_kv)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed_overlap(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class CSSP_Modulation(nn.Module):
    def __init__(self, dim, num_x, num_y):
        super().__init__()
        self.dim = dim
        self.num_x = num_x
        self.num_y = num_y
        self.spectral_attn = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, dim), nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :] 
        B, N, C = patch_tokens.shape
        spectral_weights = self.spectral_attn(patch_tokens.mean(dim=1, keepdim=True))
        patch_tokens = patch_tokens * spectral_weights
        x_2d = patch_tokens.transpose(1, 2).reshape(B, C, self.num_y, self.num_x)
        spatial_mask = self.spatial_attn(x_2d) 
        x_2d = x_2d * spatial_mask
        patch_tokens = x_2d.flatten(2).transpose(1, 2)
        return torch.cat([cls_token, patch_tokens], dim=1), spatial_mask

class TransOSS(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # 1. Patch Embeddings
        self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_SAR = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2. Tokens & PosEmbed
        self.cls_token_rgb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_rgb = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token_sar = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_sar = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. 双流 Backbone
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks_rgb = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.blocks_sar = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])

        self.cssp = CSSP_Modulation(embed_dim, self.patch_embed.num_x, self.patch_embed.num_y)
        
        # 4. 多阶段融合 (3, 6, 9, 12)
        self.fusion_layers = [3, 6, 9, 12]
        self.fusion_blocks = nn.ModuleDict({
            str(layer): CrossBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer) for layer in self.fusion_layers
        })

        self.norm_rgb = norm_layer(embed_dim)
        self.norm_sar = norm_layer(embed_dim)
        self._init_weights_custom()

    def _init_weights_custom(self):
        trunc_normal_(self.cls_token_rgb, std=.02)
        trunc_normal_(self.pos_embed_rgb, std=.02)
        trunc_normal_(self.cls_token_sar, std=.02)
        trunc_normal_(self.pos_embed_sar, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, camera_id):
        B = x.shape[0]
        rgb_idx = torch.where(camera_id == 0)[0]
        sar_idx = torch.where(camera_id == 1)[0]
        
        x_rgb, mask_rgb = None, None
        if len(rgb_idx) > 0:
            x_rgb = self.patch_embed(x[rgb_idx])
            x_rgb = torch.cat((self.cls_token_rgb.expand(len(rgb_idx), -1, -1), x_rgb), dim=1)
            x_rgb = self.pos_drop(x_rgb + self.pos_embed_rgb)

        x_sar, mask_sar = None, None
        if len(sar_idx) > 0:
            x_sar = self.patch_embed_SAR(x[sar_idx])
            x_sar = torch.cat((self.cls_token_sar.expand(len(sar_idx), -1, -1), x_sar), dim=1)
            x_sar = self.pos_drop(x_sar + self.pos_embed_sar)

        can_fuse = (x_rgb is not None) and (x_sar is not None) and (len(rgb_idx) == len(sar_idx))

        for i in range(len(self.blocks_rgb)):
            if x_rgb is not None: x_rgb = self.blocks_rgb[i](x_rgb)
            if x_sar is not None: x_sar = self.blocks_sar[i](x_sar)
            
            layer_num = i + 1
            if layer_num in [3, 6, 9] and can_fuse:
                x_rgb_f, x_sar_f = self.fusion_blocks[str(layer_num)](x_rgb, x_sar), self.fusion_blocks[str(layer_num)](x_sar, x_rgb)
                x_rgb, x_sar = x_rgb_f, x_sar_f
            
            if layer_num == 12:
                if x_rgb is not None: x_rgb, mask_rgb = self.cssp(x_rgb)
                if x_sar is not None: x_sar, mask_sar = self.cssp(x_sar)
                if can_fuse:
                    x_rgb_f, x_sar_f = self.fusion_blocks['12'](x_rgb, x_sar), self.fusion_blocks['12'](x_sar, x_rgb)
                    x_rgb, x_sar = x_rgb_f, x_sar_f

        out_feat = torch.zeros(B, self.embed_dim, device=x.device)
        if x_rgb is not None: out_feat[rgb_idx] = self.norm_rgb(x_rgb)[:, 0]
        if x_sar is not None: out_feat[sar_idx] = self.norm_sar(x_sar)[:, 0]
        return out_feat, (mask_rgb, mask_sar)

    def forward(self, x, cam_label=None):
        return self.forward_features(x, cam_label)

    def load_param(self, model_path):
        """ 核心修改：处理 768x768 -> 768x3x16x16 的映射 """
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict: param_dict = param_dict['model']
        if 'state_dict' in param_dict: param_dict = param_dict['state_dict']
            
        print(f'==> Loading weights and fixing patch_embed mismatch...')
        model_dict = self.state_dict()
        new_state_dict = {}

        for k, v in param_dict.items():
            k = k.replace('module.', '')
            if 'head' in k or 'dist' in k: continue

            # 修复 patch_embed.proj.weight 形状不匹配
            if 'patch_embed.proj.weight' in k:
                target_shape = model_dict['patch_embed.proj.weight'].shape
                if v.shape != target_shape:
                    v = v.reshape(target_shape) # 将 [768, 768] 转为 [768, 3, 16, 16]

            if 'blocks.' in k:
                rgb_k, sar_k = k.replace('blocks.', 'blocks_rgb.'), k.replace('blocks.', 'blocks_sar.')
                if rgb_k in model_dict: new_state_dict[rgb_k] = v
                if sar_k in model_dict: new_state_dict[sar_k] = v
            elif 'patch_embed' in k:
                if k in model_dict: new_state_dict[k] = v
                sar_k = k.replace('patch_embed', 'patch_embed_SAR')
                if sar_k in model_dict: new_state_dict[sar_k] = v
            elif k == 'pos_embed':
                v_res = resize_pos_embed(v, self.pos_embed_rgb, self.patch_embed.num_y, self.patch_embed.num_x)
                new_state_dict['pos_embed_rgb'], new_state_dict['pos_embed_sar'] = v_res, v_res
            elif k == 'cls_token':
                new_state_dict['cls_token_rgb'], new_state_dict['cls_token_sar'] = v, v
            elif 'norm.' in k:
                new_state_dict[k.replace('norm.', 'norm_rgb.')], new_state_dict[k.replace('norm.', 'norm_sar.')] = v, v
            elif k in model_dict:
                new_state_dict[k] = v

        self.load_state_dict(new_state_dict, strict=False)

def resize_pos_embed(posemb, posemb_new, hight, width):
    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    return torch.cat([posemb_token, posemb_grid], dim=1)

def vit_base_patch16_224_TransOSS(img_size=(224, 224), stride_size=16, **kwargs):
    model = TransOSS(img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, qkv_bias=True, **kwargs)
    return model

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l, u = norm_cdf((a - mean) / std), norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_().mul_(std * math.sqrt(2.)).add_(mean).clamp_(min=a, max=b)
        return tensor