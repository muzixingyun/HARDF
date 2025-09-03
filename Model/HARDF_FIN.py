import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights,resnet34, ResNet34_Weights,resnet50, ResNet50_Weights
from einops import rearrange
from torch.nn import functional as F
from Model import CBAM
import math
# from torchsummary import torchsummary
from thop import profile
import numpy as np


def compute_padding(size, patch_num):
    return 0 if int(size % patch_num) == 0 else int((patch_num - int(size % patch_num)) / 2)


def compute_kernel_size(size, patch_num):
    return int(size / patch_num) if int(size % patch_num) == 0 else int(
        (np.ceil(size / patch_num) * patch_num) / patch_num)


class HARDF(nn.Module):
    def __init__(self, alpha, img_len, patch_num, class_num):
        super(HARDF, self).__init__()
        self.img_len = img_len
        self.backbone = resnet50()
        self.STA1 = CBAM.STAttention(in_planes=256, reduction=16, kernel_size=7, padding=3,
                                     d_kernel_size=compute_kernel_size(56, patch_num), img_len=img_len,
                                     patch_num=patch_num,
                                     pool_padding=compute_padding(56, patch_num))

        self.STA2 = CBAM.STAttention(in_planes=512, reduction=16, kernel_size=7, padding=3,
                                     d_kernel_size=compute_kernel_size(28, patch_num), img_len=img_len,
                                     patch_num=patch_num,
                                     pool_padding=compute_padding(28, patch_num))

        self.STA3 = CBAM.STAttention(in_planes=1024, reduction=16, kernel_size=7, padding=3,
                                     d_kernel_size=compute_kernel_size(14, patch_num), img_len=img_len,
                                     patch_num=patch_num,
                                     pool_padding=compute_padding(14, patch_num))

        self.STA4 = CBAM.STAttention(in_planes=2048, reduction=16, kernel_size=7, padding=3,
                                     d_kernel_size=1, img_len=img_len, patch_num=7, pool_padding=0)
        self.FF = FeatureFusion(in_planes=2048, img_len=img_len)
        self.flatten = nn.Flatten()
        self.fusion_conv = nn.Conv2d(2048, 2048, kernel_size=7, padding=0, stride=1)
        self.decision = nn.Sequential(
            nn.Linear(1000, class_num)
        )
        self.fusion_decision = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.img_len * class_num, class_num)
        )

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.STA1(self.backbone.layer1(x))
        x = self.STA2(self.backbone.layer2(x))
        x = self.STA3(self.backbone.layer3(x))
        x = self.STA4(self.backbone.layer4(x))
        # 特征融合和决策
        x = self.FF(x)
        x = self.backbone.fc(self.flatten(self.fusion_conv(x)))
        out = self.decision(x)
        out = rearrange(out, '(b t) c -> b t c', t=self.img_len)
        return self.fusion_decision(out)


class FusionBlock(nn.Module):
    def __init__(self, inplace, img_len, emb_size, last=False):
        super(FusionBlock, self).__init__()
        self.img_len = img_len
        self.last = last
        self.GAP = nn.AdaptiveMaxPool2d((1, 1))
        self.q = nn.Linear(emb_size, emb_size, bias=False)
        self.k = nn.Linear(emb_size, emb_size, bias=False)
        self.FrontForward = nn.Sequential(
            nn.Conv2d(inplace, inplace * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(inplace * 2, inplace, kernel_size=3, stride=1, padding=1)
        )
        self.up_channels = nn.Conv2d(inplace, inplace * 2, kernel_size=1, stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, cls_token):
        # cls_token = self.avgpool(cls_token.squeeze(1)).unsqueeze(1)
        cls_token_back = cls_token
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.img_len)
        x = torch.cat([cls_token.unsqueeze(1), x], dim=1)
        x = rearrange(x, 'b t c h w -> (b t) c h w', t=self.img_len + 1)
        # 对图谱进行全局最大池化
        pool_x = self.GAP(x).squeeze(-1).squeeze(-1)
        # 分离cls_token及其他部分
        pool_x = rearrange(pool_x, '(b t) c -> b t c', t=self.img_len + 1)
        quries = self.q(pool_x)
        keys = self.k(pool_x)
        q_cls_token = quries[:, 0, ...].unsqueeze(1)
        # k_temporal_feature = keys[:, 1:, ...]
        relition_cls_temporal = torch.einsum('bqd, bkd -> bqk', q_cls_token, keys)  # 获取空间区域的上下文关系
        attention_map = F.softmax(relition_cls_temporal, dim=-1)
        attention_map = attention_map[:, 0, :, None, None, None]
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.img_len + 1)
        cls_token = torch.sum(x * attention_map, dim=1) + cls_token_back
        # x = torch.cat([cls_token, x[:, 1:, ...]], dim=1)
        cls_token = cls_token + self.FrontForward(cls_token)
        if not self.last:
            cls_token = self.up_channels(self.avgpool(cls_token))
        return cls_token


class FeatureFusion(nn.Module):
    def __init__(self, in_planes, img_len, expansion=2):
        super(FeatureFusion, self).__init__()
        self.inplace = in_planes
        self.img_len = img_len
        self.q = nn.Linear(in_planes, in_planes)
        self.k = nn.Linear(in_planes, in_planes)
        self.v = nn.Linear(in_planes, in_planes)
        self.FeedForwardBlock = nn.Sequential(
            nn.Linear(in_planes, in_planes * expansion),
            nn.ReLU(),
            nn.Linear(in_planes * expansion, in_planes),
        )

    def forward(self, x):
        bt, c, h, w = x.shape
        x = rearrange(x, 'bt c h w -> bt (h w) c')
        queries, keys, values = self.q(x), self.k(x), self.v(x)  # 获取当前输入所有区域的q,k,v
        # spatial_context = torch.einsum('bqd, bkd -> bqk', queries, keys)  # 获取空间区域的上下文关系
        # scaling = self.inplace ** (1 / 2)
        # spatial_context = F.softmax(spatial_context, dim=-1) / scaling
        # print(spatial_context.shape)
        spatial_fusion = self.scaled_dot_product_attention(queries, keys, values)
        queries = rearrange(queries, '(b t) r c -> b r t c', t=self.img_len)
        keys = rearrange(keys, '(b t) r c -> b r t c', t=self.img_len)
        values = rearrange(values, '(b t) r c -> b r t c', t=self.img_len)
        temporal_fusion = self.scaled_dot_product_attention(queries, keys, values)
        temporal_fusion = rearrange(temporal_fusion, 'b r t c -> (b t) r c')
        fusion_x = x + spatial_fusion + temporal_fusion
        fusion_x = fusion_x + self.FeedForwardBlock(fusion_x)
        return rearrange(fusion_x, 'bt (h w) c -> bt c h w', h=h)

        # Efficient implementation equivalent to the following:

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
                                     is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value


if __name__ == '__main__':
    a = torch.rand((1, 15, 3, 224, 224)).cuda()
    model = HARDF(alpha=0.3, img_len=15, patch_num=14, class_num=1).cuda()
    out = model(a)
    # print(out.shape)
    # print(return_temporal_contex3.shape)
    # torchsummary.summary(model, a)
    flops, params = profile(model.cuda(), inputs=(a,))
    print("参数量：", params)
    print("FLOPS：", flops)
