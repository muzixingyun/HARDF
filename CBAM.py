import torch
from torch import nn
from einops import rearrange, reduce, repeat
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # return self.sigmoid(x)
        return x


class STAttention(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7, padding=3, d_kernel_size=4, img_len=15, patch_num=14,
                 pool_padding=None):
        super(STAttention, self).__init__()
        self.img_len = img_len
        self.patch_num = patch_num
        self.d_kernel_size = d_kernel_size
        self.pool_padding = pool_padding
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)
        self.q = nn.Linear(in_planes, in_planes, bias=False)
        self.k = nn.Linear(in_planes, in_planes, bias=False)
        self.padding_layer = nn.ZeroPad2d((pool_padding,pool_padding,pool_padding,pool_padding))
        self.avgpool = nn.AvgPool2d(kernel_size=d_kernel_size, stride=d_kernel_size)
        self.project_spatial_contex = nn.Linear(patch_num ** 2, 1, bias=False)
        self.project_temporal_contex = nn.Linear(img_len, 1, bias=False)
        self.project_change_contex = nn.Linear(patch_num ** 2, 1, bias=False)
        self.fusion_conv = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)
        # self.f_w = nn.Parameter(torch.randn([1, 4, 1, 1]))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = x * self.ca(x)  # 从CBAM获取通道注意力并重定义输入
        out_sa = self.sa(out)  # 从CBAM获取输入的空间注意力图谱
        pool_x = rearrange(self.avgpool(self.padding_layer(out)), 'bt c h w -> bt (h w) c')  # 通过平均池化代表当前区域的特征
        # print(pool_x.shape)
        queries = self.q(pool_x)  # 获取当前输入所有区域的q
        keys = self.k(pool_x)  # 获取当前输入所有区域的k
        spatial_context = torch.einsum('bqd, bkd -> bqk', queries, keys)  # 获取空间区域的上下文关系
        spatial_context = F.softmax(spatial_context, dim=-1)
        # spatial_context = self.sigmoid(spatial_context)  # 压缩空间上下文关系
        queries = rearrange(queries, '(b t) r c -> b t r c', t=self.img_len)  # 变换以求取时间上下文关系
        keys = rearrange(keys, '(b t) r c -> b t r c', t=self.img_len)  # 变换以求取时间上下文关系
        temporal_contex = torch.einsum('brqd, brkd -> brqk', queries.transpose(1, 2),
                                       keys.transpose(1, 2))  # 获取空间区域的上下文关系
        temporal_contex = F.softmax(temporal_contex, dim=-1)
        # temporal_contex = self.sigmoid(temporal_contex)  # 压缩空间上下文关系
        change_contex = self.cosSimilarityFunc(pool_x)  # 计算时间节点关于其他时间节点的相似性
        spatial_context = self.project_spatial_contex(spatial_context.transpose(-1, -2))  # 对空间上下文进行投影
        spatial_context = rearrange(spatial_context.squeeze(-1), 'bt (rx ry) -> bt rx ry',
                                    rx=self.patch_num)  # 变换为区域形式

        temporal_contex = self.project_temporal_contex(temporal_contex.transpose(-1, -2))  # 对时间上下文进行投影
        temporal_contex = rearrange(temporal_contex.squeeze(-1), 'b (rx ry) t -> (b t) rx ry',
                                    rx=self.patch_num)  # 变换为区域形式

        change_contex = self.project_change_contex(change_contex)  # 对变换上下文进行投影
        change_contex = rearrange(change_contex.squeeze(-1), 'bt (rx ry) -> bt rx ry',
                                  rx=self.patch_num)  # 变换为区域形式
        SA = self.avgpool(self.padding_layer(out_sa)).squeeze(1)  # 从CBAM获得的空间区域重要性
        fusion_map = torch.stack([SA, spatial_context, temporal_contex, change_contex], dim=1)

        spatial_attention = self.fusion_conv(fusion_map)
        # print(spatial_attention.shape)
        spatial_attention = spatial_attention.repeat_interleave(self.d_kernel_size, dim=2).repeat_interleave(
            self.d_kernel_size, dim=3)
        if self.pool_padding > 0:
            spatial_attention = spatial_attention[::, ::, self.pool_padding:-self.pool_padding,
                                self.pool_padding:-self.pool_padding]
        return out * self.sigmoid(spatial_attention)

    def cosSimilarityFunc(self, input):
        '''
        该函数用于计算特征间的余弦相似度
        :param input:
        :return:
        '''
        # b, t, l = input.shape
        input_norm = input / (input.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.bmm(input_norm, input_norm.transpose(1, 2))


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out_sa = self.sa(out)
        return out * out_sa, out_sa


if __name__ == '__main__':
    a = torch.randn((60, 64, 14, 14))
    model = STAttention(in_planes=64)
    model(a)
