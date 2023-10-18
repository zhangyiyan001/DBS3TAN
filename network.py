# -*- coding:utf-8 -*-
"""
作者：张亦严
日期:2022年10月04日
"""
from torch import nn
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Patch_emd(nn.Module):
    def __init__(self, image_size, patch_size, channels, dim, emb_dropout):
        super(Patch_emd, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, img):
        img = self.to_patch_embedding(img)
        b, n, _ = img.shape
        # cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        # x = torch.cat((cls_tokens, img), dim=1)
        img += self.pos_embedding[:, :(n)]
        img = self.dropout(img)

        return img

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches =  (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding1 = nn.Parameter(torch.randn(1, dim+1, num_patches))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, num_patches))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer1 = Transformer(num_patches, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        #print(cls_tokens.shape, x.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        x1_class = x[:, 0]
        x1 = x[:, 1:]
        x1 = x1.permute(0, 2, 1)
        b, n, _ = x1.shape
        cls_tokens1 = repeat(self.cls_token1, '1 n d -> b n d', b=b)
        #print(cls_tokens1.shape, x1.shape)
        x2 = torch.cat((cls_tokens1, x1), dim=1)
        x2 += self.pos_embedding1[:, :(n + 1)]
        x2 = self.dropout(x2)
        x2 = self.transformer1(x2)
        x2_class = x2[:, 0]
        # x2 = x2[:, 1:]
        out = torch.cat((x1_class,x2_class),dim=-1)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        out = self.to_latent(out)

        return out

class Spa_att(nn.Module):
    def __init__(self):
        super(Spa_att, self).__init__()
    def forward(self, x):
        b, c, h, d = x.size()
        k = torch.clone(x) #(b c h w)
        q = torch.clone(x) #(b c h w)
        v = torch.clone(x) #(b c h w)
        k_hat = rearrange(k, 'b c h d -> b (h d) c') #(b f c)
        q_hat = rearrange(q, 'b c h d -> b c (h d)') #(b c f)
        v_hat = rearrange(v, 'b c h d -> b c (h d)') #(b c f)
        k_q = torch.bmm(k_hat, q_hat)
        k_q = F.softmax(k_q, dim=-1)
        q_v = torch.bmm(v_hat, k_q)
        q_v_re = rearrange(q_v, 'b c (h d) -> b c h d', h=h, d=d)
        att = x + q_v_re
        return att

class Net(nn.Module):
    def __init__(self, in_cha, patch, num_class):
        super(Net, self).__init__()
        self.spa_conv1 = nn.Sequential(
            nn.Conv2d(in_cha, 32, 3, 1, 1),

            nn.Conv2d(32, 64, 3, 1, 1, groups=32),
            nn.Conv2d(64, in_cha, 1, 1),
            nn.BatchNorm2d(in_cha),
            nn.ReLU(inplace=True),

        )
        self.spa_attention = Spa_att()

        self.spa_conv2 = nn.Sequential(
            nn.Conv2d(in_cha, 32, 3, 1, 1),

            nn.Conv2d(32, 64, 3, 1, 1, groups=32),
            nn.Conv2d(64, in_cha, 1, 1),
            nn.BatchNorm2d(in_cha),
            nn.ReLU(inplace=True),

        )
        self.linear = nn.Sequential(
            nn.Linear(3080, num_class)
        )

        self.spe_former = ViT(image_size=(patch, patch),
                               patch_size=(1, 1),
                               num_classes=num_class,
                               dim=in_cha,
                               depth=2,
                               heads=3,
                               mlp_dim=256,
                               pool='cls',
                               channels=in_cha,
                               dim_head=64,
                               dropout=0.2,
                               emb_dropout=0.1)

        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        x1 = self.spa_conv1(x)
        x2 = self.spa_attention(x1)
        x3 = x + x2
        x4 = self.spa_conv2(x3)
        x5 = self.spa_attention(x4)
        x6 = x4 + x5
        return x6

    def forward_twice(self, x):
        x = self.spe_former(x)

        return x

    def forward_third(self, x):
        x = self.linear(x)
        return x


    def forward(self, x1_spa, x2_spa, x1_band, x2_band):
        x1_spa = self.forward_once(x1_spa)
        x2_spa = self.forward_once(x2_spa)

        x1_spe = self.forward_twice(x1_band)
        x2_spe = self.forward_twice(x2_band)


        x1_spa = x1_spa.view(x1_spa.shape[0], -1)
        x2_spa = x2_spa.view(x2_spa.shape[0], -1)

        similar1 = F.pairwise_distance(x1_spa, x2_spa)
        similar2 = F.pairwise_distance(x1_spe, x2_spe)

        similar = 0.5 * similar1 + 0.5 * similar2

        # similar = self.sigmoid(similar)

        return x1_spa, x2_spa, x1_spe, x2_spe, similar

# Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive




