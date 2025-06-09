import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super(ConvLayer, self).__init__()
        groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel//2, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          if dwconv:
            layers_.append(CombConvLayer(inch, outch))
          else:
            layers_.append(ConvLayer(inch, outch))
          
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)
        
    def forward(self, x):
        layers_ = [x]
        
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:            
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
            
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, gr, grmul, n_layer, out_channels):
        super(EncoderBlock, self).__init__()

        self.hardblock = HarDBlock(in_channels, gr, grmul, n_layer)
        conv_in_ch = self.hardblock.get_out_ch()
        self.conv = ConvLayer(conv_in_ch, out_channels, kernel=1)
        
    def forward(self, x):
        x = self.hardblock(x)
        x = self.conv(x)
        return x



# class HarDNetBackbone(nn.Module):
#     def __init__(
#         self,
#         in_channels=1,
#         base_out_ch=[32, 64],
#         grmul=1.7,
#         drop_rate=0.1,
#         ch_list=[128, 256, 320, 640, 1024],
#         gr_list=[14, 16, 20, 40, 160],
#         n_layers=[8, 16, 16, 16, 4],
#         pool_layer=[1, 0, 1, 1, 0]
#     ):
#         super(HarDNetBackbone, self).__init__()

#         assert len(ch_list) == len(gr_list) == len(n_layers), "Length of ch_list, gr_list, and n_layers must match"

#         self.base_conv_1 = ConvLayer(in_channels=in_channels, out_channels=base_out_ch[0], kernel=3, stride=2, bias=False)
#         self.base_conv_2 = ConvLayer(in_channels=base_out_ch[0], out_channels=base_out_ch[1], kernel=3)
#         self.base_max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.encoder_blocks = nn.ModuleList()
#         self.encoder_pools = nn.ModuleList()
#         self.attention_channels = [base_out_ch[1]]  # First attention from base_conv2
#         self.pool_layer = pool_layer
#         in_ch = base_out_ch[1]
#         for i in range(len(ch_list)):
#             block = EncoderBlock(in_ch, gr_list[i], grmul, n_layers[i], ch_list[i])
#             self.encoder_blocks.append(block)
#             self.attention_channels.append(ch_list[i])
#             self.encoder_pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
#             in_ch = ch_list[i]

#     def forward(self, x):
#         attention_list = []
#         x = self.base_conv_1(x)
#         x = self.base_conv_2(x)
#         attention_list.append(x)
        
#         x = self.base_max_pool(x)

#         for i, block in enumerate(self.encoder_blocks):
#             x = block(x)
#             attention_list.append(x)
#             if self.pool_layer[i]:
#                 x = self.encoder_pools[i](x)

#         return attention_list
class HarDNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_out_ch=[32, 64],
        grmul=1.7,
        drop_rate=0.1,
        ch_list=[128, 256, 320, 640, 1024],
        gr_list=[14, 16, 20, 40, 160],
        n_layers=[8, 16, 16, 16, 4],
        pool_layer=[1, 0, 1, 1, 0]
    ):
        super(HarDNetBackbone, self).__init__()

        assert len(ch_list) == len(gr_list) == len(n_layers), "Length of ch_list, gr_list, and n_layers must match"

        self.base_conv_1 = ConvLayer(in_channels=in_channels, out_channels=base_out_ch[0], kernel=3, stride=2, bias=False)
        self.base_conv_2 = ConvLayer(in_channels=base_out_ch[0], out_channels=base_out_ch[1], kernel=3)
        # 加入 dropout
        self.base_dropout = nn.Dropout2d(p=drop_rate)

        self.base_max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        self.encoder_dropouts = nn.ModuleList()  # 加一個 dropout list
        self.attention_channels = [base_out_ch[1]]
        self.pool_layer = pool_layer

        in_ch = base_out_ch[1]
        for i in range(len(ch_list)):
            block = EncoderBlock(in_ch, gr_list[i], grmul, n_layers[i], ch_list[i])
            self.encoder_blocks.append(block)
            self.attention_channels.append(ch_list[i])
            self.encoder_pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder_dropouts.append(nn.Dropout2d(p=drop_rate))  # 對應 encoder block 加入 Dropout
            in_ch = ch_list[i]


    def forward(self, x):
            attention_list = []
            x = self.base_conv_1(x)
            x = self.base_conv_2(x)
            x = self.base_dropout(x)
        
            attention_list.append(x)
            
            x = self.base_max_pool(x)
    
            for i, block in enumerate(self.encoder_blocks):
                x = block(x)
                x = self.encoder_dropouts[i](x)
                attention_list.append(x)
                if self.pool_layer[i]:
                    x = self.encoder_pools[i](x)
    
            return attention_list
class AMFF(nn.Module):
    def __init__(self, n_inputs=5, in_channels = [0, 0, 0, 0, 0], out_channels = [12, 12, 12, 12, 12]):
        super(AMFF, self).__init__()

        self.n_inputs = n_inputs
        max_pool_size = (20, 20)
        self.max_pools = nn.ModuleList([
            nn.AdaptiveMaxPool2d(max_pool_size) for _ in range(n_inputs)
        ])

        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels[i], kernel_size = 3, padding=1) for i in range(n_inputs)
        ])
        
    def forward(self, x_list):
        x_outs = []
        for i in range(self.n_inputs):
            x_out = self.max_pools[i](x_list[i])
            x_out = self.conv[i](x_out)
            x_outs.append(x_out)
        x_cat = torch.cat(x_outs, dim=1)
        
        return x_cat

class PMCS(nn.Module):
    def __init__(self, in_channels):
        super(PMCS, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = nn.LayerNorm(in_channels)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        B, C, H, W = x.shape
        HW = H * W

        # Q: [B, C, H, W] -> [B, C, HW]
        Q = self.query_conv(x).reshape(B, C, HW)

        # K: [B, 1, H, W] -> [B, 1, HW] and softmax
        K = self.key_conv(x).reshape(B, 1, HW)
        K = F.softmax(K, dim=-1)

        # MatMul(Q, K^T): [B, C, HW] @ [B, HW, 1] -> [B, C, 1]
        attn = torch.bmm(Q, K.transpose(1, 2)).view(B, C, 1, 1)

        # Conv + LayerNorm + Sigmoid
        attn = self.out_conv(attn)  # [B, C, 1, 1]
        attn = self.norm(attn.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        attn = self.sigmoid(attn)

        # V: [B, C, H, W]
        V = self.value_conv(x)

        # Final Output: V * attn
        out = V * attn  # broadcasting over (H, W)
        return out




import torch
import torch.nn as nn
import torch.nn.functional as F

class PMSS(nn.Module):
    def __init__(self, in_channels, n_branches=3):
        super(PMSS, self).__init__()
        self.n_branches = n_branches
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels * n_branches, kernel_size=1)
        self.output_conv = nn.Conv2d(in_channels * n_branches, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape  # Assume x is already multi-scale fused: shape (B, 3Ck, H, W)
        Ck = C // self.n_branches  # Channel per branch
        Cv = Ck  # Value channels per branch (can be different if designed that way)

        # Step 1: Compute Q and K
        Q = self.query_conv(x)  # (B, C, H, W)
        K = self.key_conv(x)    # (B, C, H, W)

        # Step 2: Global mean pooling across spatial dimensions on K
        K_pool = F.adaptive_avg_pool2d(K, output_size=1)  # (B, C, 1, 1)
        K_pool = K_pool.view(B, C)                        # (B, C)
        K_soft = F.softmax(K_pool, dim=1)                 # (B, C)

        # Step 3: Reshape Q and K to (B, C, HW) and perform matmul
        Q_flat = Q.view(B, C, -1)                         # (B, C, HW)
        K_soft = K_soft.view(B, C, 1)                     # (B, C, 1)
        attention_scores = torch.bmm(K_soft.transpose(1, 2), Q_flat)  # (B, 1, HW)
        attention_scores = attention_scores.view(B, 1, H, W)          # (B, 1, H, W)
        attention_map = self.sigmoid(attention_scores)                # (B, 1, H, W)

        # Step 4: Value computation
        V = self.value_conv(x)                            # (B, Cv, H, W)
        V_split = torch.chunk(V, self.n_branches, dim=1)  # [(B, Cv, H, W)] * 3

        # Repeat attention for each branch and multiply
        attended = [v * attention_map for v in V_split]   # [(B, Cv, H, W)] * 3

        # Step 5: Concatenate and project
        fused = torch.cat(attended, dim=1)                # (B, Cv * 3, H, W)
        out = self.output_conv(fused)                     # (B, Cv, H, W)

        return out

class PMFS(nn.Module):
    def __init__(self, n_inputs, in_channels):
        super(PMFS, self).__init__()
        self.amff_out_channels = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12][:n_inputs]
        self.amff = AMFF(n_inputs = n_inputs, in_channels = in_channels, out_channels = self.amff_out_channels)
        
        self.attention_in_channels = sum(self.amff_out_channels)
        self.pmcs = PMCS(in_channels = self.attention_in_channels)
        self.pmss = PMSS(in_channels = self.attention_in_channels, n_branches = len(self.amff_out_channels))
        
    def forward(self, x_list):
        amff_out = self.amff(x_list)
        pmcs_out = self.pmcs(amff_out)
        # print(pmcs_out.shape)
        pmss_out = self.pmss(pmcs_out)
        return pmss_out

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, in_channels=60, out_channels=1, output_size=224, layers_num=4):
        super(Decoder, self).__init__()

        assert 1 <= layers_num <= 4, "layers_num 必須是 1 到 4 之間"

        self.layers_num = layers_num
        self.output_size = output_size

        # 全部 4 層的設計：對應 (in → out, size)
        channels = [64, 32, 16, 8][-layers_num:]
        channels = [in_channels] + channels
        # print(channels)
        self.all_ups = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=2, stride=2) for i in range(layers_num)
        ])
            

        self.all_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels[i + 1], channels[i + 1], kernel_size=3, padding=1), nn.ReLU(inplace=True)) for i in range(layers_num)
        ])

        # 根據 layers_num 只保留最後 N 層
        self.ups = self.all_ups
        self.convs = self.all_convs

        # 輸出層根據最後一層 conv 的 output channel 決定
        out_ch = 8
        self.out_conv = nn.Conv2d(out_ch, out_channels, kernel_size=1)

    def forward(self, x):
        for up, conv in zip(self.ups, self.convs):
            # print("origin")
            # print(x.shape)
            x = up(x)
            # print(x.shape)
            x = conv(x)
            # print(x.shape)

        x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        return self.out_conv(x)
        
class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, output_size=224):
        super(UpsampleConvBlock, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels, kernel=3)
        self.output_size = output_size

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        return x
        
class HybridSegModel(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 2, output_size = 224, layers_num = 5, dropout_rate = 0.0):
        super(HybridSegModel, self).__init__()

        ch_list=[128, 256, 320, 640, 1024]
        gr_list=[14, 16, 20, 40, 160]
        n_layers=[8, 16, 16, 16, 4]
        pool_layer=[1, 0, 1, 1, 0]
        
        self.layers_num = layers_num
        self.backbone = HarDNetBackbone(in_channels, ch_list = ch_list[:layers_num], gr_list = gr_list[:layers_num], n_layers = n_layers[:layers_num], pool_layer = pool_layer[:layers_num], drop_rate = dropout_rate)

        n_attention = layers_num + 1
        pmfs_in_channels = self.backbone.attention_channels
        self.pmfs = PMFS(n_inputs = n_attention, in_channels = pmfs_in_channels)

        decoder_in_channels = self.pmfs.attention_in_channels
        decoder_out_channels = 32
        decoder_layers = layers_num - 1
        self.decoder = Decoder(in_channels = decoder_in_channels, out_channels = decoder_out_channels, output_size = output_size, layers_num = decoder_layers)
        
        self.upsample_list = nn.ModuleList([
            UpsampleConvBlock(64, out_channels=32, output_size=output_size),
            UpsampleConvBlock(ch_list[0], out_channels=32, output_size=output_size),
            UpsampleConvBlock(ch_list[1], out_channels=32, output_size=output_size),
            UpsampleConvBlock(ch_list[2], out_channels=32, output_size=output_size),
            UpsampleConvBlock(ch_list[3], out_channels=32, output_size=output_size),
            UpsampleConvBlock(ch_list[4], out_channels=32, output_size=output_size)
        ])

        final_in_channels = self.layers_num * 32 + decoder_out_channels
        self.final_conv = nn.Conv2d(final_in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        attention_list = self.backbone(x)
        attention_upsample = []
        for i in range(self.layers_num):
            upsample = self.upsample_list[i](attention_list[i])
            attention_upsample.append(upsample)
        pmfs_out = self.pmfs(attention_list)
        out = self.decoder(pmfs_out)

        attention_upsample_cat = torch.cat(attention_upsample, axis = 1)
        out_cat = torch.cat([attention_upsample_cat, out], axis = 1)

        out = self.final_conv(out_cat)
        
        return out