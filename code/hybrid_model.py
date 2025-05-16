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

class HarDNetBackbone(nn.Module):
    def __init__(self, in_channels = 1):
        super(HarDNetBackbone, self).__init__()

        base_out_ch = [32, 64]
        self.base_conv_1 = ConvLayer(in_channels = in_channels, out_channels = base_out_ch[0], kernel = 3, stride = 2,  bias = False)
        self.base_conv2 = ConvLayer(in_channels = base_out_ch[0], out_channels = base_out_ch[1],  kernel = 3)
        self.base_max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        
        grmul = 1.7
        drop_rate = 0.1
        ch_list = [  128, 256, 320, 640, 1024]
        gr       = [  14, 16, 20, 40,160]
        n_layers = [   8, 16, 16, 16,  4]
        # downSamp = [   1,  0,  1,  1,  0]

        encoder_block1_input_ch = 64
        self.encoder_block1 = EncoderBlock(encoder_block1_input_ch, gr[0], grmul, n_layers[0], ch_list[0])
        self.encoder_max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        encoder_block2_input_ch = ch_list[0]
        self.encoder_block2 = EncoderBlock(encoder_block2_input_ch, gr[1], grmul, n_layers[1], ch_list[1])
        encoder_block3_input_ch = ch_list[1]
        self.encoder_block3 = EncoderBlock(encoder_block3_input_ch, gr[2], grmul, n_layers[2], ch_list[2])
        self.encoder_max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        encoder_block4_input_ch = ch_list[2]
        self.encoder_block4 = EncoderBlock(encoder_block4_input_ch, gr[3], grmul, n_layers[3], ch_list[3])
        self.encoder_max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        encoder_block5_input_ch = ch_list[3]
        self.encoder_block5 = EncoderBlock(encoder_block5_input_ch, gr[4], grmul, n_layers[4], ch_list[4])


        self.attention_channels = [base_out_ch[1], ch_list[0], ch_list[2], ch_list[3], ch_list[3], ch_list[4]]

    def forward(self, x):
        attention_list = []
        x = self.base_conv_1(x)
        x = self.base_conv2(x)
        # print(x.shape)
        attention_list.append(x)
        x1 = self.base_max_pool(x)

        x1 = self.encoder_block1(x1)
        # print(x1.shape)
        attention_list.append(x1)
        x2 = self.encoder_max_pool1(x1)

        x2 = self.encoder_block2(x2)
        x2 = self.encoder_block3(x2)
        # print(x2.shape)
        attention_list.append(x2)
        x3 = self.encoder_max_pool2(x2)

        x3 = self.encoder_block4(x3)
        # print(x3.shape)
        attention_list.append(x3)
        x4 = self.encoder_max_pool3(x3)
        # print(x4.shape)
        attention_list.append(x4)
        
        x5 = self.encoder_block5(x4)
        # print(x5.shape)
        attention_list.append(x5)


        # dimension = [
        #     x : base_out_ch[1], W/2, H/2
        #     x1 : ch_list[0], W/4, H/4
        #     x2 : ch_list[2], W/8, H/8
        #     x3 : ch_list[3], W/16, H/16
        #     x4 : ch_list[3], W/32, W/32
        #     x5 : ch_list[4], W/32, W/32
        # ]
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
        self.amff_out_channels = [12, 12, 12, 12, 12]
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

class Decoder(nn.Module):
    def __init__(self, in_channels=60, out_channels=1, output_size = 224):
        super(Decoder, self).__init__()

        self.up1 = nn.ConvTranspose2d(in_channels, 64, kernel_size=2, stride=2)  # (20 → 40)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # (40 → 80)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # (80 → 160)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)   # (160 → 320)
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(8, out_channels, kernel_size=1)  # map to [B, 1, 224, 224]

        self.output_size = output_size
    def forward(self, x):
        x = self.up1(x)       # [B, 64, 40, 40]
        x = self.conv1(x)
        x = self.up2(x)       # [B, 32, 80, 80]
        x = self.conv2(x)
        x = self.up3(x)       # [B, 16, 160, 160]
        x = self.conv3(x)
        x = self.up4(x)       # [B, 8, 320, 320]
        x = self.conv4(x)

        x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        return self.out_conv(x)  # [B, 1, 224, 224]

class HybridSegModel(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 2, output_size = 224):
        super(HybridSegModel, self).__init__()
        self.backbone = HarDNetBackbone(in_channels)

        n_attention = 5
        pmfs_in_channels = self.backbone.attention_channels
        self.pmfs = PMFS(n_inputs = n_attention, in_channels = pmfs_in_channels)

        decoder_in_channels = self.pmfs.attention_in_channels
        self.decoder = Decoder(in_channels = decoder_in_channels, out_channels = out_channels, output_size = output_size)


    def forward(self, x):
        attention_list = self.backbone(x)
        pmfs_out = self.pmfs(attention_list)
        out = self.decoder(pmfs_out)
        return out