

"""
The network has four parts:
  (A) A ResNet-18 CNN encoder that extracts multi-level local features.
  (B) A Multi-level ViT Module (MVTM): one ViT branch per encoder level,
      which refines each level's features by adding long-distance
      (global) context via self-attention.
  (C) A Multi-level Feature Fusion Module (MFFM): concatenates the
      refined multi-level features and the deepest CNN feature, then
      fuses them with a 1x1 conv.
  (D) A U-shape decoder that upsamples the fused feature and merges it
      with encoder features via skip connections to produce a per-pixel
      probability map.
 
  - We output a 1-channel logit map (ship vs background). The sigmoid is
    applied in the loss for numerical stability, not inside the model.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
 
 



#  CNN ENCODER — ResNet-18 backbone

class ResNet18Encoder(nn.Module):
    """
    Extracts multi-level features from the input image using ResNet-18.
 
    We tap the network at four stages, giving feature maps at strides
    4, 8, 16, 32. These multi-level features are what feed into the MVTM and the decoder.

    """
 
    def __init__(self, in_channels=1, pretrained=False):
        super().__init__()
        # Load a standard ResNet-18. pretrained=False because ImageNet
        # weights are for 3-channel natural images; our 1-channel thermal
        # data is different enough that we train from scratch 
        backbone = torchvision.models.resnet18(weights=None)
 
        # Replace the first conv to accept 1 channel instead of 3 
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
 
        # ResNet's four residual stages
        self.layer1 = backbone.layer1   # stride 4,  64 channels
        self.layer2 = backbone.layer2   # stride 8, 128 channels
        self.layer3 = backbone.layer3   # stride 16,256 channels
        self.layer4 = backbone.layer4   # stride 32,512 channels
 
        # Channel counts at each tapped level (used by later modules)
        self.feature_channels = [64, 128, 256, 512]
 
    def forward(self, x):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))   # stride 2
        x = self.maxpool(x)                      # stride 4
 
        f1 = self.layer1(x)    # stride 4,   64ch   (highest resolution)
        f2 = self.layer2(f1)   # stride 8,  128ch
        f3 = self.layer3(f2)   # stride 16, 256ch
        f4 = self.layer4(f3)   # stride 32, 512ch   (lowest resolution)
 
        # Return all four levels, shallow -> deep.
        return [f1, f2, f3, f4]
 
 



#  MULTI-LEVEL ViT MODULE (MVTM)

class ViTBranch(nn.Module):
    """
    A single Vision Transformer branch 
 
    Why this exists: Self-attention lets every token "see" every other token, i.e.
    it injects long-distance dependency.
 
    CRITICAL design point — PATCHIFY FIRST:
      Self-attention is O(N^2) in the number of tokens N. If we used one
      token per pixel, a stride-4 feature of a 512 input is 128x128 =
      16,384 tokens, and the attention matrix alone is ~16k^2 -> billions
      of entries -> out of memory. The paper avoids this by flattening
      features into PATCHES of size (P,P), giving a small fixed token
      grid. We do the same: every level is pooled to a fixed token_grid
      (e.g. 16x16 = 256 tokens) before attention, then upsampled back to
      its original size afterward. 
    """
 
    def __init__(self, channels, num_heads=4, mlp_ratio=2.0, token_grid=16):
        super().__init__()
        self.channels   = channels
        self.token_grid = token_grid                  # tokens per side
 
        # Pre-norm transformer block (more stable than post-norm).
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True)
 
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
 
        # Fixed-size learnable positional embedding for token_grid^2 tokens.
        # Because the token grid is fixed, this is a normal Parameter (no
        # lazy resizing), so it saves/loads cleanly in the state_dict.
        n_tokens = token_grid * token_grid
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
 
    def forward(self, x):
        B, C, H, W = x.shape
 
        # --- Patchify: pool the feature map down to a fixed token grid ---
        # Adaptive avg pool turns any HxW into token_grid x token_grid,
        # so the token count (and thus attention cost) is constant.
        g = self.token_grid
        pooled = F.adaptive_avg_pool2d(x, output_size=(g, g))   # (B,C,g,g)
 
        tokens = pooled.flatten(2).transpose(1, 2)              # (B, g*g, C)
        tokens = tokens + self.pos_embed
 
        # --- Self-attention block (Eq. 1): a = MSA(LN(x)) + x ---
        t = self.norm1(tokens)
        attn_out, _ = self.attn(t, t, t)                        # Q=K=V
        tokens = tokens + attn_out
 
        # --- MLP block (Eq. 2): b = MLP(LN(a)) + a ---
        tokens = tokens + self.mlp(self.norm2(tokens))
 
        # --- Unpatchify: reshape to (B,C,g,g) then upsample back to HxW ---
        # The attention output (global context) is broadcast back onto the
        # original resolution and ADDED to x as a residual, so we keep the
        # CNN's fine detail and inject the ViT's long-distance information.
        ctx = tokens.transpose(1, 2).reshape(B, C, g, g)
        ctx = F.interpolate(ctx, size=(H, W), mode='bilinear',
                            align_corners=False)
        return x + ctx                                          # residual
 



 
class MVTM(nn.Module):
    """
    Multi-level ViT Module: one ViTBranch per encoder level EXCEPT the
    deepest (paper applies ViT to levels 1..k-1 and keeps the deepest CNN
    feature Fk as-is, then fuses them all in the MFFM).
 
    So for our 4 encoder levels [f1,f2,f3,f4], we refine f1,f2,f3 with ViT
    and pass f4 straight through.
    """
 
    def __init__(self, feature_channels, num_heads=4):
        super().__init__()
        # One ViT branch per level except the last.
        self.branches = nn.ModuleList([
            ViTBranch(ch, num_heads=num_heads)
            for ch in feature_channels[:-1]          # f1, f2, f3
        ])
 
    def forward(self, features):
        # features = [f1, f2, f3, f4]
        refined = []
        for i, branch in enumerate(self.branches):
            refined.append(branch(features[i]))      # refine f1,f2,f3
        refined.append(features[-1])                 # keep f4 unchanged
        return refined                               # [v1, v2, v3, f4]
 
 



#  MULTI-LEVEL FEATURE FUSION MODULE (MFFM)

class MFFM(nn.Module):
    """
    Fuses the multi-level refined features 
        Mk = Conv1x1( Concat(Fk, V_{k-1}, ..., V_1) )
 
    All levels have different spatial sizes, so before concatenating we
    resize every level to the deepest level's spatial size (Hk x Wk),
    matching the paper's notation where each Vi is mapped to Hk x Wk.
 
    The 1x1 conv then mixes the concatenated channels into a single fused
    feature carrying long-distance information at the deepest resolution.
    """
 
    def __init__(self, feature_channels):
        super().__init__()
        total_ch = sum(feature_channels)            # 64+128+256+512 = 960
        out_ch   = feature_channels[-1]             # fuse down to 512
        self.fuse = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, refined):
        # refined = [v1, v2, v3, f4]; target size = deepest level's size
        target_size = refined[-1].shape[-2:]
        resized = [
            F.interpolate(r, size=target_size, mode='bilinear',
                          align_corners=False)
            for r in refined
        ]
        cat = torch.cat(resized, dim=1)             # concat along channels
        return self.fuse(cat)                       # Mk
 
 



#   U-SHAPE DECODER

class DecoderBlock(nn.Module):
    """
    One decoder step 
        M_{i-1} = Conv( Concat[ F_{i-1}, Upsample(M_i) ] )
 
    Upsamples the running feature, concatenates the matching encoder
    feature via a skip connection (which restores fine spatial detail
    lost to downsampling — essential for describing tiny ship shapes),
    then convolves.
    """
 
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x, skip):
        # Upsample x to the skip connection's spatial size, then merge.
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
 



 
class MTUNet(nn.Module):
    """
    The full MTU-Net.
 
    Output: a single-channel LOGIT map at the input resolution.
    We do NOT apply sigmoid here — the loss applies it internally
    (BCEWithLogits-style) for numerical stability. At inference you call
    torch.sigmoid(output) to get probabilities.
    """
 
    def __init__(self, in_channels=1, num_heads=4):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels=in_channels)
        ch = self.encoder.feature_channels            # [64,128,256,512]
 
        self.mvtm = MVTM(ch, num_heads=num_heads)
        self.mffm = MFFM(ch)
 
        # Decoder: start from fused feature (512ch at stride 32) and walk
        # back up, merging encoder skips f3, f2, f1.
        self.dec3 = DecoderBlock(in_ch=ch[3], skip_ch=ch[2], out_ch=256)
        self.dec2 = DecoderBlock(in_ch=256,   skip_ch=ch[1], out_ch=128)
        self.dec1 = DecoderBlock(in_ch=128,   skip_ch=ch[0], out_ch=64)
 
        # Final 1x1 conv -> single logit channel.
        self.head = nn.Conv2d(64, 1, kernel_size=1)
 
    def forward(self, x):
        input_size = x.shape[-2:]
 
        feats   = self.encoder(x)        # [f1,f2,f3,f4]
        refined = self.mvtm(feats)       # [v1,v2,v3,f4]
        fused   = self.mffm(refined)     # Mk, 512ch @ stride 32
 
        # Decoder with skip connections from the ORIGINAL encoder features
        d3 = self.dec3(fused, feats[2])  # merge f3
        d2 = self.dec2(d3,    feats[1])  # merge f2
        d1 = self.dec1(d2,    feats[0])  # merge f1
 
        logits = self.head(d1)
 
        # Upsample to the input resolution so output matches the mask size.
        logits = F.interpolate(logits, size=input_size, mode='bilinear',
                               align_corners=False)
        return logits                    # (B, 1, H, W) logits
 



 
if __name__ == '__main__':
    # Quick shape test with a dummy 1-channel 512x512 batch.
    model = MTUNet(in_channels=1)
    x = torch.randn(2, 1, 512, 512)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    print(f"Params: {n_params/1e6:.2f} M")