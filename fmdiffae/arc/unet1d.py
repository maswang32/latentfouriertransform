import torch
import torch.nn as nn
import torch.nn.functional as F
import math


"""
Peculiarities from EDM that are supported:
    - Biases are always initialized to zero
    - Attention in the decoder is only on the last of each group of blocks.
    - Attention in the encoder is on all blocks for resolutions that have it.
    - ConvTranspose scales the resampling kernel by 2.
    - One more decoder than encoder block per resolution.
    - Scale Attention Outprojection to 1e-5.

Things we can add to make it more like EDM:
    - Custom Initializations
    - Dropout
    - Different resampling filters
    - Last Conv Layer has 1e-5 initialization

Other Things we Support:
    - DAC-like dilated convolution

Other Things we can add:
    - Weight Normalization
"""


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias_init_scale=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            self.linear.bias.mul_(bias_init_scale)

    def forward(self, x):
        return self.linear(x)

    def __repr__(self):
        return self.linear.__repr__()


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        up=False,
        down=False,
        resample=True,
        weight_init_scale=1.0,
        bias_init_scale=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.up = up
        self.down = down
        self.resample = resample
        self.weight_init_scale = weight_init_scale
        self.bias_init_scale = bias_init_scale

        assert not (up and down)

        if (up or down) and resample:
            resample_filter = torch.ones(1, 1, 2) / 2
            resample_filter = resample_filter.expand(in_channels, 1, 2)
            self.register_buffer("resample_filter", resample_filter, persistent=False)

        # Actual Convolution
        if kernel_size:
            assert kernel_size % 2 == 1
            effective_kernel_size = (kernel_size - 1) * dilation + 1
            padding = effective_kernel_size // 2

            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            )

            # Initialize Weights and Biases
            with torch.no_grad():
                self.conv.weight.mul_(weight_init_scale)
                self.conv.bias.mul_(bias_init_scale)

    def forward(self, x):
        if self.up and self.resample:
            x = F.conv_transpose1d(
                x, 2 * self.resample_filter, stride=2, groups=self.in_channels
            )
        if self.down and self.resample:
            x = F.conv1d(x, self.resample_filter, stride=2, groups=self.in_channels)

        if self.kernel_size:
            x = self.conv(x)

        return x

    def __repr__(self):
        if self.kernel_size:
            return self.conv.__repr__()
        else:
            if self.up:
                return "UpResample()"
            if self.down:
                return "DownResample()"
            else:
                return "Identity()"


class GroupNorm(nn.Module):
    def __init__(self, num_channels, target_num_groups=32, min_channels_per_group=4):
        super().__init__()
        self.num_channels = num_channels
        # There can be at most (num_channels // min_channels_per_group) groups
        # To ensure there are at least min_channels_per_group channels per group
        self.num_groups = min(target_num_groups, num_channels // min_channels_per_group)
        if self.num_groups == 0:
            raise ValueError("Num. channels less than min. channels per group")
        self.gn = nn.GroupNorm(self.num_groups, num_channels, eps=1e-06)

    def forward(self, x):
        return self.gn(x)

    def __repr__(self):
        return self.gn.__repr__()


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        up=False,
        down=False,
        use_t=True,
        emb_dim=None,
        use_attention=False,
        num_heads=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.up = up
        self.down = down

        self.use_t = use_t
        self.emb_dim = emb_dim
        self.use_attention = use_attention
        self.num_heads = num_heads

        # Layers
        self.norm0 = GroupNorm(in_channels)
        self.conv0 = Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, up=up, down=down
        )

        self.norm1 = GroupNorm(out_channels)
        self.conv1 = Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=1,  # Like DAC
            weight_init_scale=1e-5,
        )

        if self.use_t:
            self.emb_proj = Linear(emb_dim, out_channels)

        # Skip
        if (in_channels != out_channels) or up or down:
            k = None if (in_channels == out_channels) else 1
            self.skip_proj = Conv1d(
                in_channels, out_channels, kernel_size=k, down=down, up=up
            )
        else:
            self.skip_proj = nn.Identity()

        # Attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                out_channels,
                num_heads=num_heads,
                bias=False,
                batch_first=True,
                dtype=torch.float32,
            )
            # Zero-initialize attention out projection
            with torch.no_grad():
                self.attention.out_proj.weight.mul_(1e-5)

    def forward(self, x, emb):
        residual_stream = x

        x = self.conv0(F.silu(self.norm0(x)))
        if emb is not None:
            x = x + self.emb_proj(emb).unsqueeze(-1)
        x = self.conv1(F.silu(self.norm1(x)))

        x = (x + self.skip_proj(residual_stream)) / math.sqrt(2)

        if self.use_attention:
            x_attn = x.permute(0, 2, 1)  # B, C, T -> B, T, C
            x_attn, _ = self.attention(x_attn, x_attn, x_attn)
            x_attn = x_attn.to(x.dtype).permute(0, 2, 1)  # B, T, C -> B, C, T
            x = (x + x_attn) / math.sqrt(2)

        return x


class PositionalEncoding(nn.Module):
    """
    Postional Encoding, as in Attention is All You Need, DDPM++.

    Attributes:
        num_sinusoids (int): number of sinusodal features. Half will be sines,
            while the other half will be cosines. The frequencies of the sinusoids
            are logarithimically spaced, so adjacent frequencies have
            a common ratio.
        max_input_value (int): corresponds to 1 radian at the lowest frequency.
            At the lowest frequency, the maximum input maps to one radian.
    """

    def __init__(self, num_sinusoids=128, max_input_value=100):
        super().__init__()
        num_freqs = num_sinusoids // 2
        min_angular_frequency = 1 / max_input_value

        angular_freqs = torch.logspace(
            0, 1, steps=num_freqs, base=min_angular_frequency
        )
        self.register_buffer("angular_freqs", angular_freqs, persistent=False)

    def forward(self, x):
        x = x.view(-1, 1) * self.angular_freqs.reshape(1, -1)
        return torch.cat((x.cos(), x.sin()), dim=-1)


class EmbeddingNetwork(nn.Module):
    def __init__(self, num_sinusoids=128, emb_dim=512, max_input_value=100):
        super().__init__()
        self.map = PositionalEncoding(
            num_sinusoids=num_sinusoids, max_input_value=max_input_value
        )
        self.linear0 = Linear(in_features=num_sinusoids, out_features=emb_dim)
        self.linear1 = Linear(in_features=emb_dim, out_features=emb_dim)

    def forward(self, x):
        x = F.silu(self.linear0(self.map(x)))
        return F.silu(self.linear1(x))


class UNet1d(nn.Module):
    def __init__(
        self,
        data_resolution,
        in_channels=80,
        out_channels=80,
        model_dim=256,
        channel_mults=[1, 2, 2, 3],
        num_blocks_per_res=3,
        kernel_size=3,
        dilation_sequence=[1, 1, 1, 1],
        use_attention=True,
        attn_resolutions=[64],
        num_heads=1,
        use_t=True,
        emb_num_sinusoids=128,
        emb_dim_mult=4,
        max_t_value=100,
    ):
        super().__init__()
        # Decoder has num_blocks_per_res + 1 blocks
        assert len(dilation_sequence) == num_blocks_per_res + 1

        # Filling out Fields
        self.data_resolution = data_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model_dim = model_dim
        self.channel_mults = channel_mults
        self.num_blocks_per_res = num_blocks_per_res

        self.kernel_size = kernel_size
        self.dilation_sequence = dilation_sequence

        self.use_attention = use_attention
        self.attn_resolutions = attn_resolutions
        self.num_heads = num_heads

        self.use_t = use_t

        # Block kwargs
        self.block_kwargs = dict(
            kernel_size=kernel_size,
            num_heads=num_heads,
            use_t=self.use_t,
        )

        if use_t:
            self.emb_num_sinusoids = emb_num_sinusoids
            self.emb_dim_mult = emb_dim_mult
            self.max_pos_value = max_t_value

            # Embedding Network
            self.emb_dim = model_dim * emb_dim_mult
            self.block_kwargs["emb_dim"] = self.emb_dim

            self.emb_network = EmbeddingNetwork(
                num_sinusoids=emb_num_sinusoids,
                emb_dim=self.emb_dim,
                max_input_value=max_t_value,
            )

        # Number of Levels
        self.num_levels = len(channel_mults)

        self._build_encoder()
        self._build_decoder()

        # Print number of params
        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"UNet1d Number of Parameters: {self.num_params:,}")

    def forward(self, x, ts=None):
        if self.use_t:
            emb = self.emb_network(ts)
        else:
            emb = None

        enc_outs = {}

        for name, module in self.enc.items():
            x = module(x, emb) if isinstance(module, ConvBlock) else module(x)
            enc_outs[name] = x

        for name, module in self.dec.items():
            if name in self.skip_dict:
                x = torch.cat((x, enc_outs[self.skip_dict[name]]), dim=-2)
            x = module(x, emb) if isinstance(module, ConvBlock) else module(x)

        return x

    def _build_encoder(self):
        self.enc = torch.nn.ModuleDict()

        for level in range(self.num_levels):
            res = self.data_resolution >> level

            res_out_channels = self.channel_mults[level] * self.model_dim

            if level == 0:
                res_in_channels = self.model_dim

                self.enc[f"{res}_conv0"] = Conv1d(
                    in_channels=self.in_channels,
                    out_channels=res_in_channels,
                    kernel_size=self.kernel_size,
                )

            else:
                res_in_channels = self.channel_mults[level - 1] * self.model_dim

                self.enc[f"{res * 2}->{res}_down"] = ConvBlock(
                    in_channels=res_in_channels,
                    out_channels=res_in_channels,
                    down=True,
                    **self.block_kwargs,
                )

            for block_idx in range(self.num_blocks_per_res):
                block_in_channels = (
                    res_in_channels if block_idx == 0 else res_out_channels
                )

                self.enc[f"{res}_block{block_idx}"] = ConvBlock(
                    in_channels=block_in_channels,
                    out_channels=res_out_channels,
                    use_attention=(res in self.attn_resolutions) and self.use_attention,
                    dilation=self.dilation_sequence[block_idx],
                    **self.block_kwargs,
                )

    def _build_decoder(self):
        self.dec = torch.nn.ModuleDict()
        self.skip_dict = {}
        self.skips_from = list(self.enc.keys())

        for level in reversed(range(self.num_levels)):
            res = self.data_resolution >> level

            res_out_channels = self.model_dim * self.channel_mults[level]

            if level == self.num_levels - 1:
                res_in_channels = self.model_dim * self.channel_mults[level]

                self.dec[f"{res}_in0"] = ConvBlock(
                    in_channels=res_in_channels,
                    out_channels=res_in_channels,
                    use_attention=self.use_attention,
                    **self.block_kwargs,
                )

                self.dec[f"{res}_in1"] = ConvBlock(
                    in_channels=res_in_channels,
                    out_channels=res_in_channels,
                    use_attention=False,
                    **self.block_kwargs,
                )

            else:
                res_in_channels = self.model_dim * self.channel_mults[level + 1]

                self.dec[f"{res // 2}->{res}_up"] = ConvBlock(
                    in_channels=res_in_channels,
                    out_channels=res_in_channels,
                    use_attention=False,
                    up=True,
                    **self.block_kwargs,
                )

            for block_idx in range(self.num_blocks_per_res + 1):
                block_name = f"{res}_block{block_idx}"
                self.skip_dict[block_name] = self.skips_from.pop()

                if block_idx == 0:
                    block_in_channels = (
                        res_in_channels
                        + self.enc[self.skip_dict[block_name]].out_channels
                    )
                else:
                    block_in_channels = (
                        res_out_channels
                        + self.enc[self.skip_dict[block_name]].out_channels
                    )

                self.dec[block_name] = ConvBlock(
                    in_channels=block_in_channels,
                    out_channels=res_out_channels,
                    use_attention=(res in self.attn_resolutions)
                    and self.use_attention
                    and (block_idx == self.num_blocks_per_res),
                    dilation=self.dilation_sequence[block_idx],
                    **self.block_kwargs,
                )

            if level == 0:
                self.dec[f"{res}_outnorm"] = GroupNorm(res_out_channels)
                self.dec[f"{res}_outsilu"] = nn.SiLU()
                self.dec[f"{res}_outconv"] = Conv1d(
                    in_channels=res_out_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                )
