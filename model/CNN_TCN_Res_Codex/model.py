import torch
import torch.nn.functional as F
import torch.nn as nn


def _make_norm3d(norm: str, channels: int) -> nn.Module:
    norm = norm.lower()
    if norm == "batch":
        return nn.BatchNorm3d(channels)
    if norm == "group":
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unknown 3D norm: {norm}")


class InputChannelPad2d(nn.Module):
    def __init__(
        self,
        padding: int = 1,
        ct_value: float = -1.024,
        proj_value: float = 0.0,
        inverse_square_value: float = 0.0,
        wed_value: float = 0.0,
    ):
        super().__init__()
        self.padding = padding
        self.pad_values = (ct_value, proj_value, inverse_square_value, wed_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding <= 0:
            return x
        if not 1 <= x.shape[1] <= len(self.pad_values):
            raise ValueError(f"InputChannelPad2d expects 1-{len(self.pad_values)} channels, got {x.shape[1]}")
        pad = int(self.padding)
        batch, channels, height, width = x.shape
        out = x.new_empty((batch, channels, height + 2 * pad, width + 2 * pad))
        for channel_idx in range(channels):
            out[:, channel_idx].fill_(float(self.pad_values[channel_idx]))
        out[:, :, pad : pad + height, pad : pad + width].copy_(x)
        return out


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        temporal_dilation: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        temporal_mode: str = "symmetric",
        dropout: float = 0.0,
        norm: str = "group",
    ):
        super().__init__()
        kt, kh, kw = kernel_size

        temporal_mode = temporal_mode.lower()
        if temporal_mode not in ("symmetric", "causal"):
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")
        if temporal_mode == "symmetric" and kt % 2 == 0:
            raise ValueError("Symmetric temporal convolution expects an odd temporal kernel size.")

        self.temporal_mode = temporal_mode
        self.temporal_pad = temporal_dilation * (kt // 2)
        self.causal_temporal_pad = temporal_dilation * (kt - 1)

        conv_padding = (
            self.temporal_pad if temporal_mode == "symmetric" else 0,
            kh // 2,
            kw // 2,
        )

        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=conv_padding,
            dilation=(temporal_dilation, 1, 1),
            bias=False,
        )
        self.norm1 = _make_norm3d(norm, channels)
        self.activation1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = _make_norm3d(norm, channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.temporal_mode == "causal":
            x = F.pad(x, (0, 0, 0, 0, self.causal_temporal_pad, 0))
        x = self.conv(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.pointwise(x)
        x = self.norm2(x)
        return self.activation(residual + x)


class TemporalConvStack(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        kernel_size: tuple[int, int, int] | tuple[tuple[int, int, int], ...] = (
            (3, 3, 3),
            (3, 3, 3),
            (3, 1, 1),
            (3, 3, 3),
            (3, 1, 1),
            (3, 3, 3),
            (3, 1, 1),
        ),
        temporal_modes: tuple[str, ...] = (
            "symmetric",
            "symmetric",
            "causal",
            "causal",
            "causal",
            "causal",
            "causal",
        ),
        dropout: float = 0.0,
        norm: str = "group",
    ):
        super().__init__()
        if len(kernel_size) > 0 and isinstance(kernel_size[0], int):
            kernel_sizes = tuple(kernel_size for _ in dilations)  # type: ignore[misc]
        else:
            kernel_sizes = kernel_size  # type: ignore[assignment]
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_size must be one tuple or one tuple per dilation.")
        if len(temporal_modes) != len(dilations):
            raise ValueError("temporal_modes must have the same length as dilations.")

        self.pre = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(
                    channels=channels,
                    temporal_dilation=dilation,
                    kernel_size=block_kernel_size,
                    temporal_mode=temporal_mode,
                    dropout=dropout,
                    norm=norm,
                )
                for dilation, block_kernel_size, temporal_mode in zip(
                    dilations, kernel_sizes, temporal_modes
                )
            ]
        )
        self.post = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_norm3d(norm, channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.blocks(x)
        return self.post(x)


class DualBranchFiLM(nn.Module):
    def __init__(self, channels: int, use_aperture: bool = True, use_angle: bool = True):
        super().__init__()
        self.channels = int(channels)
        self.use_aperture = bool(use_aperture)
        self.use_angle = bool(use_angle)
        self.collect_diagnostics = True
        self.aperture_head = nn.Linear(32, 2 * self.channels)
        self.angle_head = nn.Linear(32, 2 * self.channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.aperture_head.weight)
        nn.init.zeros_(self.aperture_head.bias)
        nn.init.zeros_(self.angle_head.weight)
        nn.init.zeros_(self.angle_head.bias)

    def _zero_delta(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = embedding.new_zeros((embedding.shape[0], self.channels))
        return zeros, zeros

    def _branch_delta(
        self,
        head: nn.Linear,
        embedding: torch.Tensor,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not enabled:
            return self._zero_delta(embedding)
        gamma_beta = head(embedding)
        return gamma_beta.chunk(2, dim=1)

    @staticmethod
    def _stats(prefix: str, gamma: torch.Tensor, beta: torch.Tensor) -> dict[str, torch.Tensor]:
        values = {}
        for name, tensor in (("gamma", gamma), ("beta", beta)):
            detached = tensor.detach()
            values[f"{prefix}_{name}_mean"] = detached.mean()
            values[f"{prefix}_{name}_std"] = detached.std(unbiased=False)
            values[f"{prefix}_{name}_abs_max"] = detached.abs().max()
        return values

    def forward(
        self,
        x: torch.Tensor,
        aperture_embedding: torch.Tensor,
        angle_embedding: torch.Tensor,
        batch: int | None = None,
        time: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ap_gamma, ap_beta = self._branch_delta(self.aperture_head, aperture_embedding, self.use_aperture)
        angle_gamma, angle_beta = self._branch_delta(self.angle_head, angle_embedding, self.use_angle)
        gamma = ap_gamma + angle_gamma
        beta = ap_beta + angle_beta

        if x.ndim == 5:
            gamma_view = gamma.view(gamma.shape[0], self.channels, 1, 1, 1)
            beta_view = beta.view(beta.shape[0], self.channels, 1, 1, 1)
        elif x.ndim == 4:
            if batch is None or time is None:
                raise ValueError("4D FiLM requires batch and time for [B,Z] flattening.")
            gamma_view = gamma[:, None, :].expand(batch, time, self.channels).reshape(batch * time, self.channels, 1, 1)
            beta_view = beta[:, None, :].expand(batch, time, self.channels).reshape(batch * time, self.channels, 1, 1)
        else:
            raise ValueError(f"DualBranchFiLM expects 4D or 5D features, got shape {tuple(x.shape)}")

        if self.collect_diagnostics:
            stats = {}
            stats.update(self._stats("aperture", ap_gamma, ap_beta))
            stats.update(self._stats("angle", angle_gamma, angle_beta))
            stats.update(self._stats("combined", gamma, beta))
        else:
            stats = {}
        return x * (1.0 + gamma_view) + beta_view, stats


class TransposeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
        )

    def forward(
        self,
        x: torch.Tensor,
        e50: torch.Tensor | None = None,
        e100: torch.Tensor | None = None,
        batch: int | None = None,
        time: int | None = None,
        aperture_embedding: torch.Tensor | None = None,
        angle_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(x)


class ZMixer2dSequence(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("ZMixer2dSequence expects a positive odd kernel size.")
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=channels,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.conv.kernel_size == (5, 1, 1):
            with torch.no_grad():
                kernel = torch.tensor(
                    [0.05, 0.20, 0.50, 0.20, 0.05],
                    dtype=self.conv.weight.dtype,
                    device=self.conv.weight.device,
                )
                self.conv.weight.zero_()
                self.conv.weight[:, 0, :, 0, 0] = kernel
                if self.conv.bias is not None:
                    self.conv.bias.zero_()
        else:
            nn.init.dirac_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, batch: int, time: int) -> torch.Tensor:
        _, channels, height, width = x.shape
        x = x.reshape(batch, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.reshape(batch * time, channels, height, width)


class ConvActAvgPoolDownsample2d(nn.Module):
    """Separate feature extraction from stride-2 downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.activation(self.conv(x)))


class BilinearOutputHead(nn.Module):
    def __init__(self, zero_init_output: bool = False):
        super().__init__()
        self.up200 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv200 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.act200 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.out = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        if zero_init_output:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_after_upsample(self.up200(x))

    def forward_after_upsample(self, x_up200: torch.Tensor) -> torch.Tensor:
        x = self.act200(self.conv200(x_up200))
        return self.out(x)


class SharedSplitOutputHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.up200 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv200 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bev_out = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.residual_out = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.activation(self.conv200(self.up200(x)))
        base_feature, residual_feature = torch.chunk(x, 2, dim=1)
        return self.bev_out(base_feature), self.residual_out(residual_feature)


class BilinearDecoder(nn.Module):
    def __init__(
        self,
        use_skips: bool = False,
        skip_gate_init: float = 0.1,
        use_skip_z_mixer: bool = False,
        skip_z_mixer_kernel: int = 5,
        use_skip50: bool = True,
        use_skip100: bool = True,
        film32: DualBranchFiLM | None = None,
        film16: DualBranchFiLM | None = None,
        output_head_mode: str = "separate_standard",
    ):
        super().__init__()
        self.use_skips = use_skips
        self.use_skip50 = bool(use_skip50)
        self.use_skip100 = bool(use_skip100)
        self.use_skip_z_mixer = use_skip_z_mixer
        self.output_head_mode = output_head_mode.lower().strip()
        self.film32 = film32
        self.film16 = film16
        self.collect_diagnostics = True
        self.share_separate_head_upsample = True
        self.last_film_stats: dict[str, torch.Tensor] = {}
        self.last_skip_stats: dict[str, torch.Tensor] = {}
        if self.output_head_mode not in ("separate_standard", "shared_split"):
            raise ValueError(
                "output_head_mode must be 'separate_standard' or 'shared_split', "
                f"got {output_head_mode!r}"
            )

        self.up50 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv50 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.act50 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.up100 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv100 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.act100 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if self.output_head_mode == "shared_split":
            self.output_heads = SharedSplitOutputHeads()
            self.bev_head = None
            self.residual_head = None
        else:
            self.output_heads = None
            self.bev_head = BilinearOutputHead()
            self.residual_head = BilinearOutputHead(zero_init_output=True)

        if use_skips:
            self.skip50_proj = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
            self.skip100_proj = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1)
            self.skip50_alpha = nn.Parameter(torch.tensor(float(skip_gate_init)))
            self.skip100_alpha = nn.Parameter(torch.tensor(float(skip_gate_init)))
            self.skip50_z_mixer = (
                ZMixer2dSequence(channels=32, kernel_size=skip_z_mixer_kernel)
                if use_skip_z_mixer
                else nn.Identity()
            )
            self.skip100_z_mixer = (
                ZMixer2dSequence(channels=16, kernel_size=skip_z_mixer_kernel)
                if use_skip_z_mixer
                else nn.Identity()
            )
        else:
            self.skip50_proj = None
            self.skip100_proj = None
            self.skip50_z_mixer = None
            self.skip100_z_mixer = None
            self.register_parameter("skip50_alpha", None)
            self.register_parameter("skip100_alpha", None)

    def forward(
        self,
        x: torch.Tensor,
        e50: torch.Tensor | None = None,
        e100: torch.Tensor | None = None,
        batch: int | None = None,
        time: int | None = None,
        aperture_embedding: torch.Tensor | None = None,
        angle_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.collect_diagnostics:
            self.last_film_stats = {}
            self.last_skip_stats = {}
        if (self.film32 is not None or self.film16 is not None) and (
            aperture_embedding is None or angle_embedding is None
        ):
            raise ValueError("BilinearDecoder FiLM requires aperture_embedding and angle_embedding.")
        x = self.conv50(self.up50(x))
        if self.use_skips and self.use_skip50:
            if e50 is None or self.skip50_proj is None or self.skip50_alpha is None:
                raise ValueError("BilinearDecoder skip50 is enabled but E50 was not provided.")
            skip50 = self.skip50_proj(e50)
            if self.use_skip_z_mixer:
                if batch is None or time is None or self.skip50_z_mixer is None:
                    raise ValueError("Skip z mixer requires batch and time dimensions.")
                skip50 = self.skip50_z_mixer(skip50, batch, time)
            skip50_addend = self.skip50_alpha * skip50
            if self.collect_diagnostics:
                skip50_abs_mean = skip50_addend.detach().abs().mean()
                feature_abs_mean = x.detach().abs().mean()
                self.last_skip_stats["skip50_alpha"] = self.skip50_alpha.detach()
                self.last_skip_stats["skip50_addend_abs_mean"] = skip50_abs_mean
                self.last_skip_stats["skip50_feature_abs_mean"] = feature_abs_mean
                self.last_skip_stats["skip50_addend_to_feature_abs_ratio"] = (
                    skip50_abs_mean / torch.clamp(feature_abs_mean, min=1e-12)
                )
            x = x + skip50_addend
        if self.film32 is not None:
            x, stats = self.film32(x, aperture_embedding, angle_embedding, batch=batch, time=time)
            if self.collect_diagnostics:
                self.last_film_stats.update({f"film32/{key}": value for key, value in stats.items()})
        x = self.act50(x)

        x = self.conv100(self.up100(x))
        if self.use_skips and self.use_skip100:
            if e100 is None or self.skip100_proj is None or self.skip100_alpha is None:
                raise ValueError("BilinearDecoder skip100 is enabled but E100 was not provided.")
            skip100 = self.skip100_proj(e100)
            if self.use_skip_z_mixer:
                if batch is None or time is None or self.skip100_z_mixer is None:
                    raise ValueError("Skip z mixer requires batch and time dimensions.")
                skip100 = self.skip100_z_mixer(skip100, batch, time)
            skip100_addend = self.skip100_alpha * skip100
            if self.collect_diagnostics:
                skip100_abs_mean = skip100_addend.detach().abs().mean()
                feature_abs_mean = x.detach().abs().mean()
                self.last_skip_stats["skip100_alpha"] = self.skip100_alpha.detach()
                self.last_skip_stats["skip100_addend_abs_mean"] = skip100_abs_mean
                self.last_skip_stats["skip100_feature_abs_mean"] = feature_abs_mean
                self.last_skip_stats["skip100_addend_to_feature_abs_ratio"] = (
                    skip100_abs_mean / torch.clamp(feature_abs_mean, min=1e-12)
                )
            x = x + skip100_addend
        if self.film16 is not None:
            x, stats = self.film16(x, aperture_embedding, angle_embedding, batch=batch, time=time)
            if self.collect_diagnostics:
                self.last_film_stats.update({f"film16/{key}": value for key, value in stats.items()})
        x = self.act100(x)

        if self.output_head_mode == "shared_split":
            if self.output_heads is None:
                raise RuntimeError("shared_split output head is not initialized.")
            return self.output_heads(x)
        if self.share_separate_head_upsample:
            if self.bev_head is None or self.residual_head is None:
                raise RuntimeError("separate_standard output heads are not initialized.")
            x_up200 = self.bev_head.up200(x)
            return self.bev_head.forward_after_upsample(x_up200), self.residual_head.forward_after_upsample(x_up200)
        return self.bev_head(x), self.residual_head(x)


def _make_decoder(
    decoder_type: str,
    use_skips: bool = False,
    skip_gate_init: float = 0.1,
    use_skip_z_mixer: bool = False,
    skip_z_mixer_kernel: int = 5,
    use_skip50: bool = True,
    use_skip100: bool = True,
    film32: DualBranchFiLM | None = None,
    film16: DualBranchFiLM | None = None,
    output_head_mode: str = "separate_standard",
) -> nn.Module:
    decoder_type = decoder_type.lower()
    if use_skips and decoder_type != "bilinear":
        raise ValueError("Decoder skips are only implemented for decoder_type='bilinear'.")
    if use_skip_z_mixer and not use_skips:
        raise ValueError("Skip z mixer requires use_skips=True.")
    if decoder_type == "transpose":
        if film32 is not None or film16 is not None:
            raise ValueError("FiLM is only implemented for decoder_type='bilinear'.")
        return TransposeDecoder()
    if decoder_type == "bilinear":
        return BilinearDecoder(
            use_skips=use_skips,
            skip_gate_init=skip_gate_init,
            use_skip_z_mixer=use_skip_z_mixer,
            skip_z_mixer_kernel=skip_z_mixer_kernel,
            use_skip50=use_skip50,
            use_skip100=use_skip100,
            film32=film32,
            film16=film16,
            output_head_mode=output_head_mode,
        )
    raise ValueError(f"Unknown decoder_type: {decoder_type}")


class CNN_TCN_Res_Codex(nn.Module):
    """CNN/TCN network with a BEV dose branch and a residual correction branch."""

    def __init__(
        self,
        input_h: int = 200,
        input_w: int = 200,
        tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        tcn_kernel_size: tuple[int, int, int] | tuple[tuple[int, int, int], ...] = (
            (3, 3, 3),
            (3, 3, 3),
            (3, 1, 1),
            (3, 3, 3),
            (3, 1, 1),
            (3, 3, 3),
            (3, 1, 1),
        ),
        tcn_temporal_modes: tuple[str, ...] = (
            "symmetric",
            "symmetric",
            "causal",
            "causal",
            "causal",
            "causal",
            "causal",
        ),
        tcn_dropout: float = 0.0,
        tcn_norm: str = "group",
        decoder_type: str = "transpose",
        use_input_channel_padding: bool = False,
        input_channels: int = 2,
        input_ct_pad_value: float = -1.024,
        input_proj_pad_value: float = 0.0,
        input_inverse_square_pad_value: float = 0.0,
        input_wed_pad_value: float = 0.0,
        use_decoder_skips: bool = False,
        decoder_skip_gate_init: float = 0.1,
        use_skip_z_mixer: bool = False,
        skip_z_mixer_kernel: int = 5,
        use_skip50: bool = True,
        use_skip100: bool = True,
        use_aperture_film: bool = True,
        use_angle_film: bool = True,
        encoder_downsample_mode: str = "stride_conv",
        output_head_mode: str = "separate_standard",
    ):
        super().__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.decoder_type = decoder_type.lower()
        self.input_channels = input_channels
        self.use_input_channel_padding = use_input_channel_padding
        self.use_decoder_skips = use_decoder_skips
        self.use_aperture_film = bool(use_aperture_film)
        self.use_angle_film = bool(use_angle_film)
        self.encoder_downsample_mode = encoder_downsample_mode.lower().strip()
        self.output_head_mode = output_head_mode.lower().strip()
        self.collect_diagnostics = True
        self.use_channels_last_2d = False
        self.use_channels_last_3d = False
        self.last_film_stats: dict[str, torch.Tensor] = {}
        self.last_skip_stats: dict[str, torch.Tensor] = {}
        if not 2 <= self.input_channels <= 4:
            raise ValueError(f"CNN_TCN_Res_Codex expects input_channels 2-4, got {self.input_channels}")
        if self.encoder_downsample_mode not in ("stride_conv", "conv_avgpool"):
            raise ValueError(
                "encoder_downsample_mode must be 'stride_conv' or 'conv_avgpool', "
                f"got {encoder_downsample_mode!r}"
            )

        first_conv_padding = 0 if use_input_channel_padding else 1
        input_pad_layer: nn.Module = (
            InputChannelPad2d(
                padding=1,
                ct_value=input_ct_pad_value,
                proj_value=input_proj_pad_value,
                inverse_square_value=input_inverse_square_pad_value,
                wed_value=input_wed_pad_value,
            )
            if use_input_channel_padding
            else nn.Identity()
        )

        if self.encoder_downsample_mode == "stride_conv":
            downsample_100_to_50 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
            downsample_100_to_50_activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            downsample_50_to_25 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
            downsample_50_to_25_activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            downsample_100_to_50 = ConvActAvgPoolDownsample2d(in_channels=32, out_channels=64)
            downsample_100_to_50_activation = nn.Identity()
            downsample_50_to_25 = ConvActAvgPoolDownsample2d(in_channels=64, out_channels=64)
            downsample_50_to_25_activation = nn.Identity()

        self.encoder = nn.Sequential(
            input_pad_layer,
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=first_conv_padding,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            downsample_100_to_50,
            downsample_100_to_50_activation,
            downsample_50_to_25,
            downsample_50_to_25_activation,
        )

        self.tcn = TemporalConvStack(
            channels=64,
            dilations=tcn_dilations,
            kernel_size=tcn_kernel_size,
            temporal_modes=tcn_temporal_modes,
            dropout=tcn_dropout,
            norm=tcn_norm,
        )

        self.aperture_condition_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
        )
        self.angle_condition_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.SiLU(),
            nn.Linear(16, 32),
            nn.SiLU(),
        )
        self.film64 = DualBranchFiLM(64, use_aperture=self.use_aperture_film, use_angle=self.use_angle_film)
        self.film32 = DualBranchFiLM(32, use_aperture=self.use_aperture_film, use_angle=self.use_angle_film)
        self.film16 = DualBranchFiLM(16, use_aperture=self.use_aperture_film, use_angle=self.use_angle_film)

        self.decoder = _make_decoder(
            self.decoder_type,
            use_skips=use_decoder_skips,
            skip_gate_init=decoder_skip_gate_init,
            use_skip_z_mixer=use_skip_z_mixer,
            skip_z_mixer_kernel=skip_z_mixer_kernel,
            use_skip50=use_skip50,
            use_skip100=use_skip100,
            film32=self.film32,
            film16=self.film16,
            output_head_mode=self.output_head_mode,
        )

    def _validate_condition_features(
        self,
        name: str,
        features: torch.Tensor,
        batch: int,
        reference: torch.Tensor,
    ) -> None:
        if features.shape != (batch, 2):
            raise ValueError(f"{name} must have shape [B,2], got {tuple(features.shape)}")
        if not torch.is_floating_point(features):
            raise TypeError(f"{name} must be floating point, got {features.dtype}")
        if features.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}, got {features.device}")
        if self.collect_diagnostics and not torch.isfinite(features).all():
            raise ValueError(f"{name} contains NaN/Inf")

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor | None = None,
        x4: torch.Tensor | None = None,
        *,
        aperture_features: torch.Tensor,
        angle_features: torch.Tensor,
    ) -> torch.Tensor:
        batch, time, height, width = x1.shape
        self._validate_condition_features("aperture_features", aperture_features, batch, x1)
        self._validate_condition_features("angle_features", angle_features, batch, x1)
        input_tensors = tuple(tensor for tensor in (x1, x2, x3, x4) if tensor is not None)
        if len(input_tensors) != self.input_channels:
            raise ValueError(f"Model was built for {self.input_channels} channels, got {len(input_tensors)} inputs")

        x_combined = x1.new_empty((batch * time, self.input_channels, height, width))
        for channel_idx, tensor in enumerate(input_tensors):
            x_combined[:, channel_idx].copy_(tensor.reshape(batch * time, height, width))
        if self.use_channels_last_2d:
            x_combined = x_combined.contiguous(memory_format=torch.channels_last)

        e100 = None
        e50 = None
        x_encoded = x_combined
        for layer_idx, layer in enumerate(self.encoder):
            x_encoded = layer(x_encoded)
            if layer_idx == 4:
                e100 = x_encoded
            elif layer_idx == 6:
                e50 = x_encoded
        _, channels, h_enc, w_enc = x_encoded.shape

        x_tcn = x_encoded.reshape(batch, time, channels, h_enc, w_enc)
        x_tcn = x_tcn.permute(0, 2, 1, 3, 4).contiguous()
        if self.use_channels_last_3d:
            x_tcn = x_tcn.contiguous(memory_format=torch.channels_last_3d)
        aperture_embedding = self.aperture_condition_encoder(aperture_features)
        angle_embedding = self.angle_condition_encoder(angle_features)
        x_tcn = self.tcn(x_tcn)
        x_tcn, film64_stats = self.film64(x_tcn, aperture_embedding, angle_embedding)
        if self.collect_diagnostics:
            self.last_film_stats = {f"film64/{key}": value for key, value in film64_stats.items()}
        x_tcn = x_tcn.permute(0, 2, 1, 3, 4).contiguous()

        x_decoder_input = x_tcn.reshape(batch * time, channels, h_enc, w_enc)
        if self.use_channels_last_2d:
            x_decoder_input = x_decoder_input.contiguous(memory_format=torch.channels_last)
        bev_decoded, residual_decoded = self.decoder(
            x_decoder_input,
            e50=e50,
            e100=e100,
            batch=batch,
            time=time,
            aperture_embedding=aperture_embedding,
            angle_embedding=angle_embedding,
        )
        if self.collect_diagnostics and hasattr(self.decoder, "last_film_stats"):
            self.last_film_stats.update(self.decoder.last_film_stats)
        if self.collect_diagnostics and hasattr(self.decoder, "last_skip_stats"):
            self.last_skip_stats = dict(self.decoder.last_skip_stats)
        bev_output = bev_decoded.reshape(batch, time, self.input_h, self.input_w)
        residual_output = residual_decoded.reshape(batch, time, self.input_h, self.input_w)

        return bev_output, residual_output
