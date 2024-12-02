import pytorch_lightning as pl
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            # nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            # nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class UNet1D(pl.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512), lr=1e-3):
        super(UNet1D, self).__init__()
        self.lr = lr

        # Downsampling layers
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            self.pools.append(nn.MaxPool1d(kernel_size=2))
            in_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Upsampling layers
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2))
            self.up_convs.append(ConvBlock(feature * 2, feature))

        # Final layer
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for up, up_conv, skip in zip(self.ups, self.up_convs, skip_connections):
            x = up(x)
            if x.size() != skip.size():
                x = nn.functional.interpolate(x, size=skip.size()[2:])
            x = torch.cat([skip, x], dim=1)
            x = up_conv(x)

        return self.final_conv(x)

    def training_step(self, batch):
        return self.generic_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self.generic_step(batch, stage='val')

    def generic_step(self, batch, stage='train'):
        x_masked = torch.cat((batch['masked_poses'], batch['mask']), dim=-1)
        y = batch['poses']
        x_masked = x_masked.permute(0, 2, 1)  # (BATCH, N_FRAMES, JOINTS) -> (BATCH, JOINTS, N_FRAMES)
        y_pred = self(x_masked)
        y_pred = y_pred.permute(0, 2, 1)  # (BATCH, JOINTS, N_FRAMES) -> (BATCH, N_FRAMES, JOINTS)
        loss = nn.MSELoss()(y_pred, y)  # Reconstruction loss
        self.log(f"{stage}_loss", loss, batch_size=y.shape[0], prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
