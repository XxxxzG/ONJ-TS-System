import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=1, num_class=2):
        super().__init__()

        ##################
        # Encoder
        ##################
        # (batch_size,in_channels,320,320) -> (batch_size,64,320,320)
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        # (batch_size,64,320,320) -> (batch_size,128,160,160)
        self.encoder_layer2 = nn.Sequential(
            nn.MaxPool2d(2),
            # (batch_size,64,160,160)
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # (batch_size,128,160,160)
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

        # (batch_size,128,160,160) -> (batch_size,256,80,80)
        self.encoder_layer3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        # (batch_size,256,80,80) —> (batch_size,512,40,40)
        self.encoder_layer4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        # (batch_size,512,40,40) —> (batch_size,1024,20,20)
        self.encoder_layer5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
        )

        ##################
        # Decoder
        ##################
        # (batch_size,1024,20,20) -> (batch_size,512,40,40)
        self.decoder_layer5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        # (batch_size,1024,40,40) -> (batch_size,256,80,80)
        self.decoder_layer4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        )

        self.decoder_layer3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )

        self.decoder_layer2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )

        self.decoder_layer1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_class, 1, padding=0)
        )


    def forward(self, x):
        # Encoder
        f1 = self.encoder_layer1(x)     # (bs,in_channels,320,320)
        f2 = self.encoder_layer2(f1)
        f3 = self.encoder_layer3(f2)
        f4 = self.encoder_layer4(f3)
        f5 = self.encoder_layer5(f4)
        # Decoder
        y = self.decoder_layer5(f5)
        y = torch.cat([f4, y], dim=1)
        y = self.decoder_layer4(y)
        y = torch.cat([f3, y], dim=1)
        y = self.decoder_layer3(y)
        y = torch.cat([f2, y], dim=1)
        y = self.decoder_layer2(y)
        y = torch.cat([f1, y], dim=1)
        y = self.decoder_layer1(y)

        return y


if __name__=='__main__':
    # 1. data
    images = torch.rand(8,1,320,320).cuda()

    # 2. model
    model = UNet(in_channels=1, num_class=2).cuda()

    # 3. forward
    logits = model(images)
    print(logits.shape)