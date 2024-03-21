
import torch
from torch import nn
import torch.nn.functional as F

from ..basemodel import BaseModel


class _EncBlk_3d(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncBlk_3d, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(p=0.5))
        layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _Blk_3d(nn.Module):
    def __init__(self, in_channels, middle_channels):
        super(_Blk_3d, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, middle_channels, kernel_size=3, 
                      padding=1, bias=False),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.decode(x)

# class _DecBlk_3d(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super(_DecBlk_3d, self).__init__()
#         self.decode = nn.Sequential(
#             nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(middle_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(middle_channels),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=2, stride=2),
#         )

#     def forward(self, x):
#         return self.decode(x)



class UNet3plus(BaseModel):

    def __init__(self, in_channels, num_classes, ini_channels, deep_sup=False):
        self.deep_sup = deep_sup

        super(UNet3plus, self).__init__()
        self.enc1 = _EncBlk_3d(in_channels, ini_channels)
        self.enc2 = _EncBlk_3d(ini_channels, 2*ini_channels)
        self.enc3 = _EncBlk_3d(2*ini_channels, 4*ini_channels) # 8*ini_channels
        self.enc4 = _EncBlk_3d(4*ini_channels, 8*ini_channels, dropout=True)

        self.center = _Blk_3d(8*ini_channels, 16*ini_channels)
        self.center_up = nn.ConvTranspose3d(16*ini_channels, 8*ini_channels,
                                            kernel_size=4, stride=2, padding=1)
        
        dec4_in_dims = (16 + 1 + 2 + 4) * ini_channels
        self.dec4 = _Blk_3d(dec4_in_dims, 8*ini_channels)
        self.dec4_up = nn.ConvTranspose3d(8*ini_channels, 4*ini_channels,
                                          kernel_size=4, stride=2, padding=1)
        
        dec3_in_dims = (8 + 1 + 2 + 16) * ini_channels
        self.dec3 = _Blk_3d(dec3_in_dims, 4*ini_channels)
        self.dec3_up = nn.ConvTranspose3d(4*ini_channels, 2*ini_channels,
                                          kernel_size=4, stride=2, padding=1)
        
        dec2_in_dims = (4 + 1 + 16 + 8) * ini_channels
        self.dec2 = _Blk_3d(dec2_in_dims, 2*ini_channels)
        self.dec2_up = nn.ConvTranspose3d(2*ini_channels, ini_channels,
                                          kernel_size=4, stride=2, padding=1)
        
        dec1_in_dims = (2 + 16 + 8 + 4) * ini_channels
        self.dec1 = nn.Sequential(
            nn.Conv3d(dec1_in_dims, ini_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm3d(ini_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(ini_channels, ini_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm3d(ini_channels),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv3d(ini_channels, num_classes, kernel_size=1)

        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # deep supervisions
        if self.deep_sup:
            self.center_ds = nn.Sequential(
                nn.Conv3d(16*ini_channels, num_classes, kernel_size=1)
                )
            self.dec4_ds = nn.Sequential(
                nn.Conv3d(8*ini_channels, num_classes, kernel_size=1)
                )
            self.dec3_ds = nn.Sequential(
                nn.Conv3d(4*ini_channels, num_classes, kernel_size=1)
                )
            self.dec2_ds = nn.Sequential(
                nn.Conv3d(2*ini_channels, num_classes, kernel_size=1)
                )

        tot_params, tot_tparams = self.param_counts
        print(f'💠 UNet3p-3D model initiated with n_classes={num_classes}, '
              f'(deep_sup={deep_sup})\n'
              f'   n_input={in_channels}, ini_chans={ini_channels}\n'
              f'   params={tot_params:,}, trainable_params={tot_tparams:,}.')

    def forward(self, x):
        # Encoding stage
        enc1 = self.enc1(x)
        enc1_pool1 = self.max_pool(enc1)
        enc1_pool2 = self.max_pool(enc1_pool1)
        enc1_pool3 = self.max_pool(enc1_pool2) 

        enc2 = self.enc2(enc1)
        enc2_pool1 = self.max_pool(enc2)
        enc2_pool2 = self.max_pool(enc2_pool1)

        enc3 = self.enc3(enc2)
        enc3_pool1 = self.max_pool(enc3)

        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        
        # Decoding stage
        center_up = self.center_up(center)
        center_dense_up2 = F.interpolate(center, scale_factor=4.0, 
                                         mode='trilinear', align_corners=True)
        center_dense_up3 = F.interpolate(center, scale_factor=8.0, 
                                         mode='trilinear', align_corners=True)
        center_dense_up4 = F.interpolate(center, scale_factor=16.0, 
                                         mode='trilinear', align_corners=True)

        dec4 = self.dec4(torch.cat([
                center_up, 
                F.interpolate(enc4, center_up.size()[2:], mode='trilinear',
                              align_corners=False),
                F.interpolate(enc1_pool3, center_up.size()[2:],
                              mode='trilinear', align_corners=False),
                F.interpolate(enc2_pool2, center_up.size()[2:], 
                              mode='trilinear', align_corners=False),
                F.interpolate(enc3_pool1, center_up.size()[2:], 
                              mode='trilinear', align_corners=False)
               ], 1))
        dec4_dense_up2 = F.interpolate(dec4, scale_factor=4.0, mode='trilinear', 
                                       align_corners=True)
        dec4_dense_up3 = F.interpolate(dec4, scale_factor=8.0, mode='trilinear', 
                                       align_corners=True)
        dec4_up = self.dec4_up(dec4)
        
        dec3 = self.dec3(torch.cat([
                dec4_up, 
                F.interpolate(enc3, dec4_up.size()[2:], 
                              mode='trilinear', align_corners=False),
                F.interpolate(enc1_pool2, dec4_up.size()[2:], 
                              mode='trilinear', align_corners=False),
                F.interpolate(enc2_pool1, dec4_up.size()[2:], 
                              mode='trilinear', align_corners=False),
                F.interpolate(center_dense_up2, dec4_up.size()[2:], 
                              mode='trilinear', align_corners=False)
               ], 1))
        dec3_dense_up2 = F.interpolate(dec3, scale_factor=4.0, mode='trilinear', 
                                       align_corners=True)
        dec3_up = self.dec3_up(dec3)
        
        dec2 = self.dec2(torch.cat([
                dec3_up, 
                F.interpolate(enc2, dec3_up.size()[2:], 
                              mode='trilinear', align_corners=False),
                F.interpolate(enc1_pool1, dec3_up.size()[2:], 
                              mode='trilinear', align_corners=False),
                F.interpolate(center_dense_up3, dec3_up.size()[2:], 
                              mode='trilinear', align_corners=False),
                F.interpolate(dec4_dense_up2, dec3_up.size()[2:], 
                              mode='trilinear', align_corners=False)
               ], 1))
        dec2_up = self.dec2_up(dec2)
        dec1 = self.dec1(torch.cat([
                dec2_up, 
                F.interpolate(enc1, dec2_up.size()[2:], 
                              mode='trilinear', align_corners=False),
                F.interpolate(center_dense_up4, dec2_up.size()[2:], 
                              mode='trilinear', align_corners=False),
                F.interpolate(dec4_dense_up3, dec2_up.size()[2:], 
                              mode='trilinear', align_corners=False),
                F.interpolate(dec3_dense_up2, dec2_up.size()[2:], 
                              mode='trilinear', align_corners=False)
               ], 1))
        
        final = self.final(dec1)
        out = F.interpolate(final, x.size()[2:], mode='trilinear',
                            align_corners=False)

        # deep supervisions: conv + trilinear up-sampling
        if self.deep_sup:
            center_ds = self.center_ds(center)
            center_ds = F.interpolate(center_ds, scale_factor=16.0, 
                                      mode='trilinear', align_corners=True)

            dec4_ds = self.dec4_ds(dec4)
            dec4_ds = F.interpolate(dec4_ds, scale_factor=8.0, 
                                      mode='trilinear', align_corners=True)

            dec3_ds = self.dec3_ds(dec3)
            dec3_ds = F.interpolate(dec3_ds, scale_factor=4.0, 
                                      mode='trilinear', align_corners=True)

            dec2_ds = self.dec2_ds(dec2) 
            dec2_ds = F.interpolate(dec2_ds, scale_factor=2.0, 
                                      mode='trilinear', align_corners=True)
            return {
                'out': out,
                '2x': dec2_ds,
                '4x': dec3_ds,
                '8x': dec4_ds,
                '16x': center_ds
            }
        return {'out': out}
