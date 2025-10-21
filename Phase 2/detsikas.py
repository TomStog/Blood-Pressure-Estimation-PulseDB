import torch
import torch.nn as nn
import sys

def conv_block(inputs, filters, activation, kernel_size=3, dilation_rate=1):
    cx = nn.Conv1d(in_channels=inputs, out_channels=filters, kernel_size=kernel_size, padding='same', dilation=dilation_rate)
    bn = nn.BatchNorm1d(filters)
    if activation == 'relu':
        act = nn.ReLU()
    elif activation == 'leaky_relu':
        act = nn.LeakyReLU()
    elif activation is None:
        act = nn.Identity()
    else:
        print('Bad activation')
        sys.exit(0)
    return nn.Sequential(cx, bn, act)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv_block(in_channels, filters, activation=activation)
        self.conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3, padding='same')
        self.res_conv = nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=1, padding='same')
        self.bn = nn.BatchNorm1d(filters)
        
        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            print('Bad activation')
            sys.exit(0)
    
    def forward(self, x):
        identity = self.res_conv(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.bn(out)
        out = self.act(out)
        return out

class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, dilation_rate=2, activation='relu'):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = conv_block(in_channels, filters, activation=activation)
        self.conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3, padding='same', dilation=dilation_rate)
        self.res_conv = nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=1, padding='same', dilation=dilation_rate)
        self.bn = nn.BatchNorm1d(filters)
        
        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            print('Bad activation')
            sys.exit(0)
    
    def forward(self, x):
        identity = self.res_conv(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.bn(out)
        out = self.act(out)
        return out

class MLP(nn.Module):
    def __init__(self, C, r=8):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(C, int(C / r))
        self.fc2 = nn.Linear(int(C / r), C)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class VisualAttentionBlock(nn.Module):
    def __init__(self, C, r=8):
        super(VisualAttentionBlock, self).__init__()
        self.mlp = MLP(C, r)
        self.spatial_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, enc_input, dec_input):
        # Channel attention for decoder input
        f_ch_avg = self.global_avg_pool(dec_input).squeeze(-1)
        m_ch = self.mlp(f_ch_avg).unsqueeze(-1)
        
        # Spatial attention for decoder input
        f_sp_avg = torch.mean(dec_input, dim=1, keepdim=True)
        m_sp = self.sigmoid(self.spatial_conv(f_sp_avg))
        
        # Ensure spatial dimensions match exactly
        if m_sp.shape[2] > enc_input.shape[2]:
            m_sp = m_sp[:, :, :enc_input.shape[2]]
        elif m_sp.shape[2] < enc_input.shape[2]:
            # Pad if m_sp is shorter
            pad_size = enc_input.shape[2] - m_sp.shape[2]
            m_sp = torch.nn.functional.pad(m_sp, (0, pad_size), mode='replicate')
        
        f_ch = m_ch * enc_input
        f_sp = f_ch * m_sp
        
        return f_sp

class DilatedVisualAttentionResidualUNet(nn.Module):
    def __init__(self, input_channels=2, starting_filters=16, activation='relu'):
        super(DilatedVisualAttentionResidualUNet, self).__init__()
        
        # Encoder path
        filters = starting_filters
        self.conv_block1 = conv_block(input_channels, filters, activation=activation)
        self.dropout1 = nn.Dropout(0.1)
        
        filters *= 2
        self.pool1 = nn.AvgPool1d(2)
        self.res_block1 = ResidualBlock(filters // 2, filters)
        self.conv_block2 = conv_block(filters, filters, activation=activation)
        self.dropout2 = nn.Dropout(0.1)
        
        filters *= 2
        self.pool2 = nn.AvgPool1d(2)
        self.res_block2 = ResidualBlock(filters // 2, filters)
        self.conv_block3 = conv_block(filters, filters, activation=activation)
        self.dropout3 = nn.Dropout(0.2)
        
        filters *= 2
        self.pool3 = nn.AvgPool1d(2)
        self.dilated_res_block1 = DilatedResidualBlock(filters // 2, filters)
        self.conv_block4 = conv_block(filters, filters, activation=activation)
        self.dropout4 = nn.Dropout(0.2)
        
        filters *= 2
        self.dilated_res_block2 = DilatedResidualBlock(filters // 2, filters, 2)
        self.dropout5 = nn.Dropout(0.2)
        
        # Bottleneck
        filters *= 2
        self.bottleneck_conv1 = conv_block(filters // 2, filters, activation=activation)
        self.bottleneck_conv2 = conv_block(filters, filters, activation=activation)
        self.bottleneck_dropout = nn.Dropout(0.3)
        
        # Decoder path
        filters //= 2
        self.dec_res_block1 = ResidualBlock(filters * 2, filters)
        self.attn1 = VisualAttentionBlock(filters)
        self.dec_conv1 = conv_block(filters * 2, filters, activation=activation)
        self.dec_dropout1 = nn.Dropout(0.2)
        
        filters //= 2
        self.dec_res_block2 = ResidualBlock(filters * 2, filters)
        self.attn2 = VisualAttentionBlock(filters)
        self.dec_conv2 = conv_block(filters * 2, filters, activation=activation)
        self.dec_dropout2 = nn.Dropout(0.2)
        
        filters //= 2
        self.upsample1 = nn.ConvTranspose1d(in_channels=filters * 2, out_channels=filters, kernel_size=2, stride=2, padding=0)
        self.dec_res_block3 = ResidualBlock(filters, filters)
        self.attn3 = VisualAttentionBlock(filters)
        self.dec_conv3 = conv_block(filters * 2, filters, activation=activation)
        self.dec_dropout3 = nn.Dropout(0.2)
        
        filters //= 2
        self.upsample2 = nn.ConvTranspose1d(in_channels=filters * 2, out_channels=filters, kernel_size=2, stride=2, padding=0)
        self.dec_res_block4 = ResidualBlock(filters, filters)
        self.attn4 = VisualAttentionBlock(filters)
        self.dec_conv4 = conv_block(filters * 2, filters, activation=activation)
        self.dec_dropout4 = nn.Dropout(0.1)
        
        filters //= 2
        self.upsample3 = nn.ConvTranspose1d(in_channels=filters * 2, out_channels=filters, kernel_size=2, stride=2, padding=0)
        self.dec_res_block5 = ResidualBlock(filters, filters)
        self.attn5 = VisualAttentionBlock(filters)
        self.dec_conv5 = conv_block(filters * 2, filters, activation=activation)
        self.dec_dropout5 = nn.Dropout(0.1)
        
        # Global pooling and regression head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(filters, 1)
    
    def forward(self, x):
        signal_length = x.shape[2]
        target_length = 1248
        if signal_length > target_length:
            x = x[:, :, :target_length]
        
        # Encoder
        filters = 16
        x = self.conv_block1(x)
        sc_1 = self.dropout1(x)
        
        filters *= 2
        x = self.pool1(sc_1)
        x = self.res_block1(x)
        x = self.conv_block2(x)
        sc_2 = self.dropout2(x)
        
        filters *= 2
        x = self.pool2(sc_2)
        x = self.res_block2(x)
        x = self.conv_block3(x)
        sc_3 = self.dropout3(x)
        
        filters *= 2
        x = self.pool3(sc_3)
        x = self.dilated_res_block1(x)
        x = self.conv_block4(x)
        sc_4 = self.dropout4(x)
        
        filters *= 2
        x = self.dilated_res_block2(x)
        sc_5 = self.dropout5(x)
        
        # Bottleneck
        filters *= 2
        x = self.bottleneck_conv1(x)
        x = self.bottleneck_conv2(x)
        x = self.bottleneck_dropout(x)
        
        # Decoder
        filters //= 2
        x = self.dec_res_block1(x)
        attn_output = self.attn1(sc_5, x)
        x = torch.cat([x, attn_output], dim=1)
        x = self.dec_conv1(x)
        x = self.dec_dropout1(x)
        
        filters //= 2
        x = self.dec_res_block2(x)
        attn_output = self.attn2(sc_4, x)
        x = torch.cat([x, attn_output], dim=1)
        x = self.dec_conv2(x)
        x = self.dec_dropout2(x)
        
        filters //= 2
        x = self.upsample1(x)
        # Match dimensions if needed
        if x.shape[2] != sc_3.shape[2]:
            x = x[:, :, :sc_3.shape[2]]
        x = self.dec_res_block3(x)
        attn_output = self.attn3(sc_3, x)
        x = torch.cat([x, attn_output], dim=1)
        x = self.dec_conv3(x)
        x = self.dec_dropout3(x)
        
        filters //= 2
        x = self.upsample2(x)
        # Match dimensions if needed
        if x.shape[2] != sc_2.shape[2]:
            x = x[:, :, :sc_2.shape[2]]
        x = self.dec_res_block4(x)
        attn_output = self.attn4(sc_2, x)
        x = torch.cat([x, attn_output], dim=1)
        x = self.dec_conv4(x)
        x = self.dec_dropout4(x)
        
        filters //= 2
        x = self.upsample3(x)
        # Match dimensions if needed
        if x.shape[2] != sc_1.shape[2]:
            x = x[:, :, :sc_1.shape[2]]
        x = self.dec_res_block5(x)
        attn_output = self.attn5(sc_1, x)
        x = torch.cat([x, attn_output], dim=1)
        x = self.dec_conv5(x)
        x = self.dec_dropout5(x)
        
        # Regression output
        x = self.global_pool(x).squeeze(-1)
        output = self.fc(x)
        
        return output