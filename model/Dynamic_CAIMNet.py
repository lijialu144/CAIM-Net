import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import crnn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_channels=16, num_groups=2)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=2)

        self.residual_conv1 = nn.Conv2d(in_channels, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.resiucal_norm1 = nn.GroupNorm(num_channels=16, num_groups=2)
        self.residual_conv2 = nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.resiucal_norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=2)

        self.conv_final1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_final1 = nn.GroupNorm(num_channels=out_channels, num_groups=2)

    def forward(self, x):
        identify = F.relu(self.resiucal_norm1(self.residual_conv1(x)))
        identify = F.relu(self.resiucal_norm2(self.residual_conv2(identify)))

        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))

        final_out = F.relu(self.norm_final1(self.conv_final1(identify + out)))

        return final_out

def createConvFunc(x, weights, bias=None, stride=1, padding=1, dilation=1, groups=1):
    shape = weights.shape
    weights = weights.view(shape[0], shape[1], -1)
    weights_clone = weights.clone()
    weights_sum = weights_clone.sum(dim=[2], keepdim=True)
    buffer_temp = weights_clone[:, :, [4]]
    buffer = weights_clone
    buffer[:, :, [4]] = (buffer_temp - weights_sum)
    buffer = buffer.view(shape[0], shape[1], 3, 3)
    y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return y

class BEConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups, bias=False):
        super(BEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channel = out_channels
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.dilation = 1
        self.groups = groups
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, self.kernel_size, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        y = createConvFunc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        out = input + y
        return out

class Spatial_Attention(nn.Module):
    def __init__(self, d_model, num_layers, nheads, hidden_size):
        super(Spatial_Attention, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nheads, dim_feedforward=hidden_size)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, input):
        output = self.transformer(input)
        return output

class Temporal_Attention(nn.Module):
    def __init__(self, batch, times, features, size, hidden_channels, ):
        super(Temporal_Attention, self).__init__()
        self.b = batch
        self.t = times
        self.c = features
        self.s = size
        self.hidden_channels = hidden_channels
        self.LSTM_diff = crnn.LSTMdistCell('LSTM', features, hidden_channels, 3, convndim=2)

    def forward(self, input):
        input = input.permute(1, 0, 3, 4, 2).view(self.t - 1, self.b * self.s * self.s, self.c)
        temporal = torch.zeros((input.shape[0], input.shape[1], self.hidden_channels)).cuda()
        for i in range(input.shape[0]):
            if i == 0:
                hx, cx = self.LSTM_diff(input[i])
            else:
                hx, cx = self.LSTM_diff(input[i], (hx, cx))
            temporal[i] = hx
        temporal = temporal.view(self.t - 1, self.b, self.s, self.s, self.hidden_channels)
        temporal = temporal.permute(1, 0, 4, 2, 3)
        return temporal

class Coarse_Change_Moment1(nn.Module):
    def __init__(self, batch, features, times, size):
        super(Coarse_Change_Moment1, self).__init__()
        self.b = batch
        self.s = size
        self.t = times
        self.c = features

        self.conv = nn.Conv2d(in_channels=features, out_channels=2, kernel_size=(1, 1))
        self.norm = nn.GroupNorm(num_channels=2, num_groups=1)


    def forward(self, input):  # 【64, 5, 64, 64, 64】
        # 该模块应该有两个输出，一个是用于变化区域，一个用于变化时刻
        input = input.permute(1, 0, 2, 3, 4)  # 【5, 64, 64, 64, 64】
        change = torch.zeros(self.t - 1, self.b, self.s, self.s).cuda()  # 【5, 64, 64, 64】
        unchange = torch.zeros(self.t - 1, self.b, self.s, self.s).cuda()  # 【5, 64, 64, 64】
        for i in range(self.t - 1):
            change_area_ave = self.norm(self.conv(input[i]))  # 【64, 2, 64, 64】
            unchange[i] = change_area_ave[:, 0, :, :]
            change[i] = change_area_ave[:, 1, :, :]

        unchange = unchange.permute(1, 0, 2, 3)  # 【64, 5, 64, 64】
        change = change.permute(1, 0, 2, 3)  # 【64, 5, 64, 64】
        unchange_min = torch.min(unchange, dim=1)[0].unsqueeze(1)  # 【64, 1, 64, 64】
        change_moment = torch.concat((unchange_min, change), dim=1)  # 【64, 6, 64, 64】
        change_moment = change_moment.permute(0, 2, 3, 1).contiguous().view(-1, self.t)
        return change_moment

class Coarse_Change_Moment2(nn.Module):
    def __init__(self, batch, features, times, size):
        super(Coarse_Change_Moment2, self).__init__()
        self.b = batch
        self.s = size
        self.t = times
        self.c = features

        self.conv1 = nn.Conv2d(in_channels=features * 5, out_channels=features, kernel_size=3, padding=1, stride=1)
        self.norm1 = nn.GroupNorm(num_channels=features, num_groups=2)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=times, kernel_size=3, padding=1, stride=1)

    def forward(self, input):  # 【64, 5, 64, 64, 64】
        input = input.contiguous().view(self.b, (self.t - 1) * self.c, self.s, self.s)
        x = self.norm1(self.conv1(input))
        change_moment = self.conv2(x)
        change_moment = change_moment.permute(0, 2, 3, 1).contiguous().view(self.b * self.s * self.s, self.t)
        return change_moment

class Fine_Change_Moment2(nn.Module):
    def __init__(self, batch, features, times, size):
        super(Fine_Change_Moment2, self).__init__()
        self.b = batch
        self.c = features
        self.t = times
        self.s = size

        self.fc2 = nn.Linear(4, 6)
        self.fc_weight2 = nn.Parameter(self.fc2.weight.t().unsqueeze(0).expand(batch * size * size // 4, -1, -1))

    def forward(self, input):
        # input【64, 6, 64, 64】
        input = input.view(self.b, self.s, self.s, self.t).permute(0, 3, 1, 2)
        feature_map2 = input.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(self.b, self.t, self.s // 2, self.s // 2, -1)  # 【64, 6, 32, 32, 4】
        feature_map2 = feature_map2.permute(0, 2, 3, 4, 1).contiguous().view(-1, 4, self.t)  # 【64*32*32/65536, 4, 6】

        output_mean2 = feature_map2.mean(2)  # 【65536, 4】
        output2 = F.softmax(self.fc2(output_mean2))  # 【65536, 6】
        feature_map2 = feature_map2.permute(0, 2, 1)  # 【65536, 6, 4】

        cam2 = torch.bmm(feature_map2, self.fc_weight2).transpose(1, 2)  # 【65536, 6, 6】
        min_val2 = torch.min(cam2, dim=2, keepdim=True)[0]  # 【65536, 6, 1]  在空间维度取最小值
        cam2 -= min_val2  # CAM 减去最小值
        max_val2 = torch.max(cam2, dim=2, keepdim=True)[0]  # 【65536, 6, 1】  在空间维度取最大值
        cam2 /= max_val2

        topk_cam2 = cam2[:, 0].view(self.b, self.s // 2, self.s // 2, self.t).contiguous().permute(0, 3, 1, 2)

        fine_change_moment2 = F.interpolate(topk_cam2, (self.s, self.s), mode='bilinear', align_corners=True)
        fine_change_moment2 = fine_change_moment2.permute(0, 2, 3, 1).view(-1, self.t)
        return fine_change_moment2

class Fine_Change_Moment3(nn.Module):
    def __init__(self, batch, features, times, size):
        super(Fine_Change_Moment3, self).__init__()
        self.b = batch
        self.c = features
        self.t = times
        self.s = size

        self.fc3 = nn.Linear(16, 6)
        self.fc_weight3 = nn.Parameter(self.fc3.weight.t().unsqueeze(0).expand(batch * size * size // 16, -1, -1))

    def forward(self, input):
        # input【64, 6, 64, 64】
        input = input.view(self.b, self.s, self.s, self.t).permute(0, 3, 1, 2)
        feature_map3 = input.unfold(2, 4, 4).unfold(3, 4, 4).contiguous().view(self.b, self.t, self.s // 4, self.s // 4, -1)  # 【64, 6, 32, 32, 4】
        feature_map3 = feature_map3.permute(0, 2, 3, 4, 1).contiguous().view(-1, 16, self.t)  # 【64*32*32/65536, 4, 6】

        output_mean3 = feature_map3.mean(2)  # 【65536, 4】
        output3 = F.softmax(self.fc3(output_mean3))  # 【65536, 6】
        feature_map3 = feature_map3.permute(0, 2, 1)  # 【65536, 6, 4】

        cam3 = torch.bmm(feature_map3, self.fc_weight3).transpose(1, 2)  # 【65536, 6, 6】
        min_val3 = torch.min(cam3, dim=2, keepdim=True)[0]  # 【65536, 6, 1]  在空间维度取最小值
        cam3 -= min_val3  # CAM 减去最小值
        max_val3 = torch.max(cam3, dim=2, keepdim=True)[0]  # 【65536, 6, 1】  在空间维度取最大值
        cam3 /= max_val3

        topk_cam3 = cam3[:, 0].view(self.b, self.s // 4, self.s // 4, self.t).contiguous().permute(0, 3, 1, 2)

        fine_change_moment3 = F.interpolate(topk_cam3, (self.s, self.s), mode='bilinear', align_corners=True)
        fine_change_moment3 = fine_change_moment3.permute(0, 2, 3, 1).view(-1, self.t)
        return fine_change_moment3

def Moment_to_Area(moment_prob):
    first_column = moment_prob[:, 0]
    second_column = moment_prob[:, 1:].max(dim=1).values
    area = torch.stack((first_column, second_column), dim=1)
    return area

class Model(nn.Module):
    def __init__(self, batch, input_channels, features, change_nums, times, size):
        super(Model, self).__init__()
        self.input_c = input_channels
        self.b = batch
        self.c = features
        self.t = times
        self.s = size
        # 无下采样的空间特征提取器
        self.Encoder = BasicBlock(in_channels=input_channels, out_channels=features)
        # 边界增强卷积
        self.BEConv = BEConv(in_channels=features*(times-1), out_channels=features*(times-1), groups=features*(times - 1))
        # 空间注意力
        self.Spatial_Attention = Spatial_Attention(d_model=features*(times-1), num_layers=1, nheads=4, hidden_size=4*features*(times-1))
        self.Temporal_Attention = Temporal_Attention(batch=batch, times=times, features=features, size=size, hidden_channels=features//2)
        # 变化区域和变化时间的分析推断
        self.Coarse_Change_Moment1 = Coarse_Change_Moment1(batch, features//2, times, self.s)
        self.Coarse_Change_Moment2 = Coarse_Change_Moment2(batch, features//2, times, self.s)
        self.Fine_Change_Moment1_2 = Fine_Change_Moment2(batch, features // 2, times, self.s)
        self.Fine_Change_Moment1_3 = Fine_Change_Moment3(batch, features // 2, times, self.s)
        self.Fine_Change_Moment2_2 = Fine_Change_Moment2(batch, features // 2, times, self.s)
        self.Fine_Change_Moment2_3 = Fine_Change_Moment3(batch, features // 2, times, self.s)

    def forward(self, input):
        # 第一阶段： 差异特征的准确提取
        # 【6, 96, 4, 64, 64】-【0 - 1】输入的特征维度
        #【576, 4, 64, 64】-【0 - 1】进行特征维度转换
        input = input.contiguous().view(-1, self.input_c, self.s, self.s)
        #【576, 64, 64, 64】-【0 - 35.4772】 空间特征提取
        features = self.Encoder(input)
        #【6, 96, 64, 64, 64】 - 【0, 35.4772】空间特征维度转换
        features = features.view(self.t, self.b, self.c, self.s, self.s)
        #【5, 64, 64, 64, 64】 - 【0 - 0】创建差异矩阵
        difference = torch.zeros(self.t - 1, self.b, self.c, self.s, self.s).cuda()
        # 【5, 64, 64, 64, 64】 - 【0 - 32.6901】计算差异矩阵
        for m in range(self.t - 1):
            difference[m] = torch.abs(features[m + 1] - features[m])
        # 【64, 320, 64, 64】 - 【0 - 32.6901】差异特征维度转换
        difference = difference.permute(1, 0, 2, 3, 4).contiguous().view(self.b, -1, self.s,  self.s)
        # 【64, 320, 64, 64】 - 【-4.4317 - 37.2407】边界增强卷积
        difference_BEConv = self.BEConv(difference)  # 【64, 320, 64, 64】
        # 【64, 4096, 320】 - 【-4.4317 - 37.2407】边界增强差异特征维度转换
        difference_BEConv = difference_BEConv.view(self.b, (self.t - 1) * self.c, self.s * self.s).permute(0, 2, 1)
        # 第二阶段：提取每个分支的粗略的变化区域或者变化时间
        # 【64, 4096, 320】 - 【-5.6722 - 11.0287】边界增强差异特征空间相关性-空间增强卷积
        difference_spatial = self.Spatial_Attention(difference_BEConv)
        # 【64, 5, 64, 64, 64】- 【-5.6722 - 11.0287】空间增强特征维度转换
        difference_spatial = difference_spatial.view(self.b, self.s, self.s, self.t - 1, self.c)
        difference_spatial = difference_spatial.permute(0, 3, 4, 1, 2)
        difference_temporal = self.Temporal_Attention(difference_spatial)

        change_moment1 = self.Coarse_Change_Moment1(difference_temporal)
        change_moment2 = self.Coarse_Change_Moment2(difference_temporal)
        moment1_prob = F.softmax(change_moment1)
        moment2_prob = F.softmax(change_moment2)

        fine_moment1_2 = self.Fine_Change_Moment1_2(moment1_prob)
        fine_moment1_3 = self.Fine_Change_Moment1_3(moment1_prob)
        fine_moment2_2 = self.Fine_Change_Moment2_2(moment2_prob)
        fine_moment2_3 = self.Fine_Change_Moment2_3(moment2_prob)

        fine_moment1_2_prob = F.softmax(fine_moment1_2)
        fine_moment1_3_prob = F.softmax(fine_moment1_3)
        fine_moment2_2_prob = F.softmax(fine_moment2_2)
        fine_moment2_3_prob = F.softmax(fine_moment2_3)
        fine_moment_prob = F.softmax(fine_moment1_2_prob + fine_moment1_3_prob + fine_moment2_2_prob + fine_moment2_3_prob)

        fine_area1_2_prob = F.softmax(Moment_to_Area(fine_moment1_2_prob))
        fine_area1_3_prob = F.softmax(Moment_to_Area(fine_moment1_3_prob))
        fine_area2_2_prob = F.softmax(Moment_to_Area(fine_moment2_2_prob))
        fine_area2_3_prob = F.softmax(Moment_to_Area(fine_moment2_3_prob))
        fine_area_prob = F.softmax(Moment_to_Area(fine_moment_prob))

        return fine_area_prob, fine_moment_prob, \
               fine_area1_2_prob, fine_moment1_2_prob, fine_area1_3_prob, fine_moment1_3_prob, \
               fine_area2_2_prob, fine_moment2_2_prob, fine_area2_3_prob, fine_moment2_3_prob
