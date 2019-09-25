from layers import create_act
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from utils_our import pad_extra_rows

class GraphConvolutionCollector(nn.Module):
    def __init__(self, gcn_num, fix_size, mode, padding_value, align_corners, **kwargs):
        super().__init__()
        self.gcn_num = gcn_num
        self.MNEResize = MNEResize(inneract=False, fix_size=fix_size,
                                    mode=mode, padding_value=padding_value,
                                    align_corners=align_corners, **kwargs)

    def forward(self, inputs, batch_data, model):
        assert (type(inputs) is list and inputs)
        assert (len(inputs) == self.gcn_num)

        gcn_ins = []  #each item is a list of similarity matricies for a given gcn output
        for gcn_layer_out in inputs:
            gcn_ins.append(torch.stack(self.MNEResize(gcn_layer_out,batch_data, model)))

        return torch.stack(gcn_ins) #gcn_num by Num_pairs by Nmax by Nmax similarity matricies len(gcn_ins)==gcn_num



class MNEResize(nn.Module):
    def __init__(self, inneract, fix_size, mode, padding_value,
                 align_corners, **kwargs):
        super().__init__()
        self.inneract = inneract
        self.fix_size = fix_size
        self.align_corners = align_corners
        self.padding_value = padding_value
        modes =  ["bilinear", "nearest", "bicubic", "area"]
        if mode < 0 or mode > len(modes):
            raise RuntimeError('Unknown MNE resize mode {}'.format(self.mode))
        self.mode = modes[mode]

    def forward(self, ins, batch_data, model):
        x = ins  # x has shape N(gs) by D
        ind_list = batch_data.merge_data["ind_list"]
        sim_mat_l = [] #list of similarity matricies (should be len(ind_list)/2 items)
        for i in range(0,len(ind_list), 2):
            g1_ind = i
            g2_ind = i + 1
            g1x = x[ind_list[g1_ind][0]: ind_list[g1_ind][1]]
            g2x = x[ind_list[g2_ind][0]: ind_list[g2_ind][1]]
            sim_mat_l.append(self._call_one_pair(g1x, g2x))
        return sim_mat_l



    def _call_one_pair(self, g1x, g2x):
        x1_pad, x2_pad = pad_extra_rows(g1x, g2x, self.padding_value)
        sim_mat_temp = torch.matmul(x1_pad, torch.t(x2_pad))

        sim_mat = sim_mat_temp.unsqueeze(0).unsqueeze(0) if not self.inneract else \
            self.inneract(sim_mat_temp).unsqueeze(0).unsqueeze(0) #need for dims for bilinear interpolation
        sim_mat_resize = F.interpolate(sim_mat,
                                       size=[self.fix_size, self.fix_size],
                                       mode=self.mode,
                                       align_corners=self.align_corners)
        return sim_mat_resize.squeeze().unsqueeze(0)


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, gcn_num, bias,
                 poolsize, act, end_cnn=False, **kwargs):
        super().__init__()

        #same padding calc
        self.kernel_size = kernel_size
        self.stride = stride
        self.convs  = nn.ModuleList()
        self.maxpools = nn.ModuleList()
        self.pool_stride =  poolsize
        self.pool_size = poolsize
        self.activation = create_act(act)
        self.end_cnn = end_cnn
        self.gcn_num=gcn_num
        self.out_channels = out_channels

        for i in range(gcn_num):
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias))
            self.convs[-1].apply(self.weights_init)
            self.maxpools.append(nn.MaxPool2d(poolsize, stride=poolsize))

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)

    def forward(self, ins, batch_data, model):
        inshape = ins[0].shape
        num_batch = inshape[0]
        H_in = inshape[2]
        W_in = inshape[3]

        pad_cnn_h = self._same_pad_calc(H_in, self.kernel_size, self.stride)
        pad_cnn_w = self._same_pad_calc(W_in, self.kernel_size, self.stride)
        pad_pool_h = self._same_pad_calc(H_in, self.pool_size, self.pool_stride)
        pad_pool_w = self._same_pad_calc(W_in, self.pool_size, self.pool_stride)

        result = []

        for i in range(self.gcn_num):
            result.append(self._conv_and_pool(ins[i, :, :, :, :], i, pad_cnn_h,
                                              pad_cnn_w, pad_pool_h,
                                              pad_pool_w))
        rtn = torch.stack(result)
        if self.end_cnn:
            rtn = rtn.squeeze(4).squeeze(3)
            rtn = rtn.permute(1,0,2)
            rtn = torch.reshape(rtn, [num_batch, self.out_channels * self.gcn_num])
        return rtn


    def _conv_and_pool(self, gcn_sim_mat, gcn_ind, pad_cnn_h, pad_cnn_w, pad_pool_h, pad_pool_w):
        gcn_sim_mat_pad = F.pad(gcn_sim_mat, (pad_cnn_w[0], pad_cnn_w[1],pad_cnn_h[0], pad_cnn_h[1]))

        conv_x = self.convs[gcn_ind](gcn_sim_mat_pad)
        conv_x = self.activation(conv_x)
       # print(conv_x.shape)

        conv_x = F.pad(conv_x, (pad_pool_w[0], pad_pool_w[1],pad_pool_h[0], pad_pool_h[1]))
        pool_x = self.maxpools[gcn_ind](conv_x)
       # print(pool_x.shape)
        return pool_x


    # to mimic the tensorflow implementation
    def _same_pad_calc(self, in_dim, kernel_size, stride):
        pad = ((math.ceil(in_dim/stride)-1)*stride-in_dim + kernel_size)
        if pad % 2 == 0:
            return (int(pad/2), int(pad/2))
        else:
            return (int(pad/2), int(pad/2)+1)

