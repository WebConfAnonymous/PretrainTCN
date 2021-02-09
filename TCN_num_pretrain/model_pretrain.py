from torch import nn
from Methods.tcn import TemporalConvNet
import numpy as np
import torch

class TimeSlicePre(nn.Module):
    def __init__(self, input_size, output_size, num_channels,kernel_size, mlp_hid_size,dropout):
        super(TimeSlicePre, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels,kernel_size=kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(2*num_channels[-1], mlp_hid_size)
        self.linear2 = nn.Linear(mlp_hid_size,int(mlp_hid_size/2))
        self.linear3 = nn.Linear(int(mlp_hid_size/2),1)
        self.leakyrelu = nn.LeakyReLU()
        #self.init_weights()

    #def init_weights(self):
        #self.linear1.weight.data.normal_(0, 0.01)
        #self.linear2.weight.data.normal_(0, 0.01)
        #self.linear3.weight.data.normal_(0, 0.01)
    def get_concat(self):
        return self.y_concat
    def forward(self, x1, x2):
        # batch_size * num_channels * seq_length
        y1_out = self.tcn(x1)
        # batch_size * num_channels
        #print("y1_out:",y1_out.size()) 
        y1_out_last = y1_out[:,:,-1]
        #print("y1_out_last:",y1_out_last.size())

        y2_out = self.tcn(x2)
        y2_out_last = y2_out[:,:,-1] 
        
        self.y_concat = torch.cat((y1_out_last,y2_out_last),1)
        #print("y_concat:",y_concat.size())
        #print("y_concat:",y_concat)
        result = self.leakyrelu(self.linear3(self.leakyrelu(self.linear2(self.leakyrelu(self.linear1(self.y_concat))))))
        #print("result:",result.size())

        #y2 = self.linear(y1[:, :, -1])
        #print(y2.size())
        # if np.sum(np.isnan(x.cpu().numpy())) > 0:
        #     print("model inputs x nan:", x.size(), "input:", x)
        # if np.sum(np.isnan(y1.detach().cpu().numpy())) > 0:
        #     print("model inputs y1 nan:", y1.size(), "input:", y1)
        # if np.sum(np.isnan(y2.detach().cpu().numpy())) > 0:
        #     print("model inputs y2 nan:", y2.size(), "input:", y2)
        return result
