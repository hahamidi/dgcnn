import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_geometric.nn import XConv, fps, global_mean_pool,knn_interpolate
from torch_geometric.profile import rename_profile_file

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--inference', action='store_true')
parser.add_argument('--profile', action='store_true')
args = parser.parse_args()


def Xconv_layer(in_chanel,out_chanel,kernel_size,hidden_chanel,dilation = 2 ):
    layer = torch.nn.Sequential(XConv(in_chanel,out_chanel,dim=3,
                                      kernel_size=kernel_size,
                                      hidden_channels=hidden_chanel,
                                      dilation=dilation),
                                torch.nn.ReLU())
    
    return layer

def down_sample_layer(x ,pose,batch,ratio = 0.375 ):
    idx = fps(pose, batch, ratio=ratio)
    x, pose, batch = x[idx], pose[idx], batch[idx]
    return x,pose,batch 



class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = XConv(0, 64, dim=3, kernel_size=8, hidden_channels=32)
        self.conv2 = XConv(64, 96, dim=3, kernel_size=12, hidden_channels=64,dilation=2)
        self.conv3 = XConv(96, 192, dim=3, kernel_size=16, hidden_channels=128,dilation=2)
        self.conv4 = XConv(192, 384, dim=3, kernel_size=16,hidden_channels=256, dilation=2)
        self.conv4_up = XConv(384 + 128 , 192 , dim=3, kernel_size=12,hidden_channels=320, dilation=2)
        self.conv3_up = XConv(192 + 192 , 96 , dim=3, kernel_size=12,hidden_channels=256, dilation=2)
        self.conv2_up = XConv(96 + 96 , 96 , dim=3, kernel_size=12,hidden_channels=125, dilation=2)
        self.conv1_up = XConv(96 + 64 , 128 , dim=3, kernel_size=8,hidden_channels=120, dilation=2)
        

        self.lin1 = Lin(384, 256)
        self.lin2 = Lin(256, 128)
        self.down_sample = down_sample_layer
        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, kernel_size=1),
            )
        self.num_classes = num_classes
    def forward(self, pos, batch):

        x1 = F.relu(self.conv1(None, pos, batch))
        x2, pos1, batch1 = self.down_sample(x1, pos, batch)
        x2 = F.relu(self.conv2(x2, pos1, batch1))
        x3, pos2, batch2 = self.down_sample(x2, pos1, batch1)
        x3 = F.relu(self.conv3(x3, pos2, batch2))
        x4 = F.relu(self.conv4(x3, pos2, batch2))
        
        x_glob = global_mean_pool(x4, batch2)
        x_glob = F.relu(self.lin1(x_glob))
        x_glob = F.relu(self.lin2(x_glob))
        x_con_glob = x_glob[batch2]
        layer_up1 = torch.cat((x_con_glob,x4),1)
        up4 = F.relu(self.conv4_up(layer_up1, pos2, batch2))
        layer_up2 = torch.cat((up4,x3),1)
        up3 = F.relu(self.conv3_up(layer_up2, pos2, batch2))
        layer_up3 = torch.cat((knn_interpolate(x = up3,pos_x=pos2,batch_x=batch2,k=4,pos_y=pos1,batch_y=batch1),x2),1)

        up2 = F.relu(self.conv2_up(layer_up3, pos1, batch1))
        layer_up4 = torch.cat((knn_interpolate(x = up2,pos_x=pos1,batch_x=batch1,k=4,pos_y=pos,batch_y=batch),x1),1)
        up1 = F.relu(self.conv1_up(layer_up4, pos, batch))
        

        out = torch.unsqueeze(up1.T, 0)
        
        out = self.fc_lyaer(out)
        
        # out_batch = torch.zeros(batch_size,point_number, self.num_classes)
        # out = out.squeeze(0).T
        # for b in range(batch_size):
        #     out_batch[b,:,:] = out[batch == b]
        return out.squeeze(0)

        











# print("runnig")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(num_classes=13).to(device)
# pos = torch.load('/content/ss/tensor_pos.pt')
# batch = torch.load('/content/ss/tensor_batch.pt')
# print(pos.size())
# print(batch.size())
# model(pos,batch)
# def get_n_params(model):
#     pp=0
#     for p in list(model.parameters()):
#         nn=1
#         # print(p)
#         for s in list(p.size()):
#             nn = nn*s
#         pp += nn
#     return pp
# print(get_n_params(model))

# torch.save(model.state_dict(), '/content/ss/model_state_dict.pt')
