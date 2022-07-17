
from turtle import forward
import torch
import torch.nn as nn
from pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule

print(PointnetFPModule)

class PointNet2SemSegSSG(nn.Module):



    def __init__(self):
        super().__init__()
        self.hparams = {"model.use_xyz" : True}
        self.flag = 1
            # self.metric_moiu = MeanIoU(num_classes=13)
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 64, 64, 64],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 128, 128, 128],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 256, 256, 256],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 512, 512, 512],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_lyaer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 22, kernel_size=1),
            

             )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features
        
    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        classification = self.fc_lyaer(l_features[0])

      

        return classification




class model_with_head(torch.nn.Module):
    def __init__(self,model_bkb):
        super().__init__()
        self.Back_bone_model = model_bkb


        self.fc_head = nn.Sequential(

            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 22, kernel_size=1),
        )

    def forward(self,pointcloud):
        bcb_out = self.Back_bone_model(pointcloud)
        out = self.fc_head(bcb_out)
        return out

