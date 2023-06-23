import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self,norm_type,norm_type_last):
        super(Net, self).__init__()
        self.norm_type = norm_type
        self.norm_type_last = norm_type_last
        self.conv1 = self.conv_block(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False, dropout=0.01,norm_type=self.norm_type)
        self.conv2 = self.conv_block(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False, dropout=0.01,norm_type=self.norm_type)
        self.conv3 = self.conv_block(in_channels=12, out_channels=24, kernel_size=1, stride=1, padding=0, bias=False, dropout=0.01,norm_type=self.norm_type)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = self.conv_block(in_channels=36, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False, dropout=0.01,norm_type=self.norm_type)
        self.conv5 = self.conv_block(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False, dropout=0.01,norm_type=self.norm_type)
        self.conv6 = self.conv_block(in_channels=12, out_channels=36, kernel_size=3, stride=1, padding=1, bias=False, dropout=0.01,norm_type=self.norm_type)
        self.conv7 = self.conv_block(in_channels=48, out_channels=96, kernel_size=1, stride=1, padding=0, bias=False, dropout=0.01,norm_type=self.norm_type)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv8 = self.conv_block(in_channels=144, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False, dropout=0.01,norm_type=self.norm_type)
        self.conv9 = self.conv_block(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, dropout=0.01,norm_type=self.norm_type)
        self.conv10 = self.conv_block(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False, dropout=0.01,norm_type=self.norm_type)
        self.avg_pool =  nn.AvgPool2d(kernel_size=8,stride=1)
        self.conv11 = self.conv_block(in_channels=72, out_channels=10, kernel_size=3, stride=1, padding=1, bias=False, dropout=0,norm_type=self.norm_type_last)


    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x,self.conv2(x)],1)
        x = torch.cat([x,self.conv3(x)],1)
        x = self.pool1(x)
        x = self.conv4(x)
        x = torch.cat([x,self.conv5(x)],1)
        x = torch.cat([x,self.conv6(x)],1)
        x = torch.cat([x,self.conv7(x)],1)
        x = self.pool2(x)
        x = self.conv8(x)
        x = torch.cat([x,self.conv9(x)],1)
        x = torch.cat([x,self.conv10(x)],1)
        x = self.avg_pool(x)
        x= self.conv11(x)
        x = torch.flatten(x,1)
        return F.log_softmax(x, dim=1)

    def conv_block(self,in_channels,out_channels,kernel_size,stride,padding,bias,dropout,norm_type="batch_norm"):
        if norm_type != None:
            if norm_type == "batch_norm":
                normalization = nn.BatchNorm2d(out_channels)
            elif norm_type == "layer_norm":
                normalization = nn.LayerNorm()
            elif norm_type == "group_norm":
                normalization = nn.GroupNorm()
            x = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=bias),
                nn.ReLU(),
                normalization,
                nn.Dropout(dropout))
            return x
        else:
            x = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride,padding=padding, bias=bias))
            return x


if __name__ == "__main__":
    config_file = r"C:\Users\vinot\Python_files\Deep_learning_algorithims\Deep learning basics\School_AI\22nd_June_2023\config.cfg"
    Net(config_file,"NETWORK1")





