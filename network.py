import torch
import torch.nn as nn
import math
import copy
import random
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from module import *
import torch.nn.utils.spectral_norm as SpectralNorm

class MNIST_Small_ConvNet(nn.Module):
    def __init__(self,
                 num_classes=10, 
                 non_negative = [True, True, True, True], 
                 norm = [False, False, False, False]):
        
        super(MNIST_Small_ConvNet, self).__init__()
        self.conv1 = RobustConv2d(1,16,4,2, padding = 1, non_negative =non_negative[0])
        if norm[0]:
            self.conv1 = SpectralNorm(self.conv1)
        self.conv2 = RobustConv2d(16,32,4,1, padding= 1, non_negative =non_negative[1])
        if norm[1]:
            self.conv2 = SpectralNorm(self.conv2)
        self.fc1 = RobustLinear(13*13*32, 100, non_negative =non_negative[2])
        if norm[2]:
            self.fc1 = SpectralNorm(self.fc1)
        self.fc2 = RobustLinear(100,10, non_negative =non_negative[3])
        if norm[3]:
            self.fc2 = SpectralNorm(self.fc2)
        
        self.activation = F.relu 
        self.score_function = self.fc2
        
    def forward_g(self,x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.activation(x)
        return x
    
    def forward(self, x):
        x = self.score_function(self.forward_g(x))
        return x
    
    
        
        
                
class MNIST_Medium_ConvNet(nn.Module):
    def __init__(self, 
                 num_classes=10, 
                 non_negative = [True, True, True, True, True, True, True], 
                 norm = [False, False, False, False, False, False, False]):
        
        super(MNIST_Medium_ConvNet, self).__init__()
        self.conv1 = RobustConv2d(1,32,3, stride = 1, padding = 1, non_negative = non_negative[0])
        if norm[0]:
            self.conv1 = SpectralNorm(self.conv1)
        self.conv2 = RobustConv2d(32,32,4, stride = 2, padding= 1, non_negative = non_negative[1])
        if norm[1]:
            self.conv2 = SpectralNorm(self.conv2)
        self.conv3 = RobustConv2d(32,64,3, stride = 1, padding = 1, non_negative = non_negative[2])
        if norm[2]:
            self.conv3 = SpectralNorm(self.conv3)
        self.conv4 = RobustConv2d(64,64,4, stride = 2, padding= 1, non_negative = non_negative[3])
        if norm[3]:
            self.conv4 = SpectralNorm(self.conv4)
        
        self.fc1 = RobustLinear(7*7*64, 512, non_negative = non_negative[4])
        if norm[4]:
            self.fc1 = SpectralNorm(self.fc1)
        self.fc2 = RobustLinear(512, 512, non_negative = non_negative[5])
        if norm[5]:
            self.fc2 = SpectralNorm(self.fc2)
        self.fc3 = RobustLinear(512,10, non_negative = non_negative[6])
        if norm[6]:
            self.fc3 = SpectralNorm(self.fc3)
        
        self.activation = F.relu
        self.score_function = self.fc3
        
    def forward_g(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x
        
    def forward(self, x):
        x = self.score_function(self.forward_g(x))
        return x
    
    
    
class MNIST_Large_ConvNet(nn.Module):
    def __init__(self, 
                 num_classes=10, 
                 non_negative = [True, True, True, True, True, True, True], 
                 norm = [False, False, False, False, False, False, False]):
        
        super(MNIST_Large_ConvNet, self).__init__()
        self.conv1 = RobustConv2d(1,64,3, stride = 1, padding = 1, non_negative = non_negative[0])
        if norm[0]:
            self.conv1 = SpectralNorm(self.conv1)
        self.conv2 = RobustConv2d(64,64,3, stride = 1, padding = 1, non_negative = non_negative[1])
        if norm[1]:
            self.conv2 = SpectralNorm(self.conv2)
        self.conv3 = RobustConv2d(64,128,3, stride = 2, padding= 1, non_negative = non_negative[2])
        if norm[2]:
            self.conv3 = SpectralNorm(self.conv3)
        self.conv4 = RobustConv2d(128,128,3, stride = 1, padding = 1, non_negative = non_negative[3])
        if norm[3]:
            self.conv4 = SpectralNorm(self.conv4)
        self.conv5 = RobustConv2d(128,128,3, stride = 1, padding= 1, non_negative = non_negative[4])
        if norm[4]:
            self.conv5 = SpectralNorm(self.conv5)
        
        self.fc1 = RobustLinear(14*14*128, 200, non_negative = non_negative[5])
        if norm[5]:
            self.fc1 = SpectralNorm(self.fc1)
        self.fc2 = RobustLinear(200, 10, non_negative = non_negative[6])
        if norm[6]:
            self.fc2 = SpectralNorm(self.fc2)
            
        self.activation = F.relu
        self.score_function = self.fc2
    
    def forward_g(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.activation(x)
        return x

    def forward(self, x):
        x = self.score_function(self.forward_g(x))
        return x

class Cifar_Small_ConvNet(nn.Module):
    def __init__(self, 
                 num_classes=10, 
                 non_negative = [True, True, True, True], 
                 norm = [False, False, False, False]):
        
        super(Cifar_Small_ConvNet, self).__init__()
        self.conv1 = RobustConv2d(3,16,4, stride = 2, padding = 0, non_negative = non_negative[0])
        if norm[0]:
            self.conv1 = SpectralNorm(self.conv1)
        self.conv2 = RobustConv2d(16,32,4, stride = 1, padding= 0, non_negative = non_negative[1])
        if norm[1]:
            self.conv2 = SpectralNorm(self.conv2)
        self.fc1 = RobustLinear(12*12*32, 100, non_negative = non_negative[2])
        if norm[2]:
            self.fc1 = SpectralNorm(self.fc1)
        self.fc2 = RobustLinear(100,10, non_negative = non_negative[3])
        if norm[3]:
            self.fc2 = SpectralNorm(self.fc2)
            
        self.deconv1 = nn.ConvTranspose2d(32,16,4, padding = 0, stride = 1)
        self.deconv2 = nn.ConvTranspose2d(16,3,4, padding = 0, stride = 2)
        
        self.activation = F.relu
        self.score_function = self.fc2
        
        self.image_norm = ImageNorm([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
    
    def forward_conv(self,x):
        x = self.image_norm(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x
        
    def forward_g(self, x):
        x = self.forward_conv(x) 
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.activation(x)
        return x
        
    def forward(self, x):
        x = self.score_function(self.forward_g(x))
        return x
        
    
    
    
class Cifar_Medium_ConvNet(nn.Module):
    def __init__(self, 
                 num_classes=10, 
                 non_negative = [True, True, True, True, True, True, True], 
                 norm = [False, False, False, False, False, False, False]):
        
        super(Cifar_Medium_ConvNet, self).__init__()
        self.conv1 = RobustConv2d(3,32,3, stride = 1, padding = 1, non_negative = non_negative[0])
        if norm[0]:
            self.conv1 = SpectralNorm(self.conv1)
        self.conv2 = RobustConv2d(32,32,4, stride = 2, padding= 1, non_negative = non_negative[1])
        if norm[1]:
            self.conv2 = SpectralNorm(self.conv2)
        self.conv3 = RobustConv2d(32,64,3, stride = 1, padding = 1, non_negative = non_negative[2])
        if norm[2]:
            self.conv3 = SpectralNorm(self.conv3)
        self.conv4 = RobustConv2d(64,64,4, stride = 2, padding= 1, non_negative = non_negative[3])
        if norm[3]:
            self.conv4 = SpectralNorm(self.conv4)
        
        self.fc1 = RobustLinear(8*8*64, 512, non_negative = non_negative[4])
        if norm[4]:
            self.fc1 = SpectralNorm(self.fc1)
        self.fc2 = RobustLinear(512, 512, non_negative = non_negative[5])
        if norm[5]:
            self.fc2 = SpectralNorm(self.fc2)
        self.fc3 = RobustLinear(512,10, non_negative = non_negative[6])
        if norm[6]:
            self.fc3 = SpectralNorm(self.fc3)
        self.activation = F.relu
        self.score_function = self.fc3
        self.image_norm = ImageNorm([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
        
    def forward_g(self, x):
        x = self.image_norm(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x
    
    def forward(self, x):
        x = self.score_function(self.forward_g(x))
        return x
    
class Cifar_Large_ConvNet(nn.Module):
    def __init__(self, 
                 num_classes=10, 
                 non_negative = [True, True, True, True, True, True, True], 
                 norm = [False, False, False, False, False, False, False]):
        
        super(Cifar_Large_ConvNet, self).__init__()
        self.conv1 = RobustConv2d(3,64,3, stride = 1, padding = 1, non_negative = non_negative[0])
        if norm[0]:
            self.conv1 = SpectralNorm(self.conv1)
        
        self.conv2 = RobustConv2d(64,64,3, stride = 1, padding = 1, non_negative = non_negative[1])
        if norm[1]:
            self.conv2 = SpectralNorm(self.conv2)
        
        self.conv3 = RobustConv2d(64,128,3, stride = 2, padding= 1, non_negative = non_negative[2])
        if norm[2]:
            self.conv3 = SpectralNorm(self.conv3)
        
        self.conv4 = RobustConv2d(128,128,3, stride = 1, padding = 1, non_negative = non_negative[3])
        if norm[3]:
            self.conv4 = SpectralNorm(self.conv4)
        
        self.conv5 = RobustConv2d(128,128,3, stride = 1, padding= 1, non_negative = non_negative[4])
        if norm[4]:
            self.conv5 = SpectralNorm(self.conv5)
        
        self.fc1 = RobustLinear(16*16*128, 200, non_negative = non_negative[5])
        if norm[5]:
            self.fc1 = SpectralNorm(self.fc1)
        
        self.fc2 = RobustLinear(200,10, non_negative = non_negative[6])
        if norm[6]:
            self.fc2 = SpectralNorm(self.fc2)
        
        self.deconv1 = nn.ConvTranspose2d(128,128,3, padding = 1, stride = 1)
        self.deconv2 = nn.ConvTranspose2d(128,128,3, padding = 1, stride = 1)
        self.deconv3 = nn.ConvTranspose2d(128,64,3, padding = 1, stride = 2, output_padding = 1)
        self.deconv4 = nn.ConvTranspose2d(64,64,3, padding = 1, stride = 1)
        self.deconv5 = nn.ConvTranspose2d(64,3,3, padding = 1, stride = 1)
        
        self.activation = F.leaky_relu
        self.score_function = self.fc2
        self.image_norm = ImageNorm([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
    
    def forward_g(self, x):
        x = self.image_norm(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.forward_conv(x)
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.activation(x)
        return x
    
    def forward(self, x):
        x =  self.score_function(self.forward_g(x))
        return x
    
    
class Cifar_VGG(nn.Module):
    def __init__(self, 
                 num_classes=10, 
                 non_negative = [True, True, True, True, True, True], 
                 norm = [False, False, False, False, False, False]):
        
        super(Cifar_VGG, self).__init__()
        self.conv1 = RobustConv2d(3,64,3, stride = 1, padding = 1, non_negative = non_negative[0])
        if norm[0]:
            self.conv1 = SpectralNorm(self.conv1)
        
        self.conv2 = RobustConv2d(64,64,3, stride = 1, padding = 1, non_negative = non_negative[1])
        if norm[1]:
            self.conv2 = SpectralNorm(self.conv2)
        
        self.conv3 = RobustConv2d(64,128,3, stride = 2, padding= 1, non_negative = non_negative[2])
        if norm[2]:
            self.conv3 = SpectralNorm(self.conv3)
        
        self.conv4 = RobustConv2d(128,128,3, stride = 1, padding = 1, non_negative = non_negative[3])
        if norm[3]:
            self.conv4 = SpectralNorm(self.conv4)
        
        self.fc1 = RobustLinear(16*16*128, 512, non_negative = non_negative[4])
        if norm[4]:
            self.fc1 = SpectralNorm(self.fc1)
        
        self.fc2 = RobustLinear(512,10, non_negative = non_negative[5])
        if norm[5]:
            self.fc2 = SpectralNorm(self.fc2)
        
        self.activation = F.relu
        self.score_function = self.fc2
        self.image_norm = ImageNorm([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
    
    def forward_g(self, x):
        x = self.image_norm(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.activation(x)
        return x
    
    def forward(self, x):
        x =  self.score_function(self.forward_g(x))
        return x
    

                
