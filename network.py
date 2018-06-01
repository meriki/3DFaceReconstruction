import torch,torch.nn as nn
import torch.legacy.nn as legacynn
import torch.autograd
import torchvision.transforms as transform


import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    expansion = 2

    def __init__(self, inplanes, outplanes):
        super(Residual, self).__init__()
        
        self.inplanes = inplanes
        self.outplanes = outplanes
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes/2, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(outplanes/2)
        self.conv2 = nn.Conv2d(outplanes/2, outplanes/2, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(outplanes/2)
        self.conv3 = nn.Conv2d(outplanes/2, outplanes, kernel_size=1, stride=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv = nn.Conv2d(inplanes,outplanes,kernel_size=1)
#	print 'res init done'        
#        

    def forward(self, x):
        
        residual = x
        if(self.inplanes != self.outplanes):
            residual = self.conv(x)
        
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out

class Hourglass(nn.Module):
    def __init__(self, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = Residual
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(depth)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1,bias=True)
        self.bn = nn.BatchNorm2d(256) 
        self.relu = nn.ReLU(inplace=True)
 #       print 'over init at hg'
        
    def _make_hour_glass(self,depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(Residual(256,256))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)
    
    def _hour_glass_forward(self, n, x):
        up1 = self.hg[3][0](x)
        up1 = self.hg[3][1](up1)
        up1 = self.hg[3][2](up1)
        if(n>1):
            up1 = self._hour_glass_forward(n-1, up1)        
        pool = nn.MaxPool2d(2,stride=2)(x)  
        low1 = self.hg[2][0](pool)
        low1 = self.hg[2][1](low1)
        low1 = self.hg[2][2](low1)
        low1 = self.hg[1][0](low1)
        low1 = self.hg[1][1](low1)
        low2 = self.hg[1][2](low1)
        if (n==1):
            low2 = self.hg[0][0](low2)
            low2 = self.hg[0][1](low2)
            low2 = self.hg[0][2](low2)
        
        sam = self.upsample(low2)
        
        out = up1 + sam
        return out
    
    def forward(self, x):
    	 return self._hour_glass_forward(self.depth, x)

class HourglassNet(nn.Module):
    
    def __init__(self, num_stacks=2):
        super(HourglassNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(64) 
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer1 = Residual(inplanes=64, outplanes=128)
        self.layer2 = Residual(inplanes=128, outplanes=128)
        self.layer3 = Residual(inplanes=128, outplanes=256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1,bias=True)
        self.bn2 = nn.BatchNorm2d(256) 
        self.conv3 = nn.Conv2d(256,200,kernel_size=1)
        self.upsamplingbilinear = nn.Upsample(scale_factor=4,mode = 'bilinear')
        self.sig = nn.Sigmoid()

        ch = 256
        hg, res, fc,  fc_ = [], [], [], []
        
        for i in range(2):
            hg.append(Hourglass(4))
            res.append(Residual(inplanes=256, outplanes=256))
            fc.append(self._make_fc(ch, ch))
            fc_.append(self._make_fc(ch, ch))
            
    	self.hg = nn.ModuleList(hg)
	self.res = nn.ModuleList(res)
	self.fc = nn.ModuleList(fc)
	self.fc_ = nn.ModuleList(fc_)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )
    
    def forward(self,x):
        
       # out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 

        x = self.layer1(x)  
        x = self.maxpool(x)
        x = self.layer2(x)  
        x = self.layer3(x) 
        
        for i in range(2):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            y = self.fc_[i](y)
            x = x+y
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        out = self.upsamplingbilinear(out)
  #      print 'after upsb'
        out = self.sig(out)
#	print 'why?'        
        return out

