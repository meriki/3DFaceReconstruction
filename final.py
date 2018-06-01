import torch,torch.nn as nn
import torch.legacy.nn as legacynn
import torch.autograd
import torchvision.transforms as transform
from network import HourglassNet
import scipy.io as io
import os,cv2
import torch.nn.parallel
import torch.optim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

gpus=[0,1]
criterion = torch.nn.MSELoss(size_average=True).cuda()

#optimizer = torch.optim.RMSprop(model.parameters(), 
#                                lr=args.lr,
#                                momentum=args.momentum,
#                                weight_decay=args.weight_decay)

model = HourglassNet(num_stacks=2)
model = torch.nn.DataParallel(model).cuda(device=gpus[0])


optimizer = torch.optim.RMSprop(model.parameters(), 
                                lr=1.0e-4,
                                momentum=0,
                                weight_decay=0)



model.train()

images = sorted(os.listdir('./train202/'))

#traini = 

tens = transform.ToTensor()
epochs =  40
while(epochs>0):
    for i,image in enumerate(images):
        img = cv2.imread('./train202/'+image);
      	img = tens(img)
        img = img.unsqueeze(0)
        print img.shape
	img = torch.autograd.Variable(img.cuda())
        output = model(img)
        print './trainmat202/'+image[:-4]+'.mat'
	mat = io.loadmat('./trainmat202/'+image[:-6]+'.mat')
#        print mat
	target_var = mat['target']
	#rint 'here', sum(target_var)
	target_var = tens(target_var)
	target_var = torch.autograd.Variable(target_var.float().cuda())
        loss = criterion(output, target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
	
        if(i%50==0):	
   	    torch.save(model.state_dict(), 'chkpts/'+str(epochs)+'_'+str(i)+'.pth.tar')
	    #print output
	    print loss
	    #s = output
            
	    

	if(epochs%2==0 and i==199):
	   torch.save(model.state_dict(), 'chkpts/'+str(epochs)+'_'+str(i)+'.pth')
	   print output
	   resu = cv2.imread('train202/image_test_0021-0.jpg')
	   resu = tens(resu)
           resu = resu.unsqueeze(0)
	   resu = torch.autograd.Variable(resu.cuda())
	   big = model(resu)
	   big = big.data.cpu().float()
	   big[big>=0.5]=1
	   #big = (big<0.5).float()
           pos = np.where(big==1)
	   ax.scatter(pos[0],pos[1],pos[2],c='red') 
	   fig.savefig('result6.jpg', dpi=fig.dpi)
          
	print i,epochs	
    epochs = epochs - 1
    print epochs
	#scipy.io.savemat(filepath, mdict={'preds' : preds})    
