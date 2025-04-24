import numpy as np
import string
import pdb
import matplotlib.pyplot as plt
import random
import torch
import os
import torch.nn as nn

#Consider Distance?
kaolvjuli=0

#Map Division, set 1 building，0 no building
blockx=40
blocky=40
NumArea=10
NumFreq=5
Freq=[1750,2750,3750,4750,5750]

StationsPosition=np.loadtxt('stations_position.txt',dtype=int)
sigma=0.1


if not os.path.exists('TopologyDepth{}'.format(sigma)):
    os.makedirs('TopologyDepth{}'.format(sigma))


for AreaIndex in range(NumArea):
	if not os.path.exists('LossProb/{}'.format(AreaIndex+1)):
		os.makedirs('LossProb/{}'.format(AreaIndex+1))
	Sampling_Strength=np.loadtxt('NewStrength{}/{}/{}.txt'.format(blockx,AreaIndex+1,Freq[0]))
	

	All_Topology=Sampling_Strength
	All_Topology[np.where(All_Topology!=0)]=-1
	All_Topology[np.where(All_Topology==0)]=1
	All_Topology[np.where(All_Topology==-1)]=0
	print(np.shape(np.where(All_Topology==0)))
	print(max((np.where(All_Topology==0)[0])))

	#Compute Loss Probability，three transmitters，three array
	Num_tri=3
	Axis_tr=np.ones((2,3),dtype=int)
	Axis_tr[:,0]=np.array(StationsPosition[AreaIndex,0:2])-1
	Axis_tr[:,1]=np.array(StationsPosition[AreaIndex,2:4])-1
	Axis_tr[:,2]=np.array(StationsPosition[AreaIndex,4:6])-1
	print(np.shape(All_Topology))

	#Patch for grids
	stride0=1
	pool=nn.AvgPool2d(stride0,stride=stride0)
	zhongjian=torch.tensor(np.ones((1,np.shape(All_Topology)[0],np.shape(All_Topology)[1])))
	print(np.shape(zhongjian))
	zhongjian[0,:,:]=torch.tensor(All_Topology)
	Avg_All_Topology_Tensor=pool(torch.tensor(zhongjian))
	Avg_All_Topology=Avg_All_Topology_Tensor[0].numpy()



	#Patch?
	#Locations x、y of transmitter after patching
	Avg_Axis_tr=np.ones((2,3),dtype=int)
	Avg_Axis_tr[:,0]=Axis_tr[:,0]/stride0
	Avg_Axis_tr[:,1]=Axis_tr[:,1]/stride0
	Avg_Axis_tr[:,2]=Axis_tr[:,2]/stride0



	Para_Avg_All_Topology=np.ones((3,3,np.shape(Avg_All_Topology)[0],np.shape(Avg_All_Topology)[1]))
	Para_Avg_All_Topology[:,2,:,:]=0
	for i in range(Num_tri):
		Para_Avg_All_Topology[i,1,np.where(Para_Avg_All_Topology[i,1,:,:]>=0)[0],np.where(Para_Avg_All_Topology[i,1,:,:]>=0)[1]]=np.sqrt((np.where(Para_Avg_All_Topology[i,1,:,:]>=0)[0]-Avg_Axis_tr[0,i])**2+(np.where(Para_Avg_All_Topology[i,1,:,:]>=0)[1]-Avg_Axis_tr[1,i])**2)
		for k in range(np.shape(Avg_All_Topology)[1]):
			print(k)
			for j in range(np.shape(Avg_All_Topology)[0]):
				#print(j)
				if k<Avg_Axis_tr[1,i]:
					X_Index=np.array(range(k,Avg_Axis_tr[1,i],1))
					Y_Index=np.array((Avg_Axis_tr[0,i]-j)/(Avg_Axis_tr[1,i]-k)*(X_Index-k)+j).astype(int)
					Para_Avg_All_Topology[i,0,j,k]=np.sum(Avg_All_Topology[Y_Index,X_Index])/np.shape(X_Index)[0]
				else:
					if k>Avg_Axis_tr[1,i]:
						X_Index=np.array(range(k,Avg_Axis_tr[1,i],-1))
						Y_Index=np.array((Avg_Axis_tr[0,i]-j)/(Avg_Axis_tr[1,i]-k)*(X_Index-k)+j).astype(int)
						Para_Avg_All_Topology[i,0,j,k]=np.sum(Avg_All_Topology[Y_Index,X_Index])/np.shape(X_Index)[0]
					else:
						X_Index=Avg_Axis_tr[1,i]
						if j<Avg_Axis_tr[0,i]:
							Y_Index=np.array(range(j,Avg_Axis_tr[0,i],1))
							Para_Avg_All_Topology[i,0,j,k]=np.sum(Avg_All_Topology[Y_Index,X_Index])/np.shape(Y_Index)
						else:
							if j>Avg_Axis_tr[0,i]:
								Y_Index=np.array(range(j,Avg_Axis_tr[0,i],-1))
								Avg_All_Topology[Y_Index.astype(int),X_Index]
								Para_Avg_All_Topology[i,0,j,k]=np.sum(Avg_All_Topology[Y_Index,X_Index])/np.shape(Y_Index)
							else:
								Y_Index=Avg_Axis_tr[0,i]
								Para_Avg_All_Topology[i,0,j,k]=np.sum(Avg_All_Topology[Y_Index,X_Index])

	#no patch
	print(np.where(Para_Avg_All_Topology[0,0,:,:]==np.max(Para_Avg_All_Topology[0,0,:,:]))) 
	if kaolvjuli==1:
		for i in range(Num_tri):
		
			Para_Avg_All_Topology[i,2,:,:]=(1-Para_Avg_All_Topology[i,0,:,:])/(Para_Avg_All_Topology[i,1,:,:]**sigma)

			

		DepthMap=np.ones((np.shape(Avg_All_Topology)[0],np.shape(Avg_All_Topology)[1]))
		DepthMap[:,:]=Para_Avg_All_Topology[0,2,:,:]+Para_Avg_All_Topology[1,2,:,:]+Para_Avg_All_Topology[2,2,:,:]
		np.savetxt('TopologyDepth{}/{}.txt'.format(sigma,AreaIndex+1),DepthMap,fmt='%f')
	else:
		for i in range(Num_tri):
			Para_Avg_All_Topology[i,2,:,:]=1-Para_Avg_All_Topology[i,0,:,:]
			np.savetxt('LossProb/{}/{}.txt'.format(AreaIndex+1,i+1),Para_Avg_All_Topology[i,2,:,:],fmt='%f')
		#LossProb=np.ones((np.shape(Avg_All_Topology)[0],np.shape(Avg_All_Topology)[1]))
		


