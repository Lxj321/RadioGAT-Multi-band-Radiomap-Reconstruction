import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
import os

from torch.nn import Linear
from sklearn.metrics import mean_squared_error
import math

import pdb


#Whether consider distance
kaolvjuli=0
if kaolvjuli==1:
	BlockX=40
	BlockY=40
	sita=0.1
	AreaNum=10
	dcor=4.5

	Num_Freq=5
	Frequency=np.array([1750,2750,3750,4750,5750])
	Carry_Freq=Frequency
	#Freq Hyperparameters
	sigma1=0.5
	chayuzhi=0.01
	Frequency0=Frequency**sigma1

	for AreaIndex in range(AreaNum):

		Spatial_Depth_Map=np.loadtxt('TopologyDepth{}/{}.txt'.format(sita,AreaIndex+1),dtype=float)


		print(np.where(Spatial_Depth_Map==float('inf')))
		mini=np.min(Spatial_Depth_Map)

		#inf to max
		io=np.where(Spatial_Depth_Map==float('inf'))
		Spatial_Depth_Map[np.where(Spatial_Depth_Map==float('inf'))]=-1
		Spatial_Depth_Map[np.where(Spatial_Depth_Map==-1)]=np.max(Spatial_Depth_Map)

		Xiao_Spatial_Depth_Map=Spatial_Depth_Map[80:200,0:120]

		plt.figure(1)
		plt.imshow(Spatial_Depth_Map[80:200,0:120])

		plt.figure(2)
		print(np.min(Spatial_Depth_Map))
		print(np.max(Spatial_Depth_Map))

		plt.imshow((Spatial_Depth_Map-mini)/(np.max(Spatial_Depth_Map)-mini))

		#Freq Matrix







		for i in range(Num_Freq):
			SF=Spatial_Depth_Map/Frequency0[i]
			np.savetxt('SFDepth0{}.txt'.format(i),SF,fmt='%f')

		print(np.shape(Spatial_Depth_Map)[1])
		print(np.shape(Spatial_Depth_Map)[0])
		SFDepth=np.ones((Num_Freq,np.shape(Spatial_Depth_Map)[0],np.shape(Spatial_Depth_Map)[1]))



		for i in range(Num_Freq):
			SFDepth[i,:,:]=np.loadtxt('SFDepth0{}.txt'.format(i),dtype=float)

		SFDepth=(SFDepth-np.min(SFDepth))/(np.max(SFDepth)-np.min(SFDepth))


		plt.figure(3)
		plt.imshow(SFDepth[2,:,:])

		plt.figure(4)
		plt.imshow(SFDepth[3,:,:])

		plt.figure(5)
		plt.imshow(SFDepth[4,:,:])

		plt.figure(6)
		plt.imshow(SFDepth[0,:,:])
		plt.show()
		
		axis=np.loadtxt('New_axis{}.txt'.format(BlockX),dtype=int)
		Num_All_Block=int(axis[AreaIndex,0]*axis[AreaIndex,1]/BlockY/BlockX)
		Num_All_Block_Y=int(axis[AreaIndex,0]/BlockY)
		Num_All_Block_X=int(axis[AreaIndex,1]/BlockX)
		Num_Node_MAX=1600
		Adj=np.ones((Num_Node_MAX,Num_Node_MAX),dtype=int)

		for BlockIndex in range(Num_All_Block):
			m=int(BlockIndex/Num_All_Block_X)
			n=int(BlockIndex-m*Num_All_Block_X)
			suoyin=np.loadtxt('NodeFeatureAxis{}/{}/{}/{}_{}.txt'.format(BlockX,AreaIndex+1,Carry_Freq[0],m+1,n+1),dtype=int)
			Num_Node=np.shape(suoyin)[1]
			for j in range(Num_Freq):
				if not os.path.exists('FreqDepth{}_{}_{}/{}/{}'.format(sita,sigma1,chayuzhi,AreaIndex+1,Frequency[j])):
					os.makedirs('FreqDepth{}_{}_{}/{}/{}'.format(sita,sigma1,chayuzhi,AreaIndex+1,Frequency[j]))
				Adj[:,:]=0
				print(j)
				
				print(chayuzhi)
				pdb.set_trace()
				for mo in range(Num_Node):
					for no in range(Num_Node):
						if np.sqrt((suoyin[0,mo]-suoyin[0,no])**2+(suoyin[1,mo]-suoyin[1,no])**2)<dcor:
						#差值设为0.005
							# print(j)
							# print(suoyin[0,mo])
							# print(suoyin[1,mo])
							# print(abs(SFDepth[j,suoyin[0,mo],suoyin[1,mo]]))
							# print(abs(SFDepth[j,suoyin[0,no],suoyin[1,no]]))
							if abs(SFDepth[j,suoyin[0,mo],suoyin[1,mo]])-abs(SFDepth[j,suoyin[0,no],suoyin[1,no]])<=chayuzhi:
								# print(abs(SFDepth[j,suoyin[0,mo],suoyin[1,mo]]))
								# print(abs(SFDepth[j,suoyin[0,no],suoyin[1,no]]))
								Adj[mo,no]=1
				print(Num_Node)
				Adj[0:Num_Node,0:Num_Node]
				np.savetxt('FreqDepth{}_{}_{}/{}/{}/{}_{}.txt'.format(sita,sigma1,chayuzhi,AreaIndex+1,Frequency[j],m+1,n+1),Adj[0:Num_Node,0:Num_Node],fmt='%d')
else:
	BlockX=40
	BlockY=40
	sita=0.1
	AreaNum=10
	dcor=4.5

	Num_Axis=3
	Num_Freq=5
	Frequency=np.array([1750,2750,3750,4750,5750])
	Carry_Freq=Frequency
	#频率超参
	sigma1=0.5
	chayufenzi=0.1
	Frequency0=Frequency**sigma1

	for AreaIndex in range(AreaNum):
	
			Spatial_Depth_Map1=np.loadtxt('LossProb/{}/1.txt'.format(AreaIndex+1),dtype=float)
			Spatial_Depth_Map2=np.loadtxt('LossProb/{}/2.txt'.format(AreaIndex+1),dtype=float)
			Spatial_Depth_Map3=np.loadtxt('LossProb/{}/3.txt'.format(AreaIndex+1),dtype=float)



			SFDepth=np.ones((Num_Axis,np.shape(Spatial_Depth_Map1)[0],np.shape(Spatial_Depth_Map1)[1]))

			#频率矩阵






			SFDepth[0,:,:]=Spatial_Depth_Map1
			SFDepth[1,:,:]=Spatial_Depth_Map2
			SFDepth[2,:,:]=Spatial_Depth_Map3


			#SFDepth[1,io[0],io[1]]=np.max(SFDepth[0,:,:])
			#SFDepth[2,io[0],io[1]]=np.max(SFDepth[0,:,:])
			#归一化



			# plt.figure(3)
			# plt.imshow(SFDepth[2,:,:])

			# plt.figure(4)
			# plt.imshow(SFDepth[0,:,:])

			# plt.figure(5)
			# plt.imshow(SFDepth[1,:,:])
			# plt.show()
			

			axis=np.loadtxt('New_axis{}.txt'.format(BlockX),dtype=int)
			Num_All_Block=int(axis[AreaIndex,0]*axis[AreaIndex,1]/BlockY/BlockX)
			Num_All_Block_Y=int(axis[AreaIndex,0]/BlockY)
			Num_All_Block_X=int(axis[AreaIndex,1]/BlockX)
			Num_Node_MAX=1600
			Adj=np.ones((Num_Node_MAX,Num_Node_MAX),dtype=int)

			for BlockIndex in range(Num_All_Block):
				m=int(BlockIndex/Num_All_Block_X)
				n=int(BlockIndex-m*Num_All_Block_X)
				suoyin=np.loadtxt('NodeFeatureAxis{}/{}/{}/{}_{}.txt'.format(BlockX,AreaIndex+1,Carry_Freq[0],m+1,n+1),dtype=int)
				Num_Node=np.shape(suoyin)[1]
				for j in range(Num_Freq):
					if not os.path.exists('FreqDepth{}/{}/{}'.format(chayufenzi,AreaIndex+1,Frequency[j])):
						os.makedirs('FreqDepth{}/{}/{}'.format(chayufenzi,AreaIndex+1,Frequency[j]))
					Adj[:,:]=0
					print(j)
					chayuzhi=chayufenzi/math.log(Frequency[j],10)
					print(chayuzhi)
					print(Frequency[j])
					#pdb.set_trace()
					for mo in range(Num_Node):
						for no in range(Num_Node):
							if np.sqrt((suoyin[0,mo]-suoyin[0,no])**2+(suoyin[1,mo]-suoyin[1,no])**2)<dcor:
							#差值设为0.005
								# print(j)
								# print(suoyin[0,mo])
								# print(suoyin[1,mo])

								if (abs(SFDepth[0,suoyin[0,mo],suoyin[1,mo]])-abs(SFDepth[0,suoyin[0,no],suoyin[1,no]])<=chayuzhi) and (abs(SFDepth[1,suoyin[0,mo],suoyin[1,mo]])-abs(SFDepth[1,suoyin[0,no],suoyin[1,no]])<=chayuzhi) and (abs(SFDepth[2,suoyin[0,mo],suoyin[1,mo]])-abs(SFDepth[2,suoyin[0,no],suoyin[1,no]])<=chayuzhi):
									# print(abs(SFDepth[j,suoyin[0,mo],suoyin[1,mo]]))
									# print(abs(SFDepth[j,suoyin[0,no],suoyin[1,no]]))
									Adj[mo,no]=1
					print(Num_Node)
					Adj[0:Num_Node,0:Num_Node]
					np.savetxt('FreqDepth{}/{}/{}/{}_{}.txt'.format(chayufenzi,AreaIndex+1,Frequency[j],m+1,n+1),Adj[0:Num_Node,0:Num_Node],fmt='%d')





