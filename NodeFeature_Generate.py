import numpy as np
import string
import pdb
import matplotlib.pyplot as plt
import random
import scipy.io as scio
import os


NumFreq=5
#单位MHz
Freq=[1750,2750,3750,4750,5750]
NumArea=10

Blockx=40
Blocky=40


for i in range(NumArea):
	for j in range(NumFreq):
		if not os.path.exists('NodeFeatureAxis{}/{}/{}'.format(Blockx,i+1,Freq[j])):
			os.makedirs('NodeFeatureAxis{}/{}/{}'.format(Blockx,i+1,Freq[j]))
		if not os.path.exists('NodeFeatureStrength{}/{}/{}'.format(Blockx,i+1,Freq[j])):
			os.makedirs('NodeFeatureStrength{}/{}/{}'.format(Blockx,i+1,Freq[j]))
		All_Area=np.loadtxt('NewStrength{}/{}/{}.txt'.format(Blockx,i+1,Freq[j]),dtype=float)
		for m in range(int(np.shape(All_Area)[0]/Blocky)):
			for n in range(int(np.shape(All_Area)[1]/Blockx)):
				Data=np.loadtxt('SplitStrength{}/{}/{}/{}_{}.txt'.format(Blockx,i+1,Freq[j],m+1,n+1),dtype=float)
#生成格式[2,N_Node],0行是排y，1行是列x
				np.savetxt('NodeFeatureAxis{}/{}/{}/{}_{}.txt'.format(Blockx,i+1,Freq[j],m+1,n+1),np.where(Data!=0),fmt='%d')
				np.savetxt('NodeFeatureStrength{}/{}/{}/{}_{}.txt'.format(Blockx,i+1,Freq[j],m+1,n+1),Data[np.where(Data!=0)],fmt='%.4f')



