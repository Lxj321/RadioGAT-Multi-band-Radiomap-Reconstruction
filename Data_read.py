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
	if not os.path.exists('Strength/{}'.format(i+1)):
		os.makedirs('Strength/{}'.format(i+1))
	if not os.path.exists('NewStrength{}/{}'.format(Blockx,i+1)):
		os.makedirs('NewStrength{}/{}'.format(Blockx,i+1))
	for j in range(NumFreq):
		FileName='receivedpower_{}MHz_mat/TOTAL_REC_{}MHz_{}.mat'.format(Freq[j],Freq[j],i+1)
		Data=scio.loadmat(FileName)
		#print(Data)
		NewData=np.array(Data['powers{}'.format(j+1)])
		#print(NewData)
		SaveDataName='Strength/{}/{}.txt'.format(i+1,Freq[j])
		np.savetxt(SaveDataName,NewData,fmt='%.4f')

		print(np.shape(NewData))

		ProcessdData=NewData[0:Blocky*int(np.shape(NewData)[0]/Blocky),0:Blockx*int(np.shape(NewData)[1]/Blockx)]
		np.savetxt('NewStrength{}/{}/{}.txt'.format(Blockx,i+1,Freq[j]),ProcessdData,fmt='%.4f')



#对新的数据集进行分割
for i in range(NumArea):
	for j in range(NumFreq):
		if not os.path.exists('SplitStrength{}/{}/{}'.format(Blockx,i+1,Freq[j])):
			os.makedirs('SplitStrength{}/{}/{}'.format(Blockx,i+1,Freq[j]))
		All_Area=np.loadtxt('NewStrength{}/{}/{}.txt'.format(Blockx,i+1,Freq[j]),dtype=float)

#先横向索引，再纵向索引，即y先不变，x向右,[y,x]

		for m in range(int(np.shape(All_Area)[0]/Blocky)):
			for n in range(int(np.shape(All_Area)[1]/Blockx)):
				np.savetxt('SplitStrength{}/{}/{}/{}_{}.txt'.format(Blockx,i+1,Freq[j],m+1,n+1),All_Area[m*Blocky:(m+1)*Blocky,n*Blockx:(n+1)*Blockx],fmt='%.4f')


















