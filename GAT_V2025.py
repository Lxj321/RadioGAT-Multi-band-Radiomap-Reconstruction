import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np

from torch.nn import Linear

from sklearn.metrics import mean_squared_error


import pdb



#GATConv Definition (You can also use "GATConv" layer in DGL)
class GATConv(nn.Module):
    def __init__(self, in_channels, out_channels, add_self_loops=False, bias=True):
        super(GATConv, self).__init__()
        self.in_channels = in_channels # Feature Input Num of Nodes
        self.out_channels = out_channels # Feature Output Num of Nodes
        self.adj = None
        self.add_self_loops = add_self_loops
        self.weight_w = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.weight_a = nn.Parameter(torch.FloatTensor(out_channels * 2, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels, 1))
        else:
            self.register_parameter('bias', None)
        
        self.leakyrelu = nn.LeakyReLU()
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight_w)
        nn.init.xavier_uniform_(self.weight_a)
        
        if self.bias != None:
            nn.init.zeros_(self.bias)
        
    def forward(self, x, edge_index):
        # 1.Calculate wh
        wh = torch.mm(x, self.weight_w)
                                  
        
        e = torch.mm(wh, self.weight_a[: self.out_channels]) + torch.matmul(wh, self.weight_a[self.out_channels:]).T

        # 3.Activation
        e = self.leakyrelu(e)
        
        # Adj Obtain
      #  if self.adj == None:
         #   self.adj = to_dense_adj(edge_index).squeeze()
            
            # 5.Add Self_loop
        #if self.add_self_loops:
        #    self.adj += torch.eye(x.shape[0])
        
        # 6. Attention array obtain
        attention = torch.where(edge_index > 0, e, -1e9 * torch.ones_like(e))
        
        # 7.Normalization
        attention = F.softmax(attention, dim=1)
        
        output = torch.mm(attention, wh)
        
        # 9.Bias
        if self.bias != None:
            return output + self.bias.flatten()
        else:
            return output                                   

#GAT Definition
class GAT(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.adj=torch.eye(num_classes)
        self.conv1 = GATConv(in_channels=num_node_features,
                                    out_channels=80)
        self.conv2 = GATConv(in_channels=80,
                                    out_channels=320)
        self.conv3 = GATConv(in_channels=320,
                                    out_channels=num_classes)
        self.conv4 = GATConv(in_channels=num_node_features,
                                    out_channels=80)
    def forward(self, inp, adj,outAdj):
        x, edge_index,out_edge_index = inp, adj,outAdj
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,p=0.5, training=self.training)

        x = self.conv2(x, out_edge_index)
        x = F.relu(x)
        x = F.dropout(x,p=0.5, training=self.training)
        x = self.conv3(x, out_edge_index)
        return x


BlockX=40
BlockY=40
axis=np.loadtxt('New_axis{}.txt'.format(BlockX),dtype=int)
Carry_Freq=np.array([1750,2750,3750,4750,5750])
Train_Freq_Index=np.array([0,1,2,3])
AreaTrain=10
Num_All_Freq=5
Num_train_Freq=1
Num_Out_Freq=1
Input_Freq_Index=0
Out_Freq_Index=0
Num_Node_Max=1600
#for percent in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]:
for percent in [0.05]:
	masklv=1-percent

	method=0   #0Obsadj，1depadj,2Aroundadj,3TransmitterAdj

	toupiao=0#0不投票，1投票


	StationsPosition=np.loadtxt('stations_position.txt',dtype=int)
	#for AreaIndex in range(1):
	for AreaIndex in range(10):
		

		#The division of training and test block
		Num_All_Block=int(axis[AreaIndex,0]*axis[AreaIndex,1]/BlockY/BlockX)
		Num_All_Block_Y=int(axis[AreaIndex,0]/BlockY)
		Num_All_Block_X=int(axis[AreaIndex,1]/BlockX)

		Train_Percent=0.5
		Num_Train_Block=int(Num_All_Block*Train_Percent)
		Num_Test_Block=Num_All_Block-Num_Train_Block
		
		np.random.seed(0)
		Block_Index0=[i for i in range(Num_All_Block)]
		Block_Index=np.array(Block_Index0)
		np.random.shuffle(Block_Index)
		print("Block Index：",Block_Index)

		#Obtain the number of nodes in each block
		N_Node=np.ones((Num_All_Block),dtype=int)
		for i in Block_Index:
			m=int(i/Num_All_Block_X)
			n=int(i-m*Num_All_Block_X)
			#print(i)
			#print(m)
			#print(n)
			k=len(np.loadtxt('NodeFeatureStrength{}/{}/{}/{}_{}.txt'.format(BlockX,AreaIndex+1,Carry_Freq[0],m+1,n+1),dtype=float))
			N_Node[i]=k

		#print(N_Node)


		#The mask procedure. (Different mask matrix for different blocks and frequencies.)
		# Total size: block num * freq num * Node num (max 1600)

		All_Mask=np.ones((Num_All_Block,Num_All_Freq,Num_Node_Max),dtype=int)

		np.random.seed(0)
		for i in Block_Index:
			train_mask0=np.ones((Num_All_Freq,N_Node[i]),dtype=int)
			train_mask0[:,0:int(N_Node[i]*masklv)]=0
			for j in range(Num_All_Freq):
				np.random.shuffle(train_mask0[j,:])

			All_Mask[i,:,0:N_Node[i]]=train_mask0


		#Feature size：Num_All_Block*N_Node(Block_Index)*Num_Feature(location:x、y+Num_All_Freq+Num_tr+Num_Modelbased_Feature)
		Num_Axis_Feature=2
		Num_tr=0
		Use_Axis_tr=False
		Num_Modelbased_Feature=0
		Num_In_Freq=1
		#In_Freq=[2115]
		Num_Feature=Num_Axis_Feature+Num_In_Freq+Num_Out_Freq+Num_In_Freq+Num_tr+Num_Modelbased_Feature

		All_Feature=np.ones((Num_All_Block,Num_Node_Max,Num_Feature,Num_All_Freq))


		#Feature Processing


		#Load depth feature
		# Depth_Map=np.loadtxt('depth4.txt',dtype=float)
		# Depth_Map=(Depth_Map-np.min(Depth_Map))/(2.037461-np.min(Depth_Map))


		#1、Location(spatial and frequency) Feature
		for i in Block_Index:
			m=int(i/Num_All_Block_X)
			n=int(i-m*Num_All_Block_X)
			for k in range(Num_All_Freq):
				print(N_Node[i])
				All_Feature[i,0:N_Node[i],0:Num_Axis_Feature,k]=np.loadtxt('NodeFeatureAxis{}/{}/{}/{}_{}.txt'.format(BlockX,AreaIndex+1,Carry_Freq[0],m+1,n+1),dtype=float).T
			#Normalization
				All_Feature[i,0:N_Node[i],0,k]=(All_Feature[i,0:N_Node[i],0,k]+40*int(i/Num_All_Block_X))/(Num_All_Block_Y)/40
				All_Feature[i,0:N_Node[i],1,k]=(All_Feature[i,0:N_Node[i],1,k]+40*(i-Num_All_Block_X*int(i/Num_All_Block_X)))/Num_All_Block_X/40

				print(All_Feature[i,0:N_Node[i],0])
			#2、Frequency
				for j in range(Num_In_Freq):
				#print(np.shape(np.loadtxt('Label{}/{}.txt'.format(Carry_Freq[j],i+1),dtype=float)[0:N_Node[i]]))
					All_Feature[i,0:N_Node[i],Num_Axis_Feature+Num_In_Freq+Num_Out_Freq+j,k]=(All_Mask[i,k,0:N_Node[i]]*np.loadtxt('NodeFeatureStrength{}/{}/{}/{}_{}.txt'.format(BlockX,AreaIndex+1,Carry_Freq[j],m+1,n+1),dtype=float))/(-100)


			#3、Transmitter feature
				if Use_Axis_tr==True:
			#relative location
					Axis_tr=np.ones((Num_Axis_Feature,Num_tr))
					Axis_tr[:,0]=np.array(StationsPosition[AreaIndex,0:2])-1
					Axis_tr[:,1]=np.array(StationsPosition[AreaIndex,2:4])-1
					Axis_tr[:,2]=np.array(StationsPosition[AreaIndex,4:6])-1

					for j in range(Num_tr):
						All_Feature[i,0:N_Node[i],Num_Axis_Feature+Num_In_Freq+Num_Out_Freq+Num_In_Freq+j,k]=np.sqrt(np.square(All_Feature[i,0:N_Node[i],0,k]*Num_All_Block_Y*40-Axis_tr[0,j])+np.square(All_Feature[i,0:N_Node[i],0,k]*Num_All_Block_X*40-Axis_tr[1,j]))
						All_Feature[i,0:N_Node[i],Num_Axis_Feature+Num_In_Freq+Num_Out_Freq+Num_In_Freq+j,k]=All_Feature[i,0:N_Node[i],Num_Axis_Feature+Num_In_Freq+Num_Out_Freq+Num_In_Freq+j,k]/max(All_Feature[i,0:N_Node[i],Num_Axis_Feature+Num_In_Freq+Num_Out_Freq+Num_In_Freq+j,k])
				
			#4、model-based feature (depth, etc.)
			#radio-depthmap
			# print(np.shape(Depth_Map))
			# All_Feature[i,0:N_Node[i],Num_Axis_Feature+Num_train_Freq+Num_tr]=Depth_Map[np.array(All_Feature[i,0:N_Node[i],0]*Num_All_Block_Y*40).astype(int)+80,np.array(All_Feature[i,0:N_Node[i],1]*Num_All_Block_X*40).astype(int)]




		

		#Output Definition
		All_Y=np.ones((Num_All_Block,Num_Node_Max,Num_All_Freq))
		for i in Block_Index:
			m=int(i/Num_All_Block_X)
			n=int(i-m*Num_All_Block_X)
			for j in range(Num_All_Freq):
				All_Y[i,0:N_Node[i],j]=(np.loadtxt('NodeFeatureStrength{}/{}/{}/{}_{}.txt'.format(BlockX,AreaIndex+1,Carry_Freq[j],m+1,n+1),dtype=float))/(-100)
#Adjacency Size：Num_All_Block*N_Node(Block_Index)*N_Node(Block_Index)，N_Node(Block_Index):=Num_Node_Max
		All_Adj=np.ones((Num_All_Freq,Num_All_Block,Num_Node_Max,Num_Node_Max),dtype=int)
		for j in range(Num_All_Freq):
			for i in Block_Index:
				m=int(i/Num_All_Block_X)
				n=int(i-m*Num_All_Block_X)
				if method==0:
					All_Adj[j,i,0:N_Node[i],0:N_Node[i]]=np.loadtxt('ObstructionAdj_6_40/{}/{}_{}.txt'.format(AreaIndex+1,m+1,n+1),dtype=int)
				else:
					if method==1:
						All_Adj[j,i,0:N_Node[i],0:N_Node[i]]=np.loadtxt('FreqDepth0.1_3/{}/{}/{}_{}.txt'.format(AreaIndex+1,Carry_Freq[j],m+1,n+1),dtype=int)
					else:
						if method==2:
							All_Adj[j,i,0:N_Node[i],0:N_Node[i]]=np.loadtxt('AroundAdj40/{}/{}_{}.txt'.format(AreaIndex+1,m+1,n+1),dtype=int)
						else:
							if method==3:
								All_Adj[j,i,0:N_Node[i],0:N_Node[i]]=np.loadtxt('TransmitterAdj40/{}/{}_{}.txt'.format(AreaIndex+1,m+1,n+1),dtype=int)
			#add selfloop
				adj0=np.eye(N_Node[i],dtype=int)
				All_Adj[j,i,0:N_Node[i],0:N_Node[i]]=adj0+All_Adj[j,i,0:N_Node[i],0:N_Node[i]]

		#Model Setting
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设备
		epochs = 250# 学习轮数
		lr = 0.0005 # 学习率

		All_Graph=np.ones((Num_All_Block_X*40,Num_All_Block_Y*40))

		#result set
		loss0=np.ones(epochs)
		acc0=np.ones(epochs)

		Num_test=1 #The total number of test 
		Num_mse=np.ones((Num_test,1))
		for o in range(Num_test):
			model = GAT(Num_Feature,Num_Out_Freq).to(device)
			optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 优化器
			loss_function = nn.MSELoss() # 损失函数
			
		#Model parameters initialization
			for m in model.modules():
				if isinstance(m, (nn.Conv2d, nn.Linear)):
					nn.init.xavier_uniform_(m.weight)
			for epoch in range(epochs):
				model.train()
				print(epoch)
				for i in Block_Index[0:Num_Train_Block]:
					for k in range(Num_train_Freq):
						Feature=All_Feature[i,0:N_Node[i],:,k]
						Feature[0:N_Node[i],Num_Axis_Feature]=Carry_Freq[Train_Freq_Index[k]]/5750
						Feature[0:N_Node[i],Num_Axis_Feature+Num_In_Freq]=Carry_Freq[Out_Freq_Index]/5750
						Adj=All_Adj[Train_Freq_Index[k],i,0:N_Node[i],0:N_Node[i]]
						Feature=torch.FloatTensor(Feature).requires_grad_(True)
						Adj=torch.FloatTensor(Adj)
						OutAdj=All_Adj[0,i,0:N_Node[i],0:N_Node[i]]
						OutAdj=torch.FloatTensor(OutAdj)
						optimizer.zero_grad()
						Pred=model(Feature.to(device),Adj.to(device),OutAdj.to(device))
						if Carry_Freq[Train_Freq_Index[k]]==Carry_Freq[Out_Freq_Index]:
							Mask_Pred=Pred[torch.tensor(All_Mask[i,Out_Freq_Index,0:N_Node[i]].T).bool()]
							Mask_Real=All_Y[i,0:N_Node[i],Out_Freq_Index][torch.tensor(All_Mask[i,Out_Freq_Index,0:N_Node[i]].T).bool()]
							
						else:
							Mask_Pred=Pred
							Mask_Real=All_Y[i,0:N_Node[i],Out_Freq_Index]
						Mask_Real=torch.FloatTensor(Mask_Real)
						loss = loss_function(Mask_Pred[:,0].to(device), Mask_Real.to(device)) # 损失
						loss.backward()
						optimizer.step()

				loss0[epoch]=loss.item()
				print(loss0[epoch])

				model.eval()
				if toupiao==1:
					if epoch%10==0:

						mse=0
						for i1 in Block_Index[Num_Train_Block:Num_All_Block]:

							Mask_Pred0=0
							for j1 in range(Num_train_Freq):
								Feature=All_Feature[i1,0:N_Node[i1],:,j1]
								Feature[0:N_Node[i1],Num_Axis_Feature]=Carry_Freq[j1]/5750
								Feature[0:N_Node[i1],Num_Axis_Feature+Num_In_Freq]=Carry_Freq[Out_Freq_Index]/5750


								Adj=All_Adj[j1,i1,0:N_Node[i1],0:N_Node[i1]]
								Feature=torch.FloatTensor(Feature).requires_grad_(True)
								Adj=torch.FloatTensor(Adj)

								OutAdj=All_Adj[Out_Freq_Index,i1,0:N_Node[i1],0:N_Node[i1]]
								OutAdj=torch.FloatTensor(OutAdj)
								Pred=model(Feature.to(device),Adj.to(device),OutAdj.to(device))
								#print(np.shape(Pred))
								Mask_Pred=Pred[torch.tensor(1-All_Mask[i1,Out_Freq_Index,0:N_Node[i1]].T).bool(),0]
								Mask_Pred0=Mask_Pred0+Mask_Pred
							Mask_Pred0=Mask_Pred0/Num_train_Freq
						
							Mask_Real=All_Y[i1,0:N_Node[i1],Out_Freq_Index][torch.tensor(1-All_Mask[i1,Out_Freq_Index,0:N_Node[i1]].T).bool()]
							mse=np.sqrt(mse*mse+mean_squared_error(Mask_Pred.cpu().detach().numpy(),Mask_Real))

				else:
				#valid
					if epoch%10==0:
						mse=0
						for i in Block_Index[Num_Train_Block:Num_All_Block]:

							Feature=All_Feature[i,0:N_Node[i],:,Input_Freq_Index]
							Feature[0:N_Node[i],Num_Axis_Feature]=Carry_Freq[Input_Freq_Index]/5750
							Feature[0:N_Node[i],Num_Axis_Feature+Num_In_Freq]=Carry_Freq[Out_Freq_Index]/5750


							Adj=All_Adj[Input_Freq_Index,i,0:N_Node[i],0:N_Node[i]]
							Feature=torch.FloatTensor(Feature).requires_grad_(True)
							Adj=torch.FloatTensor(Adj)

							OutAdj=All_Adj[Out_Freq_Index,i,0:N_Node[i],0:N_Node[i]]
							OutAdj=torch.FloatTensor(OutAdj)
							Pred=model(Feature.to(device),Adj.to(device),OutAdj.to(device))
							#print(np.shape(Pred))
							Mask_Pred=Pred[torch.tensor(1-All_Mask[i,Out_Freq_Index,0:N_Node[i]].T).bool(),0]
						


						#print(Mask_Pred)
							Mask_Real=All_Y[i,0:N_Node[i],Out_Freq_Index][torch.tensor(1-All_Mask[i,Out_Freq_Index,0:N_Node[i]].T).bool()]
						#print(Mask_Real
							mse=np.sqrt(mse*mse+mean_squared_error(Mask_Pred.cpu().detach().numpy(),Mask_Real))
						#print(mse)
				acc0[epoch]=100*mse/np.sqrt(Num_Test_Block)
				print(acc0[epoch])
			Num_mse[o]=acc0[epochs-1]
		# if method==2:
		# 	np.savetxt('predict_result/1750_{}_AroundAdj/{}_{}.txt'.format(Carry_Freq[Out_Freq_Index],AreaIndex+1,percent),Num_mse,fmt='%.4f')
		# else:
		# 	if method==1:
		# 		np.savetxt('predict_result/1750_{}/{}_{}.txt'.format(Carry_Freq[Out_Freq_Index],AreaIndex+1,percent),Num_mse,fmt='%.4f')
		# 	else:
		# 		if method==0:
		# 			np.savetxt('predict_result/1750_{}_ObstructionAdj/{}_{}.txt'.format(Carry_Freq[Out_Freq_Index],AreaIndex+1,percent),Num_mse,fmt='%.4f')
		# 		else:
		# 			if method==3:
		# 				np.savetxt('predict_result/1750_{}_TransAdj/{}_{}.txt'.format(Carry_Freq[Out_Freq_Index],AreaIndex+1,percent),Num_mse,fmt='%.4f')

		print(Num_mse)
		print(np.sum(Num_mse)/Num_test)
		#np.savetxt('SingleFreq/Result/Visualization/MSE/{}_{}_{}_{}_{}.txt'.format(AreaIndex+1,Out_Freq_Index+1,percent,Train_Percent,method),Num_mse,fmt='%.4f')






		#Model Validation
		model.eval()
		zongtu=np.ones((Num_All_Block_X*40,Num_All_Block_Y*40),dtype=float)
		for i in Block_Index:
			m=int(i/Num_All_Block_X)
			n=int(i-m*Num_All_Block_X)
			Feature=All_Feature[i,0:N_Node[i],:,Input_Freq_Index]
			Feature[0:N_Node[i],Num_Axis_Feature]=Carry_Freq[Input_Freq_Index]/5750
			Feature[0:N_Node[i],Num_Axis_Feature+Num_In_Freq]=Carry_Freq[Out_Freq_Index]/5750
			Adj=All_Adj[Input_Freq_Index,i,0:N_Node[i],0:N_Node[i]]
			Feature=torch.FloatTensor(Feature).requires_grad_(True)
			Adj=torch.FloatTensor(Adj)
			OutAdj=All_Adj[Out_Freq_Index,i,0:N_Node[i],0:N_Node[i]]
			OutAdj=torch.FloatTensor(OutAdj)
			Pred=model(Feature.to(device),Adj.to(device),OutAdj.to(device))
			#print(np.shape(Pred[:,0]))

			axistest=np.loadtxt('NodeFeatureAxis{}/{}/{}/{}_{}.txt'.format(BlockX,AreaIndex+1,Carry_Freq[0],m+1,n+1),dtype=int).T
			u=axistest[0:N_Node[i],0]+int(40*m)
			v=axistest[0:N_Node[i],1]+int(40*n)
			k=Pred[:,0]
			zongtu[v,u]=k.cpu().detach().numpy()*(-100)
			Mask_Pred=Pred[torch.tensor(1-All_Mask[i,Out_Freq_Index,0:N_Node[i]].T).bool(),0]
		zongtu[np.where(zongtu==1)]=0
		plt.figure(1)
		Norm=plt.Normalize(vmin=-100,vmax=-10)


		#np.savetxt('SingleFreq/Result/Visualization/Figure/{}_{}_{}_{}_{}.txt'.format(AreaIndex+1,Out_Freq_Index+1,percent,Train_Percent,method),zongtu,fmt='%.4f')

		plt.imshow(zongtu.T,cmap='jet',norm=Norm)

		plt.colorbar()
		#plt.savefig('SingleFreq/predict_result/{}_5.png'.format(AreaIndex+1))

		plt.show()
