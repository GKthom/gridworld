import os
import numpy as np
import params as p
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import matplotlib.patches as patches


#import priors_tabular as PR

def Qlearn_multirun_tab():
	#This function just runs multiple instances of 
	#Q-learning. Doing so helps obtain an average performance 
	#measure over multiple runs.
	retlog=[] # log of returns of all episodes, in all runs
	for i in range(p.Nruns):
		print("Run no:",i)
		Q,ret=main_Qlearning_tab()#call Q learning
		if i==0:
			retlog=ret
		else:
			retlog=np.vstack((retlog,ret))
		#retlog.append(ret)
		if (i+1)/p.Nruns==0.25:
			print('25% runs complete')
		elif (i+1)/p.Nruns==0.5:
			print('50% runs complete')
		elif (i+1)/p.Nruns==0.75:
			print('75% runs complete')
		elif (i+1)==p.Nruns:
			print('100% runs complete')
	#meanreturns=(np.mean(retlog,axis=0))
	return Q, retlog

def main_Qlearning_tab():
	#This calls the main Q learning algorithm
	Q=np.zeros((p.a,p.b,p.A)) # initialize Q function as zeros
	goal_state=p.targ#target point
	returns=[]#stores returns for each episode
	ret=0
	for i in range(p.episodes):
		if (i+1)/p.episodes==0.25:
			print('25% episodes done')
		elif (i+1)/p.episodes==0.5:
			print('50% episodes done')
		elif (i+1)/p.episodes==0.75:
			print('75% episodes done')
		elif (i+1)/p.episodes==1:
			print('100% episodes done')
		Q,ret=Qtabular(Q,i)#call Q learning
		if i%1==0:
			returns.append(ret)#compute return offline- can also be done online, but this way, a better estimate can be obtained
	return Q, returns

def Qtabular(Q,episode_no):
	initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
	#initial_state=np.array([2,8])
	rounded_initial_state=staterounding(initial_state)
	while p.world[rounded_initial_state[0],rounded_initial_state[1]]==1:
		initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
		rounded_initial_state=staterounding(initial_state)
	state=staterounding(initial_state.copy())
	count=0
	breakflag=0
	eps_live=1-(p.epsilon_decay*episode_no)
	#eps_live=0.7
	ret=0
	target_state=p.targ
	#while np.linalg.norm(state-target_state)>p.thresh:
	
	#statelog.append(state)
	for i in range(p.breakthresh):

		count=count+1
		if breakflag==1:
			break
		if count>p.breakthresh:
			breakflag=1
		if eps_live>np.random.sample():
			a=np.random.randint(p.A)
		else:
			Qmax,Qmin,a=maxQ_tab(Q,state)

		next_state=transition(state,a)
		
		roundedstate=staterounding(state)
		roundednextstate=staterounding(next_state)

		if p.world[next_state[0],next_state[1]]==0 and (next_state[0]<=p.a and next_state[0]>=0 and next_state[1]<=p.b and next_state[1]>=0):	
			if np.linalg.norm(next_state-target_state)<=p.thresh:
				R=p.highreward
			else:
				R=p.livingpenalty
		else: 
			R=p.penalty
			next_state=state.copy()

		ret=ret+R

		Qmaxnext,Qminnext,aoptnext=maxQ_tab(Q,next_state)
		Qtarget=R+(p.gamma*Qmaxnext)-Q[roundedstate[0],roundedstate[1],a]
		Q[roundedstate[0],roundedstate[1],a]=Q[roundedstate[0],roundedstate[1],a]+(p.alpha*Qtarget)
		if np.linalg.norm(next_state-target_state)<=p.thresh:
			break
		state=next_state.copy()

	return Q,ret


def maxQ_tab(Q,state):
	#get max of Q values and corresponding action
	Qlist=[]
	roundedstate=staterounding(state)
	for i in range(p.A):
		Qlist.append(Q[roundedstate[0],roundedstate[1],i])
	tab_maxQ=np.max(Qlist)
	tab_minQ=np.min(Qlist)
	maxind=[]
	for j in range(len(Qlist)):
		if tab_maxQ==Qlist[j]:
			maxind.append(j)
	#print(maxind)
	if len(maxind)>1:
		optact=maxind[np.random.randint(len(maxind))]
	else:
		optact=maxind[0]
	return tab_maxQ,tab_minQ,optact

def optpol_visualize(Qp):
	for i in range(p.a):
		for j in range(p.b):
			if p.world[i,j]==0:
				Qmaxopt,Qminopt,optact=maxQ_tab(Qp,[i,j])
				if optact==0:
					plt.scatter(i,j,color='red')
				elif optact==1:
					plt.scatter(i,j,color='green')
				elif optact==2:
					plt.scatter(i,j,color='blue')
				elif optact==3:
					plt.scatter(i,j,color='yellow')

	plotmap(p.world)
	plt.show()

def transition(state,act):
	#print(orig_state)
	#print(act)
	n1 = np.random.uniform(low=-0.2, high=0.2, size=(1,))# x noise
	n2 = np.random.uniform(low=-0.2, high=0.2, size=(1,))# y noise
	new_state=state.copy()
	if act==0:
		new_state[0]=state[0]
		new_state[1]=state[1]+1#move up
	elif act==1:
		new_state[0]=state[0]+1#move right
		new_state[1]=state[1]
	elif act==2:
		new_state[0]=state[0]
		new_state[1]=state[1]-1#move down
	elif act==3:
		new_state[0]=state[0]-1#move left
		new_state[1]=state[1]
	
	#new_state[0]=new_state[0]+n1
	#new_state[1]=new_state[1]+n2
	return new_state

########Additional functions for visualization######
def plotmap(worldmap):
	#plots the obstacle map
	for i in range(p.a):
		for j in range(p.b):
			if worldmap[i,j]>0:
				plt.scatter(i,j,color='black')
	plt.show()

def staterounding(state):
	#rounds off states
	roundedstate=[0,0]
	roundedstate[0]=int(np.around(state[0]))
	roundedstate[1]=int(np.around(state[1]))
	if roundedstate[0]>=(p.a-1):
		roundedstate[0]=p.a-2
	elif roundedstate[0]<1:
		roundedstate[0]=1
	if roundedstate[1]>=(p.b-1):
		roundedstate[1]=p.b-2
	elif roundedstate[1]<=0:
		roundedstate[1]=1
	return roundedstate

def opt_pol(Q,state,goal_state):
	#shows optimal policy
	plt.figure(0)
	plt.ion()
	for i in range(p.a):
		for j in range(p.b):
			if p.world[i,j]>0:
				plt.scatter(i,j,color='black')
	plt.show()
	pol=[]
	statelog=[]
	count=1
	while np.linalg.norm(state-goal_state)>=1:
		Qm,Qmin,a=maxQ_tab(Q,state)
		if np.random.sample()>0.9:
			a=np.random.randint(p.A)
		next_state=transition(state,a)
		roundednextstate=staterounding(next_state)
		if p.world[roundednextstate[0],roundednextstate[1]]==1:
			next_state=state.copy()
		pol.append(a)
		statelog.append(state)
		print(state)
		plt.ylim(0, p.b)
		plt.xlim(0, p.a)
		plt.scatter(state[0],state[1],(60-count*0.4),color='blue')
		plt.draw()
		plt.pause(0.1)
		state=next_state.copy()
		print(count)
		if count>=100:
			break
		count=count+1
	return statelog,pol

def mapQ(Q):
	#plots a map of the value function
	fig=plt.figure(1)
	plt.ion
	Qmap=np.zeros((p.a,p.b))
	for i in range(p.a):
		for j in range(p.b):
 			Qav=0
 			for k in range(p.A):
 				Qav=Qav+Q[i,j,k]
 			Qmap[i,j]=Qav
	#Qfig=plt.imshow(np.rot90(Qmap))
	Qmap=Qmap-np.min(Qmap)
	if np.max(Qmap)>0:
		Qmap=Qmap/np.max(Qmap)
	Qmap=np.rot90(Qmap)

	return Qmap



#######################################
if __name__=="__main__":
	#w,Qimall=Qlearn_main_vid()

	Q,retlog=Qlearn_multirun_tab()

	mr=(np.mean(retlog,axis=0))
	csr=[]
	for i in range(len(mr)):
		if i>0:			
			csr.append(np.sum(mr[0:i])/i)
	np.savez("DQN"+str(p.Nruns)+"_runs.npy.npz",retlog,Q)
	s_retlog=np.shape(retlog)
	x=range(s_retlog[1])
	mn=np.mean(retlog,axis=0)
	st_err=np.std(retlog,axis=0)/np.sqrt(p.Nruns)
	plt.xlabel('Steps',fontsize=15) 
	plt.ylabel('Average sum of rewards' ,fontsize=15)
	plt.gca().legend(('Q-learning'),frameon=False)
	plt.grid(linestyle='-')
	plt.plot(x,mn,'r')
	plt.fill_between(x,mn-st_err, mn+st_err,color='darksalmon',alpha=0.3)
	plt.show()
