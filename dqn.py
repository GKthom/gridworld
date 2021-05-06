import os
import numpy as np
import params as p
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import matplotlib.patches as patches
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#import priors_tabular as PR

def Qlearn_multirun_tab():
	model = Sequential()
	model.add(Dense(24, input_shape=(3,), activation="relu"))
	model.add(Dense(24, activation="relu"))
	model.add(Dense(1, activation="linear"))
	model.compile(loss="mse", optimizer=Adam(lr=0.001))

	modelcopy = Sequential()
	modelcopy.add(Dense(24, input_shape=(3,), activation="relu"))
	modelcopy.add(Dense(24, activation="relu"))
	modelcopy.add(Dense(1, activation="linear"))
	modelcopy.compile(loss="mse", optimizer=Adam(lr=0.001))

	
	#This function just runs multiple instances of 
	#Q-learning. Doing so helps obtain an average performance 
	#measure over multiple runs.
	retlog=[] # log of returns of all episodes, in all runs
	bumpcountlog=[]
	for i in range(p.Nruns):
		print("Run no:",i)
		tracebuffer=[]
		tracebuffer_neg=[]
		Q,ret,betamatrix,bumpcountret,tracemap,model,tracebuffer,tracebuffer_neg=main_Qlearning_tab(model,modelcopy,tracebuffer,tracebuffer_neg)#call Q learning
		if i==0:
			retlog=ret
			bumpcountlog=bumpcountret
		else:
			retlog=np.vstack((retlog,ret))
			bumpcountlog=np.vstack((bumpcountlog,bumpcountret))
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
	return Q, retlog,betamatrix,bumpcountlog,tracemap,model

def main_Qlearning_tab(model,modelcopy,tracebuffer,tracebuffer_neg):
	#This calls the main Q learning algorithm
	Q=np.zeros((p.a,p.b,p.A)) # initialize Q function as zeros
	tracemap=np.zeros((p.a,p.b,p.A))
	betamatrix=np.ones((p.a,p.b,p.A))
	#betamatrix=np.random.rand(p.a,p.b,p.A)
	visitmap=np.zeros((p.a,p.b,p.A))
	#statevisitslog=np.zeros((p.a,p.b)) # initialize counter for visits
	goal_state=p.targ#target point
	returns=[]#stores returns for each episode
	bumpcountret=[]
	ret=0
	Qimall=[]
	totcnt=0
	success_attempts=0
	tot_attempts=0
	for i in range(p.episodes):
		if (i+1)/p.episodes==0.25:
			print('25% episodes done')
		elif (i+1)/p.episodes==0.5:
			print('50% episodes done')
		elif (i+1)/p.episodes==0.75:
			print('75% episodes done')
		elif (i+1)/p.episodes==1:
			print('100% episodes done')
		Q,ret,betamatrix,bumpcount,visitmap,totcnt,tracemap,model,tracebuffer,tracebuffer_neg,success_attempts,tot_attempts=Qtabular(Q,i,betamatrix,visitmap,totcnt,tracemap,model,modelcopy,tracebuffer,tracebuffer_neg,success_attempts,tot_attempts)#call Q learning
		print(totcnt)
		if i%1==0:
			returns.append(ret)#compute return offline- can also be done online, but this way, a better estimate can be obtained
			bumpcountret.append(bumpcount)
	print("Successes:"+str(success_attempts))
	time.sleep(5)
	return Q, returns,betamatrix,bumpcountret,tracemap,model,tracebuffer,tracebuffer_neg

def build_tracebuffer(statelog,tracebuffer,success_flag,tracemap):
	sz=np.shape(statelog)
	if success_flag==1:
		lambd=1
	else:
		lambd=0
	fliplog=np.flipud(statelog)
	for i in range(sz[0]):
		x=fliplog[i][0]-1
		y=fliplog[i][1]-1
		a=fliplog[i][2]
		tracemap[x+1,y+1,a]+=lambd
		if len(tracebuffer)==0:
			tracebuffer=np.array([x,y,a,lambd])
		elif len(tracebuffer)>p.tracebuffersize:
			tracebuffer=np.vstack((tracebuffer,np.array([x,y,a,lambd])))
			tracebuffer=np.delete(tracebuffer,0,axis=0)
		else:
			tracebuffer=np.vstack((tracebuffer,np.array([x,y,a,lambd])))
		lambd=lambd*0.9
	return tracebuffer,tracemap

def learn_trace_model(model,modelcopy,totcnt,tracebuffer,tracebuffer_neg):
	if totcnt%p.tracefreeze_freq==0:
		model.save_weights("frozen.h5")
		modelcopy.load_weights("frozen.h5")


	#sample from buffer
	if len(tracebuffer)>p.tracebuffer_batchsize and len(tracebuffer_neg)>p.tracebuffer_batchsize:
		x=[]
		y=[]
		for jj in range(p.tracebuffer_batchsize):
			if np.random.rand()>0.5:
				curr_buffer=tracebuffer_neg
			else:
				curr_buffer=tracebuffer
			ind=np.random.randint(len(curr_buffer))
			if len(x)==0:
				x=np.array([curr_buffer[ind][0],curr_buffer[ind][1],curr_buffer[ind][2]])
				y=np.array([curr_buffer[ind][3]])
			else:
				x=np.vstack((x,np.array([curr_buffer[ind][0],curr_buffer[ind][1],curr_buffer[ind][2]])))
				y=np.vstack((y,np.array([curr_buffer[ind][3]])))
		#print('model training..')
		#model.fit(x,y+modelcopy.predict(x),verbose=0)
		model.fit(x,y,verbose=0)

	return model,modelcopy

def Qtabular(Q,episode_no,betamatrix,visitmap,totcnt,tracemap,model,modelcopy,tracebuffer,tracebuffer_neg,success_attempts,tot_attempts):
	#initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
	initial_state=np.array([2,8])
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
	bumpcount=0
	target_state=p.targ
	subgoal_targs=[]
	#while np.linalg.norm(state-target_state)>p.thresh:
	statelog=[]
	
	#statelog.append(state)
	for i in range(p.breakthresh):
		'''
		s_sg=np.shape(subgoal_targs)
		if s_sg[0]>0:
			#execute subgoals till termination
			for kk in range(s_sg[0]):
				state=execute_pol(sub_pol[i],subgoal_targs,state)


		if success_attempts>5 and (success_attempts/tot_attempts)<0.1:
			#find subgoal
			#sample from unsuccessful buffer:
			s_tracebuffer_neg=np.shape(tracebuffer_neg)
			subgoal_candidates=[]
			for i in range(100):
				randint=np.random.randint(s_tracebuffer_neg[0])
				subgoal_candidates.append(tracebuffer_neg[randint])
			subgoal_candidates=np.flipud(subgoal_candidates)[:,[0,1]]
			maxind=np.argmax(model.predict(subgoal_candidates))
			subgoal=subgoal_candidates[maxind]
			while p.world[int(subgoal[0]),int(subgoal[1])]==1:
				subgoal_candidates=np.delete(subgoal_candidates,maxind,axis=0)
				maxind=np.argmax(model.predict(subgoal_candidates))
				subgoal=subgoal_candidates[maxind]

			print(subgoal)
			time.sleep(22)
			subgoal_targs.append(subgoal)
			success_attempts=0
			tot_attempts=0
		'''

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
		if len(statelog)>=(p.tracelim):
			statelog=statelog[1:(p.tracelim)]
		statelog.append(np.array([state[0],state[1],a]))
		visitmap[roundednextstate[0],roundednextstate[1],a]=visitmap[roundednextstate[0],roundednextstate[1],a]+1
		if p.world[next_state[0],next_state[1]]==0 and (next_state[0]<=p.a and next_state[0]>=0 and next_state[1]<=p.b and next_state[1]>=0):	
			if np.linalg.norm(next_state-target_state)<=p.thresh:
				R=p.highreward
				success_attempts+=1
				success_flag=1
				#if success_attempts<2:
					#tracebuffer,tracemap=build_tracebuffer(statelog,tracebuffer,success_flag,tracemap)
			else:
				R=p.livingpenalty
		else: 
			R=p.penalty
			next_state=state.copy()
			bumpcount=bumpcount+1

		ret=ret+R
		totcnt+=1

		#model,modelcopy=learn_trace_model(model,modelcopy,totcnt,tracebuffer,tracebuffer_neg)

		Qmaxnext,Qminnext,aoptnext=maxQ_tab(Q,next_state)
		Qtarget=R+(p.gamma*Qmaxnext)-Q[roundedstate[0],roundedstate[1],a]
		Q[roundedstate[0],roundedstate[1],a]=Q[roundedstate[0],roundedstate[1],a]+(p.alpha*Qtarget)
		if np.linalg.norm(next_state-target_state)<=p.thresh:
			break
		state=next_state.copy()

	if count==p.breakthresh:
		success_flag=0
		#tracebuffer_neg,tracemap_neg=build_tracebuffer(statelog,tracebuffer_neg,success_flag,tracemap)
	tot_attempts+=1

	return Q,ret,betamatrix,bumpcount,visitmap,totcnt,tracemap,model,tracebuffer,tracebuffer_neg,success_attempts,tot_attempts


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

def maptrace(Qmap):
	#plots a map of the value function
	fig=plt.figure(1)
	plt.ion
	Qmap=Qmap-np.min(Qmap)
	if np.max(Qmap)>0:
		Qmap=Qmap/np.max(Qmap)
	Qmap=np.rot90(Qmap)

	return Qmap

def findsubgoal(model,init_state,maxtrace):
	state=init_state
	dummy_state=state
	states=[]
	for i in range(100):
		a=np.random.randint(p.A)
		next_state=transition(state,a)
		if p.world[next_state[0],next_state[1]]==0 and (next_state[0]<=p.a and next_state[0]>=0 and next_state[1]<=p.b and next_state[1]>=0):
			b=1
		else:
			next_state=state.copy()
		if len(states)==0:
			states=state
		else:
			states=np.vstack((states,state))
		state=next_state.copy()
	return states




#######################################
if __name__=="__main__":
	#w,Qimall=Qlearn_main_vid()

	Q,retlog,betamatrix,bumpcountlog,tracemap,model=Qlearn_multirun_tab()

	mr=(np.mean(retlog,axis=0))
	csr=[]
	for i in range(len(mr)):
		if i>0:			
			csr.append(np.sum(mr[0:i])/i)
	np.savez("DQN"+str(p.Nruns)+"_runs.npy.npz",retlog,bumpcountlog,betamatrix,Q,tracemap)

	states=[]
	cnt=0
	tracepred=np.zeros((p.a,p.b,p.A))
	for i in range(p.a):
		for j in range(p.b):
			for k in range(p.A):
				states=np.vstack((np.array([0,0,0]),np.array([i,j,k])))
				pred=model.predict(states)
				tracepred[i,j,k]=pred[1]
	#tracepred[p.world==1]=-2
	meantrace=np.mean(tracepred,axis=2)
	plt.imshow(np.rot90(meantrace))
	plt.show()
	'''
	for i in range(p.a):
		for j in range(p.b):
			if cnt==0:
				states=np.array([i,j])
			else:
				states=np.vstack((states,np.array([i,j])))
			cnt+=1		
	model.predict(states)
	'''