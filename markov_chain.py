import numpy as np
from scipy import optimize
from scipy import sparse
import multiprocessing as mp
import itertools
import time
import math

def state_function(n,x):
	state = np.array(list(map(float,list(np.base_repr(x,3)))))
	return(np.pad(state,(n-len(state),0),'constant',constant_values=(0,0)))

# T_function: produces elements of the T matrix as given in equation 3.
def T_function(q,n,x):
	x1 = x//(3**n)
	x2 = x%(3**n)
	state1 = state_function(n,x1)
	state2 = state_function(n,x2)
	return(np.prod((1-np.prod(q**np.transpose(np.broadcast_to(state1==1,(n,n))),
		axis=0))**((state1+state2)==3))*np.prod(np.prod(q**
		(np.transpose(np.broadcast_to(state1==1,(n,n)))*
			np.broadcast_to((state1+state2)==4,(n,n))),axis=0)))

# configuration_probabilities: function to get the probabilities of each subset 
# of populations experiencing an epidemic.
# INPUTS
# L is a (n,n) array of rates of travel between populations.
# R0 is a (n,1) array of R0 in each popluation.
# MU is a (n,1) array  of the recovery rate in each population.
# Sinit is a (n,1) array  of the number of initial susceptibles in each population.
# Iinit is a (n,1) array  the number of initial infected in each population.
# We require the populations with initial infecteds to be listed first.
# parallel tells the function whether to perform computations in parallel.
# size tells the function whether to output expected final size of the outbreak.
# OUTPUTS
# If size=False then the function outputs a (2**n) array of probabilities for
# each subset of populations experiencing an epidemic, in order from all populations
# to no populations. The order is given by assigning a 0 to populations with an
# epidemic, a 1 to populations with no epidemic, putting the digits in order and
# reading it as a binary number - this gives the index of that scenario.
# If size=True then a two-element array is returned, with the array described above 
# in the first position and a scalar indicating the expected total final size of
# the outbreak, summing over all scenarios and populations.
def configuration_probabilities(L,R0,MU,Sinit,Iinit, parallel=False, size=False):
	n = L.shape[1]

	r0 = np.broadcast_to(R0,(n,n))
	mu = np.broadcast_to(MU,(n,n))
	l = np.broadcast_to(np.sum(L,axis=1),(n,n))
	Q = np.minimum(np.transpose(((1/R0)**Iinit)),1)

	def Rinf_function(x):
		x.shape = (n,1)
		y=x - Sinit + Sinit*np.exp(-x*R0/Sinit)
		y.shape = (n,)
		return(y)

	Rinf = optimize.fsolve(Rinf_function,(Sinit-1))
	Rinf = np.array(Rinf)
	Rinf.shape = (n,1)
	rinf = np.broadcast_to(Rinf,(n,n))

	#make infection probability matrix
	q = np.minimum(((L/(r0*(l+mu)))+(l+mu-L)/(l+mu))**rinf,1)

	#define the states - 2 is N, 1 is U, and 0 is R
	state_strings = np.array([np.base_repr(x,3,padding=n-1-
		int(max([np.log(x+0.1),-np.log(3)])/np.log(3))) for x in range(3**n)])
	state_arrays = [np.array(list(map(float,list(state)))) 
	for state in state_strings]
	

	#initial probabilities of states from native infecteds
	P = np.prod(np.array([state for state in state_arrays])
		,axis=1).astype(bool).astype(int)*np.array([np.prod(Q**
			(state-1).clip(min=0))*np.prod((1-Q)**
			(1-(state-1).astype(bool).astype(int))) for state in state_arrays])

	#The transition matrix is in general very large, but sparse.
	#Here we filter out just the elements that could be non-zero, as well as
	# elements that will never be needed due to the initial state of the system
	# - the initial unresolved states can never be unresolved at the same time 
	# as anything else.
	startnz = time.time()
	v=np.array(list(itertools.product([2*3**n + 2,2*3**n + 1,3**n,0],repeat=n)))
	nonzero_indices = np.dot(v,np.array([3**x for x in range(n)]))
	endnz = time.time()
	#print('nonzero_indices takes '+str(endnz-startnz))

	startt = time.time()
	#transition matrix
	if parallel:
		pool = mp.Pool(mp.cpu_count())
		#The values of T at the nonzero indices are calcualted
		T_flat = pool.starmap(T_function,[tuple([q,n,x]) for x in nonzero_indices])
		pool.close()
	else:
		T_flat = map(lambda x: T_function(q,n,x), nonzero_indices)
		T_flat = list(T_flat)
	endt = time.time()

	#print('T takes '+str(endt-startt))


	startm = time.time()
	#The calculated values of T and the corresponding indices are turned into a
	#sparse matrix.
	T = sparse.csr_matrix((T_flat,([i%(3**n) for i in nonzero_indices],
		[i//(3**n) for i in nonzero_indices])),shape=(3**n,3**n))
	#Probabilities are derived from the T matrix and initial conditions.
	probabilities = (T**n).dot(P)
	probabilities = probabilities[[np.all(state!=1) for state in state_arrays]]
	endm = time.time()

	#print('multiplication takes '+str(endm-startm))

	if size:
		population_powerset = list(powerset(range(n)))
		population_powerset.reverse()
		bignesses = np.array([int(sum([Rinf[i] for i in s])) 
			for s in population_powerset])
		bignesses.shape = (2**n)
		size = sum(probabilities*bignesses)
		return((probabilities,size))
	else:
		return(probabilities)

def probability_row_function(i,n,R0Y,R0Y_hat,L,MUY,SinitY,IinitY, size=False):
	#convert string to binary vector
	config = np.binary_repr(i,width=n)
	epidemic_bools = np.zeros(n)
	epidemic_bools.shape = (n,1)
	for place in range(n):
		epidemic_bools[place] = int(config[place])
	#Assign the correct R0 to each population
	R0 = R0Y*epidemic_bools + R0Y_hat*(1-epidemic_bools)
	if size:
		(P,S) = configuration_probabilities(L,R0,MUY,SinitY,IinitY)
		return((P,S))
	else:
		P = configuration_probabilities(L,R0,MUY,SinitY,IinitY)
		return(P)
 
# second_configuration_probabilities: get the probabilities of each subset of 
# populations experiencing an epidemic of pathogen Y following an outbreak of 
# pathogen X.
# L is a (n,n) array of rates of travel between populations.
# R0X and R0Y are (n,1) arrays of R0 in each popluation, for each pathogen.
# MUX and MUY are (n,1) arrays of the recovery rate in each population, for 
# each pathogen.
# SinitX and SinitY are (n,1) arrays of the number of initial susceptibles 
# in each population, for each pathogen.
# IinitX and IinitY are (n,1) arrays the number of initial infected in each
# population, for each pathogen.
# We require the populations with initial infecteds to be listed first.
# ProbsX is the output of configuraiton_probabilities for pathogen X.
# alpha is a scalar giving the degree of cross-immunity.
# size tells the function whether to output expected final size of epidemics.
# sample_size tells the function whether to perform a non-exhaustive sample 
# of scenarios, and give an approximate probaibility vector.
# OUTPUTS
# If size=False then the function outputs a (2**n) array of probabilities for
# each subset of populations experiencing an epidemic of pathogen Y, in order 
# from all populations to no populations. The order is given by assigning a 0 
# to populations with an epidemic, a 1 to populations with no epidemic, putting 
# the digits in order and reading it as a binary number - this gives the index 
# of that scenario.
# If size=True then a two-element array is returned, with the array described above 
# in the first position and a scalar indicating the expected total final size of
# the outbreak, summing over all scenarios and populations.
def second_configuration_probabilities(L,R0X,MUX,R0Y,MUY,SinitX,SinitY,IinitX,
	IinitY,ProbsX,alpha,size=False,sample_size=False):
	n = L.shape[1]
	N = SinitX+IinitX

	#Find R0Y_hat
	def Rinf_function_X(x):
		x.shape = (n,1)
		y=x - SinitX + SinitX*np.exp(-x*R0X/SinitX)
		y.shape = (n,)
		return(y)

	RinfX = optimize.fsolve(Rinf_function_X,(SinitX-1))
	RinfX = np.array(RinfX)
	RinfX.shape = (n,1)

	R0Y_hat = np.maximum((1- alpha*(RinfX/N))*R0Y,0)

	if sample_size:
		cases_to_test=np.random.choice(range(2**n),sample_size,
			replace=False,p=ProbsX[0])
	else:
		cases_to_test=np.arange(2**n)[ProbsX[0,:]!=0]

	#matrix of probabilties of each configuration of Y epdiemics given 
	#a configuration of X epidemics
	pool = mp.Pool(mp.cpu_count())
	probability_sample = pool.starmap(probability_row_function,
		[tuple([i,n,R0Y,R0Y_hat,L,MUY,SinitY,IinitY]) for i	in cases_to_test])
	pool.close()
	probability_matrix = np.zeros((2**n,2**n))
	for j in range(len(cases_to_test)):
		probability_matrix[cases_to_test[j],:] = probability_sample[j]
	probability_matrix = probability_matrix

	#multiply each configuration probability by the probability of the X configuration
	probability_matrix_weighted = probability_matrix*\
	np.broadcast_to(np.transpose(ProbsX),(2**n,2**n))
	if sample_size:
		probability_matrix_weighted=probability_matrix_weighted/\
		sum(sum(probability_matrix_weighted))
	probabilities = np.sum(probability_matrix_weighted,axis=0)

	if size:
		def Rinf_function_Y(x):
			x.shape = (n,1)
			y=x - SinitY + SinitY*np.exp(-x*R0Y/SinitY)
			y.shape = (n,)
			return(y)

		RinfY = optimize.fsolve(Rinf_function_Y,(SinitY-1))
		RinfY = np.array(RinfY)
		RinfY.shape = (n,1)

		def Rinf_function_Y_hat(x):
			x.shape = (n,1)
			y=x - SinitY + SinitY*np.exp(-x*R0Y_hat/SinitY)
			y.shape = (n,)
			return(y)

		RinfY_hat = optimize.fsolve(Rinf_function_Y_hat,(SinitY-1))
		RinfY_hat = np.array(RinfY_hat)
		RinfY_hat.shape = (n,1)
		RinfY_hat[R0Y_hat<1]=0

		size_matrix = np.zeros((2**n,2**n))
		epidemic_bools = np.zeros((n,2**n))
		for i in range(2**n):
			config = np.binary_repr(i,width=n)
			for place in range(n):
				epidemic_bools[place,i] = int(config[place])

		#Assign the correct Rinf to each population
		Rinf = np.broadcast_to(RinfY,(n,2**n))*epidemic_bools +\
		np.broadcast_to(RinfY_hat,(n,2**n))*(1-epidemic_bools)
		bignesses = np.row_stack([[sum([Rinf[int(j),i]*epidemic_bools[j,k] 
			for j in range(n)]) for k in reversed(range(2**n))] for i in range(2**n)])
		size_matrix = probability_matrix_weighted*bignesses
		size = sum(sum(size_matrix))
		return((probabilities,size))
	else:
		return(probabilities)
