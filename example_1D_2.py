import numpy as np
import PC 
from matplotlib import pylab as pl                         
import matplotlib as mpl                                   
import matplotlib.pyplot as plt                            
import seaborn as sns                                      #
import scipy.io as sio

def model(x): 
	'''
		This model is a 20th order PCE of another model.
		Parameter
		----------
		x : ndarray
			Uncertain input: oil droplet diameters between 200 and 500 micrometers.
		Returns
		-------
		surrogate : ndarray
			Oil concentration.
	'''
	#import oil model outputs sampled at quadrature points for a 20th order PCE
	fmat = sio.loadmat('PC_20th_order_data.mat')
	obs = fmat['observables'][:]   
	# build PCE 
	uq = PC.PCE(np.array([200,500])[None,:],20)
	uq.computePCE(obs)  
	# return surrogate samples at x points
	return np.squeeze(PC.surrogate(uq,x)) 

# 
polOrder = 8 # polynomial expansion order
figName = 'example_1D_2.png' 
#
x = np.arange(200,500.1,1) # input parameters
y = model(x) # model outputs at x points

 # initialize PC components
uq = PC.PCE(np.array([x.min(),x.max()])[None,:],polOrder)

# sample model at quadrature points
y_qp = model(uq.input_qp.squeeze()) 

# compute expansion coefficients
uq.computePCE(y_qp) 

#sample surrogate at x points
surrogate = PC.surrogate(uq,x,1) 

# absolute error of the surrogate
absError = np.abs(surrogate - y) 


# plot results
FS = 14
figW = 10
figH = 8 
left = [0.07,0.58]
width = 0.41 
bottom = [0.07,0.57] #[0.08,0.39,0.7]
height = 0.41 #0.27 

fig = plt.figure(figsize=(figW,figH))
plot1=fig.add_axes([left[0],bottom[1],width,height])
plot1.plot(x,y,'r',x,surrogate,'b',uq.input_qp.squeeze(),y_qp,'or')   
plot1.tick_params(axis='both',which='major',labelsize=FS)
plot1.set_ylabel('Output',fontsize=FS)
plot1.set_xlabel('Input',fontsize=FS)
plt.legend([r'model output',r'PC surrogate',r'model output at quadrature pts'],loc=4)

plot2=fig.add_axes([left[1],bottom[1],width,height])
plot2.plot(x,absError,'k')   
plot2.set_yscale('log')
plot2.tick_params(axis='both',which='major',labelsize=FS)
plot2.set_ylabel('Surrogate absolute error',fontsize=FS)
plot2.set_xlabel('Input',fontsize=FS)

plot3=fig.add_axes([left[0],bottom[0],width,height])
sns.distplot(surrogate,10) 
plot3.tick_params(axis='both',which='major',labelsize=FS)
plot3.set_ylabel('Surrogate PDF',fontsize=FS)
plot3.set_xlabel('Output',fontsize=FS)

plot4=fig.add_axes([left[1],bottom[0],width,height])
plot4.semilogy(np.abs(uq.coef))   
plot4.tick_params(axis='both',which='major',labelsize=FS)
plot4.set_ylabel('PC coefficient',fontsize=FS)
plot4.set_xlabel('Polynomial order',fontsize=FS)

pl.savefig(figName)