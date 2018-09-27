import numpy as np
import PC 
from matplotlib import pylab as pl                         
import matplotlib as mpl                                   
import matplotlib.pyplot as plt                            
import seaborn as sns                                      #

def model(x): 
	'''
		Arbitrary 1D function to emulate the response of a model output to an input parameter x.
	'''
	return 0.5**2*np.exp(-(x-3)/2)*x**4 + np.cos(np.pi*x/0.5)

# 
polOrder = 16 # polynomial expansion order
figName = 'example_1D.png' 
#
x = np.arange(-1,1.1,0.01) # input parameters
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
plt.legend([r'model output',r'PC surrogate',r'model output at quadrature pts'],loc=1)

plot2=fig.add_axes([left[1],bottom[1],width,height])
plot2.plot(x,absError,'k')   
plot2.set_yscale('log')
plot2.tick_params(axis='both',which='major',labelsize=FS)
plot2.set_ylabel('Surrogate absolute error',fontsize=FS)
plot2.set_xlabel('Input',fontsize=FS)

plot3=fig.add_axes([left[0],bottom[0],width,height])
sns.distplot(surrogate,20) 
plot3.tick_params(axis='both',which='major',labelsize=FS)
plot3.set_ylabel('Surrogate PDF',fontsize=FS)
plot3.set_xlabel('Output',fontsize=FS)

plot4=fig.add_axes([left[1],bottom[0],width,height])
plot4.semilogy(np.abs(uq.coef))   
plot4.tick_params(axis='both',which='major',labelsize=FS)
plot4.set_ylabel('PC coefficient',fontsize=FS)
plot4.set_xlabel('Polynomial order',fontsize=FS)

pl.savefig(figName)