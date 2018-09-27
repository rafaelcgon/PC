import numpy as np
import PC 
from matplotlib import pylab as pl                         
import matplotlib as mpl                                   
import matplotlib.pyplot as plt                            
import seaborn as sns                                      #
import scipy.io as sio

def model(X,lx=3, A=1.5): 
	'''
		Generates a N-D Gaussian surface.

		Parameter
		----------
		X : ndarray
			M by N array, where M is the number of sampling points, and N is the number of input parameters.
		lx : ndarray
			N size array with correlation scales for each dimension of X
		A = amplitude of the Gaussian function

		Returns
		-------
		output : scalar model output
	'''
	if np.shape(lx) == ():
		lx = np.array([lx])
	if lx.size!= X.shape[0]:
		lx = np.repeat(lx[0],X.shape[0])
	B = np.array([((X[n,:]-X[n,:].mean())/lx[n])**2 for n in range(X.shape[0])])
	return A*np.exp(B.sum(axis=0)/(-2))


# 
polOrder = 6 # polynomial expansion order
figName = 'example_2D.png' 
#
x1 = np.arange(20,50.1,1) # input parameters
x2 = np.arange(-10,40.1,1) # input parameters
X1,X2= np.meshgrid(x1,x2)
X = np.concatenate([X1.reshape([1,-1]),X2.reshape([1,-1])],axis=0)
lx = np.array([10,20])
Y = model(X,lx) # model outputs at X points
Y = Y.reshape([x2.size,x1.size])
V = np.linspace(0,Y.max(),30)

 # initialize PC components
inLim = np.array([[x1.min(),x1.max()],[x2.min(),x2.max()]])
uq = PC.PCE(inputLim=inLim,polOrder=polOrder)

# sample model at quadrature points
Xqp1,Xqp2 = np.meshgrid(uq.input_qp[0,:],uq.input_qp[1,:])
Xqp = np.concatenate([Xqp1.reshape([1,-1]),Xqp2.reshape([1,-1])],axis=0)
Y_qp = model(Xqp,lx) 

# compute expansion coefficients
uq.computePCE(Y_qp) 

#sample surrogate at x points
surrogate = PC.surrogate(uq,X,1) 
surrogate = surrogate.reshape([x2.size,x1.size])

# absolute error of the surrogate
absError = np.abs(surrogate - Y) 
V2 = np.linspace(0,absError.max(),30)


# plot results
FS = 14
figW = 16
figH = 8
left = [0.05,0.37,0.69]
width = 0.27 
bottom = 0.14
height = 0.8 #0.27 

fig = plt.figure(figsize=(figW,figH))
plot1=fig.add_axes([left[0],bottom,width,height],aspect = 'equal')
plot1.contourf(x1,x2,Y,V,cmap=plt.get_cmap('jet'))   
plot1.plot(Xqp[0,:],Xqp[1,:],'ok')
plot1.tick_params(axis='both',which='major',labelsize=FS)
plot1.set_ylabel('Input 2',fontsize=FS)
plot1.set_xlabel('Input 1',fontsize=FS)
plot1.set_title('Model output',fontsize=FS)
plt.legend([r'Quadrature points'],loc=4)

plot2=fig.add_axes([left[1],bottom,width,height],aspect = 'equal')
plot2.contourf(x1,x2,surrogate,V,cmap=plt.get_cmap('jet'))   
plot2.plot(Xqp[0,:],Xqp[1,:],'ok')
plot2.tick_params(axis='both',which='major',labelsize=FS)
#plot2.set_ylabel('Input 2',fontsize=FS)
plot2.set_xlabel('Input 1',fontsize=FS)
plot2.set_title('PC surrogate',fontsize=FS)

plot3=fig.add_axes([left[2],bottom,width,height],aspect = 'equal')
plot3.contourf(x1,x2,absError,V,cmap=plt.get_cmap('jet'))   
plot3.plot(Xqp[0,:],Xqp[1,:],'ok')
plot3.tick_params(axis='both',which='major',labelsize=FS)
#plot3.set_ylabel('Input 2',fontsize=FS)
plot3.set_xlabel('Input 1',fontsize=FS)
plot3.set_title('Absolute error',fontsize=FS)

## INDEPENDENT COLORBAR
axbar = fig.add_axes([0.055, 0.045, 0.9, 0.02])
cb = mpl.colorbar.ColorbarBase(axbar, orientation = 'horizontal', boundaries = V,cmap='jet')
#    cb.set_ticks(Vt) #[-4, -3, -2, -1])
##    cb.set_ticklabels(VtL1)
cb.ax.tick_params(labelsize = FS)


pl.savefig(figName)