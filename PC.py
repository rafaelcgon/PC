from basisFunctions import *

class PCE(object): 
	def __init__(self,inputLim,polOrder,basis = 'Legendre'):    
		'''
		Initialize polynomial chaos expansion (PCE) variables
		inputLims : ND by 2 numpy array where ND is the number input parameters considered.
			inputLims contain the limits of each input parameters distribution.
		polOrder  : order of the polynomial expansions.
		basis     : polynomial basis function. 'Legendre' or 'Hermite'

		'''
		self.basis = basis
		self.quadpoint,self.weight = gaussQuad(basis,polOrder)
		self.inputLim = inputLim # limits of each input variable
		self.input_qp = getQuadPoint(inputLim,self.quadpoint) # input values at quadrature pts 
		self.pol,self.pol2 = getBasis(self.quadpoint,polOrder,basis)
		self.coef = np.array([])
		self.obs = np.array([])
		self.var= np.array([])
		self.pvar = np.array([])
		self.order = polOrder

	def computePCE(self, Uq):
		'''
		Compute polynomial expansion coefficients, total variance and marginal variance.

		Parameters
		----------
		Uq : ndarray
			2-D ndarray containing the quantity of interest sampled at quadrature points.
			1st dimension contains data sampled at quadrature points of uncertain inputs
			2nd dimension corresponds to other input parameters accounted in the PCE)

		'''
		if Uq.ndim==1:
			Uq = Uq[:,None]
		M = np.size(self.pol[0,:]) # highest order + 1
		Q = np.size(self.pol[:,0]) # number of 1D quadrature points
		ND = self.inputLim.shape[0]
		Pmq = self.pol*self.weight[:,None] / self.pol2   
		print Uq.shape,M,Q,ND
		self.coef = np.ndarray(np.concatenate([np.repeat(M,ND),np.array([Uq.shape[1]])]).tolist())*0
		if self.input_qp.shape[0]>1:
			ind=np.zeros((ND,Q**ND))	# Build the indices for each dimension
			for n in range(ND):
				ind[n,:]=np.mod(np.array(range(Q**ND))/Q**(ND-(n+1)),Q)
			ind = ind.astype('int64')

			for n in range(Q**ND): # loop over all quadrature runs
				aux = np.dot(Pmq[ind[0,n],:][:,None], Pmq[ind[1,n],:][None,:])
				matShape = np.array([M,M,1])
				if ND > 2:
					for i in range(2,ND):
						aux = np.dot(aux.reshape(matShape),Pmq[ind[i,n],:][None,:])
						matShape = np.concatenate([np.array(M),matShape])
				self.coef += np.dot(np.reshape(aux,np.concatenate([matShape,[1]])),np.reshape(Uq[n,:],[1,-1]))
		else:
			self.coef = np.dot(Pmq.T,Uq)
		self.obs = Uq
		self.var,self.pvar = getVar(self)

#####################################################################################
def getVar(pc):
	'''
		var,pvar = getVar(pc)
		Estimate ensamble variance from the PC coefficients.

		Parameters
		----------
		pc : PC object

		Returns
		-------
		var : ndarray
			1-D ndarray containing truncated variance.
		pvar : ndarray
			N-D ndarray containing marginal variance of each N input parameters.
	'''
	ND = pc.inputLim.shape[0]  # Number of dimensions
	M = np.size(pc.coef,ND)  # Number of grid points
	Q = np.size(pc.coef,0)   # Number of quadrature points in each dimension
	var = np.zeros(M)        # Total ensemble variance
	pvar = np.zeros((ND,M))  # Marginal variances 
	ind=np.zeros((ND,Q**ND)) # Matrix of indices 
	string1 = '(pc.coef['
	string2 = ''
	for n in range(ND):
		ind[n,:]=np.mod(np.array(range(Q**ND))/Q**(ND-(n+1)),Q)
		string1 = string1+'ind['+str(n)+',i],'
		string2 = string2+'*pc.pol2[ind['+str(n)+',i]]'
	string1 = string1+':]**2)'
	ind = ind.astype('int64')
	for i in range(Q**ND):
		if (np.sum(ind[0:ND,i],0) <= np.size(pc.coef,0)-1): # TRIANG. TRUNC. 
			if (i>0):
				exec('var+= '+string1+string2)
				for j in range(ND): # calculate marginal variance 
					if (ind[j,i]>0):
						exec('pvar[j,:] += '+string1+string2)
	return var,pvar

#####################################################################################
def surrogate(pc,xs,norm=1):
	'''
		surrogate = surrogate(pc,xs)
		Sample PC surrogate at xs input points

		Parameters
		----------
		pc : PC object
		xs : ndarray
			N by S ndarray containing the N uncertain inputs at the S sampling points.
		norm: normalize inputs
		Returns
		-------
		surrogate : ndarray
			1-D ndarray containing S PC outputs.
	'''
	if norm == 1:
		xs = getStandardParam(pc.inputLim,xs)
	ND = pc.inputLim.shape[0]
	M = np.size(pc.coef,ND) #  number of PC expansions
	Q = np.size(pc.coef,0) # number of quadrature points in each dimension

	if np.ndim(xs)==2:
		pol = np.zeros(((ND,Q,np.size(xs,1)))) 
	else:
		pol1,pol2 = getBasis(xs,Q-1,pc.basis)
		pol = np.reshape(pol1,[np.size(pol1,0),np.size(pol1,1),1])
 	ind=np.zeros((ND,Q**ND))
	string1 = 'surrogate+=np.dot(pc.coef['
	string2 = ':][:,None], '
    
	for n in range(ND): 
		ind[n,:]=np.mod(np.array(range(Q**ND))/Q**(ND-(n+1)),Q) 
		string1 = string1+'ind[' + str(n) + ',IND[ii]],'
		string2 = string2+'pol['+str(n)+',ind[' + str(n) + ',IND[ii]],:]*'
		if (np.ndim(xs)==2):
			pn,pn2 = getBasis(xs[n,:],Q-1,pc.basis)
			pol[n,:,:] = pn.transpose()
	ind = ind.astype('int64')
	command = string1 + string2[:-1] + '[None,:])'
	surrogate = np.zeros((M,np.size(pol,2)))

	for i in range(Q): 		# Here is the actual sum of coef*pol
		IND = np.where(np.sum(ind,0)==i)[0].tolist() 
		for ii in range(np.size(IND)): # Triangular truncation. No pol will have order higher than "i"
			exec(command)

#surrogate+=np.dot( uq.coef[ind[0,IND[ii]],         ind[1,IND[ii]],:][:,None],  
#               pol[0,ind[0,IND[ii]],:]*pol[1,ind[1,IND[ii]],:][None,:])      

	return surrogate

###########################################################################################
def getQuadPoint(xlim,xi):
	'''
		Get the quadrature point x from a standardized random variable xi within xlim
		For ND dimensions, xlim[:ND,0] and xlim[:ND,1] are the lower and upper limits, respectively.
	'''
	if xlim.ndim==1:
		xlim = xlim[None,:]
	if xi.ndim==1:
		xi = xi[None,:]
	return ((xi+1)*(xlim[:,1]-xlim[:,0])/2) + xlim[:,0]
###########################################################################################
def getStandardParam(xlim,x): # Considers xi E [-1,1]
	if xlim.ndim == 1:
		xlim = xlim[None,:]
	if x.ndim == 1:
		x = x[None,:]
	return (2*(x-xlim[:,0])/(xlim[:,1]-xlim[:,0]))-1
