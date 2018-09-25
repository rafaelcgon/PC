import numpy as np
import scipy.special as sp
from math import factorial

def gaussQuad(basis,polOrder):
	'''
	return sample points and weigth for the Gauss quadrature.
	'''
	if (basis == 'legendre')|(basis == 'Legendre'):
		return np.polynomial.legendre.leggauss(polOrder+1)
	elif (basis == 'hermite')|(basis == 'Hermite'):
		return np.polynomial.hermite.hermgauss(polOrder+1)
	else:
		return np.polynomial.laguerre.laggauss(polOrder+1)

#####################################################################################
def getBasis(x,N,basis='Legendre'):
	'''
	return polynomial basis function.
    x : standardized inputs
    N : truncation order of the polynomial series
    basis: orthogonal polynomial basis
	'''
	if (basis == 'legendre')|(basis == 'Legendre'):
		P = legendrePol(x,N)
		P2 = (2./(2*np.array(range(N+1))+1.))
	elif (basis == 'hermite')|(basis == 'Hermite'):
		P = spPol(x,N)
		P2 = np.array([np.sqrt(np.pi)*(2**n)*factorial(n) for n in range(N+1)])
	elif (basis == 'laguerre')|(basis == 'Laguerre'):
		P = laguerrePol(x,N)
		P2 = np.array([sp.gamma(n+1)/factorial(n) for n in range(N+1)])
	return P,P2
#####################################################################################
def legendrePol(x,N):
	'''
		L = legendrePol(x,N)

		computes the Legendre polynomial of degree 0:N at points specified by x
		using the recurrence relationship.
		Usage is L = legendrepol(x,N)
    '''
	L = np.array(np.zeros((np.size(x),N+1)))
	L[:,0] = 1.0 # Legendre polynomial of degree zero
	if (N>0):
		L[:,1] = x
		for n in range(2,N+1):
			a = float((2*n-1))/float(n) 
			b = float((n-1))/float(n)
			L[:,n] = a*x*L[:,n-1] - b*L[:,n-2]
	return L
#####################################################################################
def laguerrePol(x,N):
	'''
		L = laguerrePol(x,N)

		computes the Laguerre polynomial of degree 0:N at points specified by x
		using the recurrence relationship.
    '''
	L = np.array(np.zeros((np.size(x),N+1)))
	L[:,0] = 1.0 # Legendre polynomial of degree zero
	if (N>0):
		L[:,1] = 1 - x

		for n in np.arange(2,N+1):
			a = (2.*n-1.-x)/n 
			b = (n-1.)/n
			L[:,n] = a*L[:,n-1] - b*L[:,n-2]
	return L
#####################################################################################
def spPol(x,N,basis):
	'''
	Get polynomial from scipy.special
  	P = spPol(x,N,basis)
	x = array with input values
	N = max. polynomial order
	basis = type of polynomial (Hermite or Laguerre)
	'''
	
	P = np.array(np.zeros((np.size(x),N+1)))
	if (basis == 'hermite')|(basis == 'Hermite'):
		command = 'P[:,n]=sp.hermite(n)(x)'
	elif (basis == 'laguerre')|(basis == 'Laguerre'):
		command = 'P[:,n]=sp.laguerre(n)(x)'
	else:
		command = 'P[:,n]=sp.legendre(n)(x)'

	for n in range(N+1):
		exec(command)
	return P