# PC
The polynomial chaos expansion is an uncertainty quantification methodology in which the response of model outputs to uncertain input parameters is established by spectral projection with orthogonal polynomial basis functions.  

This repository contains Python 2 scripts to perform polynomial chaos expansion in multi-dimensional problems.

  - The expansion coefficients are computed by tensor product quadrature.
  - At the moment, Legendre polynomials are the only basis function available, which assume uniform input distributions.
  - Other 3 orthogonal polynomial basis functions will be implemented: Hermit, for Gaussian distributions; Jacobi, for beta distributions; and Laguerre, for gamma distributions. 
  


References:

Gonçalves, R. C., Iskandarani, M., Srinivasan, A., Thacker, W. C., Chassignet, E., & Knio, O. M. (2016). A framework to quantify uncertainty in simulations of oil transport in the ocean. Journal of Geophysical Research: Oceans, 121(4), 2058–2077. https://doi.org/10.1002/2015JC011311

Knio, O. M., & Le Maître, O. P. (2006). Uncertainty propagation in CFD using polynomial chaos decomposition. Fluid Dynamics Research, 38(9), 616–640. https://doi.org/10.1016/j.fluiddyn.2005.12.003

Thacker, W. C., Iskandarani, M., Gonçalves, R. C., Srinivasan, A., & Knio, O. M. (2015). Pragmatic aspects of uncertainty propagation: A conceptual review. Ocean Modelling, 95, 25–36. https://doi.org/10.1016/j.ocemod.2015.09.001
