from __future__ import division
import numpy as np


#----------------------------- COSMOS galaxy bias (arXiv:1205.1064) ----------------------------------------
def bias_Amara(z, bias_type, zcutoff):
	y = z/(1+z)
	ycutoff = zcutoff/(1+zcutoff)

	#bias_type --> nn:fifth nearest neighbor or gs: gaussian smoothing
	if(bias_type=='nn'): b0=1.25; f=-1.4 
	if(bias_type=='gs'): b0=0.7 ; f=-1.615

	bmax = b0/(1 + (f*ycutoff) )

	if(z<=zcutoff): return b0/(1+ (f*y))
	if(z>zcutoff): return bmax


#----------------- Galaxy bias (1.> arXiv:1101.2453 (pg. 4), 2.> arXiv:0810.0003 (pg. 7, Table 3)) ---------
def bias_Porto(z):
	return np.sqrt(1+z)