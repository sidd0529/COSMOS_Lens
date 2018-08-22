from __future__ import division
import numpy as np


#----------- Fill locations with less density with appropriate (block-averaged) values ---------------------
def low_density_fill(indx_x, indx_y, neighb, matrx, edge, bin_num_RA, bin_num_Dec):
	x_start = edge
	x_end = (bin_num_Dec-edge)
	y_start = edge
	y_end = (bin_num_RA-edge)

	if( (indx_x-neighb) > x_start ): x_start = indx_x-neighb 
	if( (indx_x+neighb+1) < x_end ): x_end = indx_x+neighb+1 
	if( (indx_y-neighb) > y_start ): y_start = indx_y-neighb 
	if( (indx_y+neighb+1) < y_end ): y_end = indx_y+neighb+1 

	matt = matrx[ x_start : x_end, y_start : y_end]

	return np.mean(matt)