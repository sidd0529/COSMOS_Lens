#include "potential.h"


void pot_field(double* Kap, double* pott, double* Dec_bin_mid, double* RA_bin_mid, double softening, int Dec_len, int RA_len)
{
	int posn, xcell, ycell, xiter, yiter;
	double convergence[RA_len][Dec_len], RA_mid[RA_len], Dec_mid[Dec_len], RA_cell, Dec_cell, RA_iter, Dec_iter, r, val;

	/* Use pointers to read and store elements of convergence matrix, Kappa. */
	for(int i=0; i<RA_len; i++)
	{
		for(int j=0; j<Dec_len; j++)
		{
			posn = i*Dec_len +j;
			convergence[i][j] = *(Kap+posn);
		}
	}


	/* Use pointers to read and store elements of midpoints of RA. */
	for(int i=0; i<RA_len; i++)
	{
		RA_mid[i] = *(RA_bin_mid+i);
	}


	/* Use pointers to read and store elements of midpoints of RA. */
	for(int i=0; i<Dec_len; i++)
	{
		Dec_mid[i] = *(Dec_bin_mid+i);
	}


	/* Compute the potential field using convergence field (convergence). */
	for(xcell=0; xcell<Dec_len; xcell++)
	{
		for(ycell=0; ycell<RA_len; ycell++)
		{
			Dec_cell = Dec_mid[xcell];
			RA_cell = RA_mid[ycell];

			for(xiter=0; xiter<Dec_len; xiter++)
			{
				for(yiter=0; yiter<RA_len; yiter++)
				{
					if( (xcell!=xiter) || (ycell!=yiter) )
					{
						Dec_iter = Dec_mid[xiter];
						RA_iter = RA_mid[yiter];

						r = sqrt( pow( (Dec_cell-Dec_iter) , 2 ) + pow( (RA_cell-RA_iter) , 2 ) ) + softening;
						val = (2*convergence[xiter][yiter])/r;

						posn = xcell*RA_len + ycell;
						*( pott+posn ) += val;
					}
				}
			}
		}
	}

}