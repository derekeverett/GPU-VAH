/*
 * SemiDiscreteKurganovTadmorScheme.cpp
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#include "../include/SemiDiscreteKurganovTadmorScheme.h"
#include "../include/DynamicalVariables.h"
#include "../include/EnergyMomentumTensor.h"
#include "../include/LocalPropagationSpeed.h"

int flux(const PRECISION * const __restrict__ data, PRECISION * const __restrict__ result,
		PRECISION (* const rightHalfCellExtrapolation)(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp),
		PRECISION (* const leftHalfCellExtrapolation)(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp),
		PRECISION (* const spectralRadius)(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un),
		PRECISION (* const fluxFunction)(PRECISION q, PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un),
		PRECISION t, PRECISION ePrev, PRECISION uT, int i, int j, int k, double x, double y, double z
) {
	// left and right cells
	PRECISION qR[NUMBER_CONSERVED_VARIABLES], qL[NUMBER_CONSERVED_VARIABLES];

	// left and right extrapolated values of the conserved variables
	int ptr = 0;
	PRECISION qmm, qm, q, qp, qpp;
	for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
		qmm 	= *(data+ptr);
		qm 	= *(data+ptr+1);
		q 		= *(data+ptr+2);
		qp 	= *(data+ptr+3);
		qpp 	= *(data+ptr+4);
		ptr+=5;
		qR[n]	= rightHalfCellExtrapolation(qmm, qm, q, qp, qpp);
		qL[n]	= leftHalfCellExtrapolation(qmm, qm, q, qp, qpp);
	}

	// left and right extrapolated values of the primary variables
	PRECISION eR,pR,utR,uxR,uyR,unR;
	int statusR = getInferredVariables(t,qR,ePrev,uT,&eR,&pR,&utR,&uxR,&uyR,&unR, i, j, k, x,y,z,0);
	PRECISION eL,pL,utL,uxL,uyL,unL;
	int statusL = getInferredVariables(t,qL,ePrev,uT,&eL,&pL,&utL,&uxL,&uyL,&unL, i, j, k,x,y,z,0);

	PRECISION a,qR_n,qL_n,FqR,FqL,res;
	a = localPropagationSpeed(utR,uxR,uyR,unR,utL,uxL,uyL,unL,spectralRadius);
	for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
		qR_n = qR[n];
		qL_n = qL[n];
		FqR = fluxFunction(qR_n, utR, uxR, uyR, unR);
		FqL = fluxFunction(qL_n, utL, uxL, uyL, unL);
		res = FqR + FqL - a * (qR_n - qL_n);
		res /= 2;
		result[n] = res;
	}

	if(statusR==0 && statusL==0) return 0;
	else return 1;
}
