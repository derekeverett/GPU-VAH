/*
 * EnergyMomentumTensor.cpp
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */
#include <stdlib.h>
#include <stdio.h> // for printf

#include <math.h> // for math functions

#include "../include/EnergyMomentumTensor.h"
#include "../include/DynamicalVariables.h"
#include "../include/LatticeParameters.h"

#include "../include/FullyDiscreteKurganovTadmorScheme.h" // for const params
#include "../include/EquationOfState.h"
#include "../include/AnisotropicDistributionFunctions.h"

#define MAX_ITERS 10000
#define VBAR 0.563624
#define EPS 0.1

PRECISION getTransverseFluidVelocityMagnitude(const FLUID_VELOCITY * const __restrict__ u, int s) {
		PRECISION u1 = u->ux[s];
		PRECISION u2 = u->uy[s];
		return sqrt(fabs(u1*u1+u2*u2));
}

int transverseFluidVelocityFromConservedVariables(PRECISION t, PRECISION ePrev, PRECISION uT_0,
PRECISION MB0, PRECISION MBT, PRECISION MB3, PRECISION PL, PRECISION Pi, double Ft, double x, double *uT,
int i, int jj, int k, double xi, double yj, double zk,
int fullTimeStepInversion
) {
	PRECISION uT0 = uT_0;	// initial guess for uT

	// Constants
	double Ft2 = Ft*Ft;
	double bT = x*MBT;
	double bL = x*x*MB0-Ft2*PL;
	double b = x*x+Ft2;

	double f,fp,DF;

	for(int j = 0; j < MAX_ITERS; ++j) {
		double e = MB0 - t*Ft*MB3 - uT0/sqrt(1 + uT0*uT0)*x*MBT;
		if(e < 0.0) return -1;
		double p = equilibriumPressure(e);
		double PtHat = 0.5*(e-PL);
		double Pt = PtHat + 1.5*Pi;

		double deduT = -x*MBT/pow(1 + uT0*uT0,1.5);
		double dPtduT = 0.5*deduT;

		f = uT0/sqrt(1+uT0*uT0)*(bL+b*Pt) - bT;
		fp = 1/pow(1 + uT0*uT0,1.5)*(bL+b*Pt)+uT0/sqrt(1+uT0*uT0)*b*dPtduT;

		if(fabs(fp)==0.0) fp = 1.e-16;

		DF = f/fp;

		*uT = uT0 - DF;

		if(isnan(*uT) || isinf(*uT) || *uT < 0 || *uT > 9.0072e+15) return -1;

		double DUT = fabs(*uT-uT0);
		double UT = fabs(*uT);
		if(DUT <=  1.e-7 * UT) return 0;
		uT0 = *uT;
	}
	return -1;
}

int getInferredVariables(PRECISION t, const PRECISION * const __restrict__ q, PRECISION ePrev, PRECISION uT_0,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
PRECISION * const __restrict__ ut, PRECISION * const __restrict__ ux, PRECISION * const __restrict__ uy, PRECISION * const __restrict__ un,
int i, int j, int k, double xi, double yj, double zk,
int fullTimeStepInversion
) {
	PRECISION ttt = q[0];
	PRECISION ttx = q[1];
	PRECISION tty = q[2];
	PRECISION ttn = q[3];
	PRECISION pl = q[4];
	// \pi^{\mu\nu}_{\perp}
#ifdef PIMUNU
	PRECISION pitt = q[5];
	PRECISION pitx = q[6];
	PRECISION pity = q[7];
	PRECISION pitn = q[8];
#else
	PRECISION pitt = 0;
	PRECISION pitx = 0;
	PRECISION pity = 0;
	PRECISION pitn = 0;
#endif
	// W^{\mu}_{\perp z}
#ifdef W_TZ_MU
	PRECISION WtTz = q[15];
	PRECISION WxTz = q[16];
	PRECISION WyTz = q[17];
	PRECISION WnTz = q[18];
#else
	PRECISION WtTz = 0;
	PRECISION WxTz = 0;
	PRECISION WyTz = 0;
	PRECISION WnTz = 0;
#endif
	// \Pi
#ifdef PI
	PRECISION Pi = q[NUMBER_CONSERVED_VARIABLES-1];
#else
	PRECISION Pi = 0;
#endif

	PRECISION M0 = ttt-pitt;
	PRECISION M1 = ttx-pitx;
	PRECISION M2 = tty-pity;
	PRECISION M3 = ttn-pitn;

	double t2 = t*t;

	double M0PL = M0+pl;
	if(M0PL==0.0) M0PL=1.e-16;

	PRECISION A = M3/M0PL;
	PRECISION At = t*A;
	double B = WtTz/M0PL/t;
	double Bt = t*B;
	double At2 = At*At;
	double Bt2 = Bt*Bt;
	double F = (A-fabs(B)*sqrt(fabs(1-At2+Bt2)))/(1+Bt2);
	double Ft = t*F;
	double Ft2 = Ft*Ft;
	PRECISION x = sqrt(fabs(1.-Ft2));

	double MB0 = M0-2*WtTz*Ft/x;
	double MB1 = M1-WxTz*Ft/x;
	double MB2 = M2-WyTz*Ft/x;
	double MB3 = M3-(1+Ft2)*WtTz/t/x;

	double MBT = sqrt(MB1*MB1+MB2*MB2);
	if(MBT==0.0) MBT=1.e-16;

	double uT;
	int status = -1;

	status = transverseFluidVelocityFromConservedVariables(t, ePrev, uT_0, MB0, MBT, MB3, pl, Pi, Ft, x, &uT, i, j, k, xi, yj, zk, fullTimeStepInversion);

	if(status == 0) {
		double C2 = 1.0+pow(uT,2.);
		double C = sqrt(C2);
		double U = uT/C;

		*ux=uT*MB1/MBT;
		*uy=uT*MB2/MBT;
		*un = F*C/x;
		*ut = C/x;
		//*ut = sqrt(C2+t2*pow((*un),2.));

		*e = MB0 - t*Ft*MB3 - U*x*MBT;
		*p = equilibriumPressure(*e);
	}	else {
		*e = ePrev*.999;
		*p = equilibriumPressure(*e);
		*ux=0.0;
		*uy=0.0;
//		*un = F/x;
//		*ut = sqrt(1.0+t2*pow((*un),2.));
		*un = 0.0;
		*ut = 1.0;
	}
//	if(*e > 1.1*ePrev && ePrev <= 0.1) {
//		*e = ePrev*.999;
//		*p = equilibriumPressure(*e);
//	}

	if (isnan(*e) || isnan(*ut) || isnan(*ux) || isnan(*uy) || isnan(*un)) {
		printf("=======================================================================================\n");
		printf("found NaN in getInferredVariables.\n");
		printf("Grid point = (%d, %d, %d) = (%.3f, %.3f, %.3f)\n", i, j, k, xi, yj, zk);
		if(fullTimeStepInversion==0) printf("From semiDiscreteKurganovTadmorAlgorithm.\n");
		printf("t=%.3f\n",t);
		printf("uT=%.9f\n",uT);
		printf("ePrev=%.9f\n",ePrev);
		printf("A=%.9f;B=%.9f;F=%.9f;x=%.9f;\n",A,B,F,x);
		printf("e=%.9f;p=%.9f;\n",*e,*p);
		printf("ut=%.9f;ux=%.9f;uy=%.9f;un=%.9f;\n",*ut,*ux,*uy,*un);
		printf("MB1=%.9f\n",MB1);
		printf("MB2=%.9f\n",MB2);
		printf("MB0=%.3f,\t MBT=%.3f,\t MB3=%.3f,\tPL=%.3f,ePrev=%.3f\t,uT_0=%.3f\n", MB0, MBT, MB3,pl,ePrev,uT_0);
		printf("=======================================================================================\n");
		exit(-1);
	}

	return status;
}

void setInferredVariablesKernel(const CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ up, FLUID_VELOCITY * const __restrict__ u,
PRECISION t, void * latticeParams, PRECISION *fTSolution
) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;

	int ncx,ncy,ncz;
	ncx = lattice->numComputationalLatticePointsX;
	ncy = lattice->numComputationalLatticePointsY;
	ncz = lattice->numComputationalLatticePointsRapidity;

	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;
	double dx = lattice->latticeSpacingX;
	double dy = lattice->latticeSpacingY;
	double dz = lattice->latticeSpacingRapidity;

	for(int i = 2; i < ncx-2; ++i) {
		double x = (i-2 - (nx-1)/2.)*dx;
		for(int j = 2; j < ncy-2; ++j) {
			double y = (j-2 - (ny-1)/2.)*dy;
			for(int k = 2; k < ncz-2; ++k) {
				double z = (k-2 - (nz-1)/2.)*dz;
				int s = columnMajorLinearIndex(i, j, k, ncx, ncy);

				PRECISION q_s[NUMBER_CONSERVED_VARIABLES],_e,_p,ut,ux,uy,un;
				q_s[0] = q->ttt[s];
				q_s[1] = q->ttx[s];
				q_s[2] = q->tty[s];
				q_s[3] = q->ttn[s];
				q_s[4] = q->pl[s];
				// \pi^{\mu\nu}_{\perp}
#ifdef PIMUNU
				q_s[5] = q->pitt[s];
				q_s[6] = q->pitx[s];
				q_s[7] = q->pity[s];
				q_s[8] = q->pitn[s];
#endif
				// w^{\mu}_{\perp z}
#ifdef W_TZ_MU
				q_s[15] = q->WtTz[s];
				q_s[16] = q->WxTz[s];
				q_s[17] = q->WyTz[s];
				q_s[18] = q->WnTz[s];
#endif
				// \Pi
#ifdef PI
				q_s[NUMBER_CONSERVED_VARIABLES-1] = q->Pi[s];
#endif

				PRECISION uT = getTransverseFluidVelocityMagnitude(up, s);

				int status = getInferredVariables(t,q_s,e[s],uT,&_e,&_p,&ut,&ux,&uy,&un,i,j,k,x,y,z,1);
				if(status==0) {
					fTSolution[s] = 0.0;
				} else {
					fTSolution[s] = 1.0;
				}
				e[s] = _e;
				p[s] = _p;
				u->ut[s] = ut;
				u->ux[s] = ux;
				u->uy[s] = uy;
				u->un[s] = un;
			}
		}
	}
}

//===================================================================
// Components of T^{\mu\nu} in (\tau,x,y,\eta_s)-coordinates
//===================================================================
PRECISION Ttt(PRECISION e, PRECISION p, PRECISION ut, PRECISION pitt) {
	return (e+p)*ut*ut-p+pitt;
}

PRECISION Ttx(PRECISION e, PRECISION p, PRECISION ut, PRECISION ux, PRECISION pitx) {
	return (e+p)*ut*ux+pitx;
}

PRECISION Tty(PRECISION e, PRECISION p, PRECISION ut, PRECISION uy, PRECISION pity) {
	return (e+p)*ut*uy+pity;
}

PRECISION Ttn(PRECISION e, PRECISION p, PRECISION ut, PRECISION un, PRECISION pitn) {
	return (e+p)*ut*un+pitn;
}
