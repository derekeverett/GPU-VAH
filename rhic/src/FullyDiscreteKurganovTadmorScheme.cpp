/*
 * FullyDiscreteKurganovTadmorScheme.cpp
 *
 *  Created on: Oct 23, 2015
 *      Author: bazow
 */

#include <stdlib.h>
#include <stdio.h> // for printf
#include <math.h>

#include "../include/FullyDiscreteKurganovTadmorScheme.h"
#include "../include/LatticeParameters.h"
#include "../include/DynamicalVariables.h"
#include "../include/SemiDiscreteKurganovTadmorScheme.h"
#include "../include/HalfSiteExtrapolation.h"
#include "../include/FluxFunctions.h"
#include "../include/SpectralRadius.h"
#include "../include/SourceTerms.h"
#include "../include/EnergyMomentumTensor.h"
#include "../include/HydroParameters.h"

#include "../include/FiniteDifference.h" //temp

#include "../include/HydrodynamicValidity.h"

#include "../include/EquationOfState.h"

#include "../include/AnisotropicDistributionFunctions.h"

/**************************************************************************************************************************************************/
void setNeighborCellsJK2(const PRECISION * const __restrict__ in, PRECISION * const __restrict__ out,
int s, int ptr, int smm, int sm, int sp, int spp
) {
	PRECISION data_ns = in[s];
	*(out + ptr		) = in[smm];
	*(out + ptr + 1) = in[sm];
	*(out + ptr + 2) = data_ns;
	*(out + ptr + 3) = in[sp];
	*(out + ptr + 4) = in[spp];
}

void eulerStepKernelSource(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p,
const FLUID_VELOCITY * const __restrict__ u, const FLUID_VELOCITY * const __restrict__ up,
int ncx, int ncy, int ncz, PRECISION dt, PRECISION dx, PRECISION dy, PRECISION dz, PRECISION etabar
) {
	for(int i = 2; i < ncx-2; ++i) {
		for(int j = 2; j < ncy-2; ++j) {
			for(int k = 2; k < ncz-2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, ncx, ncy);
				double x = (i-2 - (ncx-2-1)/2.)*dx;
				double y = (j-2 - (ncy-2-1)/2.)*dy;
				double z = (k-2 - (ncz-2-1)/2.)*dz;
				PRECISION Q[NUMBER_CONSERVED_VARIABLES];
				PRECISION S[NUMBER_CONSERVED_VARIABLES];

				Q[0] = currrentVars->ttt[s];
				Q[1] = currrentVars->ttx[s];
				Q[2] = currrentVars->tty[s];
				Q[3] = currrentVars->ttn[s];
				Q[4] = currrentVars->pl[s];
#ifdef PIMUNU
				Q[5] = currrentVars->pitt[s];
				Q[6] = currrentVars->pitx[s];
				Q[7] = currrentVars->pity[s];
				Q[8] = currrentVars->pitn[s];
				Q[9] = currrentVars->pixx[s];
				Q[10] = currrentVars->pixy[s];
				Q[11] = currrentVars->pixn[s];
				Q[12] = currrentVars->piyy[s];
				Q[13] = currrentVars->piyn[s];
				Q[14] = currrentVars->pinn[s];
#endif
#ifdef W_TZ_MU
				Q[15] = currrentVars->WtTz[s];
				Q[16] = currrentVars->WxTz[s];
				Q[17] = currrentVars->WyTz[s];
				Q[18] = currrentVars->WnTz[s];
#endif
#ifdef PI
				Q[NUMBER_CONSERVED_VARIABLES-1] = currrentVars->Pi[s];
#endif

				loadSourceTerms(Q, S, u, up->ut[s], up->ux[s], up->uy[s], up->un[s], t, e, p, s, ncx, ncy, ncz, etabar, dt, dx, dy, dz,i,j,k,x,y,z, currrentVars);

				PRECISION result[NUMBER_CONSERVED_VARIABLES];
				for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
					*(result+n) = *(Q+n) + dt * ( *(S+n) );
				}

				updatedVars->ttt[s] = result[0];
				updatedVars->ttx[s] = result[1];
				updatedVars->tty[s] = result[2];
				updatedVars->ttn[s] = result[3];
				updatedVars->pl[s] = result[4];
#ifdef PIMUNU
				updatedVars->pitt[s] = result[5];
				updatedVars->pitx[s] = result[6];
				updatedVars->pity[s] = result[7];
				updatedVars->pitn[s] = result[8];
				updatedVars->pixx[s] = result[9];
				updatedVars->pixy[s] = result[10];
				updatedVars->pixn[s] = result[11];
				updatedVars->piyy[s] = result[12];
				updatedVars->piyn[s] = result[13];
				updatedVars->pinn[s] = result[14];
#endif
#ifdef W_TZ_MU
				updatedVars->WtTz[s] = result[15];
				updatedVars->WxTz[s] = result[16];
				updatedVars->WyTz[s] = result[17];
				updatedVars->WnTz[s] = result[18];
#endif
#ifdef PI
				updatedVars->Pi[s] = result[NUMBER_CONSERVED_VARIABLES-1];
#endif
			}
		}
	}
}

void eulerStepKernelX(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const FLUID_VELOCITY * const __restrict__ u, const FLUID_VELOCITY * const __restrict__ up,
const PRECISION * const __restrict__ e,
int ncx, int ncy, int ncz, PRECISION dt, PRECISION dx, PRECISION dy, PRECISION dz,
double *fTSol
) {
	for(int i = 2; i < ncx-2; ++i) {
		for(int j = 2; j < ncy-2; ++j) {
			for(int k = 2; k < ncz-2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, ncx, ncy);
				PRECISION I[5 * NUMBER_CONSERVED_VARIABLES];
				PRECISION H[NUMBER_CONSERVED_VARIABLES];

				double x = (i-2 - (ncx-4-1)/2.)*dx;
				double y = (j-2 - (ncy-4-1)/2.)*dy;
				double z = (k-2 - (ncz-4-1)/2.)*dz;

				// calculate neighbor cell indices;
				int sim = s-1;
				int simm = sim-1;
				int sip = s+1;
				int sipp = sip+1;

				int ptr=0;
				setNeighborCellsJK2(currrentVars->ttt,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->ttx,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->tty,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->ttn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pl,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
#ifdef PIMUNU
				setNeighborCellsJK2(currrentVars->pitt,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pitx,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pity,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pitn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pixx,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pixy,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pixn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->piyy,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->piyn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pinn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
#endif
#ifdef W_TZ_MU
				setNeighborCellsJK2(currrentVars->WtTz,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->WxTz,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->WyTz,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
				setNeighborCellsJK2(currrentVars->WnTz,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
#endif
#ifdef PI
				setNeighborCellsJK2(currrentVars->Pi,I,s,ptr,simm,sim,sip,sipp);
#endif

				PRECISION uT = getTransverseFluidVelocityMagnitude(u, s);

				PRECISION result[NUMBER_CONSERVED_VARIABLES];
				int status1 = flux(I, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusX, &Fx, t, e[s], uT,i,j,k,x,y,z);
				for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
					*(result+n) = - *(H+n);
				}
				int status2 = flux(I, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusX, &Fx, t, e[s], uT,i,j,k,x,y,z);
				for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
					*(result+n) += *(H+n);
					*(result+n) /= dx;
				}

				if(status1==0 && status2==0) fTSol[s]=0.0;
				else fTSol[s]=1.0;

#ifndef IDEAL
				loadSourceTermsX(I, H, u, s, dx);
				for (unsigned int n = 0; n < NUMBER_CONSERVATION_LAWS; ++n) {
					*(result+n) += *(H+n);
					*(result+n) *= dt;
				}
#else
				for (unsigned int n = 0; n < NUMBER_CONSERVATION_LAWS; ++n) {
					*(result+n) *= dt;
				}
#endif
				for (unsigned int n = NUMBER_CONSERVATION_LAWS; n < NUMBER_CONSERVED_VARIABLES; ++n) {
					*(result+n) *= dt;
				}

				updatedVars->ttt[s] += result[0];
				updatedVars->ttx[s] += result[1];
				updatedVars->tty[s] += result[2];
				updatedVars->ttn[s] += result[3];
				updatedVars->pl[s] += result[4];
#ifdef PIMUNU
				updatedVars->pitt[s] += result[5];
				updatedVars->pitx[s] += result[6];
				updatedVars->pity[s] += result[7];
				updatedVars->pitn[s] += result[8];
				updatedVars->pixx[s] += result[9];
				updatedVars->pixy[s] += result[10];
				updatedVars->pixn[s] += result[11];
				updatedVars->piyy[s] += result[12];
				updatedVars->piyn[s] += result[13];
				updatedVars->pinn[s] += result[14];
#endif
#ifdef W_TZ_MU
				updatedVars->WtTz[s] += result[15];
				updatedVars->WxTz[s] += result[16];
				updatedVars->WyTz[s] += result[17];
				updatedVars->WnTz[s] += result[18];
#endif
#ifdef PI
				updatedVars->Pi[s] += result[NUMBER_CONSERVED_VARIABLES-1];
#endif
			}
		}
	}
}

void eulerStepKernelY(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const FLUID_VELOCITY * const __restrict__ u, const FLUID_VELOCITY * const __restrict__ up,
const PRECISION * const __restrict__ e,
int ncx, int ncy, int ncz, PRECISION dt, PRECISION dx, PRECISION dy, PRECISION dz,
double *fTSol
) {
	for(int i = 2; i < ncx-2; ++i) {
		for(int j = 2; j < ncy-2; ++j) {
			for(int k = 2; k < ncz-2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, ncx, ncy);
				PRECISION J[5* NUMBER_CONSERVED_VARIABLES];
				PRECISION H[NUMBER_CONSERVED_VARIABLES];

				double x = (i-2 - (ncx-4-1)/2.)*dx;
				double y = (j-2 - (ncy-4-1)/2.)*dy;
				double z = (k-2 - (ncz-4-1)/2.)*dz;

				// calculate neighbor cell indices;
				int sjm = s-ncx;
				int sjmm = sjm-ncx;
				int sjp = s+ncx;
				int sjpp = sjp+ncx;

				int ptr=0;
				setNeighborCellsJK2(currrentVars->ttt,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->ttx,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->tty,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->ttn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pl,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
#ifdef PIMUNU
				setNeighborCellsJK2(currrentVars->pitt,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pitx,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pity,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pitn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pixx,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pixy,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pixn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->piyy,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->piyn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pinn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
#endif
#ifdef W_TZ_MU
				setNeighborCellsJK2(currrentVars->WtTz,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->WxTz,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->WyTz,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->WnTz,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
#endif
#ifdef PI
				setNeighborCellsJK2(currrentVars->Pi,J,s,ptr,sjmm,sjm,sjp,sjpp);
#endif

				PRECISION uT = getTransverseFluidVelocityMagnitude(u, s);

				PRECISION result[NUMBER_CONSERVED_VARIABLES];
				int status1 = flux(J, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusY, &Fy, t, e[s], uT,i,j,k,x,y,z);
				for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
					*(result+n) = - *(H+n);
				}
				int status2 = flux(J, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusY, &Fy, t, e[s], uT,i,j,k,x,y,z);
				for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
					*(result+n) += *(H+n);
					*(result+n) /= dy;
				}

				if(status1==0 && status2==0) fTSol[s]=0.0;
				else fTSol[s]=1.0;

#ifndef IDEAL
				loadSourceTermsY(J, H, u, s, dy);
				for (unsigned int n = 0; n < NUMBER_CONSERVATION_LAWS; ++n) {
					*(result+n) += *(H+n);
					*(result+n) *= dt;
				}
#else
				for (unsigned int n = 0; n < NUMBER_CONSERVATION_LAWS; ++n) {
					*(result+n) *= dt;
				}
#endif
				for (unsigned int n = NUMBER_CONSERVATION_LAWS; n < NUMBER_CONSERVED_VARIABLES; ++n) {
					*(result+n) *= dt;
				}

				updatedVars->ttt[s] += result[0];
				updatedVars->ttx[s] += result[1];
				updatedVars->tty[s] += result[2];
				updatedVars->ttn[s] += result[3];
				updatedVars->pl[s] += result[4];
#ifdef PIMUNU
				updatedVars->pitt[s] += result[5];
				updatedVars->pitx[s] += result[6];
				updatedVars->pity[s] += result[7];
				updatedVars->pitn[s] += result[8];
				updatedVars->pixx[s] += result[9];
				updatedVars->pixy[s] += result[10];
				updatedVars->pixn[s] += result[11];
				updatedVars->piyy[s] += result[12];
				updatedVars->piyn[s] += result[13];
				updatedVars->pinn[s] += result[14];
#endif
#ifdef W_TZ_MU
				updatedVars->WtTz[s] += result[15];
				updatedVars->WxTz[s] += result[16];
				updatedVars->WyTz[s] += result[17];
				updatedVars->WnTz[s] += result[18];
#endif
#ifdef PI
				updatedVars->Pi[s] += result[NUMBER_CONSERVED_VARIABLES-1];
#endif
			}
		}
	}
}

void eulerStepKernelZ(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const FLUID_VELOCITY * const __restrict__ u, const FLUID_VELOCITY * const __restrict__ up,
const PRECISION * const __restrict__ e,
int ncx, int ncy, int ncz, PRECISION dt, PRECISION dx, PRECISION dy, PRECISION dz
) {
	for(int i = 2; i < ncx-2; ++i) {
		for(int j = 2; j < ncy-2; ++j) {
			for(int k = 2; k < ncz-2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, ncx, ncy);
				PRECISION K[5 * NUMBER_CONSERVED_VARIABLES];
				PRECISION H[NUMBER_CONSERVED_VARIABLES];

				double x = (i-2 - (ncx-4-1)/2.)*dx;
				double y = (j-2 - (ncy-4-1)/2.)*dy;
				double z = (k-2 - (ncz-4-1)/2.)*dz;

				// calculate neighbor cell indices;
				int stride = ncx * ncy;
				int skm = s-stride;
				int skmm = skm-stride;
				int skp = s+stride;
				int skpp = skp+stride;

				int ptr=0;
				setNeighborCellsJK2(currrentVars->ttt,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->ttx,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->tty,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->ttn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pl,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
#ifdef PIMUNU
				setNeighborCellsJK2(currrentVars->pitt,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pitx,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pity,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pitn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pixx,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pixy,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pixn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->piyy,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->piyn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->pinn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
#endif
#ifdef W_TZ_MU
				setNeighborCellsJK2(currrentVars->WtTz,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->WxTz,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->WyTz,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
				setNeighborCellsJK2(currrentVars->WnTz,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
#endif
#ifdef PI
				setNeighborCellsJK2(currrentVars->Pi,K,s,ptr,skmm,skm,skp,skpp);
#endif

				PRECISION uT = getTransverseFluidVelocityMagnitude(u, s);

				PRECISION result[NUMBER_CONSERVED_VARIABLES];
				flux(K, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusZ, &Fz, t, e[s], uT,i,j,k,x,y,z);
				for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
					*(result+n) = -*(H+n);
				}
				flux(K, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusZ, &Fz, t, e[s], uT,i,j,k,x,y,z);
				for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
					*(result+n) += *(H+n);
					*(result+n) /= dz;
				}

#ifndef IDEAL
				loadSourceTermsZ(K, H, u, s, t, dz);
				for (unsigned int n = 0; n < NUMBER_CONSERVATION_LAWS; ++n) {
					*(result+n) += *(H+n);
					*(result+n) *= dt;
				}
#else
				for (unsigned int n = 0; n < NUMBER_CONSERVATION_LAWS; ++n) {
					*(result+n) *= dt;
				}
#endif
				for (unsigned int n = NUMBER_CONSERVATION_LAWS; n < NUMBER_CONSERVED_VARIABLES; ++n) {
					*(result+n) *= dt;
				}

				updatedVars->ttt[s] += result[0];
				updatedVars->ttx[s] += result[1];
				updatedVars->tty[s] += result[2];
				updatedVars->ttn[s] += result[3];
				updatedVars->pl[s] += result[4];
#ifdef PIMUNU
				updatedVars->pitt[s] += result[5];
				updatedVars->pitx[s] += result[6];
				updatedVars->pity[s] += result[7];
				updatedVars->pitn[s] += result[8];
				updatedVars->pixx[s] += result[9];
				updatedVars->pixy[s] += result[10];
				updatedVars->pixn[s] += result[11];
				updatedVars->piyy[s] += result[12];
				updatedVars->piyn[s] += result[13];
				updatedVars->pinn[s] += result[14];
#endif
#ifdef W_TZ_MU
				updatedVars->WtTz[s] += result[15];
				updatedVars->WxTz[s] += result[16];
				updatedVars->WyTz[s] += result[17];
				updatedVars->WnTz[s] += result[18];
#endif
#ifdef PI
				updatedVars->Pi[s] += result[NUMBER_CONSERVED_VARIABLES-1];
#endif
			}
		}
	}
}
/**************************************************************************************************************************************************\

/**************************************************************************************************************************************************/
void convexCombinationEulerStepKernel(const CONSERVED_VARIABLES * const __restrict__ q, CONSERVED_VARIABLES * const __restrict__ Q,
int ncx, int ncy, int ncz
) {
	for(int i = 2; i < ncx-2; ++i) {
		for(int j = 2; j < ncy-2; ++j) {
			for(int k = 2; k < ncz-2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, ncx, ncy);
				Q->ttt[s] += q->ttt[s];
				Q->ttt[s] /= 2;
				Q->ttx[s] += q->ttx[s];
				Q->ttx[s] /= 2;
				Q->tty[s] += q->tty[s];
				Q->tty[s] /= 2;
				Q->ttn[s] += q->ttn[s];
				Q->ttn[s] /= 2;
				Q->pl[s] += q->pl[s];
				Q->pl[s] /= 2;
#ifdef PIMUNU
				Q->pitt[s] += q->pitt[s];
				Q->pitt[s] /= 2;
				Q->pitx[s] += q->pitx[s];
				Q->pitx[s] /= 2;
				Q->pity[s] += q->pity[s];
				Q->pity[s] /= 2;
				Q->pitn[s] += q->pitn[s];
				Q->pitn[s] /= 2;
				Q->pixx[s] += q->pixx[s];
				Q->pixx[s] /= 2;
				Q->pixy[s] += q->pixy[s];
				Q->pixy[s] /= 2;
				Q->pixn[s] += q->pixn[s];
				Q->pixn[s] /= 2;
				Q->piyy[s] += q->piyy[s];
				Q->piyy[s] /= 2;
				Q->piyn[s] += q->piyn[s];
				Q->piyn[s] /= 2;
				Q->pinn[s] += q->pinn[s];
				Q->pinn[s] /= 2;
#endif
#ifdef W_TZ_MU
				Q->WtTz[s] += q->WtTz[s];
				Q->WtTz[s] /= 2;
				Q->WxTz[s] += q->WxTz[s];
				Q->WxTz[s] /= 2;
				Q->WyTz[s] += q->WyTz[s];
				Q->WyTz[s] /= 2;
				Q->WnTz[s] += q->WnTz[s];
				Q->WnTz[s] /= 2;
#endif
#ifdef PI
				Q->Pi[s] += q->Pi[s];
				Q->Pi[s] /= 2;
#endif
			}
		}
	}
}

/**************************************************************************************************************************************************/
void
regulateDissipativeCurrents(PRECISION t,
CONSERVED_VARIABLES * const __restrict__ currrentVars,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
const FLUID_VELOCITY * const __restrict__ u,
int ncx, int ncy, int ncz, PRECISION dx, PRECISION dy, PRECISION dz,
VALIDITY_DOMAIN * const __restrict__ validityDomain
) {
	for(int i = 2; i < ncx-2; ++i) {
		for(int j = 2; j < ncy-2; ++j) {
			for(int k = 2; k < ncz-2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, ncx, ncy);

				//===================================================
				// Longitudinal pressure regularization
				//===================================================
				double eps=1.e-7;
				if(e[s]<0.0) {
					e[s] = 1.e-7;
					p[s] = 1.e-7;
				}
				if(currrentVars->pl[s]<0.0) currrentVars->pl[s]=1.e-7;
//				double P = p[s]+currrentVars->Pi[s];
//				if(currrentVars->pl[s] > 3*P) currrentVars->pl[s] = 3*P;
//				if(currrentVars->pl[s] > e[s]) currrentVars->pl[s] = e[s]*(1.0-1.e-5);

				PRECISION xi0 = 0.1;
				PRECISION rhomax = 1.0;

//xi0=1.0;
//rhomax=10.0;

xi0=1.0;
rhomax=1.0;

				PRECISION e_s = e[s];
				PRECISION p_s = p[s];
				PRECISION pl = currrentVars->pl[s];
				double ptHat = transversePressureHat(e_s, p_s, pl);

//===============================REGULARIZATION ON PL=============================================//
//double rhoL = fabs(pl/0.1/e_s);
//double facL = tanh(rhoL)/rhoL; if(rhoL<1.e-7) facL=1.0;
//currrentVars->pl[s] *= facL;
//===============================END REGULARIZATION ON PL=========================================//

				//===================================================
				// transverse shear stress regularization
				//===================================================
#ifdef PIMUNU
				PRECISION pitt = currrentVars->pitt[s];
				PRECISION pitx = currrentVars->pitx[s];
				PRECISION pity = currrentVars->pity[s];
				PRECISION pitn = currrentVars->pitn[s];
				PRECISION pixx = currrentVars->pixx[s];
				PRECISION pixy = currrentVars->pixy[s];
				PRECISION pixn = currrentVars->pixn[s];
				PRECISION piyy = currrentVars->piyy[s];
				PRECISION piyn = currrentVars->piyn[s];
				PRECISION pinn = currrentVars->pinn[s];

				PRECISION ut = u->ut[s];
				PRECISION ux = u->ux[s];
				PRECISION uy = u->uy[s];
				PRECISION un = u->un[s];

				PRECISION uT2 = ux*ux+uy*uy;
				PRECISION F = 1+uT2;
				double z0 = t*un/sqrt(F);
				double z3 = ut/t/sqrt(F);

				PRECISION t2 = t*t;

				double TrsTrs = e_s*e_s+pl*pl+2*ptHat*ptHat;
//				double TrsTrs = e_s*e_s+3*p_s*p_s;
//				double TrsTrs = e_s*e_s+3*ptHat*ptHat;
				double sTrsTrs = sqrt(fabs(TrsTrs));

				PRECISION pipi = pitt*pitt-2*pitx*pitx-2*pity*pity+pixx*pixx+2*pixy*pixy+piyy*piyy-2*pitn*pitn*t2+2*pixn*pixn*t2+2*piyn*piyn*t2+pinn*pinn*t2*t2;
				PRECISION spipi = sqrt(fabs(pipi));
				// g_{\mu\nu}\pi^{\mu\nu}_{\perp}
				PRECISION pimumu = pitt - pixx - piyy - pinn*t*t;
				// u_{\mu}\pi^{\mu\nu}_{\perp}
				PRECISION piu0 = -pitn*t2*un + pitt*ut - pitx*ux - pity*uy;
				PRECISION piu1 = -pixn*t2*un + pitx*ut - pixx*ux - pixy*uy;
				PRECISION piu2 = -piyn*t2*un + pity*ut - pixy*ux - piyy*uy;
				PRECISION piu3 = -pinn*t2*un + pitn*ut - pixn*ux - piyn*uy;
				// z_{\mu}\pi^{\mu\nu}_{\perp}
				PRECISION piz0 = z0*pitt-t2*z3*pitn;
				PRECISION piz1 = z0*pitx-t2*z3*pixn;
				PRECISION piz2 = z0*pity-t2*z3*piyn;
				PRECISION piz3 = z0*pitn-t2*z3*pinn;

				PRECISION a1 = spipi/rhomax/sTrsTrs;
				PRECISION den = xi0*rhomax*spipi;
/*
				PRECISION a2 = fabs(pimumu/den);
				PRECISION a3 = fabs(piu0/den);
				PRECISION a4 = fabs(piu1/den);
				PRECISION a5 = fabs(piu2/den);
				PRECISION a6 = fabs(piu3/den);
				PRECISION a7 = fabs(piz0/den);
				PRECISION a8 = fabs(piz1/den);
				PRECISION a9 = fabs(piz2/den);
				PRECISION a10 = fabs(piz3/den);
//*/
///*
				PRECISION a2 = pimumu/den;
				PRECISION a3 = piu0/den;
				PRECISION a4 = piu1/den;
				PRECISION a5 = piu2/den;
				PRECISION a6 = piu3/den;
				PRECISION a7 = piz0/den;
				PRECISION a8 = piz1/den;
				PRECISION a9 = piz2/den;
				PRECISION a10 = piz3/den;
//*/
				double rho = a1;
				if(a2 > rho) rho = a2;
				if(a3 > rho) rho = a3;
				if(a4 > rho) rho = a4;
				if(a5 > rho) rho = a5;
				if(a6 > rho) rho = a6;
				if(a7 > rho) rho = a7;
				if(a8 > rho) rho = a8;
				if(a9 > rho) rho = a9;
				if(a10 > rho) rho = a10;

				PRECISION fac = tanh(rho)/rho;
				if(rho<1.e-7) fac = 1.0;

				currrentVars->pitt[s] *= fac;
				currrentVars->pitx[s] *= fac;
				currrentVars->pity[s] *= fac;
				currrentVars->pitn[s] *= fac;
				currrentVars->pixx[s] *= fac;
				currrentVars->pixy[s] *= fac;
				currrentVars->pixn[s] *= fac;
				currrentVars->piyy[s] *= fac;
				currrentVars->piyn[s] *= fac;
				currrentVars->pinn[s] *= fac;

				validityDomain->regulations[s] = fac;

//===============================TEMPORARY DEBUGGING=============================================//
///*
				// magnitude
				double facMag = tanh(a1)/a1; if(a1<1.e-7) facMag=1.0;
				validityDomain->regMag[s] = facMag;
				// tracelessness
				double facTr = tanh(a2)/a2; if(a2<1.e-7) facTr=1.0;
				validityDomain->regTr[s] = facTr;
				// orthoganality to u
				double facU0 = tanh(a3)/a3; if(a3<1.e-7) facU0=1.0;
				double facU1 = tanh(a4)/a4; if(a4<1.e-7) facU1=1.0;
				double facU2 = tanh(a5)/a5; if(a5<1.e-7) facU2=1.0;
				double facU3 = tanh(a6)/a6; if(a6<1.e-7) facU3=1.0;
				validityDomain->regU0[s] = facU0;
				validityDomain->regU1[s] = facU1;
				validityDomain->regU2[s] = facU2;
				validityDomain->regU3[s] = facU3;
				// orthoganality to z
				double facZ0 = tanh(a7)/a7; if(a7<1.e-7) facZ0=1.0;
				double facZ1 = tanh(a8)/a8; if(a8<1.e-7) facZ1=1.0;
				double facZ2 = tanh(a9)/a9; if(a9<1.e-7) facZ2=1.0;
				double facZ3 = tanh(a10)/a10; if(a10<1.e-7) facZ3=1.0;
				validityDomain->regZ0[s] = facZ0;
				validityDomain->regZ1[s] = facZ1;
				validityDomain->regZ2[s] = facZ2;
				validityDomain->regZ3[s] = facZ3;
//*/
//===============================END TEMPORARY DEBUGGING=========================================//
#endif
#ifdef W_TZ_MU
				PRECISION WtTz = currrentVars->WtTz[s];
				PRECISION WxTz = currrentVars->WxTz[s];
				PRECISION WyTz = currrentVars->WyTz[s];
				PRECISION WnTz = currrentVars->WnTz[s];

				double WW = WtTz*WtTz-WxTz*WxTz-WyTz*WyTz-t2*WnTz*WnTz;
				WW = -2*(WtTz*WtTz-WxTz*WxTz-WyTz*WyTz-t2*WnTz*WnTz);
				double sWW = sqrt(fabs(WW));
				// u_{\mu}W^{\mu}_{\perp z}
				double Wu = ut*WtTz - ux*WxTz - uy*WyTz - t2*un*WnTz;
				// z_{\mu}W^{\mu}_{\perp z}
				double Wz = un*WtTz - ut*WnTz;

				double aW1 = sWW/rhomax/sTrsTrs;
				double denW = xi0*rhomax*sWW;
/*
				double aW2 = fabs(Wu/denW);
				double aW3 = fabs(Wz/denW);
*/
				double aW2 = Wu/denW;
				double aW3 = Wz/denW;

				double rhoW = aW1;
				if(aW2 > rhoW) rhoW = aW2;
				if(aW3 > rhoW) rhoW = aW3;

				double facW = tanh(rhoW)/rhoW;
				if(rhoW<1.e-7) facW = 1.0;

				currrentVars->WtTz[s] *= facW;
				currrentVars->WxTz[s] *= facW;
				currrentVars->WyTz[s] *= facW;
				currrentVars->WnTz[s] *= facW;
#endif
			}
		}
	}
}
/**************************************************************************************************************************************************/

void
rungeKutta2(PRECISION t, PRECISION dt, CONSERVED_VARIABLES * __restrict__ q, CONSERVED_VARIABLES * __restrict__ Q,
void * latticeParams, void * hydroParams
) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	struct HydroParameters * hydro = (struct HydroParameters *) hydroParams;

	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;
	int ncx = lattice->numComputationalLatticePointsX;
	int ncy = lattice->numComputationalLatticePointsY;
	int ncz = lattice->numComputationalLatticePointsRapidity;

	PRECISION dx = (PRECISION)(lattice->latticeSpacingX);
	PRECISION dy = (PRECISION)(lattice->latticeSpacingY);
	PRECISION dz = (PRECISION)(lattice->latticeSpacingRapidity);

	PRECISION etabar = (PRECISION)(hydro->shearViscosityToEntropyDensity);

	//===================================================
	// STEP 1:
	//===================================================
	eulerStepKernelSource(t, q, qS, e, p, u, up, ncx, ncy, ncz, dt, dx, dy, dz, etabar);
	eulerStepKernelX(t, q, qS, u, up, e, ncx, ncy, ncz, dt, dx, dy, dz, fTSol_X1);
	eulerStepKernelY(t, q, qS, u, up, e, ncx, ncy, ncz, dt, dx, dy, dz, fTSol_Y1);
	eulerStepKernelZ(t, q, qS, u, up, e, ncx, ncy, ncz, dt, dx, dy, dz);

	t+=dt;

	setInferredVariablesKernel(qS, e, p, u, uS, t, latticeParams, fTSol_1);

	regulateDissipativeCurrents(t, qS, e, p, uS, ncx, ncy, ncz, dx, dy, dz, validityDomain);

	setGhostCells(qS, e, p, uS, latticeParams);

	//===================================================
	// STEP 2:
	//===================================================
	eulerStepKernelSource(t, qS, Q, e, p, uS, u, ncx, ncy, ncz, dt, dx, dy, dz, etabar);
	eulerStepKernelX(t, qS, Q, uS, u, e, ncx, ncy, ncz, dt, dx, dy, dz, fTSol_X2);
	eulerStepKernelY(t, qS, Q, uS, u, e, ncx, ncy, ncz, dt, dx, dy, dz, fTSol_Y2);
	eulerStepKernelZ(t, qS, Q, uS, u, e, ncx, ncy, ncz, dt, dx, dy, dz);

	convexCombinationEulerStepKernel(q, Q, ncx, ncy, ncz);

	swapFluidVelocity(&up, &u);
	setInferredVariablesKernel(Q, e, p, up, u, t, latticeParams, fTSol_2);

	regulateDissipativeCurrents(t, Q, e, p, u, ncx, ncy, ncz, dx, dy, dz, validityDomain);

	setGhostCells(Q, e, p, u, latticeParams);

	checkValidity(t, validityDomain, q, e, p, u, up, ncx, ncy, ncz, etabar, dt, dx, dy, dz);
}
