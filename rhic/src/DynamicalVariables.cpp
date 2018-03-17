/*
 * DynamicalVariables.cpp
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */
#include <stdlib.h>
#include <math.h> // for math functions

#include "../include/DynamicalVariables.h"
#include "../include/LatticeParameters.h"
#include "../include/EnergyMomentumTensor.h"
#include "../include/FullyDiscreteKurganovTadmorScheme.h" // for ghost cells

#include "../include/EquationOfState.h" // TEMPORARY for sloppy implementation of xi/Lambda initial conditions

#include "../include/AnisotropicDistributionFunctions.h"

CONSERVED_VARIABLES *q,*Q,*qS;

FLUID_VELOCITY *u,*up,*uS;

PRECISION *e, *p;

VALIDITY_DOMAIN *validityDomain;

double *fTSol_X1,*fTSol_Y1,*fTSol_1,*fTSol_X2,*fTSol_Y2,*fTSol_2;

int columnMajorLinearIndex(int i, int j, int k, int nx, int ny) {
	return i + nx * (j + ny * k);
}

void allocateHostMemory(int len) {
	size_t bytes = sizeof(PRECISION);

	//=======================================================
	// Primary variables
	//=======================================================
	e = (PRECISION *)calloc(len, bytes);
	p = (PRECISION *)calloc(len,bytes);
	// fluid velocity at current time step
	u = (FLUID_VELOCITY *)calloc(1, sizeof(FLUID_VELOCITY));
	u->ut = (PRECISION *)calloc(len,bytes);
	u->ux = (PRECISION *)calloc(len,bytes);
	u->uy = (PRECISION *)calloc(len,bytes);
	u->un = (PRECISION *)calloc(len,bytes);
	// fluid velocity at previous time step
	up = (FLUID_VELOCITY *)calloc(1, sizeof(FLUID_VELOCITY));
	up->ut = (PRECISION *)calloc(len,bytes);
	up->ux = (PRECISION *)calloc(len,bytes);
	up->uy = (PRECISION *)calloc(len,bytes);
	up->un = (PRECISION *)calloc(len,bytes);
	// fluid velocity at intermediate time step
	uS = (FLUID_VELOCITY *)calloc(1, sizeof(FLUID_VELOCITY));
	uS->ut = (PRECISION *)calloc(len,bytes);
	uS->ux = (PRECISION *)calloc(len,bytes);
	uS->uy = (PRECISION *)calloc(len,bytes);
	uS->un = (PRECISION *)calloc(len,bytes);

	//=======================================================
	// Conserved variables
	//=======================================================
	q = (CONSERVED_VARIABLES *)calloc(1, sizeof(CONSERVED_VARIABLES));
	q->ttt = (PRECISION *)calloc(len, bytes);
	q->ttx = (PRECISION *)calloc(len, bytes);
	q->tty = (PRECISION *)calloc(len, bytes);
	q->ttn = (PRECISION *)calloc(len, bytes);
	q->pl = (PRECISION *)calloc(len, bytes);
#ifdef PIMUNU
	q->pitt = (PRECISION *)calloc(len, bytes);
	q->pitx = (PRECISION *)calloc(len, bytes);
	q->pity = (PRECISION *)calloc(len, bytes);
	q->pitn = (PRECISION *)calloc(len, bytes);
	q->pixx = (PRECISION *)calloc(len, bytes);
	q->pixy = (PRECISION *)calloc(len, bytes);
	q->pixn = (PRECISION *)calloc(len, bytes);
	q->piyy = (PRECISION *)calloc(len, bytes);
	q->piyn = (PRECISION *)calloc(len, bytes);
	q->pinn = (PRECISION *)calloc(len, bytes);
#endif
	// allocate space for W_Tz
#ifdef W_TZ_MU
	q->WtTz = (PRECISION *)calloc(len, bytes);
	q->WxTz = (PRECISION *)calloc(len, bytes);
	q->WyTz = (PRECISION *)calloc(len, bytes);
	q->WnTz = (PRECISION *)calloc(len, bytes);
#endif
	// allocate space for \Pi
#ifdef PI
	q->Pi = (PRECISION *)calloc(len, bytes);
#endif
	// upated variables at the n+1 time step
	Q = (CONSERVED_VARIABLES *)calloc(1, sizeof(CONSERVED_VARIABLES));
	Q->ttt = (PRECISION *)calloc(len, bytes);
	Q->ttx = (PRECISION *)calloc(len, bytes);
	Q->tty = (PRECISION *)calloc(len, bytes);
	Q->ttn = (PRECISION *)calloc(len, bytes);
	Q->pl = (PRECISION *)calloc(len, bytes);
#ifdef PIMUNU
	Q->pitt = (PRECISION *)calloc(len, bytes);
	Q->pitx = (PRECISION *)calloc(len, bytes);
	Q->pity = (PRECISION *)calloc(len, bytes);
	Q->pitn = (PRECISION *)calloc(len, bytes);
	Q->pixx = (PRECISION *)calloc(len, bytes);
	Q->pixy = (PRECISION *)calloc(len, bytes);
	Q->pixn = (PRECISION *)calloc(len, bytes);
	Q->piyy = (PRECISION *)calloc(len, bytes);
	Q->piyn = (PRECISION *)calloc(len, bytes);
	Q->pinn = (PRECISION *)calloc(len, bytes);
#endif
	// allocate space for W_Tz
#ifdef W_TZ_MU
	Q->WtTz = (PRECISION *)calloc(len, bytes);
	Q->WxTz = (PRECISION *)calloc(len, bytes);
	Q->WyTz = (PRECISION *)calloc(len, bytes);
	Q->WnTz = (PRECISION *)calloc(len, bytes);
#endif
	// allocate space for \Pi
#ifdef PI
	Q->Pi = (PRECISION *)calloc(len, bytes);
#endif
	// updated variables at the intermediate time step
	qS = (CONSERVED_VARIABLES *)calloc(1, sizeof(CONSERVED_VARIABLES));
	qS->ttt = (PRECISION *)calloc(len, bytes);
	qS->ttx = (PRECISION *)calloc(len, bytes);
	qS->tty = (PRECISION *)calloc(len, bytes);
	qS->ttn = (PRECISION *)calloc(len, bytes);
	qS->pl = (PRECISION *)calloc(len, bytes);
#ifdef PIMUNU
	qS->pitt = (PRECISION *)calloc(len, bytes);
	qS->pitx = (PRECISION *)calloc(len, bytes);
	qS->pity = (PRECISION *)calloc(len, bytes);
	qS->pitn = (PRECISION *)calloc(len, bytes);
	qS->pixx = (PRECISION *)calloc(len, bytes);
	qS->pixy = (PRECISION *)calloc(len, bytes);
	qS->pixn = (PRECISION *)calloc(len, bytes);
	qS->piyy = (PRECISION *)calloc(len, bytes);
	qS->piyn = (PRECISION *)calloc(len, bytes);
	qS->pinn = (PRECISION *)calloc(len, bytes);
#endif
	// allocate space for W_Tz
#ifdef W_TZ_MU
	qS->WtTz = (PRECISION *)calloc(len, bytes);
	qS->WxTz = (PRECISION *)calloc(len, bytes);
	qS->WyTz = (PRECISION *)calloc(len, bytes);
	qS->WnTz = (PRECISION *)calloc(len, bytes);
#endif
	// allocate space for \Pi
#ifdef PI
	qS->Pi = (PRECISION *)calloc(len, bytes);
#endif

	//=======================================================
	// Validity domain
	//=======================================================
	validityDomain = (VALIDITY_DOMAIN *)calloc(1, sizeof(VALIDITY_DOMAIN));
	validityDomain->knudsenNumberTaupiT = (PRECISION *)calloc(len, bytes);
	validityDomain->knudsenNumberTaupiL = (PRECISION *)calloc(len, bytes);
	validityDomain->knudsenNumberTaupi = (PRECISION *)calloc(len, bytes);
	validityDomain->knudsenNumberTauPi = (PRECISION *)calloc(len, bytes);
	validityDomain->Rpi = (PRECISION *)calloc(len, bytes);
	validityDomain->RPi = (PRECISION *)calloc(len, bytes);
	validityDomain->Rw = (PRECISION *)calloc(len, bytes);
	validityDomain->Rpi2 = (PRECISION *)calloc(len, bytes);
	validityDomain->RPi2 = (PRECISION *)calloc(len, bytes);
	validityDomain->Rw2 = (PRECISION *)calloc(len, bytes);
	validityDomain->fTSolution = (PRECISION *)calloc(len, bytes);
	validityDomain->regulations = (PRECISION *)calloc(len, bytes);
	validityDomain->regMag = (PRECISION *)calloc(len, bytes);
	validityDomain->regTr = (PRECISION *)calloc(len, bytes);
	validityDomain->regU0 = (PRECISION *)calloc(len, bytes);
	validityDomain->regU1 = (PRECISION *)calloc(len, bytes);
	validityDomain->regU2 = (PRECISION *)calloc(len, bytes);
	validityDomain->regU3 = (PRECISION *)calloc(len, bytes);
	validityDomain->regZ0 = (PRECISION *)calloc(len, bytes);
	validityDomain->regZ1 = (PRECISION *)calloc(len, bytes);
	validityDomain->regZ2 = (PRECISION *)calloc(len, bytes);
	validityDomain->regZ3 = (PRECISION *)calloc(len, bytes);
	for(int s=0; s<len; ++s) validityDomain->regulations[s] = (PRECISION) 1.0;
	validityDomain->stt = (PRECISION *)calloc(len, bytes);
	validityDomain->sxx = (PRECISION *)calloc(len, bytes);
	validityDomain->syy = (PRECISION *)calloc(len, bytes);
	validityDomain->snn = (PRECISION *)calloc(len, bytes);
	validityDomain->taupi = (PRECISION *)calloc(len, bytes);
	validityDomain->dxux = (PRECISION *)calloc(len, bytes);
	validityDomain->dyuy = (PRECISION *)calloc(len, bytes);
	validityDomain->theta = (PRECISION *)calloc(len, bytes);

	fTSol_X1 = (double *)calloc(len,bytes);
	fTSol_Y1 = (double *)calloc(len,bytes);
	fTSol_1 = (double *)calloc(len,bytes);
	fTSol_X2 = (double *)calloc(len,bytes);
	fTSol_Y2 = (double *)calloc(len,bytes);
	fTSol_2 = (double *)calloc(len,bytes);
}

void setConservedVariables(double t, void * latticeParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;

	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;
	int ncx = lattice->numComputationalLatticePointsX;
	int ncy = lattice->numComputationalLatticePointsY;

	for (int k = N_GHOST_CELLS_M; k < nz+N_GHOST_CELLS_M; ++k) {
		for (int j = N_GHOST_CELLS_M; j < ny+N_GHOST_CELLS_M; ++j) {
			for (int i = N_GHOST_CELLS_M; i < nx+N_GHOST_CELLS_M; ++i) {
				int s = columnMajorLinearIndex(i, j, k, ncx, ncy);

				PRECISION ux = u->ux[s];
				PRECISION uy = u->uy[s];
				PRECISION un = u->un[s];
				PRECISION ut = u->ut[s];
				PRECISION eeq = e[s];
				PRECISION peq = p[s];

				PRECISION pitt = 0;
				PRECISION pitx = 0;
				PRECISION pity = 0;
				PRECISION pitn = 0;
				// W
				PRECISION WtTz = 0;
				PRECISION WxTz = 0;
				PRECISION WyTz = 0;
				PRECISION WnTz = 0;
				// Pi
				PRECISION Pi = 0;
#ifdef PIMUNU
				pitt = q->pitt[s];
				pitx = q->pitx[s];
				pity = q->pity[s];
				pitn = q->pitn[s];
#endif
#ifdef W_TZ_MU
				q->WtTz[s] = 0;
				q->WxTz[s] = 0;
				q->WyTz[s] = 0;
				q->WnTz[s] = 0;
#endif
#ifdef PI
				Pi = q->Pi[s];
#endif

				double R220_100 = 0.000686059;
				double R220_10 = 0.0154483;
				double R220_0 = 1./3.;
				PRECISION pl = R220_10*eeq;

				PRECISION ptHat = transversePressureHat(eeq, peq, pl);
				ptHat = 0.5*(eeq-pl);
				PRECISION pt = ptHat + 1.5*Pi;
				PRECISION DP = pl-pt;

				PRECISION uT2 = ux*ux+uy*uy;
				PRECISION uT = sqrt(uT2);
				PRECISION F = 1+uT2;
				PRECISION FS = sqrt(1+uT2);

				double z0 = t*un/FS;
				double z3 = ut/t/FS;
				// L functions
				PRECISION Ltt = DP*t*t*un*un/F;
				PRECISION Ltx = 0;
				PRECISION Lty = 0;
				PRECISION Ltn = DP*ut*un/F;
				// W functions
				double Wtt = 2*WtTz*z0;
				double Wtx = WxTz*z0;
				double Wty = WyTz*z0;
				double Wtn = WtTz*z3+WnTz*z0;

				q->ttt[s] = (eeq+pt)*ut*ut - pt + Ltt + Wtt + pitt;
				q->ttx[s] = (eeq+pt)*ut*ux + Ltx + Wtx + pitx;
				q->tty[s] = (eeq+pt)*ut*uy + Lty + Wty + pity;
				q->ttn[s] = (eeq+pt)*ut*un + Ltn + Wtn + pitn;
				q->pl[s] = pl;

				// set up to u
				up->ut[s] = ut;
				up->ux[s] = ux;
				up->uy[s] = uy;
				up->un[s] = un;
			}
		}
	}
}

void setConservedVariables_old(double t, void * latticeParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;

	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;
	int ncx = lattice->numComputationalLatticePointsX;
	int ncy = lattice->numComputationalLatticePointsY;

	for (int k = N_GHOST_CELLS_M; k < nz+N_GHOST_CELLS_M; ++k) {
		for (int j = N_GHOST_CELLS_M; j < ny+N_GHOST_CELLS_M; ++j) {
			for (int i = N_GHOST_CELLS_M; i < nx+N_GHOST_CELLS_M; ++i) {
				int s = columnMajorLinearIndex(i, j, k, ncx, ncy);

				PRECISION ux = u->ux[s];
				PRECISION uy = u->uy[s];
				PRECISION un = u->un[s];
				PRECISION ut = u->ut[s];
				PRECISION eeq = e[s];
				PRECISION peq = p[s];

				PRECISION pitt = 0;
				PRECISION pitx = 0;
				PRECISION pity = 0;
				PRECISION pitn = 0;
#ifdef PIMUNU
				pitt = q->pitt[s];
				pitx = q->pitx[s];
				pity = q->pity[s];
				pitn = q->pitn[s];
#endif

				PRECISION Pi = 0;
#ifdef PI
				Pi = q->Pi[s];
#endif
				PRECISION P = peq+Pi;
				q->ttt[s] = Ttt(eeq, peq + Pi, ut, pitt);
				q->ttx[s] = Ttx(eeq, peq + Pi, ut, ux, pitx);
				q->tty[s] = Tty(eeq, peq + Pi, ut, uy, pity);
				q->ttn[s] = Ttn(eeq, peq + Pi, ut, un, pitn);
				// ISOTROPIC
				q->pl[s] = peq;
#ifdef W_TZ_MU
				q->WtTz[s] = 0;
				q->WxTz[s] = 0;
				q->WyTz[s] = 0;
				q->WnTz[s] = 0;
#endif

				// set up to u
				up->ut[s] = ut;
				up->ux[s] = ux;
				up->uy[s] = uy;
				up->un[s] = un;
			}
		}
	}
}

void setGhostCells(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u, void * latticeParams
) {
	setGhostCellsKernelI(q,e,p,u,latticeParams);
	setGhostCellsKernelJ(q,e,p,u,latticeParams);
	setGhostCellsKernelK(q,e,p,u,latticeParams);
}

void setGhostCellVars(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u,
int s, int sBC) {
	e[s] = e[sBC];
	p[s] = p[sBC];
	u->ut[s] = u->ut[sBC];
	u->ux[s] = u->ux[sBC];
	u->uy[s] = u->uy[sBC];
	u->un[s] = u->un[sBC];
	q->ttt[s] = q->ttt[sBC];
	q->ttx[s] = q->ttx[sBC];
	q->tty[s] = q->tty[sBC];
	q->ttn[s] = q->ttn[sBC];
	q->pl[s] = q->pl[sBC];
	// set \pi^\mu\nu ghost cells if evolved
#ifdef PIMUNU
	q->pitt[s] = q->pitt[sBC];
	q->pitx[s] = q->pitx[sBC];
	q->pity[s] = q->pity[sBC];
	q->pitn[s] = q->pitn[sBC];
	q->pixx[s] = q->pixx[sBC];
	q->pixy[s] = q->pixy[sBC];
	q->pixn[s] = q->pixn[sBC];
	q->piyy[s] = q->piyy[sBC];
	q->piyn[s] = q->piyn[sBC];
	q->pinn[s] = q->pinn[sBC];
#endif
	// set W^{\mu}_{\perp z} ghost cells if evolved
#ifdef W_TZ_MU
	q->WtTz[s] = q->WtTz[sBC];
	q->WxTz[s] = q->WxTz[sBC];
	q->WyTz[s] = q->WyTz[sBC];
	q->WnTz[s] = q->WnTz[sBC];
#endif
	// set \Pi ghost cells if evolved
#ifdef PI
	q->Pi[s] = q->Pi[sBC];
#endif
}

void setGhostCellsKernelI(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u, void * latticeParams
) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;

	int nx,ncx,ncy,ncz;
	nx = lattice->numLatticePointsX;
	ncx = lattice->numComputationalLatticePointsX;
	ncy = lattice->numComputationalLatticePointsY;
	ncz = lattice->numComputationalLatticePointsRapidity;

	int iBC,s,sBC;
	for(int j = 2; j < ncy; ++j) {
		for(int k = 2; k < ncz; ++k) {
			iBC = 2;
			for (int i = 0; i <= 1; ++i) {
				s = columnMajorLinearIndex(i, j, k, ncx, ncy);
				sBC = columnMajorLinearIndex(iBC, j, k, ncx, ncy);
				setGhostCellVars(q,e,p,u,s,sBC);
			}
			iBC = nx + 1;
			for (int i = nx + 2; i <= nx + 3; ++i) {
				s = columnMajorLinearIndex(i, j, k, ncx, ncy);
				sBC = columnMajorLinearIndex(iBC, j, k, ncx, ncy);
				setGhostCellVars(q,e,p,u,s,sBC);
			}
		}
	}
}

void setGhostCellsKernelJ(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u, void * latticeParams
) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;

	int ny,ncx,ncy,ncz;
	ny = lattice->numLatticePointsY;
	ncx = lattice->numComputationalLatticePointsX;
	ncy = lattice->numComputationalLatticePointsY;
	ncz = lattice->numComputationalLatticePointsRapidity;

	int jBC,s,sBC;
	for(int i = 2; i < ncx; ++i) {
		for(int k = 2; k < ncz; ++k) {
			jBC = 2;
			for (int j = 0; j <= 1; ++j) {
				s = columnMajorLinearIndex(i, j, k, ncx, ncy);
				sBC = columnMajorLinearIndex(i, jBC, k, ncx, ncy);
				setGhostCellVars(q,e,p,u,s,sBC);
			}
			jBC = ny + 1;
			for (int j = ny + 2; j <= ny + 3; ++j) {
				s = columnMajorLinearIndex(i, j, k, ncx, ncy);
				sBC = columnMajorLinearIndex(i, jBC, k, ncx, ncy);
				setGhostCellVars(q,e,p,u,s,sBC);
			}
		}
	}
}

void setGhostCellsKernelK(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u, void * latticeParams
) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;

	int nz,ncx,ncy;
	nz = lattice->numLatticePointsRapidity;
	ncx = lattice->numComputationalLatticePointsX;
	ncy = lattice->numComputationalLatticePointsY;

	int kBC,s,sBC;
	for(int i = 2; i < ncx; ++i) {
		for(int j = 2; j < ncy; ++j) {
			kBC = 2;
			for (int k = 0; k <= 1; ++k) {
				s = columnMajorLinearIndex(i, j, k, ncx, ncy);
				sBC = columnMajorLinearIndex(i, j, kBC, ncx, ncy);
				setGhostCellVars(q,e,p,u,s,sBC);
			}
			kBC = nz + 1;
			for (int k = nz + 2; k <= nz + 3; ++k) {
				s = columnMajorLinearIndex(i, j, k, ncx, ncy);
				sBC = columnMajorLinearIndex(i, j, kBC, ncx, ncy);
				setGhostCellVars(q,e,p,u,s,sBC);
			}
		}
	}
}

void swap(CONSERVED_VARIABLES **arr1, CONSERVED_VARIABLES **arr2) {
	CONSERVED_VARIABLES *tmp = *arr1;
	*arr1 = *arr2;
	*arr2 = tmp;
}

void setCurrentConservedVariables() {
	swap(&q, &Q);
}

void swapFluidVelocity(FLUID_VELOCITY **arr1, FLUID_VELOCITY **arr2) {
	FLUID_VELOCITY *tmp = *arr1;
	*arr1 = *arr2;
	*arr2 = tmp;
}

void freeFluidVelocityHostMemory(FLUID_VELOCITY * u) {
	free(u->ut);
	free(u->ux);
	free(u->uy);
	free(u->un);
	free(u);
}

void freeConservedVariablesHostMemory(CONSERVED_VARIABLES * q) {
	free(q->ttt);
	free(q->ttx);
	free(q->tty);
	free(q->ttn);
	// free \pi^\mu\nu
#ifdef PIMUNU
	free(q->pixx);
	free(q->pixy);
	free(q->piyy);
#endif
	// free W^{\mu}_{\perp z}
#ifdef W_TZ_MU
	free(q->WxTz);
	free(q->WyTz);
#endif
	// free \Pi
#ifdef PI
	free(q->Pi);
#endif
	free(q);
}

void freeHostMemory() {
	free(e);
	free(p);
	// free fluid velocity
	freeFluidVelocityHostMemory(u);
	freeFluidVelocityHostMemory(up);
	freeFluidVelocityHostMemory(uS);
	// free conserved variables
	freeConservedVariablesHostMemory(q);
	freeConservedVariablesHostMemory(qS);
	freeConservedVariablesHostMemory(Q);
}
