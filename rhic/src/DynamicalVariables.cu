/*
 * DynamicalVariables.cu
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/DynamicalVariables.cuh"
#include "../include/LatticeParameters.h"
#include "../include/EnergyMomentumTensor.cuh"
#include "../include/CudaConfiguration.cuh"

#include "../include/EquationOfState.cuh" // TEMPORARY for sloppy implementation of xi/Lambda initial conditions

#include "../include/AnisotropicDistributionFunctions.cuh"

CONSERVED_VARIABLES *q;
CONSERVED_VARIABLES *d_q, *d_Q, *d_qS;

FLUID_VELOCITY *u;
FLUID_VELOCITY *d_u, *d_up, *d_uS;

PRECISION *e, *p;
PRECISION *d_e, *d_p;
PRECISION *d_ut, *d_ux, *d_uy, *d_un, *d_utp, *d_uxp, *d_uyp, *d_unp;
PRECISION *d_ttt, *d_ttx, *d_tty, *d_ttn, *d_pitt, *d_pitx, *d_pity, *d_pitn, *d_pixx, *d_pixy, *d_pixn, *d_piyy, *d_piyn, *d_pinn, *d_Pi;

VALIDITY_DOMAIN *validityDomain, *d_validityDomain;
PRECISION *d_regulations, *d_knudsenNumberTaupi, *d_knudsenNumberTauPi, *d_knudsenNumberTaupiT, *d_knudsenNumberTaupiL, *d_inverseReynoldsNumberPimunu, *d_inverseReynoldsNumber2Pimunu,
		*d_inverseReynoldsNumberTilde2Pimunu, *d_inverseReynoldsNumberPi, *d_inverseReynoldsNumber2Pi, *d_inverseReynoldsNumberTilde2Pi,
		d_Rpi, d_RPi, d_Rpi2, d_RPi2, d_Rw, d_fTSolution, d_fTSol_1, d_fTSol_2, d_fTSol_X1, d_fTSol_X2, d_fTSol_Y1, d_fTSol_Y2,
		d_regMag, d_regU0, d_regU1, d_regU2, d_regU3, d_regZ0, d_regZ1, d_regZ2, d_regZ3, d_regTr;
// for debugging
PRECISION *d_taupi, *d_dxux, *d_dyuy, *d_theta;

double *fTSol_X1,*fTSol_Y1,*fTSol_1,*fTSol_X2,*fTSol_Y2,*fTSol_2;

__host__ __device__
int columnMajorLinearIndex(int i, int j, int k, int nx, int ny) {
	return i + nx * (j + ny * k);
}

void allocateHostMemory(int len) {
	size_t bytes = sizeof(PRECISION);
	e = (PRECISION *) calloc(len, bytes);
	p = (PRECISION *) calloc(len, bytes);

	u = (FLUID_VELOCITY *) calloc(1, sizeof(FLUID_VELOCITY));
	u->ut = (PRECISION *) calloc(len, bytes);
	u->ux = (PRECISION *) calloc(len, bytes);
	u->uy = (PRECISION *) calloc(len, bytes);
	u->un = (PRECISION *) calloc(len, bytes);

	q = (CONSERVED_VARIABLES *) calloc(1, sizeof(CONSERVED_VARIABLES));
	q->ttt = (PRECISION *) calloc(len, bytes);
	q->ttx = (PRECISION *) calloc(len, bytes);
	q->tty = (PRECISION *) calloc(len, bytes);
	q->ttn = (PRECISION *) calloc(len, bytes);
	// allocate space for \pi^\mu\nu
	#ifdef PIMUNU
	q->pitt = (PRECISION *) calloc(len, bytes);
	q->pitx = (PRECISION *) calloc(len, bytes);
	q->pity = (PRECISION *) calloc(len, bytes);
	q->pitn = (PRECISION *) calloc(len, bytes);
	q->pixx = (PRECISION *) calloc(len, bytes);
	q->pixy = (PRECISION *) calloc(len, bytes);
	q->pixn = (PRECISION *) calloc(len, bytes);
	q->piyy = (PRECISION *) calloc(len, bytes);
	q->piyn = (PRECISION *) calloc(len, bytes);
	q->pinn = (PRECISION *) calloc(len, bytes);
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
	q->Pi = (PRECISION *) calloc(len, bytes);
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

void allocateIntermidateFluidVelocityDeviceMemory(FLUID_VELOCITY *d_u, size_t size2) {
	PRECISION *d_ut, *d_ux, *d_uy, *d_un;
	cudaMalloc((void **) &d_ut, size2);
	cudaMalloc((void **) &d_ux, size2);
	cudaMalloc((void **) &d_uy, size2);
	cudaMalloc((void **) &d_un, size2);

	cudaMemcpy(&(d_u->ut), &d_ut, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_u->ux), &d_ux, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_u->uy), &d_uy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_u->un), &d_un, sizeof(PRECISION*), cudaMemcpyHostToDevice);
}

void allocateIntermidateConservedVarDeviceMemory(CONSERVED_VARIABLES *d_q, size_t bytes) {
	//=======================================================
	// Conserved variables
	//=======================================================
	PRECISION *d_ttt, *d_ttx, *d_tty, *d_ttn, *d_pitt, *d_pitx, *d_pity, *d_pitn, *d_pixx, *d_pixy, *d_pixn, *d_piyy, *d_piyn, *d_pinn, *d_Pi;
	cudaMalloc((void **) &d_ttt, bytes);
	cudaMalloc((void **) &d_ttx, bytes);
	cudaMalloc((void **) &d_tty, bytes);
	cudaMalloc((void **) &d_ttn, bytes);

	cudaMemcpy(&(d_q->ttt), &d_ttt, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->ttx), &d_ttx, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->tty), &d_tty, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->ttn), &d_ttn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	// allocate space for \pi^\mu\nu
#ifdef PIMUNU
	cudaMalloc((void **) &d_pitt, bytes);
	cudaMalloc((void **) &d_pitx, bytes);
	cudaMalloc((void **) &d_pity, bytes);
	cudaMalloc((void **) &d_pitn, bytes);
	cudaMalloc((void **) &d_pixx, bytes);
	cudaMalloc((void **) &d_pixy, bytes);
	cudaMalloc((void **) &d_pixn, bytes);
	cudaMalloc((void **) &d_piyy, bytes);
	cudaMalloc((void **) &d_piyn, bytes);
	cudaMalloc((void **) &d_pinn, bytes);

	cudaMemcpy(&(d_q->pitt), &d_pitt, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pitx), &d_pitx, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pity), &d_pity, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pitn), &d_pitn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pixx), &d_pixx, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pixy), &d_pixy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pixn), &d_pixn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->piyy), &d_piyy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->piyn), &d_piyn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pinn), &d_pinn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
#endif
// allocate space for W_Tz
#ifdef W_TZ_MU
cudaMalloc((void **) &d_WtTz, bytes);
cudaMalloc((void **) &d_WxTz, bytes);
cudaMalloc((void **) &d_WyTz, bytes);
cudaMalloc((void **) &d_WnTz, bytes);

cudaMemcpy(&(d_q->WtTz), &d_WtTz, sizeof(PRECISION*), cudaMemcpyHostToDevice);
cudaMemcpy(&(d_q->WxTz), &d_WxTz, sizeof(PRECISION*), cudaMemcpyHostToDevice);
cudaMemcpy(&(d_q->WyTz), &d_WyTz, sizeof(PRECISION*), cudaMemcpyHostToDevice);
cudaMemcpy(&(d_q->WnTz), &d_WnTz, sizeof(PRECISION*), cudaMemcpyHostToDevice);

#endif
	// allocate space for \Pi
#ifdef PI
	cudaMalloc((void **) &d_Pi, bytes);

	cudaMemcpy(&(d_q->Pi), &d_Pi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
#endif
}

void allocateDeviceMemory(size_t bytes) {
	//=======================================================
	// Energy density and pressure
	//=======================================================
	cudaMalloc((void **) &d_e, bytes);
	cudaMalloc((void **) &d_p, bytes);

	//=======================================================
	// Fluid velocity
	//=======================================================
	cudaMalloc((void **) &d_ut, bytes);
	cudaMalloc((void **) &d_ux, bytes);
	cudaMalloc((void **) &d_uy, bytes);
	cudaMalloc((void **) &d_un, bytes);
	cudaMalloc((void **) &d_utp, bytes);
	cudaMalloc((void **) &d_uxp, bytes);
	cudaMalloc((void **) &d_uyp, bytes);
	cudaMalloc((void **) &d_unp, bytes);

	cudaMalloc((void**) &d_u, sizeof(FLUID_VELOCITY));
	cudaMalloc((void**) &d_up, sizeof(FLUID_VELOCITY));
	cudaMalloc((void**) &d_uS, sizeof(FLUID_VELOCITY));

	cudaMemcpy(&(d_u->ut), &d_ut, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_u->ux), &d_ux, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_u->uy), &d_uy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_u->un), &d_un, sizeof(PRECISION*), cudaMemcpyHostToDevice);

	cudaMemcpy(&(d_up->ut), &d_utp, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_up->ux), &d_uxp, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_up->uy), &d_uyp, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_up->un), &d_unp, sizeof(PRECISION*), cudaMemcpyHostToDevice);

	//=======================================================
	// Conserved variables
	//=======================================================
	cudaMalloc((void **) &d_ttt, bytes);
	cudaMalloc((void **) &d_ttx, bytes);
	cudaMalloc((void **) &d_tty, bytes);
	cudaMalloc((void **) &d_ttn, bytes);

	cudaMalloc((void**) &d_q, sizeof(CONSERVED_VARIABLES));
	cudaMalloc((void**) &d_Q, sizeof(CONSERVED_VARIABLES));
	cudaMalloc((void**) &d_qS, sizeof(CONSERVED_VARIABLES));

	cudaMemcpy(&(d_q->ttt), &d_ttt, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->ttx), &d_ttx, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->tty), &d_tty, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->ttn), &d_ttn, sizeof(PRECISION*), cudaMemcpyHostToDevice);

	// allocate space for \pi^\mu\nu
#ifdef PIMUNU
	cudaMalloc((void **) &d_pitt, bytes);
	cudaMalloc((void **) &d_pitx, bytes);
	cudaMalloc((void **) &d_pity, bytes);
	cudaMalloc((void **) &d_pitn, bytes);
	cudaMalloc((void **) &d_pixx, bytes);
	cudaMalloc((void **) &d_pixy, bytes);
	cudaMalloc((void **) &d_pixn, bytes);
	cudaMalloc((void **) &d_piyy, bytes);
	cudaMalloc((void **) &d_piyn, bytes);
	cudaMalloc((void **) &d_pinn, bytes);

	cudaMemcpy(&(d_q->pitt), &d_pitt, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pitx), &d_pitx, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pity), &d_pity, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pitn), &d_pitn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pixx), &d_pixx, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pixy), &d_pixy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pixn), &d_pixn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->piyy), &d_piyy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->piyn), &d_piyn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pinn), &d_pinn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
#endif
// allocate space for W_Tz
#ifdef W_TZ_MU
	cudaMalloc((void **) &d_WtTz, bytes);
	cudaMalloc((void **) &d_WxTz, bytes);
	cudaMalloc((void **) &d_WyTz, bytes);
	cudaMalloc((void **) &d_WnTz, bytes);

	cudaMemcpy(&(d_q->WtTz), &d_WtTz, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->WxTz), &d_WtTz, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->WyTz), &d_WtTz, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->WnTz), &d_WtTz, sizeof(PRECISION*), cudaMemcpyHostToDevice);
#endif
	// allocate space for \Pi
#ifdef PI
	cudaMalloc((void **) &d_Pi, bytes);

	cudaMemcpy(&(d_q->Pi), &d_Pi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
#endif

	//=======================================================
	// Intermediate fluid velocity/conserved variables
	//=======================================================
	allocateIntermidateConservedVarDeviceMemory(d_Q, bytes);
	allocateIntermidateConservedVarDeviceMemory(d_qS, bytes);
	allocateIntermidateFluidVelocityDeviceMemory(d_uS, bytes);

	//=======================================================
	// Hydrodynamic validity
	//=======================================================
	cudaMalloc((void **) &d_regulations, bytes);
	cudaMalloc((void **) &d_knudsenNumberTaupi, bytes);
	cudaMalloc((void **) &d_knudsenNumberTaupiT, bytes);
	cudaMalloc((void **) &d_knudsenNumberTaupiL, bytes);
	cudaMalloc((void **) &d_knudsenNumberTauPi, bytes);
	cudaMalloc((void **) &d_Rpi, bytes);
	cudaMalloc((void **) &d_RPi, bytes);
	cudaMalloc((void **) &d_Rw, bytes);
	cudaMalloc((void **) &d_Rpi2, bytes);
	cudaMalloc((void **) &d_RPi2, bytes);
	cudaMalloc((void **) &d_fTSolution, bytes);
	cudaMalloc((void **) &d_regMag, bytes);
	cudaMalloc((void **) &d_regTr, bytes);
	cudaMalloc((void **) &d_regU0, bytes);
	cudaMalloc((void **) &d_regU1, bytes);
	cudaMalloc((void **) &d_regU2, bytes);
	cudaMalloc((void **) &d_regU3, bytes);
	cudaMalloc((void **) &d_regZ0, bytes);
	cudaMalloc((void **) &d_regZ1, bytes);
	cudaMalloc((void **) &d_regZ2, bytes);
	cudaMalloc((void **) &d_regZ3, bytes);

	cudaMalloc((void **) &d_inverseReynoldsNumberPimunu, bytes);
	cudaMalloc((void **) &d_inverseReynoldsNumber2Pimunu, bytes);
	cudaMalloc((void **) &d_inverseReynoldsNumberTilde2Pimunu, bytes);
	cudaMalloc((void **) &d_inverseReynoldsNumberPi, bytes);
	cudaMalloc((void **) &d_inverseReynoldsNumber2Pi, bytes);
	cudaMalloc((void **) &d_inverseReynoldsNumberTilde2Pi, bytes);

	cudaMalloc((void **) &d_fTSol_X1, bytes);
	cudaMalloc((void **) &d_fTSol_Y1, bytes);
	cudaMalloc((void **) &d_fTSol_1, bytes);
	cudaMalloc((void **) &d_fTSol_X2, bytes);
	cudaMalloc((void **) &d_fTSol_Y2, bytes);
	cudaMalloc((void **) &d_fTSol_2, bytes);

	// for debugging purposes
	cudaMalloc((void **) &d_taupi, bytes);
	cudaMalloc((void **) &d_dxux, bytes);
	cudaMalloc((void **) &d_dyuy, bytes);
	cudaMalloc((void **) &d_theta, bytes);

	cudaMalloc((void**) &d_validityDomain, sizeof(VALIDITY_DOMAIN));

	cudaMemcpy(&(d_validityDomain->regulations), &d_regulations, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->knudsenNumberTaupi), &d_knudsenNumberTaupi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->knudsenNumberTaupiT), &d_knudsenNumberTaupiT, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->knudsenNumberTaupiL), &d_knudsenNumberTaupiL, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->knudsenNumberTauPi), &d_knudsenNumberTauPi, sizeof(PRECISION*), cudaMemcpyHostToDevice);

	cudaMemcpy(&(d_validityDomain->Rpi), &d_Rpi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->RPi), &d_RPi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->Rw), &d_Rw, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->Rpi2), &d_Rpi2, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->RPi2), &d_RPi2, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->fTSolution), &d_fTSolution, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->regMag), &d_regMag, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->regTr), &d_regTr, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->regU0), &d_regU0, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->regU1), &d_regU1, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->regU2), &d_regU2, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->regU3), &d_regU3, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->regZ0), &d_regZ0, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->regZ1), &d_regZ1, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->regZ2), &d_regZ2, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->regZ3), &d_regZ3, sizeof(PRECISION*), cudaMemcpyHostToDevice);

	//cudaMemcpy(&(d_validityDomain->inverseReynoldsNumberPimunu), &d_inverseReynoldsNumberPimunu, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(d_validityDomain->inverseReynoldsNumber2Pimunu), &d_inverseReynoldsNumber2Pimunu, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(d_validityDomain->inverseReynoldsNumberTilde2Pimunu), &d_inverseReynoldsNumberTilde2Pimunu, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(d_validityDomain->inverseReynoldsNumberPi), &d_inverseReynoldsNumberPi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(d_validityDomain->inverseReynoldsNumber2Pi), &d_inverseReynoldsNumber2Pi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(d_validityDomain->inverseReynoldsNumberTilde2Pi), &d_inverseReynoldsNumberTilde2Pi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	// for debugging
	cudaMemcpy(&(d_validityDomain->taupi), &d_taupi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->dxux), &d_dxux, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->dyuy), &d_dyuy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->theta), &d_theta, sizeof(PRECISION*), cudaMemcpyHostToDevice);

	//cudaMemcpy(&(d_validityDomain->fTSol_X1), &d_fTSol_X1, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(d_validityDomain->fTSol_Y1), &d_fTSol_Y1, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(d_validityDomain->fTSol_1), &d_fTSol_1, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(d_validityDomain->fTSol_X2), &d_fTSol_X2, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(d_validityDomain->fTSol_Y2), &d_fTSol_Y2, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(d_validityDomain->fTSol_2), &d_fTSol_2, sizeof(PRECISION*), cudaMemcpyHostToDevice);


}

void copyHostToDeviceMemory(size_t bytes) {
	cudaMemcpy(d_e, e, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_p, p, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_ut, u->ut, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ux, u->ux, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_uy, u->uy, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_un, u->un, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_utp, u->ut, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_uxp, u->ux, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_uyp, u->uy, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_unp, u->un, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_ttt, q->ttt, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ttx, q->ttx, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tty, q->tty, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ttn, q->ttn, bytes, cudaMemcpyHostToDevice);
	// copy \pi^\mu\nu to device memory
#ifdef PIMUNU
	cudaMemcpy(d_pitt, q->pitt, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pitx, q->pitx, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pity, q->pity, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pitn, q->pitn, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixx, q->pixx, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixy, q->pixy, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixn, q->pixn, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_piyy, q->piyy, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_piyn, q->piyn, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pinn, q->pinn, bytes, cudaMemcpyHostToDevice);
#endif
// copy \pi^\mu\nu to device memory
#ifdef W_TZ_MU
cudaMemcpy(d_WtTz, q->WtTz, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_WxTz, q->WxTz, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_WyTz, q->WyTz, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_WnTz, q->WnTz, bytes, cudaMemcpyHostToDevice);
#endif
	// copy \Pi to device memory
#ifdef PI
	cudaMemcpy(d_Pi, q->Pi, bytes, cudaMemcpyHostToDevice);
#endif
	cudaMemcpy(d_regulations, validityDomain->regulations, bytes, cudaMemcpyHostToDevice);
}

void copyDeviceToHostMemory(size_t bytes) {
	cudaMemcpy(e, d_e, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(p, d_p, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(u->ut, d_ut, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(u->ux, d_ux, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(u->uy, d_uy, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(u->un, d_un, bytes, cudaMemcpyDeviceToHost);
#ifdef PIMUNU
	cudaMemcpy(q->pitt, d_pitt, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pitx, d_pitx, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pity, d_pity, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pitn, d_pitn, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pixx, d_pixx, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pixy, d_pixy, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pixn, d_pixn, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->piyy, d_piyy, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->piyn, d_piyn, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pinn, d_pinn, bytes, cudaMemcpyDeviceToHost);
#endif
#ifdef W_TZ_MU
	cudaMemcpy(q->WtTz, d_WtTz, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->WxTz, d_WxTz, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->WyTz, d_WyTz, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->WnTz, d_WnTz, bytes, cudaMemcpyDeviceToHost);
#endif
#ifdef PI
	cudaMemcpy(q->Pi, d_Pi, bytes, cudaMemcpyDeviceToHost);
#endif

	cudaMemcpy(validityDomain->regulations, d_regulations, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->knudsenNumberTaupi, d_knudsenNumberTaupi, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->knudsenNumberTauPi, d_knudsenNumberTauPi, bytes, cudaMemcpyDeviceToHost);
	//cudaMemcpy(validityDomain->inverseReynoldsNumberPimunu, d_inverseReynoldsNumberPimunu, bytes, cudaMemcpyDeviceToHost);
	//cudaMemcpy(validityDomain->inverseReynoldsNumber2Pimunu, d_inverseReynoldsNumber2Pimunu, bytes, cudaMemcpyDeviceToHost);
	//cudaMemcpy(validityDomain->inverseReynoldsNumberTilde2Pimunu, d_inverseReynoldsNumberTilde2Pimunu, bytes, cudaMemcpyDeviceToHost);
	//cudaMemcpy(validityDomain->inverseReynoldsNumberPi, d_inverseReynoldsNumberPi, bytes, cudaMemcpyDeviceToHost);
	//cudaMemcpy(validityDomain->inverseReynoldsNumber2Pi, d_inverseReynoldsNumber2Pi, bytes, cudaMemcpyDeviceToHost);
	//cudaMemcpy(validityDomain->inverseReynoldsNumberTilde2Pi, d_inverseReynoldsNumberTilde2Pi, bytes, cudaMemcpyDeviceToHost);
	// for debugging
	cudaMemcpy(validityDomain->taupi, d_taupi, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->dxux, d_dxux, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->dyuy, d_dyuy, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->theta, d_theta, bytes, cudaMemcpyDeviceToHost);
}

void setConservedVariables(double t, void * latticeParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;

	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;
	int ncx = lattice->numComputationalLatticePointsX;
	int ncy = lattice->numComputationalLatticePointsY;

	for (int k = N_GHOST_CELLS_M; k < nz + N_GHOST_CELLS_M; ++k) {
		for (int j = N_GHOST_CELLS_M; j < ny + N_GHOST_CELLS_M; ++j) {
			for (int i = N_GHOST_CELLS_M; i < nx + N_GHOST_CELLS_M; ++i) {
				int s = columnMajorLinearIndex(i, j, k, ncx, ncy);

				PRECISION ux_s = u->ux[s];
				PRECISION uy_s = u->uy[s];
				PRECISION un_s = u->un[s];
				PRECISION ut_s = u->ut[s];
				PRECISION e_s = e[s];
				PRECISION p_s = p[s];

				PRECISION pitt_s = 0;
				PRECISION pitx_s = 0;
				PRECISION pity_s = 0;
				PRECISION pitn_s = 0;

				PRECISION WtTz_s = 0;
				PRECISION WxTz_s = 0;
				PRECISION WyTz_s = 0;
				PRECISION WnTz_s = 0;

#ifdef PIMUNU
				pitt_s = q->pitt[s];
				pitx_s = q->pitx[s];
				pity_s = q->pity[s];
				pitn_s = q->pitn[s];
#endif

#ifdef W_TZ_MU
				WtTz_s = q->WtTz[s];
				WxTz_s = q->WxTz[s];
				WyTz_s = q->WyTz[s];
				WnTz_s = q->WnTz[s];
#endif
				PRECISION Pi_s = 0;
#ifdef PI
				Pi_s = q->Pi[s];
#endif

				double R220_100 = 0.000686059;
				double R220_10 = 0.0154483;
				double R220_0 = 1./3.;
				PRECISION pl = R220_10 * e_s;

				PRECISION ptHat = transversePressureHat(e_s, p_s, pl);
				ptHat = 0.5 * (e_s - pl);
				PRECISION pt = ptHat + 1.5 * Pi_s;
				PRECISION DP = pl - pt;

				PRECISION uT2 = ux_s * ux_s + uy_s * uy_s;
				PRECISION uT = sqrt(uT2);
				PRECISION F = 1.0 + uT2;
				PRECISION FS = sqrt(1.0 + uT2);

				double z0 = t * un_s / FS;
				double z3 = ut_s / t / FS;
				// L functions
				PRECISION Ltt = DP * t * t * un_s * un_s / F;
				PRECISION Ltx = 0.0;
				PRECISION Lty = 0.0;
				PRECISION Ltn = DP * ut_s * un_s / F;
				// W functions
				double Wtt = 2 * WtTz_s * z0;
				double Wtx = WxTz_s * z0;
				double Wty = WyTz_s * z0;
				double Wtn = WtTz_s * z3 + WnTz_s * z0;

				/*
				q->ttt[s] = Ttt(e_s, p_s + Pi_s, ut_s, pitt_s);
				q->ttx[s] = Ttx(e_s, p_s + Pi_s, ut_s, ux_s, pitx_s);
				q->tty[s] = Tty(e_s, p_s + Pi_s, ut_s, uy_s, pity_s);
				q->ttn[s] = Ttn(e_s, p_s + Pi_s, ut_s, un_s, pitn_s);
				*/

				q->ttt[s] = (e_s + pt) * ut_s * ut_s - pt + Ltt + Wtt + pitt_s;
				q->ttx[s] = (e_s + pt) * ut_s * ux_s + Ltx + Wtx + pitx_s;
				q->tty[s] = (e_s + pt) * ut_s * uy_s + Lty + Wty + pity_s;
				q->ttn[s] = (e_s + pt) * ut_s * un_s + Ltn + Wtn + pitn_s;
				q->pl[s] = pl;

				// set up to u
				//there is no host variable up, is this accomplished somewhere else?
				/*
				up->ut[s] = ut_s;
				up->ux[s] = ux_s;
				up->uy[s] = uy_s;
				up->un[s] = un_s;
				*/
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
	swap(&d_q, &d_Q);
}

void swapFluidVelocity(FLUID_VELOCITY **arr1, FLUID_VELOCITY **arr2) {
	FLUID_VELOCITY *tmp = *arr1;
	*arr1 = *arr2;
	*arr2 = tmp;
}

void freeHostMemory() {
	free(e);
	free(p);
	free(u->ut);
	free(u->ux);
	free(u->uy);
	free(u->un);
	free(u);

	free(q->ttt);
	free(q->ttx);
	free(q->tty);
	free(q->ttn);
	// free \pi^\mu\nu
#ifdef PIMUNU
	free(q->pitt);
	free(q->pitx);
	free(q->pity);
	free(q->pitn);
	free(q->pixx);
	free(q->pixy);
	free(q->pixn);
	free(q->piyy);
	free(q->piyn);
	free(q->pinn);
#endif
#ifdef W_TZ_MU
	free(q->WtTz);
	free(q->WxTz);
	free(q->WyTz);
	free(q->WnTz);
#endif
	// free \Pi
#ifdef PI
	free(q->Pi);
#endif
	free(q);
}

void freeDeviceMemory() {
	cudaFree(d_e);
	cudaFree(d_p);

	cudaFree(d_ut);
	cudaFree(d_ux);
	cudaFree(d_uy);
	cudaFree(d_un);
	cudaFree(d_utp);
	cudaFree(d_uxp);
	cudaFree(d_uyp);
	cudaFree(d_unp);

	cudaFree(d_u);
	cudaFree(d_up);
	cudaFree(d_uS);

	cudaFree(d_ttt);
	cudaFree(d_ttx);
	cudaFree(d_tty);
	cudaFree(d_ttn);
	// free \pi^\mu\nu
#ifdef PIMUNU
	cudaFree(d_pitt);
	cudaFree(d_pitx);
	cudaFree(d_pity);
	cudaFree(d_pitn);
	cudaFree(d_pixx);
	cudaFree(d_pixy);
	cudaFree(d_pixn);
	cudaFree(d_piyy);
	cudaFree(d_piyn);
	cudaFree(d_pinn);
#endif
#ifdef W_TZ_MU
	cudaFree(d_WtTz);
	cudaFree(d_WxTz);
	cudaFree(d_WyTz);
	cudaFree(d_WnTz);
#endif
	// free \Pi
#ifdef PI
	cudaFree(d_Pi);
#endif

	cudaFree(d_q);
	cudaFree(d_Q);
	cudaFree(d_qS);
}
