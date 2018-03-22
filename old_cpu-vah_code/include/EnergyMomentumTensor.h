/*
 * EnergyMomentumTensor.h
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef ENERGYMOMENTUMTENSOR_H_
#define ENERGYMOMENTUMTENSOR_H_

#include "DynamicalVariables.h"

PRECISION getTransverseFluidVelocityMagnitude(const FLUID_VELOCITY * const __restrict__ u, int s);

int getInferredVariables(PRECISION t, const PRECISION * const __restrict__ q, PRECISION ePrev, PRECISION uT_0,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
PRECISION * const __restrict__ ut, PRECISION * const __restrict__ ux, PRECISION * const __restrict__ uy, PRECISION * const __restrict__ un,
int i, int j, int k, double xi, double yj, double zk,
int fullTimeStepInversion
);

void setInferredVariablesKernel(const CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ up, FLUID_VELOCITY * const __restrict__ u,
PRECISION t, void * latticeParams, PRECISION *fTSolution
);

PRECISION Ttt(PRECISION e, PRECISION p, PRECISION ut, PRECISION pitt);
PRECISION Ttx(PRECISION e, PRECISION p, PRECISION ut, PRECISION ux, PRECISION pitx);
PRECISION Tty(PRECISION e, PRECISION p, PRECISION ut, PRECISION uy, PRECISION pity);
PRECISION Ttn(PRECISION e, PRECISION p, PRECISION ut, PRECISION un, PRECISION pitn);

#endif /* ENERGYMOMENTUMTENSOR_H_ */
