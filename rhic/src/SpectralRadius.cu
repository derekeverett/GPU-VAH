/*
 * SpectralRadius.cu
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/SpectralRadius.cuh"
#include "../include/EnergyMomentumTensor.cuh"
#include "../include/DynamicalVariables.cuh"

__device__
PRECISION spectralRadiusX(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un) {
	return fabsf(ux/ut);
}

__device__
PRECISION spectralRadiusY(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un) {
	return fabsf(uy/ut);
}

__device__
PRECISION spectralRadiusZ(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un) {
	return fabsf(un/ut);
}
