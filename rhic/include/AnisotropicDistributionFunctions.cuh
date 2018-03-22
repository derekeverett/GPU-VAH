#ifndef ANISOTROPICDISTRIBUTIONFUNCTIONS_H_
#define ANISOTROPICDISTRIBUTIONFUNCTIONS_H_

#include "DynamicalVariables.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

__host__ __device__
PRECISION transversePressureHat(double e, double p, double pl);

__host__ __device__
PRECISION Rbar0_fun(double a);

__host__ __device__
PRECISION Rbar0P_fun(double a);

__host__ __device__
void secondOrderTransportCoefficientsZ(double e, double p, double pl, double cs2, double T,
double *beta_lPi, double *delta_lPi, double *lambda_piPi, double *beta_PiPi, double *delta_PiPi, double *lambda_Pipi);

#endif /* ANISOTROPICDISTRIBUTIONFUNCTIONS_H_ */
