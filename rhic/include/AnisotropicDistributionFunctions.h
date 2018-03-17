#ifndef ANISOTROPICDISTRIBUTIONFUNCTIONS_H_
#define ANISOTROPICDISTRIBUTIONFUNCTIONS_H_

#include "DynamicalVariables.h"

double transversePressureHat(double e, double p, double pl);
double Rbar0_fun(double a);
double Rbar0P_fun(double a);

void secondOrderTransportCoefficientsZ(double e, double p, double pl, double cs2, double T,
double *beta_lPi, double *delta_lPi, double *lambda_piPi, double *beta_PiPi, double *delta_PiPi, double *lambda_Pipi);

#endif /* ANISOTROPICDISTRIBUTIONFUNCTIONS_H_ */
