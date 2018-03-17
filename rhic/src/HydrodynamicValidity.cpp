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

#include "../include/HydrodynamicValidity.h"

// paramters for the analytic parameterization of the bulk viscosity \zeta/S
#define A_1 -13.77
#define A_2 27.55
#define A_3 13.45

#define LAMBDA_1 0.9
#define LAMBDA_2 0.25
#define LAMBDA_3 0.9
#define LAMBDA_4 0.22

#define SIGMA_1 0.025
#define SIGMA_2 0.13
#define SIGMA_3 0.0025
#define SIGMA_4 0.022

inline PRECISION bulkViscosityToEntropyDensity(PRECISION T) {
	PRECISION x = T/1.01355;
	if(x > 1.05)
		return LAMBDA_1*exp(-(x-1)/SIGMA_1) + LAMBDA_2*exp(-(x-1)/SIGMA_2)+0.001;
	else if(x < 0.995)
		return LAMBDA_3*exp((x-1)/SIGMA_3)+ LAMBDA_4*exp((x-1)/SIGMA_4)+0.03;
	else
		return A_1*x*x + A_2*x - A_3;
}

void checkValidityKernel(PRECISION t, const VALIDITY_DOMAIN * const __restrict__ v, const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p, const FLUID_VELOCITY * const __restrict__ u,
		const FLUID_VELOCITY * const __restrict__ up,
		int s, int d_ncx, int d_ncy, int d_ncz, PRECISION d_etabar, PRECISION d_dt, PRECISION d_dx, PRECISION d_dy, PRECISION d_dz
) {
		PRECISION e_s = e[s];
		PRECISION p_s = p[s];

		PRECISION *utvec = u->ut;
		PRECISION *uxvec = u->ux;
		PRECISION *uyvec = u->uy;
		PRECISION *unvec = u->un;

		PRECISION ut = utvec[s];
		PRECISION ux = uxvec[s];
		PRECISION uy = uyvec[s];
		PRECISION un = unvec[s];

		PRECISION utp = up->ut[s];
		PRECISION uxp = up->ux[s];
		PRECISION uyp = up->uy[s];
		PRECISION unp = up->un[s];

		//=========================================================
		// spatial derivatives of primary variables
		//=========================================================
		PRECISION facX = 1 / d_dx / 2;
		PRECISION facY = 1 / d_dy / 2;
		PRECISION facZ = 1 / d_dz / 2;
		// dx of u^{\mu} components
		PRECISION dxut = (*(utvec + s + 1) - *(utvec + s - 1)) * facX;
		PRECISION dxux = (*(uxvec + s + 1) - *(uxvec + s - 1)) * facX;
		PRECISION dxuy = (*(uyvec + s + 1) - *(uyvec + s - 1)) * facX;
		PRECISION dxun = (*(unvec + s + 1) - *(unvec + s - 1)) * facX;
		// dy of u^{\mu} components
		PRECISION dyut = (*(utvec + s + d_ncx) - *(utvec + s - d_ncx)) * facY;
		PRECISION dyux = (*(uxvec + s + d_ncx) - *(uxvec + s - d_ncx)) * facY;
		PRECISION dyuy = (*(uyvec + s + d_ncx) - *(uyvec + s - d_ncx)) * facY;
		PRECISION dyun = (*(unvec + s + d_ncx) - *(unvec + s - d_ncx)) * facY;
		// dn of u^{\mu} components
		int stride = d_ncx * d_ncy;
		PRECISION dnut = (*(utvec + s + stride) - *(utvec + s - stride)) * facZ;
		PRECISION dnux = (*(uxvec + s + stride) - *(uxvec + s - stride)) * facZ;
		PRECISION dnuy = (*(uyvec + s + stride) - *(uyvec + s - stride)) * facZ;
		PRECISION dnun = (*(unvec + s + stride) - *(unvec + s - stride)) * facZ;

		// time derivatives of u
		PRECISION dtut = (ut - utp) / d_dt;
		PRECISION dtux = (ux - uxp) / d_dt;
		PRECISION dtuy = (uy - uyp) / d_dt;
		PRECISION dtun = (un - unp) / d_dt;

//		PRECISION zDzu = dnun+ut/t+dtut;
//		PRECISION thetaT = dxux+dyuy;

	/*********************************************************\
	 * shear tensor
	/*********************************************************/
	PRECISION ut2 = ut * ut;
	PRECISION un2 = un * un;
	PRECISION t2 = t * t;
	PRECISION t3 = t*t2;

	PRECISION Dut = ut*dtut + ux*dxut + uy*dyut + un*dnut + t*un*un;
	PRECISION DuxUpper = ut*dtux + ux*dxux + uy*dyux + un*dnux;
	PRECISION Dux = -DuxUpper;
	PRECISION DuyUpper = ut*dtuy + ux*dxuy + uy*dyuy + un*dnuy;
	PRECISION Duy = -DuyUpper;
	PRECISION DunUpper = ut*dtun + ux*dxun + uy*dyun + un*dnun + 2*ut*un/t;
	PRECISION Dun = -t2*DunUpper;

	PRECISION dut = Dut -t*un*un;
	PRECISION dux = ut*dtux + ux*dxux + uy*dyux + un*dnux;
	PRECISION duy = ut*dtuy + ux*dxuy + uy*dyuy + un*dnuy;
	PRECISION dun = ut*dtun + ux*dxun + uy*dyun + un*dnun;

	PRECISION theta = ut / t + dtut + dxux + dyuy + dnun;

	PRECISION stt = -t * ut * un2 + (dtut - ut * dut) + (ut2 - 1) * theta / 3;
	PRECISION stx = -(t * un2 * ux) / 2 + (dtux - dxut) / 2 - (ux * dut + ut * dux) / 2 + ut * ux * theta / 3;
	PRECISION sty = -(t * un2 * uy) / 2 + (dtuy - dyut) / 2 - (uy * dut + ut * duy) / 2 + ut * uy * theta / 3;
	PRECISION stn = -un * (2 * ut2 + t2 * un2) / (2 * t) + (dtun - dnut / t2) / 2 - (un * dut + ut * dun) / 2 + ut * un * theta / 3;
	PRECISION sxx = -(dxux + ux * dux) + (1 + ux*ux) * theta / 3;
	PRECISION sxy = -(dxuy + dyux) / 2 - (uy * dux + ux * duy) / 2	+ ux * uy * theta / 3;
	PRECISION sxn = -ut * ux * un / t - (dxun + dnux / t2) / 2 - (un * dux + ux * dun) / 2 + ux * un * theta / 3;
	PRECISION syy = -(dyuy + uy * duy) + (1 + uy*uy) * theta / 3;
	PRECISION syn = -ut * uy * un / t - (dyun + dnuy / t2) / 2 - (un * duy + uy * dun) / 2 + uy * un * theta / 3;
	PRECISION snn = -ut * (1 + 2 * t2 * un2) / t3 - dnun / t2 - un * dun + (1 / t2 + un2) * theta / 3;

	// vorticity tensor
	PRECISION wtx = (dtux + dxut) / 2 + (ux * dut - ut * dux) / 2 + t * un2 * ux / 2;
	PRECISION wty = (dtuy + dyut) / 2 + (uy * dut - ut * duy) / 2 + t * un2 * uy / 2;
	PRECISION wtn = (t2 * dtun + 2 * t * un + dnut) / 2 + (t2 * un * dut - ut * Dun) + t3 * un*un2 / 2;
	PRECISION wxy = (dyux - dxuy) / 2 + (uy * dux - ux * duy) / 2;
	PRECISION wxn = (dnux - t2 * dxun) / 2 + (t2 * un * dux - ux * Dun) / 2;
	PRECISION wyn = (dnuy - t2 * dyun) / 2 + (t2 * un * duy - uy * Dun) / 2;
	// anti-symmetric vorticity components
	PRECISION wxt = wtx;
	PRECISION wyt = wty;
	PRECISION wnt = wtn / t2;
	PRECISION wyx = -wxy;
	PRECISION wnx = -wxn / t2;
	PRECISION wny = -wyn / t2;

	PRECISION uT2 = ux*ux+uy*uy;
	PRECISION uT = sqrt(uT2);

	PRECISION F = 1+uT2;
	PRECISION F2 = F*F;
	PRECISION FS = sqrt(1+uT2);

	double z0 = t*un/sqrt(F);
	double z3 = ut/t/sqrt(F);
	double z02 = z0*z0;
	double z0z3 = z0*z3;
	double z32 = z3*z3;

	double A = z02*stt-2*t2*z0z3*stn+t2*t2*z32*snn;
	// B
	double B0 = z0*stt-t2*z3*stn;
	double B1 = z0*stx-t2*z3*sxn;
	double B2 = z0*sty-t2*z3*syn;
	double B3 = z0*stn-t2*z3*snn;
	// Bw
	double Bw0 = z3*wtn;
	double Bw1 = z0*wtx+z3*wxn;
	double Bw2 = z0*wty+z3*wyn;
	double Bw3 = z0*wtn/t2;

	// transverse and longitudinal expansion scalars
	PRECISION zDzu = theta/3-z02*stt+2*z0z3*t2*stn-t2*t2*z32*snn;
	PRECISION thetaT = theta-zDzu;
	thetaT = 2*theta/3+A;
	zDzu = theta/3-A;
	zDzu = theta-thetaT;

	// transverse shear tensor
	double sttT = stt+2*z0*B0+z02*A-0.5*(1-ut*ut+z02)*A;
	double stxT = stx+z0*B1-0.5*(-ut*ux)*A;
	double styT = sty+z0*B2-0.5*(-ut*uy)*A;
	double stnT = stn+z0*B3+z3*B0+z0z3*A-0.5*(-ut*un+z0z3)*A;
	double sxxT = sxx-0.5*(-1-ux*ux)*A;
	double sxyT = sxy-0.5*(-ux*uy)*A;
	double sxnT = sxn+z3*B1-0.5*(-ux*un)*A;
	double syyT = syy-0.5*(-1-uy*uy)*A;
	double synT = syn+z3*B2-0.5*(-uy*un)*A;
	double snnT = snn+2*z3*B3+z32*A-0.5*(-1/t2-un*un+z32)*A;

	v->stt[s] = sttT;
	v->sxx[s] = sxxT;
	v->syy[s] = syyT;
	v->snn[s] = snnT;

	v->stt[s] = stt-sxx-syy-t2*snn;
	v->sxx[s] = sttT-sxxT-syyT-t2*snnT;
	v->syy[s] = z02*stt-2*t2*z0z3*stn+t2*t2*z32*snn;
	v->snn[s] = snnT;

#ifdef PI
	PRECISION Pi = currrentVars->Pi[s];
#else
	PRECISION Pi = 0;
#endif
	PRECISION cs2 = speedOfSoundSquared(e_s);

	/*********************************************************\
	 *Validity
	/*********************************************************/
		PRECISION eeq_T = e_s;
		PRECISION peq_T = equilibriumPressure(eeq_T);
		PRECISION T = effectiveTemperature(eeq_T);
		PRECISION taupiInv = 0.2 * T/d_etabar;

		v->knudsenNumberTaupiT[s] = 5 * d_etabar * fabs(thetaT) / T;
		v->knudsenNumberTaupiL[s] = 5 * d_etabar * fabs(zDzu) / T;
		v->knudsenNumberTaupi[s] = 5 * d_etabar * fabs(theta) / T;
#ifdef PIMUNU
		// inverse reynolds number
		PRECISION pl = currrentVars->pl[s];

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

		PRECISION pipi = pitt * pitt - 2 * pitx * pitx - 2 * pity * pity + pixx * pixx + 2 * pixy * pixy + piyy * piyy - 2 * pitn * pitn * t2
				+ 2 * pixn * pixn * t2 + 2 * piyn * piyn * t2 + pinn * pinn * t2 * t2;

		double ptHat = transversePressureHat(e_s, p_s, pl);

		v->Rpi[s] = sqrt(fabs(pipi))/fabs(ptHat);
#endif
#ifdef PI
		// Kn for Pi
		PRECISION b = 0.333333 - cs2;
		PRECISION b2 = b * b;
		PRECISION zetabar = bulkViscosityToEntropyDensity(T);
		PRECISION tauPiInv = 15 * b2 * T/zetabar;
		v->knudsenNumberTauPi[s] = fabs(thetaT+zDzu) / tauPiInv;
		// R^-1 for Pi
//		v->RPi[s] = fabs(Pi)/p[s];
		v->RPi[s] = fabs(Pi)/fabs(ptHat);
#endif
#ifdef W_TZ_MU
		PRECISION WtTz = currrentVars->WtTz[s];
		PRECISION WxTz = currrentVars->WxTz[s];
		PRECISION WyTz = currrentVars->WyTz[s];
		PRECISION WnTz = currrentVars->WnTz[s];
		v->Rw[s] = fabs(WtTz*WtTz-WxTz*WxTz-WyTz*WyTz-t2*WnTz*WnTz)/fabs(ptHat);
#endif

	/*********************************************************\
	 * FOR DEBUGGING PURPOSES
	/*********************************************************/
	v->taupi[s] = 5*d_etabar/T;
	v->dxux[s] = dxux;
	v->dyuy[s] = dyuy;
	v->theta[s] = thetaT+zDzu;

	/*********************************************************************************\
	 * Second order inverse Reynolds numbers
	/*********************************************************************************/
#ifdef PIMUNU
	double a = pl/e_s;
	double a2 = a*a;
	double a3 = a2*a;
	double a4 = a3*a;
	double a5 = a4*a;
	double a6 = a5*a;
	double a7 = a6*a;
	double a8 = a7*a;
	double a9 = a8*a;
	double a10 = a9*a;
	double a11 = a10*a;
	double a12 = a11*a;
	double a13 = a12*a;
	double a14 = a13*a;
	double a15 = a14*a;

	PRECISION Rtilde = (-6.674731906076046e-6 + 0.004617789933500251*a + 0.7207562721999754*a2 + 9.097427250602184*a3 - 4.475814747302824*a4 - 36.37501529319408*a5 +
     46.868405146729316*a6 - 15.833867583743228*a7)/
   (0.06856675185266 + 2.9181587012768597*a + 11.951184087839218*a2 - 29.708257843442173*a3 - 2.618233802059826*a4 + 34.646239784689065*a5 -
     19.62596366454439*a6 + 2.374808442453899*a7);

	PRECISION Rhat = (0.0024792827625583747 + 1.943027171680747*a + 53.46970495217282*a2 + 19.989171951866325*a3 - 347.1285593126723*a4 + 412.2647882672885*a5 -
     140.53693383827797*a6)/(0.5061402347582388 + 29.466067530916984*a + 126.07947638942892*a2 - 334.420268508072*a3 + 86.57706367583984*a4 +
     183.53625188578846*a5 - 91.68259808111912*a6);

	PRECISION Rbar0 = (0.015267955823446243 + 7.725572805021035*a + 421.0063884634789*a2 + 3422.877939650926*a3 - 5785.670846299543*a4 - 12261.66452089229*a5 +
     31491.409484673808*a6 - 22737.05146992673*a7 + 5441.373392185447*a8)/
   (0.05470696094814806 + 14.505878005231883*a + 522.6643024173569*a2 + 2731.7776413939037*a3 - 6161.1991042880445*a4 -
     3989.4375208972588*a5 + 15008.260526258282*a6 - 10243.036679405379*a7 + 2116.74060159494*a8);

	PRECISION Rgamma = (0.0001373796340585521 - 0.6149634004722385*a + 3.1968875683514253*a2 + 246.35973783799196*a3 - 764.319750320186*a4 +
     834.160165071214*a5 - 371.64955466673234*a6 + 52.87411921963021*a7)/
   (-1.673322188488071 + 13.343782941396997*a + 561.7566534476223*a2 - 1790.2296622275915*a3 + 1896.4688704912812*a4 -
     658.7933063369629*a5 - 85.96181900698849*a6 + 65.09739194472589*a7);

	// second-order transport coefficients
	double beta_lPi=0;
	double delta_lPi=0;
	double lambda_piPi=0;
	double beta_PiPi=0;
	double delta_PiPi=0;
	double lambda_Pipi=0;
	secondOrderTransportCoefficientsZ(e_s, p_s, pl, cs2, T, &beta_lPi, &delta_lPi, &lambda_piPi, &beta_PiPi, &delta_PiPi, &lambda_Pipi);

	double etaBar_TC = ((3-Rtilde)*e_s-2*pl)/8;
	double delta_pipi = (3+Rgamma)/2;
	double tau_pipi = 4*(delta_pipi-1/2)/3;
	double lambda_pipi = Rgamma-1;

	PRECISION I2tt = thetaT * pitt;
	PRECISION I2tx = thetaT * pitx;
	PRECISION I2ty = thetaT * pity;
	PRECISION I2tn = thetaT * pitn;
	PRECISION I2xx = thetaT * pixx;
	PRECISION I2xy = thetaT * pixy;
	PRECISION I2xn = thetaT * pixn;
	PRECISION I2yy = thetaT * piyy;
	PRECISION I2yn = thetaT * piyn;
	PRECISION I2nn = thetaT * pinn;

	// I4
	PRECISION ux2 = ux * ux;
	PRECISION uy2 = uy * uy;
	PRECISION ps = pitt * sttT - 2 * pitx * stxT - 2 * pity * styT + pixx * sxxT + 2 * pixy * sxyT + piyy * syyT - 2 * pitn * stnT * t2 + 2 * pixn * sxnT * t2
			+ 2 * piyn * synT * t2 + pinn * snnT * t2 * t2;
	PRECISION ps2 = ps / 2;
	PRECISION I4tt = (pitt * sttT - pitx * stxT - pity * styT - t2 * pitn * stnT) - (1 - ut2+z02) * ps2;
	PRECISION I4tx = (pitt * stxT + pitx * sttT) / 2 - (pitx * sxxT + pixx * stxT) / 2 - (pity * sxyT + pixy * styT) / 2 - t2 * (pitn * sxnT + pixn * stnT) / 2
			+ (ut * ux) * ps2;
	PRECISION I4ty = (pitt * styT + pity * sttT) / 2 - (pitx * sxyT + pixy * stxT) / 2 - (pity * syyT + piyy * styT) / 2 - t2 * (pitn * synT + piyn * stnT) / 2
			+ (ut * uy) * ps2;
	PRECISION I4tn = (pitt * stnT + pitn * sttT) / 2 - (pitx * sxnT + pixn * stxT) / 2 - (pity * synT + piyn * styT) / 2 - t2 * (pitn * snnT + pinn * stnT) / 2
			+ (ut * un-z0z3) * ps2;
	PRECISION I4xx = (pitx * stxT - pixx * sxxT - pixy * sxyT - t2 * pixn * sxnT) + (1 + ux2) * ps2;
	PRECISION I4xy = (pitx * styT + pity * stxT) / 2 - (pixx * sxyT + pixy * sxxT) / 2 - (pixy * syyT + piyy * sxyT) / 2 - t2 * (pixn * synT + piyn * sxnT) / 2
			+ (ux * uy) * ps2;
	PRECISION I4xn = (pitx * stnT + pitn * stxT) / 2 - (pixx * sxnT + pixn * sxxT) / 2 - (pixy * synT + piyn * sxyT) / 2 - t2 * (pixn * snnT + pinn * sxnT) / 2
			+ (ux * un) * ps2;
	PRECISION I4yy = (pity * styT - pixy * sxyT - piyy * syyT - t2 * piyn * synT) + (1 + uy2) * ps2;
	PRECISION I4yn = (pity * stnT + pitn * styT) / 2 - (pixy * sxnT + pixn * sxyT) / 2 - (piyy * synT + piyn * syyT) / 2 - t2 * (piyn * snnT + pinn * synT) / 2
			+ (uy * un) * ps2;
	PRECISION I4nn = (pitn * stnT - pixn * sxnT - piyn * synT - t2 * pinn * snnT) + (1 / t2 + un2-z32) * ps2;

	PRECISION I5tt = zDzu * pitt;
	PRECISION I5tx = zDzu * pitx;
	PRECISION I5ty = zDzu * pity;
	PRECISION I5tn = zDzu * pitn;
	PRECISION I5xx = zDzu * pixx;
	PRECISION I5xy = zDzu * pixy;
	PRECISION I5xn = zDzu * pixn;
	PRECISION I5yy = zDzu * piyy;
	PRECISION I5yn = zDzu * piyn;
	PRECISION I5nn = zDzu * pinn;

	PRECISION Jtt = delta_pipi * I2tt + tau_pipi * I4tt - lambda_pipi * I5tt - lambda_piPi * Pi * sttT;
	PRECISION Jtx = delta_pipi * I2tx + tau_pipi * I4tx - lambda_pipi * I5tx - lambda_piPi * Pi * sttT;
	PRECISION Jty = delta_pipi * I2ty + tau_pipi * I4ty - lambda_pipi * I5ty - lambda_piPi * Pi * sttT;
	PRECISION Jtn = delta_pipi * I2tn + tau_pipi * I4tn - lambda_pipi * I5tn - lambda_piPi * Pi * sttT;
	PRECISION Jxx = delta_pipi * I2xx + tau_pipi * I4xx - lambda_pipi * I5xx - lambda_piPi * Pi * sttT;
	PRECISION Jxy = delta_pipi * I2xy + tau_pipi * I4xy - lambda_pipi * I5xy - lambda_piPi * Pi * sttT;
	PRECISION Jxn = delta_pipi * I2xn + tau_pipi * I4xn - lambda_pipi * I5xn - lambda_piPi * Pi * sttT;
	PRECISION Jyy = delta_pipi * I2yy + tau_pipi * I4yy - lambda_pipi * I5yy - lambda_piPi * Pi * sttT;
	PRECISION Jyn = delta_pipi * I2yn + tau_pipi * I4yn - lambda_pipi * I5yn - lambda_piPi * Pi * sttT;
	PRECISION Jnn = delta_pipi * I2nn + tau_pipi * I4nn - lambda_pipi * I5nn - lambda_piPi * Pi * sttT;

	PRECISION ss = sttT * sttT - 2 * stxT * stxT - 2 * styT * styT + sxxT * sxxT + 2 * sxyT * sxyT + syyT * syyT - 2 * stnT * stnT * t2 + 2 * sxnT * sxnT * t2
				+ 2 * synT * synT * t2 + snnT * snnT * t2 * t2;

	PRECISION JJ = Jtt * Jtt - 2 * Jtx * Jtx - 2 * Jty * Jty + Jxx * Jxx + 2 * Jxy * Jxy + Jyy * Jyy - 2 * Jtn * Jtn * t2 + 2 * Jxn * Jxn * t2
				+ 2 * Jyn * Jyn * t2 + Jnn * Jnn * t2 * t2;

	v->Rpi2[s] = sqrt(fabs(JJ / ss))/2/etaBar_TC;
#endif
#ifdef PI
	double zeta_z = (Rhat-Rbar0)*(e_s-3*p_s)/3;
	double zeta_T = -(Rbar0+Rhat)*(e_s-3*p_s)/6;
	PRECISION J = - beta_PiPi * Pi * zDzu - delta_PiPi * Pi * thetaT + lambda_Pipi * ps;
	v->RPi2[s] = fabsf(J / (- zeta_z*zDzu -zeta_T*thetaT));
#endif
}

void checkValidity(PRECISION t, const VALIDITY_DOMAIN * const __restrict__ v, const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p, const FLUID_VELOCITY * const __restrict__ u,
		const FLUID_VELOCITY * const __restrict__ up,
		int ncx, int ncy, int ncz, PRECISION etabar, PRECISION dt, PRECISION dx, PRECISION dy, PRECISION dz
) {
	for(int i = 2; i < ncx-2; ++i) {
		for(int j = 2; j < ncy-2; ++j) {
			for(int k = 2; k < ncz-2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, ncx, ncy);
					checkValidityKernel(t, v, currrentVars, e, p, u, up, s, ncx, ncy, ncz, etabar, dt, dx, dy, dz);
			}
		}
	}
}
