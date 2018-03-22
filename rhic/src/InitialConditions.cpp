/*
 * InitialConditions.c
 *
 *  Created on: Oct 23, 2015
 *      Author: bazow
 */

#include <math.h> // for math functions
#include <stdio.h> // for printf
#include <stdlib.h> //TEMP

#include "../include/InitialConditions.h"
#include "../include/DynamicalVariables.cuh"
#include "../include/LatticeParameters.h"
#include "../include/InitialConditionParameters.h"
#include "../include/GlauberModel.h"
#include "../include/MonteCarloGlauberModel.h"
#include "../include/HydroParameters.h"
#include "../include/EquationOfState.cuh"

#define THETA_FUNCTION(X) ((double)X < (double)0 ? (double)0 : (double)1)

/*********************************************************************************************************\
 * Set initial flow profile
 *		- u^\mu = (1, 0, 0, 0)
 * 	- No transverse flow (ux = uy = 0)
 *		- Longitudinal scaling flow (u_z = z/t, i.e. un = 0)
/*********************************************************************************************************/
void setFluidVelocityInitialCondition(void * latticeParams, void * hydroParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	struct HydroParameters * hydro = (struct HydroParameters *) hydroParams;
	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;

	double t0 = hydro->initialProperTimePoint;

	for(int i = 2; i < nx+2; ++i) {
		for(int j = 2; j < ny+2; ++j) {
			for(int k = 2; k < nz+2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, nx+4, ny+4);
				PRECISION ux = 0;
				PRECISION uy = 0;
				PRECISION un = 0;
				u->ux[s] = 0;
				u->uy[s] = 0;
				u->un[s] = 0;
				u->ut[s] = sqrt(1+ux*ux+uy*uy+t0*t0*un*un);
			}
		}
	}
}

/*********************************************************************************************************\
 * Set initial shear-stress tensor \pi^\mu\nu
 *		- Navier-Stokes value, i.e. \pi^\mu\nu = 2 * (\epsilon + P) / T * \eta/S * \sigma^\mu\nu
 * 	- No initial pressure anisotropies (\pi^\mu\nu = 0)
/*********************************************************************************************************\
void setPimunuNavierStokesInitialCondition(void * latticeParams, void * initCondParams, void * hydroParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	struct InitialConditionParameters * initCond = (struct InitialConditionParameters *) initCondParams;
	struct HydroParameters * hydro = (struct HydroParameters *) hydroParams;
	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;

	PRECISION dx = (PRECISION)(lattice->latticeSpacingX);
	PRECISION dz = (PRECISION)(lattice->latticeSpacingRapidity);

	PRECISION etabar = (PRECISION)(hydro->shearViscosityToEntropyDensity);
	PRECISION t = hydro->initialProperTimePoint;

	PRECISION e0 = initCond->initialEnergyDensity;

	for(int i = 2; i < nx+2; ++i) {
		for(int j = 2; j < ny+2; ++j) {
			for(int k = 2; k < nz+2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, nx+4, ny+4);
//				double T = pow(e[s]/e0, 0.25);
				PRECISION T = effectiveTemperature(e[s]);
				if (T == 0) T = 1.e-3;
				PRECISION pinn = -2/(3*t*t*t)*etabar*(e[s]+p[s])/T;
#ifdef PIMUNU
				q->pitt[s] = 0;
				q->pitx[s] = 0;
				q->pity[s] = 0;
				q->pitn[s] = 0;
				q->pixx[s] = -t*t*pinn/2;
				q->pixy[s] = 0;
				q->pixn[s] = 0;
				q->piyy[s] = -t*t*pinn/2;
				q->piyn[s] = 0;
				q->pinn[s] = pinn;
#endif
#ifdef PI
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
				PRECISION x = T/1.01355;
				PRECISION zetabar = A_1*x*x + A_2*x - A_3;
				if(x > 1.05)
					zetabar = LAMBDA_1*exp(-(x-1)/SIGMA_1) + LAMBDA_2*exp(-(x-1)/SIGMA_2)+0.001;
				else if(x < 0.995)
					zetabar = LAMBDA_3*exp((x-1)/SIGMA_3)+ LAMBDA_4*exp((x-1)/SIGMA_4)+0.03;
				q->Pi[s] = -zetabar*(e[s]+p[s])/T/t;
#endif
			}
		}
	}
}
/*********************************************************************************************************/
void setPimunuInitialCondition(void * latticeParams, void * initCondParams, void * hydroParams) {
	struct HydroParameters * hydro = (struct HydroParameters *) hydroParams;
	int initializePimunuNavierStokes = hydro->initializePimunuNavierStokes;
	if (initializePimunuNavierStokes==1) {
		printf("Initialize \\pi^\\mu\\nu to its asymptotic Navier-Stokes value.\n");
#ifdef PI
		printf("Initialize \\Pi to its asymptotic Navier-Stokes value.\n");
#endif
//		setPimunuNavierStokesInitialCondition(latticeParams, initCondParams, hydroParams);
		return;
	}
	else {
		printf("Initialize \\pi^\\mu\\nu to zero.\n");
		struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
		int nx = lattice->numLatticePointsX;
		int ny = lattice->numLatticePointsY;
		int nz = lattice->numLatticePointsRapidity;
		for(int i = 2; i < nx+2; ++i) {
			for(int j = 2; j < ny+2; ++j) {
				for(int k = 2; k < nz+2; ++k) {
					int s = columnMajorLinearIndex(i, j, k, nx+4, ny+4);
#ifdef PIMUNU
			  		q->pitt[s] = 0;
			  		q->pitx[s] = 0;
			  		q->pity[s] = 0;
			  		q->pitn[s] = 0;
			  		q->pixx[s] = 0;
			  		q->pixy[s] = 0;
			  		q->pixn[s] = 0;
			  		q->piyy[s] = 0;
			  		q->piyn[s] = 0;
			  		q->pinn[s] = 0;
#endif
#ifdef PI
			  		q->Pi[s] = 0;
#endif
				}
			}
		}
		return;
	}
}

/*********************************************************************************************************\
 * Constant initial energy density distribution
/*********************************************************************************************************/
void setConstantEnergyDensityInitialCondition(void * latticeParams, void * initCondParams) {
	struct InitialConditionParameters * initCond = (struct InitialConditionParameters *) initCondParams;
	double initialEnergyDensity = initCond->initialEnergyDensity;

	double T0 = 3.05;
	double ed = equilibriumEnergyDensity(T0);

	// temp
//	ed = 93.2104;

	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;
	for(int i = 2; i < nx+2; ++i) {
		for(int j = 2; j < ny+2; ++j) {
			for(int k = 2; k < nz+2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, nx+4, ny+4);
				e[s] = (PRECISION) ed;
				p[s] = equilibriumPressure(e[s]);
			}
		}
	}
}

/*********************************************************************************************************\
 * Longitudinal initial energy density distribution
/*********************************************************************************************************/
void longitudinalEnergyDensityDistribution(double * const __restrict__ eL, void * latticeParams, void * initCondParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	struct InitialConditionParameters * initCond = (struct InitialConditionParameters *) initCondParams;

	int nz = lattice->numLatticePointsRapidity;

	double dz = lattice->latticeSpacingRapidity;

	double etaFlat = initCond->rapidityMean;
	double etaVariance = initCond->rapidityVariance;
	etaFlat=5.9;
	etaVariance=0.16;

	for(int k = 0; k < nz; ++k) {
		double eta = (k - (nz-1)/2)*dz;
		double etaScaled = fabs(eta) - etaFlat/2;
		double arg = -etaScaled * etaScaled / etaVariance / 2 * THETA_FUNCTION(etaScaled);
		eL[k] = exp(arg);
	}
}

/*********************************************************************************************************\
 * Continuous optical glauber Glauber initial energy density distribution
/*********************************************************************************************************/
void setGlauberInitialCondition(void * latticeParams, void * initCondParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	struct InitialConditionParameters * initCond = (struct InitialConditionParameters *) initCondParams;

	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;

	double dx = lattice->latticeSpacingX;
	double dy = lattice->latticeSpacingY;
	double dz = lattice->latticeSpacingRapidity;

	double e0 = initCond->initialEnergyDensity;
	double T0 = 3.05;
//	e0 *= pow(T0,4);
	e0 = (double) equilibriumEnergyDensity(T0);

	double eT[nx*ny], eL[nz];
	energyDensityTransverseProfileAA(eT, nx, ny, dx, dy, initCondParams);
	longitudinalEnergyDensityDistribution(eL, latticeParams, initCondParams);

	for(int i = 2; i < nx+2; ++i) {
		for(int j = 2; j < ny+2; ++j) {
			double energyDensityTransverse = e0 * eT[i-2+(j-2)*nx];
			for(int k = 2; k < nz+2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, nx+4, ny+4);
				double energyDensityLongitudinal = eL[k-2];
				double ed = (energyDensityTransverse * energyDensityLongitudinal) + 1.e-3;
				e[s] = (PRECISION) ed;
				p[s] = equilibriumPressure(e[s]);
			}
		}
	}
}

/*********************************************************************************************************\
 * Monte carlo Glauber initial energy density distribution
/*********************************************************************************************************/
void setMCGlauberInitialCondition(void * latticeParams, void * initCondParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	struct InitialConditionParameters * initCond = (struct InitialConditionParameters *) initCondParams;

	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;

	double dx = lattice->latticeSpacingX;
	double dy = lattice->latticeSpacingY;
	double dz = lattice->latticeSpacingRapidity;

	double e0 = initCond->initialEnergyDensity;
//	double T0 = 3.05;
	double T0 = 2.03;
//	e0 *= pow(T0,4);
	e0 = (double) equilibriumEnergyDensity(T0);

	double eT[nx*ny], eL[nz];
	monteCarloGlauberEnergyDensityTransverseProfile(eT, nx, ny, dx, dy, initCondParams);
	longitudinalEnergyDensityDistribution(eL, latticeParams, initCondParams);

	for(int i = 2; i < nx+2; ++i) {
		for(int j = 2; j < ny+2; ++j) {
			double energyDensityTransverse = e0 * eT[i-2 + nx*(j-2)];
			for(int k = 2; k < nz+2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, nx+4, ny+4);
				double energyDensityLongitudinal = eL[k-2];
				double ed = (energyDensityTransverse * energyDensityLongitudinal) + 1.e-3;
				e[s] = (PRECISION) ed;
				p[s] = equilibriumPressure(e[s]);
			}
		}
	}
}

void setMCGlauberInitialCondition_FromMS_1d(void * latticeParams, void * initCondParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	struct InitialConditionParameters * initCond = (struct InitialConditionParameters *) initCondParams;

	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;

	double dx = lattice->latticeSpacingX;
	double dy = lattice->latticeSpacingY;
	double dz = lattice->latticeSpacingRapidity;

	double e0 = initCond->initialEnergyDensity;
//	double T0 = 3.05;
	double T0 = 2.03;
//	e0 *= pow(T0,4);
	e0 = (double) equilibriumEnergyDensity(T0);

double T [201] = {0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,
   0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,
   0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,
   0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,
   0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,
   0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,
   0.000599999983433382,0.0005999998354381784,0.000599999786106444,0.000599999884769913,0.00060000018076032,
   0.0005999941011173602,0.0005999899730378173,0.0005999895014264355,0.0005999943911879592,0.0006000063472271326,
   0.000599877946588576,0.0005997973040021889,0.0005998034063377464,0.0005999352404650242,0.0006002317932537975,
   0.000598531998290537,0.0005976249090491488,0.0005980995257202346,0.0006005448484943958,0.0006055498775622337,
   0.0005929250312914075,0.0005892325371511969,0.0006002560407879391,0.0006317791878479711,0.00068958562397763,
   0.0006586336190203106,0.0006957385383760284,0.0008368903716418574,0.0011180791084148677,0.0015752947382921326,
   0.0018502728711497886,0.002471821471236077,0.003574494123078321,0.005292844411203807,0.00776142592013985,0.010610427580255245,
   0.014604858793775426,0.020005364308767412,0.02707258887329805,0.03606717723543432,0.04794849148960996,0.06210377970093336,
   0.07861900728088003,0.09758013964092493,0.11907314219254352,0.14592531989125102,0.17479596371847328,0.20508570419967448,
   0.23619517186032044,0.26752499722587636,0.2996607604657861,0.3305219050505423,0.3592128240946144,0.3848379107124737,
   0.4065015580185907,0.4175742803214973,0.4243278192430877,0.42730003759931706,0.42702879820614076,0.42405196387951427,
   0.41619954971895473,0.40739422818596527,0.39885082402561056,0.39178416198295535,0.38740906680306453,0.3980978976384968,
   0.4111185612249494,0.4248964987056141,0.43785715122368213,0.448425959922345,0.4492893254608111,0.44604648958725057,
   0.43855765356585064,0.42668301866079844,0.4102827861362813,0.3847054548080138,0.3554508540007741,0.3235071105908668,
   0.2898623514545971,0.25550470346827014,0.22331697029189684,0.19191893282215056,0.1618250487394093,0.13354977572405166,
   0.10760757145645614,0.08717845484153713,0.06944493202900376,0.054255070393100166,0.041456937308070894,0.03089860014816049,
   0.022867599188902847,0.016662660677930922,0.012021983764166723,0.008683767596532416,0.006386211323950163,0.004303866307926745,
   0.0028794914316535936,0.0019921977909066707,0.0015210964814619799,0.0013452985990955251,0.0009800641972685757,
   0.0007593181746505598,0.0006531343875961552,0.0006315866924600479,0.0006647489455969237,0.0006209835593325001,
   0.0005975036950576741,0.0005898110701343721,0.000593407401924522,0.0006037944077900512,0.0006002026085996673,
   0.000598472717331823,0.0005981742504717511,0.0005988767245046843,0.000600149655915855,0.0005999480834887839,
   0.0005998596208358438,0.0005998574038676954,0.0005999145684949993,0.0006000042506284163,0.0005999965223188895,
   0.0005999933493017264,0.0005999936334525172,0.0005999962766468517,0.00060000018076032,0.0005999999854066514,
   0.0005999999202887618,0.0005999999528477066,0.0006000000505245409,0.00060000018076032,0.00060000018076032,0.00060000018076032,
   0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,
   0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,
   0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,
   0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,
   0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,0.00060000018076032,
   0.00060000018076032,0.00060000018076032,0.00060000018076032};

	int ind = 0;
	for(int i = 2; i < nx+2; ++i) {
		for(int j = 2; j < ny+2; ++j) {
			for(int k = 2; k < nz+2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, nx+4, ny+4);
				double ed = (double) equilibriumEnergyDensity(T[ind]/0.197326938);
++ind;
				e[s] = (PRECISION) ed;
				p[s] = equilibriumPressure(e[s]);
			}
		}
	}
}

/*********************************************************************************************************\
 * Initial conditions for the Gubser ideal hydro test
 *		- set energy density, pressure, fluid velocity u^\mu, and \pi^\mu\ny
/*********************************************************************************************************/
void setIdealGubserInitialCondition(void * latticeParams, void * initCondParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	struct InitialConditionParameters * initCond = (struct InitialConditionParameters *) initCondParams;

	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;

	double dx = lattice->latticeSpacingX;
	double dy = lattice->latticeSpacingY;
	double dz = lattice->latticeSpacingRapidity;

	double e0 = initCond->initialEnergyDensity;

	for(int i = 2; i < nx+2; ++i) {
		double x = (i-2 - (nx-1)/2.)*dx;
		for(int j = 2; j < ny+2; ++j) {
			double y = (j-2 - (ny-1)/2.)*dy;

			double T = 1.9048812623618392/pow(1 + pow(1 - pow(x,2) - pow(y,2),2) + 2*(1 + pow(x,2) + pow(y,2)),0.3333333333333333);
			double r = sqrt(x*x+y*y);
			double phi = atanh(2*1*r/(1+1+x*x+y*y));

			for(int k = 2; k < nz+2; ++k) {
				int s = columnMajorLinearIndex(i, j, k, nx+4, ny+4);

				e[s] = (PRECISION) (e0 * pow(T,4));
				p[s] = e[s]/3;
				u->ux[s] = (PRECISION) (sinh(phi)*x/r);
				u->uy[s] = (PRECISION) (sinh(phi)*y/r);
				u->un[s] = 0;
				u->ut[s] = sqrt(1 + u->ux[s]*u->ux[s] + u->uy[s]*u->uy[s]);
#ifdef PIMUNU
				q->pixx[s] = 0;
				q->pixy[s] = 0;
				q->piyy[s] = 0;
#endif
			}
		}
	}
}

/*********************************************************************************************************\
 * Initial conditions to use.
 *	Set the energy density, pressure, fluid velocity u^\mu, and \pi^\mu\ny.
 * 	0 - constant energy density
 *		1 - Isreal-Stewart hydrodynamic Gubser flow test
 *		2 - Continous optical Glauber
 *		3 - Ideal hydrodynamic Gubser flow test
 *		4 - Monte carlo Glauber
 *		5 - Relativistic Sod shock-tube test
/*********************************************************************************************************/
void setInitialConditions(void * latticeParams, void * initCondParams, void * hydroParams, const char *rootDirectory) {
	struct InitialConditionParameters * initCond = (struct InitialConditionParameters *) initCondParams;
	int initialConditionType = initCond->initialConditionType;
	printf("Setting initial conditions: ");
	switch (initialConditionType) {
		case 0: {
			printf("constant energy density.\n");
			setConstantEnergyDensityInitialCondition(latticeParams, initCondParams);
			setFluidVelocityInitialCondition(latticeParams, hydroParams);
			setPimunuInitialCondition(latticeParams, initCondParams, hydroParams);
			return;
		}
		case 2: {
			printf("Continous optical Glauber.\n");
			setGlauberInitialCondition(latticeParams, initCondParams);
			setFluidVelocityInitialCondition(latticeParams, hydroParams);
			setPimunuInitialCondition(latticeParams, initCondParams, hydroParams);
			return;
		}
		case 3: {
			printf("Ideal hydrodynamic Gubser flow test.\n");
			setIdealGubserInitialCondition(latticeParams, initCondParams);
			return;
		}
		case 4: {
			printf("Monte carlo Glauber.\n");
			setMCGlauberInitialCondition(latticeParams, initCondParams);
			setFluidVelocityInitialCondition(latticeParams, hydroParams);
			setPimunuInitialCondition(latticeParams, initCondParams, hydroParams);
			return;
		}
		default: {
			printf("Initial condition type not defined. Exiting ...\n");
			exit(-1);
		}
	}
}
