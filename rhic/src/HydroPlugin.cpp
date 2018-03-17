/*
 * HydroPlugin.c
 *
 *  Created on: Oct 23, 2015
 *      Author: bazow
 */

#include <stdlib.h>
#include <stdio.h> // for printf

// for timing
#include <ctime>
#include <iostream>

#include "../include/HydroPlugin.h"
#include "../include/DynamicalVariables.h"
#include "../include/LatticeParameters.h"
#include "../include/InitialConditionParameters.h"
#include "../include/HydroParameters.h"
#include "../include/FileIO.h"
#include "../include/InitialConditions.h"
#include "../include/FullyDiscreteKurganovTadmorScheme.h"
#include "../include/EnergyMomentumTensor.h"
#include "../include/EquationOfState.h"

#include "../include/HydrodynamicValidity.h"

#define FREQ 10

void outputDynamicalQuantities(double t, const char *outputDir, void * latticeParams) {
///*
	output(e, t, outputDir, "e", latticeParams);
	output(p, t, outputDir, "p", latticeParams);
//	output(a->xi, t, outputDir, "xi", latticeParams);
	output(u->ux, t, outputDir, "ux", latticeParams);
	output(u->uy, t, outputDir, "uy", latticeParams);
	output(u->un, t, outputDir, "un", latticeParams);
	output(u->ut, t, outputDir, "ut", latticeParams);
//	output(q->ttt, t, outputDir, "ttt", latticeParams);
//	output(q->ttx, t, outputDir, "ttx", latticeParams);
//	output(q->tty, t, outputDir, "tty", latticeParams);
//	output(q->ttn, t, outputDir, "ttn", latticeParams);
	output(q->pl, t, outputDir, "pl", latticeParams);
	output(validityDomain->knudsenNumberTaupiT, t, outputDir, "knTaupiT", latticeParams);
	output(validityDomain->knudsenNumberTaupiL, t, outputDir, "knTaupiL", latticeParams);
	output(validityDomain->knudsenNumberTaupi, t, outputDir, "knTaupi", latticeParams);
	output(validityDomain->regulations, t, outputDir, "regulations", latticeParams);
//
	output(validityDomain->regMag, t, outputDir, "regMag", latticeParams);
	output(validityDomain->regTr, t, outputDir, "regTr", latticeParams);
	output(validityDomain->regU0, t, outputDir, "regU0", latticeParams);
	output(validityDomain->regU1, t, outputDir, "regU1", latticeParams);
	output(validityDomain->regU2, t, outputDir, "regU2", latticeParams);
	output(validityDomain->regU3, t, outputDir, "regU3", latticeParams);
	output(validityDomain->regZ0, t, outputDir, "regZ0", latticeParams);
	output(validityDomain->regZ1, t, outputDir, "regZ1", latticeParams);
	output(validityDomain->regZ2, t, outputDir, "regZ2", latticeParams);
	output(validityDomain->regZ3, t, outputDir, "regZ3", latticeParams);
//
	output(validityDomain->stt, t, outputDir, "stt", latticeParams);
	output(validityDomain->sxx, t, outputDir, "sxx", latticeParams);
	output(validityDomain->syy, t, outputDir, "syy", latticeParams);
	output(validityDomain->snn, t, outputDir, "snn", latticeParams);
	output(validityDomain->taupi, t, outputDir, "taupi", latticeParams);
	output(validityDomain->dxux, t, outputDir, "dxux", latticeParams);
	output(validityDomain->dyuy, t, outputDir, "dyuy", latticeParams);
	output(validityDomain->theta, t, outputDir, "theta", latticeParams);
//	output(validityDomain->fTSolution, t, outputDir, "fTSolution", latticeParams);
	output(fTSol_X1, t, outputDir, "fTSol_X1", latticeParams);
	output(fTSol_Y1, t, outputDir, "fTSol_Y1", latticeParams);
	output(fTSol_1, t, outputDir, "fTSol_1", latticeParams);
	output(fTSol_X2, t, outputDir, "fTSol_X2", latticeParams);
	output(fTSol_Y2, t, outputDir, "fTSol_Y2", latticeParams);
	output(fTSol_2, t, outputDir, "fTSol_2", latticeParams);
#ifdef PIMUNU
	output(q->pixx, t, outputDir, "pixx", latticeParams);
	output(q->pixy, t, outputDir, "pixy", latticeParams);
	output(q->pixn, t, outputDir, "pixn", latticeParams);
	output(q->piyy, t, outputDir, "piyy", latticeParams);
	output(q->piyn, t, outputDir, "piyn", latticeParams);

	output(q->pitt, t, outputDir, "pitt", latticeParams);
	output(q->pitx, t, outputDir, "pitx", latticeParams);
	output(q->pity, t, outputDir, "pity", latticeParams);
	output(q->pitn, t, outputDir, "pitn", latticeParams);
	output(q->pinn, t, outputDir, "pinn", latticeParams);
	output(validityDomain->Rpi, t, outputDir, "Rpi", latticeParams);
	output(validityDomain->Rpi2, t, outputDir, "Rpi2", latticeParams);
#endif
#ifdef W_TZ_MU
	output(q->WtTz, t, outputDir, "WtTz", latticeParams);
	output(q->WxTz, t, outputDir, "WxTz", latticeParams);
	output(q->WyTz, t, outputDir, "WyTz", latticeParams);
	output(q->WnTz, t, outputDir, "WnTz", latticeParams);
	output(validityDomain->Rw, t, outputDir, "Rw", latticeParams);
#endif
#ifdef PI
	output(q->Pi, t, outputDir, "Pi", latticeParams);
	output(validityDomain->knudsenNumberTauPi, t, outputDir, "knTauPi", latticeParams);
	output(validityDomain->RPi, t, outputDir, "RPi", latticeParams);
	output(validityDomain->RPi2, t, outputDir, "RPi2", latticeParams);
#endif
//*/
}

void run(void * latticeParams, void * initCondParams, void * hydroParams, const char *rootDirectory, const char *outputDir) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	struct InitialConditionParameters * initCond = (struct InitialConditionParameters *) initCondParams;
	struct HydroParameters * hydro = (struct HydroParameters *) hydroParams;

	/************************************************************************************\
	 * System configuration
	/************************************************************************************/
	int nt = lattice->numProperTimePoints;
	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;
	int ncx = lattice->numComputationalLatticePointsX;
	int ncy = lattice->numComputationalLatticePointsY;
	int ncz = lattice->numComputationalLatticePointsRapidity;
	int nElements = ncx * ncy * ncz;

	double t0 = hydro->initialProperTimePoint;
	double dt = lattice->latticeSpacingProperTime;

	double e0 = initCond->initialEnergyDensity;

	double freezeoutTemperatureGeV = hydro->freezeoutTemperatureGeV;
	const double hbarc = 0.197326938;
	const double freezeoutTemperature = freezeoutTemperatureGeV/hbarc;
//	const double freezeoutEnergyDensity = e0*pow(freezeoutTemperature,4);
	const double freezeoutEnergyDensity = equilibriumEnergyDensity(freezeoutTemperature);
	printf("Grid size = %d x %d x %d\n", nx, ny, nz);
	printf("spatial resolution = (%.3f, %.3f, %.3f)\n", lattice->latticeSpacingX, lattice->latticeSpacingY, lattice->latticeSpacingRapidity);
	printf("Grid size [fm] = %.3f x %.3f x %.3f\n", (nx-1)/2*lattice->latticeSpacingX, (ny-1)/2*lattice->latticeSpacingY, (nz-1)/2*lattice->latticeSpacingRapidity);
	printf("freezeout temperature = %.3f [fm^-1] (eF = %.3f [fm^-4])\n", freezeoutTemperature, freezeoutEnergyDensity);
#ifdef CONFORMAL_EOS
	printf("Using conformal EOS: e=3p.\n");
#else
	printf("Using QCD EOS.\n");
#endif
	printf("eta/s = %.6f\n", hydro->shearViscosityToEntropyDensity);

	// allocate memory
	allocateHostMemory(nElements);

	/************************************************************************************\
	 * Fluid dynamic initialization
	/************************************************************************************/
	double t = t0;
	// generate initial conditions
	setInitialConditions(latticeParams, initCondParams, hydroParams, rootDirectory);
	// Calculate conserved quantities
	setConservedVariables(t, latticeParams);
	// impose boundary conditions with ghost cells
	setGhostCells(q,e,p,u,latticeParams);

	// Validity
	PRECISION dx = (PRECISION)(lattice->latticeSpacingX);
	PRECISION dy = (PRECISION)(lattice->latticeSpacingY);
	PRECISION dz = (PRECISION)(lattice->latticeSpacingRapidity);
	PRECISION etabar = (PRECISION)(hydro->shearViscosityToEntropyDensity);
	checkValidity(t, validityDomain, q, e, p, u, up, ncx, ncy, ncz, etabar, dt, dx, dy, dz);

	/************************************************************************************\
	 * Evolve the system in time
	/************************************************************************************/
	int ictr = (nx % 2 == 0) ? ncx/2 : (ncx-1)/2;
	int jctr = (ny % 2 == 0) ? ncy/2 : (ncy-1)/2;
	int kctr = (nz % 2 == 0) ? ncz/2 : (ncz-1)/2;
	int sctr = columnMajorLinearIndex(ictr, jctr, kctr, ncx, ncy);

	std::clock_t t1,t2;

	double totalTime = 0;
	int nsteps = 0;

	// evolve in time
	for (int n = 1; n <= nt+1; ++n) {
		// copy variables back to host and write to disk
		if ((n-1) % FREQ == 0) {
			printf("n = %d:%d (t = %.3f),\t (e, p, pl) = (%.3f, %.3f, %.3f) [fm^-4],\t (T = %.3f [GeV]),\t a = %.3f\n",
				n - 1, nt, t, e[sctr], p[sctr], q->pl[sctr], effectiveTemperature(e[sctr])*hbarc, q->pl[sctr]/e[sctr]);
			outputDynamicalQuantities(t, outputDir, latticeParams);
			// end hydrodynamic simulation if the temperature is below the freezeout temperature
			if(e[sctr] < freezeoutEnergyDensity) {
				printf("\nReached freezeout temperature at the center.\n");
				break;
			}
		}

		t1 = std::clock();
		rungeKutta2(t, dt, q, Q, latticeParams, hydroParams);
		t2 = std::clock();
		double delta_time = (t2 - t1) / (double)(CLOCKS_PER_SEC / 1000);
//		if ((n-1) % FREQ == 0) printf("(Elapsed time: %.3f ms)\n",delta_time);
		totalTime+=delta_time;
		++nsteps;

		setCurrentConservedVariables();

		t = t0 + n * dt;
	}
	printf("Average time/step: %.3f ms\n",totalTime/((double)nsteps));

	/************************************************************************************\
	 * Deallocate host memory
	/************************************************************************************/
	freeHostMemory();
}
