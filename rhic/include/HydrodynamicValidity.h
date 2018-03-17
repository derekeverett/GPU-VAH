/*
 * HydrodynamicValidity.h
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef HYDRODYNAMICVALIDITY_H_
#define HYDRODYNAMICVALIDITY_H_

#include "DynamicalVariables.h"

void checkValidity(PRECISION t, const VALIDITY_DOMAIN * const __restrict__ v, const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p, const FLUID_VELOCITY * const __restrict__ u,
		const FLUID_VELOCITY * const __restrict__ up,
		int ncx, int ncy, int ncz, PRECISION etabar, PRECISION dt, PRECISION dx, PRECISION dy, PRECISION dz
);

#endif /* HYDRODYNAMICVALIDITY_H_ */
