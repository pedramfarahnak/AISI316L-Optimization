# AISI 316L Post-Necking Hardening Optimization

## Overview

This repository contains a Python–ABAQUS framework for finite element-based inverse identification of the post-necking hardening behavior of Directed Energy Deposition (DED)-manufactured AISI 316L stainless steel.

The framework compares six derivative-free optimization strategies for identifying a single post-necking hardening exponent using experimental uniaxial tensile force–displacement measurements.

## Optimization Methods

The following methods are investigated:

- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Gaussian Process-based Bayesian Optimization (BO)
- Tree-structured Parzen Estimator (TPE)
- Grid Search
- Bounded Brent Optimization

## Methodology

The automated calibration procedure consists of:

1. Importing experimental tensile data and identifying the post-necking displacement interval.
2. Generating a hybrid hardening law combining experimental pre-necking data with a parameterized post-necking power law.
3. Updating the ABAQUS input file with the candidate plastic hardening curve.
4. Executing an ABAQUS/Standard finite element simulation.
5. Extracting reaction force and displacement from the output database.
6. Comparing experimental and simulated responses at 125 common displacement locations using PCHIP interpolation.
7. Evaluating the post-necking normalized root mean square error (NRMSE).
8. Updating the candidate parameter according to the selected optimization algorithm.

## Numerical Sensitivity

The research also examines the sensitivity of the FE-based objective function to the maximum increment size. Numerical differences between consecutive FE solutions are assessed separately from the experimental fitting error.

This distinction is important when interpreting small differences in optimization outcomes.

## Software

- Python
- ABAQUS/Standard 2024
- NumPy, SciPy and relevant optimization libraries

ABAQUS is commercial software and must be installed and licensed separately.

## Research Scope

The framework focuses on uniaxial tensile calibration of post-necking hardening behavior.

The available optimization results represent individual algorithm runs. They should not be interpreted as evidence of statistical superiority or independent validation of the identified constitutive model under other loading conditions.

## Data Availability

The experimental dataset is associated with the work of Urbánek et al. (2022). Redistribution of the experimental data is subject to the applicable permissions.

## Reference

Urbánek, M., et al. (2022). Prediction of Behaviour of Thin-Walled DED-Processed Structure: Experimental-Numerical Approach. *Materials*, 15(3), 806.

https://doi.org/10.3390/ma15030806

## Related Publication

Manuscript on inverse post-necking hardening identification and numerical sensitivity assessment.

Publication information will be added when available.
