# Lanegvin/Brownian Dynamics Simulation
Taken directly from [Two for One](https://github.com/microsoft/two-for-one-diffusion) codebase.

This folder contains the files that are used to run Langevin Dynamics.  
- `langevin_cgnet.py`: This code defines the Langevin Dynamics, it has been extracted from [CGnet](https://github.com/wutianyiRosun/CGNet).
- `langevin.py`: Defines the the Force Field wrapper around the diffusion model, and the Langevin Dynamics based on the previous file.