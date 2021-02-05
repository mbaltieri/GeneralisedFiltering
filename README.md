# GeneralisedFiltering
General framework for active Bayesian inversion of continuous hierarchical models

The framework is heavily inspired by spm/DEM and is so far limited to inference + control for a single layer (in hierarchical models) and linear systems.


### TODO:
- implement full support to nonlinear functions
- implement hierarchical structures (i.e., stacking layers)
- implement learning (update of parameters) and attention (update of hyperparameters)
- implement tests
- consider moving everything to JAX if the code remains heavily reliant on jacobian/hessian functions
- optimise code and restructure it as needed
