# GeneralisedFiltering
General framework for active Bayesian inversion of continuous hierarchical dynamic models (HDM), see "Hierarchical models in the brain", Friston, 2008.

The framework is heavily inspired by spm/DEM and is so far limited to inference + control for a single layer (in hierarchical models) and linear systems.

Numerical integration of stochastic differential equations is carried out using a method first proposed by Ozaki in "A bridge between nonlinear time series models and nonlinear stochastic dynamical systems: A local linearization approach", Ozaki, 1992 (see also "Simulation of stochastic differential equations through the local linearization method. A comparative study", Jimenez, Shoji and Ozaki, 1999). This method provides a robust integration scheme that allows to simulate continuous-time SDEs using an approximation based on the Jacobian of the integrand between each pair of time steps (cf. a constant used in the basic Euler-Maruyama). It is not perhaps a well known method, but it is certainly used in different fields, some references:
- "Seeing the Wood for the Trees: A Critical Evaluation of Methods to Estimate the Parameters of Stochastic Differential Equations", Hurn, Jeisman and Lindsay
- "Simulation and Inference for Stochastic Differential Equations", Iacus, 2008
- "Stochastic Differential Equations: An Introduction with Applications in Population Dynamics Modeling", Panik, 2017
- "Stochastic differential equation based on a multimodal potential to model movement data in ecology", Gloaguen, Etienne, Le Corff
- "Applied Stochastic Differential Equations", Särkkä and Solin, 2019


### TODO:
- implement full support to nonlinear functions (i.e., transition and observation laws in a state-space model)
- implement hierarchical structures (i.e., stacking layers)
- implement learning (update of parameters) and attention (update of hyperparameters)
- implement tests
- consider moving everything to JAX if the code remains heavily reliant on jacobian/hessian functions
- optimise code and restructure it as needed

Regarding nonlinear functions support, the number one priority at the moment, the current issues are: 
1. how should functions be provided by the user? 
	* as arguments to a predefined function (as lambda functions? explicit ones?)
	* as templates to be filled in within a file that has to be imported by the current classes
	* as some symbolic expression to be decoded
2. the current structure is limiting and will only allow (nonlinear) functions in the form f(x,u,a) = f(x) + f(u) + f(a) because of the computation of the Jacobians in the Local Linearisation step (what should J_x, J_u, J_a actually be multiplied by?)
3. the speed for differentiating these functions is atrocious at the moment, and it gets worse over iterations (is the computational graph growing for some reason? should some operations be within torch.no_grad()?)

### TODO (one day...):
- provide some interface with torch.nn so to exploit the full potential of pytorch's neural networks in HDMs
- look into other SDE solvers (although LL might be a fantastic choice actually, moving everything to discrete time with time increments whose intepretation is arbitrarily small but with a finite implementation); for example see ['torchsde'](https://github.com/google-research/torchsde)