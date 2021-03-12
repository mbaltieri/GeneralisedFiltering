# GeneralisedFiltering
General framework for active Bayesian inversion of continuous hierarchical dynamic models (HDM), see "Hierarchical models in the brain", Friston, 2008, with the relaxed assumptions driving to the the development of "Generalised filtering", Friston et al., 2010.

The framework is heavily inspired by spm/DEM and is so far limited to inference + control for a single layer (in hierarchical models) and linear systems.

## Features

### GPU support
The library is currently developed using pytorch, with the option of effectively running heavy simulations on GPU. Pytorch was the first, and one if not the more natural, choice at the beginning of this package's development due to the flexibility of dynamic computational graphs (the first idea to build this library came about before Tensorflow 2.0 was released). 

The natural support for automated differentiation was also one of the main reasons to try and provide an alternative to the original code in spm/DEM which, to date, relies on numerical approximations. Over time however, it has become increasingly clear that the use of specialised methods for numerical integration of SDEs, i.e., Local Linearisation (see next section), is hardly made any easier by pytorch and its automated differentiation system (at least up to March 2021 when this file was last updated) due to the strong reliance of the method here adopted on derivatives of vector functions (i.e., Jacobians rather than gradients). Due to the performance of the LL algorithm however, it is hard to imagine switching to different integration schemes and it is then perhaps worth investigating if moving to a different backend (JAX?) is a better option.


### Numerical integration
Numerical integration of stochastic differential equations is carried out using a method first proposed by Ozaki in "A bridge between nonlinear time series models and nonlinear stochastic dynamical systems: A local linearization approach", Ozaki, 1992 (see also "Simulation of stochastic differential equations through the local linearization method. A comparative study", Jimenez, Shoji and Ozaki, 1999). This method was originally devised as bridge between (discrete-time) stochastic difference equations and (continuous-time) stochastic differential equations, but has later gained a certain amount of success also as an integration method for the problemic nature of SDEs simulations (Numerical solution of SDE through computer experiments, Kloeden et al., 2012). 

Local linearisation (LL) provides a robust integration scheme that allows to simulate continuous-time SDEs using an approximation based on the Jacobian of the integrand between pairs of time steps (cf. a constant used in the more basic Euler-Maruyama). Some references:
- "Seeing the Wood for the Trees: A Critical Evaluation of Methods to Estimate the Parameters of Stochastic Differential Equations", Hurn, Jeisman and Lindsay
- "Simulation and Inference for Stochastic Differential Equations", Iacus, 2008
- "Stochastic differential equation based on a multimodal potential to model movement data in ecology", Gloaguen, Etienne, Le Corff, 2015
- "Stochastic Differential Equations: An Introduction with Applications in Population Dynamics Modeling", Panik, 2017
- "Applied Stochastic Differential Equations", Särkkä and Solin, 2019

The method works really well, but should not be seen as a 'free lunch' approach, since the main idea is to simply discard the continuous time formulation in favour of a discrete one with *arbitrarily small* step, but crucially not a step that tends to zero to recover the continuous time limit. If properties from the continuous time formulation are required (for example for some analytical calculation), this is NOT the way to go.

#### Local linearisation
To gain an undestanding of the basic implementation of LL, we first look at its application on an ODE example (i.e., without noise), see also sec. 9.2.4 in Ozaki, Tohru. Time series modeling of neuroscience data. CRC press, 2012. (the generalisation to SDEs is rather straightforward and can be found in section 9.2.5, while in section 9.2.6 we find an application to SDEs with driving inputs):

$$\dot{x}(t) = \frac{d x(t)}{d t} = f(x(t))$$

with $x \in R^n$ (for n>1, see section 9.3 in Ozaki, Tohru. Time series modeling of neuroscience data. CRC press, 2012). After differentiating both sides with respect to $t$

$$\frac{d}{d t} \dot{x}(t) = \frac{d}{d t} f(x(t))$$

using the chain rule, we can rewrite the equation as

$$\ddot{x}(t) = \frac{d}{d x} f(x(t)) \frac{d x(t)}{d t}$$

and with the Jacobian $J_f(x(t)) = \frac{d}{d x} f(x(t))$, obtain

$$\ddot{x}(t) = J_f(x(t)) \dot{x}(t)$$Applied Stochastic Differential Equations

The key assumption behind the LL method is to fix the Jacobian $J_f(x(t))$ on a (discrete) time interval $[t_0, t_0 + \Delta t)$, defining thus $J_f^{t_0} = J_f(x(t)) \big\vert_{t_0 \le t < t_0 + \Delta t}$. After replacing $J_f^{t_0}$ in the above equation, we have
$$
\begin{align}
    \ddot{x}(t) & = J_f^{t_0} \dot{x}(t) \\
    \Rightarrow \ddot{x}(t) & = J_f^{t_0} \dot{x}(t) \\
    \Rightarrow \frac{d \dot{x}(t)}{d t} & = J_f^{t_0} \dot{x}(t) \\
    \Rightarrow \frac{d \dot{x}(t)}{\dot{x}(t)} & = J_f^{t_0} d t
\end{align}
$$

whose solution for $\dot{x}(t)$ over the interval $[t_0, t_0 + \tau)$ with $\tau \in [0, \Delta t)$ will be
$$
\begin{align}
    \int_{t_0}^{t_0+\Delta t} \frac{d \dot{x}(t)}{\dot{x}(t)} & = \int_{t_0}^{t_0+\Delta t} J_f^{t_0} \, d t \\
    \Rightarrow \ln \dot{x}(t_0+\Delta t) - \ln \dot{x}(t_0) & = J_f^{t_0} \Delta t \\
    \Rightarrow \ln \dot{x}(t_0+\Delta t) & = J_f^{t_0} \Delta t + \ln \dot{x}(t_0) \\
    \Rightarrow \dot{x}(t_0+\Delta t) & = \exp^{J_f^{t_0} \Delta t} \dot{x}(t_0)
\end{align}
$$

while for $x(t)$, over the interval of $\tau \in [0, \Delta t)$, we will have
$$
\begin{align}
    \int_{0}^{\Delta t} \dot{x}(t_0+\tau) \, d \tau & = \int_{0}^{\tau} \exp^{J_f^{t_0} \tau} \dot{x}(t_0) \, d \tau \\
    \Rightarrow x(t_0+\Delta t) - x(t_0) & = \dot{x}(t_0) {J_f^{t_0}}^{-1} \int_{0}^{\tau} J_f^{t_0} \exp^{J_f^{t_0} \tau} \, d \tau \\
    \Rightarrow x(t_0+\Delta t) & = x(t_0) + {J_f^{t_0}}^{-1} \left( \exp^{J_f^{t_0} \Delta t} - \exp^{J_f^{t_0} 0 } \right) \dot{x}(t_0) \\
    \Rightarrow x(t_0+\Delta t) & = x(t_0) + {J_f^{t_0}}^{-1} \left( \exp^{J_f^{t_0} \Delta t} - I \right) \dot{x}(t_0) \\
    \Rightarrow x(t_0+\Delta t) & = x(t_0) + {J_f^{t_0}}^{-1} \left( \exp^{J_f^{t_0} \Delta t} - I \right) f(x(t_0)) \\
\end{align}
$$

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

## Welcomed contributions

#### Nonlinear functions
The number one priority, at the moment, is a full treatment for nonlinear functions. This should unlock somewhat basic applications of the package to robotics and control theoretic problems.

As suggested by Conor Heins, the key might to move to a fully matrix update of every equation, i.e., instead of

xdot = A x + B u

xdot = C v with, C = [[A, 0], [0, B]] and v = [[x],[u]]