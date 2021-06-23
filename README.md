# Generalised Filtering in Pytorch
General framework for active Bayesian inversion of continuous hierarchical dynamic models (HDM) for inference, see e.g.,
- [Friston, Karl. "Hierarchical models in the brain." PLoS Comput Biol 4.11 (2008)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000211),
- [Roweis, Sam, and Zoubin Ghahramani. "A unifying review of linear Gaussian models." Neural computation 11.2 (1999)](https://direct.mit.edu/neco/article/11/2/305/6249/A-Unifying-Review-of-Linear-Gaussian-Models),

and control
- [Friston, Karl J., et al. "Action and behavior: a free-energy formulation." Biological cybernetics 102.3 (2010)](https://link.springer.com/article/10.1007/s00422-010-0364-z)
- [Kappen, Hilbert J., Vicenç Gómez, and Manfred Opper. "Optimal control as a graphical model inference problem." Machine learning 87.2 (2012)](https://link.springer.com/article/10.1007/s10994-012-5278-7)

with the particular structure advocated by active inference models, compared and discussed in more detail in
- [Baltieri, Manuel, and Christopher L. Buckley. "On Kalman-Bucy filters, linear quadratic control and active inference." arXiv preprint arXiv:2005.06269 (2020)](https://arxiv.org/abs/2005.06269) (for continuous time models), and
- [Millidge, Beren, et al. "On the Relationship Between Active Inference and Control as Inference." International Workshop on Active Inference. Springer, Cham, 2020.](https://link.springer.com/chapter/10.1007/978-3-030-64919-7_1) (for discrete time models).

The present implementation further follows the relaxed assumptions driving to the the development of [Friston, Karl, et al. "Generalised filtering." Mathematical Problems in Engineering (2010)](https://www.hindawi.com/journals/mpe/2010/621670/). The framework is also heavily inspired by spm/DEM (see [Statistical Parametric Mapping](https://www.fil.ion.ucl.ac.uk/spm/doc/)).

## Features

### GPU support
The library is currently developed using pytorch, with the option of running large-scale simulations on GPUs. Pytorch was the first, and one if not the more natural choice at the beginning of this package's development due to the flexibility of dynamic computational graphs (the first idea to build this library and subsequent prototype came about before Tensorflow 2.0 was released). 

The natural support for automated differentiation was also one of the main reasons to try and provide an alternative to the original code in spm/DEM which, to date, relies on numerical differentiation. Over time however, it has become increasingly clear that the use of specialised methods for numerical integration of SDEs, i.e., Local Linearisation (see next section), is hardly made any easier by pytorch and its automated differentiation system (at least up to March 2021 when this file was last updated) due to the strong reliance of the method here adopted on derivatives of vector functions (i.e., Jacobians rather than gradients). Due to the performance of the LL algorithm however, it is hard to imagine switching to different integration schemes and it is then perhaps worth investigating if moving to a different backend (JAX?) is a better option.

### Arbitrary embedding orders for non-Markovian continuous-time processes
The treatment of non-Markovian stochastic processes is swiftly handled in discrete time via 'state augmentation', a technique that allows the conversion of non-Markovian variables, or rather Markov of order n (i.e., with non-zero autocorrelation), to Markovian ones, Markov of order 1, by augmenting the dimension of the state space. In continuous time however, this state augmentation technique can be more problematic to implement since an infinite of extra orders might have to be added for continuous autocorrelations, so approximations are usually in place, i.e., the number of extra states is truncated. A formulation of some of these issues is provided in [Friston, Karl. "Hierarchical models in the brain." PLoS Comput Biol 4.11 (2008)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000211), and a comparison between state augmentation analogous to the discrete time case and a (linearised) Taylor expansion of non-Markovian varibles is provided in the section 'State space models in generalised coordinates of motion' (here 'generalised coordinates of motion' := embedding orders). This follows classical treatments of continuous-time stochastic processes (in both time and frequency domains) found, for example, in [Cox, David Roxbee, and Hilton David Miller. The theory of stochastic processes. Vol. 134. CRC press, 1977](https://books.google.co.jp/books?hl=ja&lr=lang_ja%7Clang_en&id=NeR5JEunGYwC&oi=fnd&pg=PR9&dq=stochastic+processes+miller&ots=VdIlQzq9EE&sig=FU5P5KOWrZ9wVjikqEzlaYErEfM&redir_esc=y#v=onepage&q=stochastic%20processes%20miller&f=false).

### Effective numerical integration
Inspired by [Friston, Karl. "Hierarchical models in the brain." PLoS Comput Biol 4.11 (2008)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000211), the numerical integration of stochastic differential equations is carried out using a method first proposed by [Ozaki, Tohru. "A bridge between nonlinear time series models and nonlinear stochastic dynamical systems: a local linearization approach." Statistica Sinica (1992)](https://www.jstor.org/stable/24304122) (see also [Jimenez, J. C., I. Shoji, and T. Ozaki. "Simulation of stochastic differential equations through the local linearization method. A comparative study." Journal of Statistical Physics 94.3 (1999): 587-602.](https://link.springer.com/article/10.1023/A:1004504506041)). This method was originally devised as bridge between (discrete-time) stochastic difference equations and (continuous-time) stochastic differential equations, but has later gained a certain amount of success also as an integration scheme for the problemic nature of SDEs simulations in continuous time ([Kloeden, Peter Eris, Eckhard Platen, and Henri Schurz. Numerical solution of SDE through computer experiments. Springer Science & Business Media, 2012](https://books.google.co.jp/books?hl=ja&lr=lang_ja%7Clang_en&id=DOIRBwAAQBAJ&oi=fnd&pg=PR7&dq=Numerical+solution+of+SDE+through+computer+experiments&ots=tHscaPc649&sig=FycRXeMhlP3VZoeUBsqHXm7Zn0s&redir_esc=y#v=onepage&q=Numerical%20solution%20of%20SDE%20through%20computer%20experiments&f=false)). 

Local linearisation (LL) provides a robust integration scheme that allows to simulate continuous-time SDEs using an approximation based on the Jacobian of the integrand between pairs of time steps (cf. a constant used in the more basic Euler-Maruyama). Some references containing more details:
- [Hurn, A. Stan, J. I. Jeisman, and Kenneth A. Lindsay. "Seeing the wood for the trees: A critical evaluation of methods to estimate the parameters of stochastic differential equations." Journal of Financial Econometrics 5.3 (2007)](https://academic.oup.com/jfec/article-abstract/5/3/390/805841)
- [Iacus, Stefano M. Simulation and inference for stochastic differential equations: with R examples. Springer Science & Business Media, 2009](https://books.google.co.jp/books?hl=ja&lr=lang_ja%7Clang_en&id=ryCMlNVV8EAC&oi=fnd&pg=PR7&dq=Simulation+and+Inference+for+Stochastic+Differential+Equations&ots=yMTruIbG5f&sig=Z9W163fJOMRzvkpfERQgAH9UGvs&redir_esc=y#v=onepage&q=Simulation%20and%20Inference%20for%20Stochastic%20Differential%20Equations&f=false)
- [Gloaguen, Pierre, Marie‐Pierre Etienne, and Sylvain Le Corff. "Stochastic differential equation based on a multimodal potential to model movement data in ecology." Journal of the Royal Statistical Society: Series C (Applied Statistics) 67.3 (2018)](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssc.12251)
- [Oksendal, Bernt. Stochastic differential equations: an introduction with applications. Springer Science & Business Media, 2013](https://books.google.co.jp/books?hl=ja&lr=lang_ja%7Clang_en&id=gizqCAAAQBAJ&oi=fnd&pg=PA1&dq=Stochastic+Differential+Equations:+An+Introduction+with+Applications+in+Population+Dynamics+Modeling&ots=3lub6KC-BJ&sig=MhFbpZAD1I-N_7mG-tOcVHZWUks&redir_esc=y#v=onepage&q&f=false)
- [Särkkä, Simo, and Arno Solin. Applied stochastic differential equations. Vol. 10. Cambridge University Press, 2019](https://users.aalto.fi/~ssarkka/pub/sde_book.pdf)

The method is extremely effective, but should not be seen as a 'free lunch', since the main idea is to simply discard the continuous time formulation in favour of a discrete one with *arbitrarily small* step, but crucially not a step that tends (in the limit) to zero so to recover the continuous time limit. If properties from the continuous time formulation are required (for example for some analytical calculation), this implementation will obviously just provide a perhaps useful approximation.

<!-- #### Local linearisation
To gain an undestanding of the basic implementation of LL, we first look at its application on an ODE example (i.e., without noise), see also sec. 9.2.4 in Ozaki, Tohru. Time series modeling of neuroscience data. CRC press, 2012. (the generalisation to SDEs is rather straightforward and can be found in section 9.2.5, while in section 9.2.6 we find an application to SDEs with driving inputs):

$$
\dot{x}(t) = \frac{d x(t)}{d t} = f(x(t))
$$

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
$$ -->

## Current limitations
The current implementation is so far limited by:
- the presence of inference + control, but no learning (:= inference on a slower time scale) for slowly changing (hyper)parameters
- a single layer in terms of hierarchical structures
- state space models treatment of only linear systems.

### TODO:
- fix the problems introduced with Pytorch 1.9.0 to Autograd's 'jacobian' (version rolled back to 1.7.0 without problems for now)
- implement full support to nonlinear functions (i.e., transition and observation laws in a state-space model)
- implement hierarchical structures (i.e., stacking layers)
- implement learning (update of parameters) and attention (update of hyperparameters)
- implement tests
- consider moving everything to JAX if the code remains heavily reliant on jacobian/hessian functions
- optimise code and restructure it as needed

### TODO (one day...):
- provide some interface with torch.nn so to exploit the full potential of pytorch's neural networks in HDMs
- look into other SDE solvers (although LL might be a fantastic choice actually, moving everything to discrete time with time increments whose intepretation is arbitrarily small but with a finite implementation); for example see ['torchsde'](https://github.com/google-research/torchsde)

## Welcomed contributions
Everything in the list above + use cases + things not in the above list, but especially:
### Nonlinear functions
The number one priority, at the moment, is a full treatment for nonlinear functions. This should unlock somewhat widespread applications of the package to robotics and control theoretic problems.

The current issues are:
1. how should functions be provided by the user? 
	* as arguments to a predefined function (as lambda functions? explicit ones?)
	* as templates to be filled in within a file that has to be imported by the current classes
	* as some symbolic expression to be decoded
2. the current structure is limiting and will only allow (nonlinear) functions in the form f(x,u,a) = f(x) + f(u) + f(a) because of the computation of the Jacobians in the Local Linearisation step (what should J_x, J_u, J_a actually be multiplied by?)
3. the speed for differentiating these functions is not great at the moment, and it gets worse for an increasing number of iterations (is the computational graph growing for some reason? should some operations be within torch.no_grad()? ...)

As suggested by Conor Heins @conorheins, the key might to move to a fully matrix update of every equation, i.e., instead of

xdot = A x + B u

vdot = C v with, C = [[A, 0], [0, B]] and v = [[x],[u]]