# Numerical approximation of Dynkin games with asymmetric information

In this repository, I implement a numerical scheme that compute the value function of a Dynkin game $u(t,x,p)$, which satisfies to the following obstacle problem 

$$
\max\Big\\{\max\Big\\{ \min\Big\\{  -\partial_t u -\tfrac12\mathrm{Tr}[aa^\mathrm{T}D_x^2 u] - b\cdot D_x u, u - p^\mathrm{T} h \Big\\}, u - p^\mathrm{T} h\Big\\}, -\lambda(p,D_x^2 u)\Big\\}  = 0,\quad u(T,x,p) = p^\mathrm{T}g(x).
$$

We use a combination of 3 discretization methods. A probabilistic method for the time-discretization, the Quickhull algorithm for the computation of the convex envelope, and a Feedforward network for the evaluation of the value function on the state space. For a detailed description of the problem and and an extensive analysis, see 
- [Numerical approximation of Dynkin games with asymmetric information](https://doi.org/10.1137/23M1621216),
- [On Dynkin games with incomplete information](https://doi.org/10.1137/120891800)
