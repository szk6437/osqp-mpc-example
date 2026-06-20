# Linear MPC Tracking with OSQP

This repository demonstrates a computationally efficient implementation of Linear Model Predictive Control (MPC) for time-varying reference tracking. The core objective is to showcase the translation of a receding-horizon optimal control problem into a standard quadratic programming (QP) form, solved via the **OSQP** (Operator Splitting Quadratic Program) algorithm.

## Overview

The controller anticipates upcoming set-point changes by evaluating a known reference trajectory over the prediction horizon. The implementation prioritizes numerical efficiency by leveraging sparse matrix operations and OSQP's warm-starting capabilities for high-frequency control loops.

## QP Formulation

OSQP requires the problem to be cast into the standard form:

$$\text{minimize} \quad \frac{1}{2} z^T P z + q^T z$$
$$\text{subject to} \quad l \le G z \le u$$

The MPC decision vector $z$ concatenates the predicted states and control inputs over the horizon $N$: 
$z = [x_0, \dots, x_N, u_0, \dots, u_{N-1}]^T$.

The code programmatically constructs the block-diagonal Hessian $P$, the dynamic constraint matrix $G$, and the boundary vectors using `scipy.sparse`. 

## OSQP Implementation Highlights

This implementation exercises several advanced features of the OSQP API essential for real-time control:

* **Sparse Matrix Mapping:** Formulation of $P$ and $G$ as sparse Compressed Sparse Column (CSC) matrices to minimize memory footprint and factorization time.
* **Receding Horizon Updates:** Dynamic updates to the linear cost vector $q$ at each time step to reflect the shifting reference trajectory.
* **In-Place Warm-Starting:** Reusing the initial matrix factorization. At each control step, `prob.update(q=q, l=l, u=u)` is called instead of redefining the problem, utilizing the previous solution to significantly reduce solver iterations.
* **Explicit Solver Configuration:** Direct tuning of absolute and relative tolerances (`eps_abs`, `eps_rel`) and maximum iterations to guarantee deterministic execution bounds.

## Performance Diagnostics

The receding-horizon loop generates detailed diagnostics to verify solver stability:
* Validation of `res.info.status` at every step.
* Tracking of first-solve iterations vs. warm-started iterations.
* Monitoring of mean and maximum solve times to ensure real-time feasibility.

## Requirements
* `numpy`
* `scipy`
* `osqp`
* `matplotlib` (for tracking diagnostics)
