"""
Linear MPC with OSQP — time-varying reference tracking
====================================================================

Tracks a *moving* set-point target with a discrete-time linear system using
Model Predictive Control, solved with the OSQP quadratic-programming solver.

OSQP solves the standard form

        minimize     (1/2) z^T P z + q^T z
        subject to        l <= G z <= u

At every control step the reference is known over the whole prediction
horizon, so the MPC cost penalises each predicted state against the reference
at the corresponding future time:

        min   sum_{i=0}^{N-1} (x_i - r_{k+i})^T Q (x_i - r_{k+i}) + u_i^T R u_i
              + (x_N - r_{k+N})^T Qf (x_N - r_{k+N})
        s.t.  x_{i+1} = A x_i + B u_i,      i = 0 ... N-1   (dynamics)
              x_0     = x                                   (current state)
              x_min <= x_i <= x_max                         (state bounds)
              u_min <= u_i <= u_max                         (input bounds)

Decision vector: z = [ x_0, ..., x_N, u_0, ..., u_{N-1} ].

Because the controller sees the reference across the whole horizon, it
*anticipates* upcoming set-point changes and starts moving before they happen.

OSQP features exercised here:
  * standard-form translation (P, q, G, l, u as sparse CSC matrices);
  * `prob.setup(...)` with explicit solver settings;
  * status checking via `res.info.status`;
  * IN-PLACE WARM-STARTED updates with `prob.update(...)` every step:
        - l[:nx], u[:nx]  <- the new measured state  (x_0 = x);
        - q               <- the reference shifted one step along the horizon.
    The matrix data (P, G) never changes, so the factorisation is reused and
    each re-solve warm-starts from the previous solution.

Requires: pip install osqp scipy numpy matplotlib
"""

import numpy as np
import scipy.sparse as sparse
import osqp

# ===========================================================================
# 1. Discrete-time linear plant:  x_{k+1} = A x_k + B u_k
#    Double integrator: state = [position, velocity], input = acceleration.
# ===========================================================================
dt = 0.1
A = np.array([[1.0, dt],
              [0.0, 1.0]])
B = np.array([[0.5 * dt**2],
              [dt]])
nx, nu = B.shape

# ===========================================================================
# 2. Time-varying reference  r(t) = [position, velocity]
#    A sequence of position set-points (velocity target 0) that steps every
#    T_STEP seconds. The controller must move to each new level and settle.
# ===========================================================================
LEVELS = [0.0, 0.7, -0.5, 0.4, -0.8, 0.6, -0.3, 0.0]
T_STEP = 3.0                             # seconds between set-point changes


def reference(t):
    """Reference state [position, velocity] at time t."""
    level = LEVELS[int(t // T_STEP) % len(LEVELS)]
    return np.array([level, 0.0])


# ===========================================================================
# 3. MPC weights, horizon and constraints
# ===========================================================================
N = 40                                  # prediction horizon (sees ~1 step ahead)
Q = sparse.diags([10.0, 0.5])           # position / velocity error weight
R = sparse.diags([1.0])                 # input weight (smooth, gentle moves)
Qf = sparse.diags([10.0, 0.5])          # terminal-state weight

umin, umax = np.array([-1.0]), np.array([1.0])                   # |accel| <= 1
xmin, xmax = np.array([-np.inf, -2.0]), np.array([np.inf, 2.0])  # |vel| <= 2

n_var = (N + 1) * nx + N * nu           # total decision variables in z
u_start = (N + 1) * nx                  # index in z where the inputs begin

x0 = np.array([0.0, 0.0])               # start at rest at the origin

# ===========================================================================
# 4. Translate the MPC problem into OSQP standard form  (P, q, G, l, u)
# ===========================================================================

# --- Quadratic cost  P  (block-diagonal; independent of the reference) ----
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), Qf,
                       sparse.kron(sparse.eye(N), R)], format="csc")


def build_q(t_now):
    """Linear cost term q for the reference window starting at t_now.

    Predicted state x_i is penalised against r(t_now + i*dt); the per-state
    block of q is -Q @ r_i  (and -Qf @ r_N for the terminal state).
    """
    refs = np.array([reference(t_now + i * dt) for i in range(N + 1)])  # (N+1, nx)
    qx = -(Q @ refs[:N].T).T.ravel()        # blocks for x_0 ... x_{N-1}
    qxN = -(Qf @ refs[N])                   # block for x_N
    return np.hstack([qx, qxN, np.zeros(N * nu)])


q = build_q(0.0)

# --- Equality constraints: dynamics + initial condition  (l = u) ---------
Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx)) \
   + sparse.kron(sparse.eye(N + 1, k=-1), A)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), B)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N * nx)])
ueq = leq.copy()

# --- Inequality constraints: state & input box bounds --------------------
Aineq = sparse.eye(n_var)
lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])

# --- Stack into  l <= G z <= u -------------------------------------------
G = sparse.vstack([Aeq, Aineq], format="csc")
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# ===========================================================================
# 5. Configure and set up the OSQP solver
# ===========================================================================
prob = osqp.OSQP()
prob.setup(P, q, G, l, u,
           warm_start=True, eps_abs=1e-5, eps_rel=1e-5,
           max_iter=4000, verbose=False)

# ===========================================================================
# 6. Receding-horizon loop
#    Each step: update q (reference) and the initial-state bounds, solve,
#    apply u_0, advance the plant.
# ===========================================================================
nsim = 200                               # 20 s of simulation
x = x0.copy()
state_hist, input_hist, ref_hist = [x.copy()], [], [reference(0.0)]
iters, solve_times, statuses = [], [], set()

for k in range(nsim):
    t_now = k * dt

    # --- update the parts of the QP that change, then warm-start ---------
    q = build_q(t_now)
    l[:nx] = -x
    u[:nx] = -x
    prob.update(q=q, l=l, u=u)

    res = prob.solve()
    statuses.add(res.info.status)
    if res.info.status != "solved":
        raise RuntimeError(f"OSQP did not solve at step {k}: {res.info.status}")
    iters.append(res.info.iter)
    solve_times.append(res.info.solve_time)

    u0 = res.x[u_start:u_start + nu]            # first input of the sequence
    x = A.dot(x) + B.dot(u0)                    # apply input, advance plant

    input_hist.append(u0.copy())
    state_hist.append(x.copy())
    ref_hist.append(reference((k + 1) * dt))

state_hist = np.array(state_hist)
input_hist = np.array(input_hist)
ref_hist = np.array(ref_hist)

# ===========================================================================
# 7. Diagnostics
# ===========================================================================
pos_rms = np.sqrt(np.mean((state_hist[:, 0] - ref_hist[:, 0]) ** 2))
print("OSQP / MPC tracking summary")
print("-" * 50)
print(f"control steps            : {nsim}")
print(f"decision variables       : {n_var}")
print(f"constraints (rows of G)  : {G.shape[0]}")
print(f"solve statuses           : {statuses}")
print(f"position tracking RMS    : {pos_rms:.3e}")
print(f"peak |input|             : {np.max(np.abs(input_hist)):.3f}  (limit 1.0)")
print(f"first-solve iterations   : {iters[0]}")
print(f"warm-started avg iters   : {np.mean(iters[1:]):.1f}")
print(f"mean / max solve time    : {1e3*np.mean(solve_times):.3f} / "
      f"{1e3*np.max(solve_times):.3f} ms")


# ===========================================================================
# 8. Plot
# ===========================================================================
def plot_results(t, states, inputs, refs, vel_lim, in_lim, save_path):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#444", "axes.linewidth": 1.0, "axes.grid": True,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.9, "font.size": 11,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "legend.frameon": False,
    })
    c_act, c_ref, c_in, c_err, c_lim = "#0b6fb8", "#e4572e", "#1b998b", "#7b4bc4", "#e4572e"

    fig, ax = plt.subplots(4, 1, figsize=(8.6, 10), sharex=True,
                           gridspec_kw={"height_ratios": [1.3, 1, 1, 0.9]})

    def despine(a):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    # --- position ---
    ax[0].plot(t, refs[:, 0], ls="--", lw=2.2, color=c_ref, label="reference")
    ax[0].plot(t, states[:, 0], lw=2.4, color=c_act, label="actual", alpha=0.9)
    ax[0].fill_between(t, states[:, 0], refs[:, 0], color=c_act, alpha=0.10)
    ax[0].set_ylabel("position")
    ax[0].set_title("MPC tracking of a time-varying reference (OSQP)")
    ax[0].legend(loc="upper right", ncol=2)

    # --- velocity (with limit band) ---
    ax[1].axhspan(-vel_lim, vel_lim, color=c_lim, alpha=0.05)
    for s in (vel_lim, -vel_lim):
        ax[1].axhline(s, ls=":", lw=1.2, color=c_lim)
    ax[1].plot(t, states[:, 1], lw=2.2, color=c_act, alpha=0.9)
    ax[1].set_ylabel("velocity")
    ax[1].text(0.01, 0.92, "velocity limits", transform=ax[1].transAxes,
               color=c_lim, fontsize=9, va="top")

    # --- input (with limit band) ---
    ax[2].axhspan(-in_lim, in_lim, color=c_lim, alpha=0.05)
    for s in (in_lim, -in_lim):
        ax[2].axhline(s, ls=":", lw=1.2, color=c_lim)
    ax[2].step(t[:-1], inputs[:, 0], where="post", lw=2.0, color=c_in)
    ax[2].set_ylabel("input (accel.)")
    ax[2].text(0.01, 0.92, "input limits", transform=ax[2].transAxes,
               color=c_lim, fontsize=9, va="top")

    # --- tracking error ---
    err = states[:, 0] - refs[:, 0]
    rms = np.sqrt(np.mean(err ** 2))
    ax[3].axhline(0, color="#999", lw=1)
    ax[3].plot(t, err, lw=1.8, color=c_err)
    ax[3].fill_between(t, err, 0, color=c_err, alpha=0.15)
    ax[3].set_ylabel("position error")
    ax[3].set_xlabel("time [s]")
    ax[3].text(0.99, 0.06, f"position RMS = {rms:.2e}", transform=ax[3].transAxes,
               ha="right", va="bottom", fontsize=10, color=c_err,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c_err, lw=1))

    for a in ax:
        despine(a)
    fig.align_ylabels(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    t = np.arange(nsim + 1) * dt
    try:
        plot_results(t, state_hist, input_hist, ref_hist,
                     vel_lim=2.0, in_lim=1.0, save_path="mpc_tracking.png")
        import matplotlib.pyplot as plt
        plt.show()
    except ImportError:
        print("\nmatplotlib not installed; skipping plot.")
