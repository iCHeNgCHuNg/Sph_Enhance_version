import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

"""
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton University, @PMocz

Simulate the structure of a star with SPH
"""


def W(x, y, z, h):
    """
    Gausssian Smoothing kernel (3D)
        x     is a vector/matrix of x positions
        y     is a vector/matrix of y positions
        z     is a vector/matrix of z positions
        h     is the smoothing length
        w     is the evaluated smoothing function
    """

    r = np.sqrt(x**2 + y**2 + z**2)

    w = (1.0 / (h * np.sqrt(np.pi))) ** 3 * np.exp(-(r**2) / h**2)

    return w


def gradW(x, y, z, h):
    """
    Gradient of the Gausssian Smoothing kernel (3D)
    x     is a vector/matrix of x positions
    y     is a vector/matrix of y positions
    z     is a vector/matrix of z positions
    h     is the smoothing length
    wx, wy, wz     is the evaluated gradient
    """

    r = np.sqrt(x**2 + y**2 + z**2)

    n = -2 * np.exp(-(r**2) / h**2) / h**5 / (np.pi) ** (3 / 2)
    wx = n * x
    wy = n * y
    wz = n * z

    return wx, wy, wz


def getPairwiseSeparations(ri, rj):
    """
    Get pairwise desprations between 2 sets of coordinates
    ri    is an M x 3 matrix of positions
    rj    is an N x 3 matrix of positions
    dx, dy, dz   are M x N matrices of separations
    """

    M = ri.shape[0]
    N = rj.shape[0]

    # positions ri = (x,y,z)
    rix = ri[:, 0].reshape((M, 1))
    riy = ri[:, 1].reshape((M, 1))
    riz = ri[:, 2].reshape((M, 1))

    # other set of points positions rj = (x,y,z)
    rjx = rj[:, 0].reshape((N, 1))
    rjy = rj[:, 1].reshape((N, 1))
    rjz = rj[:, 2].reshape((N, 1))

    # matrices that store all pairwise particle separations: r_i - r_j
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T

    return dx, dy, dz


def getDensity(r, pos, m, h):
    """
    Get Density at sampling locations from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of densities
    """

    M = r.shape[0]

    dx, dy, dz = getPairwiseSeparations(r, pos)
    rho = np.sum(m * W(dx, dy, dz, h), 1).reshape((M, 1))

    return rho


def getPressure(rho, k, n):
    """
    Equation of State
    rho   vector of densities
    k     equation of state constant
    n     polytropic index
    P     pressure
    """

    P = k * rho ** (1 + 1 / n)

    return P


def getAcc(pos, vel, m, h, k, n, nu, g, Gx, sigma):
    """
    Calculate the acceleration on each SPH particle
    pos   is an N x 3 matrix of positions
    vel   is an N x 3 matrix of velocities
    m     is the particle mass
    h     is the smoothing length
    k     equation of state constant
    n     polytropic index
    lmbda external force constant
    nu    viscosity
    a     is N x 3 matrix of accelerations
    """

    N = pos.shape[0]

    # Calculate densities at the position of the particles
    rho = getDensity(pos, pos, m, h)

    # Get the pressures
    P = getPressure(rho, k, n)

    # Get pairwise distances and gradients
    dx, dy, dz = getPairwiseSeparations(pos, pos)
    dWx, dWy, dWz = gradW(dx, dy, dz, h)

    # Add Pressure contribution to accelerations
    ax = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWx, 1).reshape((N, 1))
    ay = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWy, 1).reshape((N, 1))
    az = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWz, 1).reshape((N, 1))

    # pack together the acceleration components
    a = np.hstack((ax, ay, az))

    # Add external potential force
    #a -= lmbda * pos

    # gravity
    a[:, 1] -= g

    # pressure gradient (driving force)
    a[:, 0] += Gx

    # surface tension (simple cohesive force)
    if sigma > 0:
        dx, dy, dz = getPairwiseSeparations(pos, pos)
        r  = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-12

        mask = r < h
        F = -sigma * (h - r) / h
        F = F * mask

        ax_st = np.sum(F * dx / r, axis=1).reshape((N, 1))
        ay_st = np.sum(F * dy / r, axis=1).reshape((N, 1))
        az_st = np.sum(F * dz / r, axis=1).reshape((N, 1))

        a += np.hstack((ax_st, ay_st, az_st))

    # viscosity (damping)
    a -= nu * vel

    return a



def main():
    """SPH simulation"""

    # Simulation parameters
    N = 400  # Number of particles
    t = 0  # current time of the simulation
    tEnd = 12  # time at which simulation ends
    dt = 0.04  # timestep
    M = 2  # star mass
    R = 0.75  # star radius
    k = 0.1  # equation of state constant
    n = 1  # polytropic index
    plotRealTime = True  # switch on for plotting as the simulation goes along
    # ===== control parameters =====
    g = 1.5          # gravity
    Gx = 0.9         # pressure gradient
    h = 0.12         # smoothing length
    nu = 0.2         # viscosity
    sigma = 0.0      # surface tension strength
    # ==============================

    # Generate Initial Conditions
    np.random.seed(42)  # set the random number generator seed

    lmbda = (
        2
        * k
        * (1 + n)
        * np.pi ** (-3 / (2 * n))
        * (M * gamma(5 / 2 + n) / R**3 / gamma(1 + n)) ** (1 / n)
        / R**2
    )  # ~ 2.01
    m = M / N  # single particle mass
    # rectangular domain initial condition
    Lx = 2.0
    Ly = 1.0
    # ===== channel flow velocity profile setup =====
    nbin = 25
    vmean_list = []
    t_list = []

    bins = np.linspace(-Ly/2, Ly/2, nbin + 1)
    y_center = 0.5 * (bins[:-1] + bins[1:])

    vx_accum = np.zeros(nbin)
    nsample = 0

    pos = np.zeros((N, 3))
    pos[:, 0] = np.random.uniform(-Lx/2, Lx/2, N)
    pos[:, 1] = np.random.uniform(-Ly/2, Ly/2, N)
    pos[:, 2] = 0.0

    vel = np.zeros(pos.shape)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, vel, m, h, k, n, nu, g, Gx, sigma)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # prep figure
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])
    

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt / 2

        # drift
        pos += vel * dt
        # simple box boundary
        for d, (xmin, xmax) in enumerate([(-Lx/2, Lx/2), (-Ly/2, Ly/2)]):
            mask_low = pos[:, d] < xmin
            mask_high = pos[:, d] > xmax
            vel[mask_low | mask_high, d] *= -0.5
            pos[mask_low, d] = xmin
            pos[mask_high, d] = xmax

        # update accelerations
        acc = getAcc(pos, vel, m, h, k, n, nu, g, Gx, sigma)

        # (1/2) kick
        vel += acc * dt / 2

        # update time
        t += dt
        vmean_list.append(np.mean(vel[:, 0]))
        t_list.append(t)

        # ===== accumulate steady velocity profile =====
        if i > Nt // 2:   # 只算後半段已經穩態的流
            idx = np.digitize(pos[:, 1], bins) - 1
            vx_tmp = np.zeros(nbin)

            for b in range(nbin):
                mask = idx == b
                if np.any(mask):
                    vx_tmp[b] = np.mean(vel[mask, 0])

            vx_accum += vx_tmp
            nsample += 1

        # get density for plotting
        rho = getDensity(pos, pos, m, h)

        # plot in real time
        if plotRealTime or (i == Nt - 1):
            plt.sca(ax1)
            plt.cla()
            cval = np.minimum((rho - 3) / 3, 1).flatten()
            plt.scatter(
                pos[:, 0], pos[:, 1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5
            )
            ax1.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
            ax1.set_aspect("equal", "box")
            ax1.set_xticks([-1, 0, 1])
            ax1.set_yticks([-1, 0, 1])
            ax1.set_facecolor("black")
            ax1.set_facecolor((0.1, 0.1, 0.1))

            plt.sca(ax2)
            plt.cla()

            plt.plot(t_list, vmean_list, linewidth=2)
            plt.xlabel("time")
            plt.ylabel("mean v_x")


            plt.pause(0.001)

    # add labels/legend
    plt.sca(ax2)
    plt.xlabel("time")
    plt.ylabel("mean v_x")

    # Save figure
    plt.savefig("sph.png", dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
