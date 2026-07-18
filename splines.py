import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.interpolate import CubicSpline
import time

# ==========================================================================
# 1. PARÁMETROS
# ==========================================================================
h_grueso = 5.0
Nx_g, Ny_g = 81, 9                       # malla gruesa: 81 x 9 nodos
h_fino = 1.0                             # resolución real del canal
Lx, Ly = (Nx_g - 1) * h_grueso, (Ny_g - 1) * h_grueso   # 400 x 40 (igual en ambas mallas)

U0 = 1.0
nu = 1.5                                 # viscosidad cinemática (controla la vorticidad)
Re = U0 * Ly / nu

print("=" * 70)
print("  Canal con obstáculos de tamaño real - vorticidad + recuperación con spline")
print("=" * 70)
print(f"  Dominio real:  {Lx:.0f} x {Ly:.0f}")
print(f"  Malla gruesa (cálculo):     {Nx_g} x {Ny_g}  (h = {h_grueso})")
print(f"  Malla fina (recuperación):  {int(Lx/h_fino)+1} x {int(Ly/h_fino)+1}  (h = {h_fino})")
print(f"  Re = U0*Ly/nu = {Re:.1f}")
print("=" * 70)


# ==========================================================================
# 2. SOLVER GENÉRICO: FUNCIÓN DE CORRIENTE - VORTICIDAD
# ==========================================================================
def resolver_vorticidad(Nx, Ny, h, U0, nu, dt, max_iter, tol):
    i_idx, j_idx = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    X, Y = i_idx * h, j_idx * h

    # --- obstáculos de tamaño REAL (Newton-Raphson) ---
    obstaculo1 = (X <= 100.0) & (Y >= 25.0)          # x:[0,100]   y:[25,40]
    obstaculo2 = (X >= 150.0) & (X <= 250.0) & (Y <= 15.0)  # x:[150,250] y:[0,15]
    solid = obstaculo1 | obstaculo2

    y_open = np.argmax(obstaculo1[0, :]) * h if obstaculo1[0, :].any() else Ly
    Psi_top, Psi_bottom = U0 * y_open, 0.0

    # --- condiciones de frontera para psi ---
    category = np.full((Nx, Ny), "interior", dtype=object)
    dir_val = np.zeros((Nx, Ny))
    category[Nx - 1, :] = "neumann"
    category[0, :] = "dirichlet"; dir_val[0, :] = U0 * (j_idx[0, :] * h)
    category[obstaculo1] = "dirichlet"; dir_val[obstaculo1] = Psi_top
    category[obstaculo2] = "dirichlet"; dir_val[obstaculo2] = Psi_bottom
    category[:, Ny - 1] = "dirichlet"; dir_val[:, Ny - 1] = Psi_top
    category[:, 0] = "dirichlet"; dir_val[:, 0] = Psi_bottom

    # --- matriz de Poisson (se arma y factoriza una sola vez) ---
    idx = i_idx * Ny + j_idx
    N = Nx * Ny
    rows, cols, data = [], [], []
    b_fixed = np.zeros(N)

    dmask = category == "dirichlet"
    dnodes = idx[dmask]
    rows += list(dnodes); cols += list(dnodes); data += [1.0] * len(dnodes)
    b_fixed[dnodes] = dir_val[dmask]

    nmask = category == "neumann"
    nnodes = idx[nmask]
    wnodes = idx[Nx - 2, j_idx[nmask]]
    rows += list(nnodes); cols += list(nnodes); data += [1.0] * len(nnodes)
    rows += list(nnodes); cols += list(wnodes); data += [-1.0] * len(nnodes)

    imask = category == "interior"
    ii, jj = np.where(imask)
    node = idx[ii, jj]
    E, W = idx[ii + 1, jj], idx[ii - 1, jj]
    Nn, S = idx[ii, jj + 1], idx[ii, jj - 1]
    rows += list(node); cols += list(node); data += [-4.0] * len(node)
    rows += list(node); cols += list(E); data += [1.0] * len(node)
    rows += list(node); cols += list(W); data += [1.0] * len(node)
    rows += list(node); cols += list(Nn); data += [1.0] * len(node)
    rows += list(node); cols += list(S); data += [1.0] * len(node)

    A = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsc()
    lu = splu(A)

    psi = dir_val.copy()
    omega = np.zeros((Nx, Ny))

    wallish = solid.copy()
    wallish[:, 0] = True
    wallish[:, -1] = True
    evolve = (~wallish).copy()
    evolve[0, :] = False
    evolve[-1, :] = False

    north_fluid = np.zeros_like(wallish); north_fluid[:, :-1] = ~wallish[:, 1:]
    south_fluid = np.zeros_like(wallish); south_fluid[:, 1:] = ~wallish[:, :-1]
    east_fluid = np.zeros_like(wallish); east_fluid[:-1, :] = ~wallish[1:, :]
    west_fluid = np.zeros_like(wallish); west_fluid[1:, :] = ~wallish[:-1, :]

    t0 = time.time()
    for it in range(max_iter):
        u_full = np.zeros_like(psi); u_full[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * h)
        v_full = np.zeros_like(psi); v_full[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2 * h)

        domega_dx = np.zeros_like(omega); domega_dx[1:-1, :] = (omega[2:, :] - omega[:-2, :]) / (2 * h)
        domega_dy = np.zeros_like(omega); domega_dy[:, 1:-1] = (omega[:, 2:] - omega[:, :-2]) / (2 * h)
        lap_omega = np.zeros_like(omega)
        lap_omega[1:-1, 1:-1] = (
            omega[2:, 1:-1] + omega[:-2, 1:-1] + omega[1:-1, 2:] + omega[1:-1, :-2] - 4 * omega[1:-1, 1:-1]
        ) / h ** 2

        rhs = -(u_full * domega_dx + v_full * domega_dy) + nu * lap_omega
        omega_new = omega.copy()
        omega_new[evolve] = omega[evolve] + dt * rhs[evolve]

        psi_n = np.zeros_like(psi); psi_n[:, :-1] = psi[:, 1:]
        psi_s = np.zeros_like(psi); psi_s[:, 1:] = psi[:, :-1]
        psi_e = np.zeros_like(psi); psi_e[:-1, :] = psi[1:, :]
        psi_w = np.zeros_like(psi); psi_w[1:, :] = psi[:-1, :]

        term_n = 2 * (psi - psi_n) / h ** 2 * north_fluid
        term_s = 2 * (psi - psi_s) / h ** 2 * south_fluid
        term_e = 2 * (psi - psi_e) / h ** 2 * east_fluid
        term_w = 2 * (psi - psi_w) / h ** 2 * west_fluid
        cnt = north_fluid.astype(float) + south_fluid + east_fluid + west_fluid
        omega_wall = np.where(cnt > 0, (term_n + term_s + term_e + term_w) / np.maximum(cnt, 1), 0.0)
        omega_new[wallish] = omega_wall[wallish]

        omega_new[0, :][~wallish[0, :]] = 0.0
        omega_new[-1, :] = omega_new[-2, :]

        b = b_fixed.copy()
        b[node] = -omega_new[ii, jj] * h ** 2
        psi_new = lu.solve(b).reshape(Nx, Ny)

        diff = np.max(np.abs(psi_new - psi)) / (np.max(np.abs(psi_new)) + 1e-12)
        psi, omega = psi_new, omega_new
        if it % 200 == 0:
            print(f"    it={it:5d}  cambio relativo = {diff:.3e}")
        if diff < tol:
            print(f"    Convergió en {it} iteraciones ({time.time()-t0:.2f} s)")
            break

    return psi, omega, solid


# ==========================================================================
# 3. RESOLVER EN LA MALLA GRUESA (h = 5)
# ==========================================================================
print("\nResolviendo en la malla gruesa (Gauss-Seidel reemplazado por vorticidad-psi)...")
psi_grueso, omega_grueso, solid_grueso = resolver_vorticidad(
    Nx_g, Ny_g, h_grueso, U0=U0, nu=nu, dt=1.0, max_iter=4000, tol=1e-6
)

dpsidx_g, dpsidy_g = np.gradient(psi_grueso, h_grueso)
u_grueso, v_grueso = dpsidy_g.copy(), -dpsidx_g.copy()
u_grueso[solid_grueso] = 0.0
v_grueso[solid_grueso] = 0.0
speed_grueso = np.sqrt(u_grueso ** 2 + v_grueso ** 2)

# ==========================================================================
# 4. RECUPERACIÓN DEL TAMAÑO ORIGINAL CON SPLINE CÚBICO NATURAL
# ==========================================================================
print("\n" + "=" * 70)
print("  Recuperando tamaño original (h = 1) con Spline Cúbico Natural")
print("=" * 70)

x_grueso = np.arange(Nx_g) * h_grueso
y_grueso = np.arange(Ny_g) * h_grueso
x_fino = np.arange(0, Lx + 1, h_fino)
y_fino = np.arange(0, Ly + 1, h_fino)

# --- paso 1: spline natural a lo largo de x, para cada fila j ---
psi_paso_x = np.zeros((len(x_fino), Ny_g))
for j in range(Ny_g):
    spline_x = CubicSpline(x_grueso, psi_grueso[:, j], bc_type="natural")
    psi_paso_x[:, j] = spline_x(x_fino)

# --- paso 2: spline natural a lo largo de y ---
psi_fino = np.zeros((len(x_fino), len(y_fino)))
for i in range(len(x_fino)):
    spline_y = CubicSpline(y_grueso, psi_paso_x[i, :], bc_type="natural")
    psi_fino[i, :] = spline_y(y_fino)

# (opcional, solo para graficar) se recupera también la vorticidad con el
# mismo procedimiento, para poder mostrar el mapa de vorticidad en h = 1
omega_paso_x = np.zeros((len(x_fino), Ny_g))
for j in range(Ny_g):
    spline_x = CubicSpline(x_grueso, omega_grueso[:, j], bc_type="natural")
    omega_paso_x[:, j] = spline_x(x_fino)
omega_fino = np.zeros((len(x_fino), len(y_fino)))
for i in range(len(x_fino)):
    spline_y = CubicSpline(y_grueso, omega_paso_x[i, :], bc_type="natural")
    omega_fino[i, :] = spline_y(y_fino)

# --- velocidades en la malla fina, derivadas del psi YA interpolado ---
dpsidx, dpsidy = np.gradient(psi_fino, h_fino)
u_fino, v_fino = dpsidy.copy(), -dpsidx.copy()

# --- máscara de obstáculos en la malla fina (mismo tamaño real, sin reescalar) ---
X_fino, Y_fino = np.meshgrid(x_fino, y_fino, indexing="ij")
obstaculo1_fino = (X_fino <= 100.0) & (Y_fino >= 25.0)
obstaculo2_fino = (X_fino >= 150.0) & (X_fino <= 250.0) & (Y_fino <= 15.0)
solid_fino = obstaculo1_fino | obstaculo2_fino

u_fino[solid_fino] = 0.0
v_fino[solid_fino] = 0.0
speed_fino = np.sqrt(u_fino ** 2 + v_fino ** 2)

print(f"  Malla original recuperada: {len(x_fino)} x {len(y_fino)} puntos "
      f"({Lx:.0f} x {Ly:.0f} unidades, h = {h_fino})")
print("\nResultados en la malla de tamaño original (interpolada)")
print("  Velocidad mínima :", np.nanmin(speed_fino))
print("  Velocidad máxima :", np.nanmax(speed_fino))
print("  Velocidad media  :", np.nanmean(speed_fino[~solid_fino]))


# ==========================================================================
# 5. VISUALIZACIÓN
# ==========================================================================
def dibujar_obstaculos(ax):
    ax.add_patch(patches.Rectangle((0, 25), 100, 15, facecolor="black", zorder=5))
    ax.add_patch(patches.Rectangle((150, 0), 100, 15, facecolor="black", zorder=5))


# --- malla gruesa (h = 5), tal como salió del solver ---
fig1, ax1 = plt.subplots(figsize=(18, 4))
im1 = ax1.imshow(
    speed_grueso.T, origin="lower", extent=[0, Lx, 0, Ly],
    cmap="turbo", aspect="auto", interpolation="nearest", vmin=0, vmax=np.nanmax(speed_fino),
)
fig1.colorbar(im1, ax=ax1, label="Velocidad |u| [m/s]")
dibujar_obstaculos(ax1)
ax1.set_title(f"Malla gruesa discretizada (h = {h_grueso:.0f})")
ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
ax1.set_xlim(0, Lx); ax1.set_ylim(0, Ly)
fig1.tight_layout()

# --- malla fina recuperada (h = 1), con líneas de corriente ---
fig2, ax2 = plt.subplots(figsize=(18, 4))
im2 = ax2.imshow(
    speed_fino.T, origin="lower", extent=[0, Lx, 0, Ly],
    cmap="turbo", aspect="auto", interpolation="bilinear", vmin=0, vmax=np.nanmax(speed_fino),
)
fig2.colorbar(im2, ax=ax2, label="Velocidad |u| [m/s]")

u_plot = u_fino.T.copy(); v_plot = v_fino.T.copy()
u_plot[solid_fino.T] = np.nan
v_plot[solid_fino.T] = np.nan
ax2.streamplot(x_fino, y_fino, u_plot, v_plot, color="white", density=1.4, linewidth=0.6, arrowsize=0.8)

dibujar_obstaculos(ax2)
ax2.set_title(f"Tamaño original recuperado (h = {h_fino:.0f}) - Spline Cúbico Natural")
ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
ax2.set_xlim(0, Lx); ax2.set_ylim(0, Ly)
fig2.tight_layout()

# --- vorticidad: mismo rango de color en las dos mallas, para poder comparar ---
vlim = np.nanpercentile(np.abs(omega_fino[~solid_fino]), 98)

# malla gruesa (h = 5)
fig3, ax3 = plt.subplots(figsize=(18, 4))
im3 = ax3.imshow(
    omega_grueso.T, origin="lower", extent=[0, Lx, 0, Ly],
    cmap="RdBu_r", aspect="auto", interpolation="nearest", vmin=-vlim, vmax=vlim,
)
fig3.colorbar(im3, ax=ax3, label="Vorticidad omega [1/s]")
dibujar_obstaculos(ax3)
ax3.set_title(f"Campo de vorticidad - malla gruesa (h = {h_grueso:.0f})")
ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]")
ax3.set_xlim(0, Lx); ax3.set_ylim(0, Ly)
fig3.tight_layout()

# malla fina recuperada (h = 1)
fig4, ax4 = plt.subplots(figsize=(18, 4))
im4 = ax4.imshow(
    omega_fino.T, origin="lower", extent=[0, Lx, 0, Ly],
    cmap="RdBu_r", aspect="auto", interpolation="bilinear", vmin=-vlim, vmax=vlim,
)
fig4.colorbar(im4, ax=ax4, label="Vorticidad omega [1/s]")
dibujar_obstaculos(ax4)
ax4.set_title(f"Campo de vorticidad recuperado (h = {h_fino:.0f}) - Spline Cúbico Natural")
ax4.set_xlabel("x [m]"); ax4.set_ylabel("y [m]")
ax4.set_xlim(0, Lx); ax4.set_ylim(0, Ly)
fig4.tight_layout()

plt.show()
