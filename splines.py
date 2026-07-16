import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import CubicSpline
from scipy.interpolate import RegularGridInterpolator

# ==========================================================================
# 1. PARÁMETROS GEOMÉTRICOS
#    (medidas correctas, tomadas del solucionador de Newton-Raphson)
# ==========================================================================
h_real = 5.0                      # tamaño de celda de la malla de cálculo
Nx = 81                           # nodos en x de la malla de cálculo
Ny = 9                            # nodos en y de la malla de cálculo
Lx = (Nx - 1) * h_real            # 400 -> longitud real del canal en x
Ly = (Ny - 1) * h_real            # 40  -> longitud real del canal en y

U0 = 1.0                          # velocidad de entrada
omega = 0.85                      # factor de relajación (SOR)
g = 0.3                           # coeficiente de convección simplificado
tol = 1e-6
max_iter = 10000

print("=" * 70)
print("  Recuperación de tamaño original - Spline Cúbico Natural")
print("=" * 70)
print(f"  Malla de cálculo (gruesa): {Nx} x {Ny} nodos, h = {h_real}")
print(f"  Dominio real del canal:    {Lx:.0f} x {Ly:.0f} unidades")
print("=" * 70)


# ==========================================================================
# 2. MÁSCARA DE OBSTÁCULOS EN LA MALLA GRUESA
#    (misma geometría que el código original de Gauss-Seidel:
#     superior izquierdo -> i <= 9 y j >= 6
#     inferior central   -> 35 <= i <= 44 y j <= 1)
# ==========================================================================
def build_solid_mask_grueso():
    solid = np.zeros((Nx, Ny), dtype=bool)
    solid[0:10, 6:9] = True       # obstáculo superior izquierdo (i<=9, j>=6)
    solid[35:45, 0:2] = True      # obstáculo inferior central (35<=i<=44, j<=1)
    return solid


obstaculo = build_solid_mask_grueso()
fijo = obstaculo.copy()
fijo[0, :] = True                 # entrada del canal (condición fija)

# ==========================================================================
# 3. CAMPO DE VELOCIDADES INICIAL
# ==========================================================================
u = np.ones((Nx, Ny)) * U0
u[0, :] = U0
u[obstaculo] = 0.0

# ==========================================================================
# 4. SOLUCIÓN ITERATIVA (GAUSS-SEIDEL / SOR) SOBRE LA MALLA GRUESA
# ==========================================================================
print("\nResolviendo con Gauss-Seidel (SOR) en la malla gruesa...")
for it in range(max_iter):
    # --------------------------------------
    # paredes impermeables (gradiente nulo)
    # --------------------------------------
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    # --------------------------------------
    # salida con gradiente nulo
    # --------------------------------------
    u[-1, :] = u[-2, :]
    u[-2, :] = u[-3, :]
    u[-3, :] = u[-4, :]

    error = 0.0
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if fijo[i, j]:
                continue

            laplace = (
                u[i + 1, j] +
                u[i - 1, j] +
                u[i, j + 1] +
                u[i, j - 1]
            ) / 4.0
            laplace += g * (u[i - 1, j] - u[i, j])

            nuevo = omega * laplace + (1 - omega) * u[i, j]
            error = max(error, abs(nuevo - u[i, j]))
            u[i, j] = nuevo

    if it % 500 == 0:
        print(f"  Iteración {it:5d}   Error = {error:.3e}")
    if error < tol:
        print(f"\n  Convergió en {it} iteraciones (Error = {error:.3e})")
        break

u[obstaculo] = 0.0
u_grueso = u.copy()

print("\nResultados en la malla gruesa (81 x 9)")
print("  Velocidad mínima :", np.nanmin(u_grueso))
print("  Velocidad máxima :", np.nanmax(u_grueso))
print("  Velocidad media  :", np.nanmean(u_grueso))

# ==========================================================================
# 5. RECUPERACIÓN DEL TAMAÑO ORIGINAL MEDIANTE SPLINE CÚBICO NATURAL
#    Se interpola de forma separable (2D = spline en x seguido de spline
#    en y), usando bc_type="natural" en ambos pasos, para llevar la
#    solución de la malla gruesa (h = 5) a la resolución original del
#    canal (h = 1), es decir, de 400 x 40 unidades reales.
# ==========================================================================
print("\n" + "=" * 70)
print("  Interpolando con Spline Cúbico Natural (recuperando tamaño original)")
print("=" * 70)

# Coordenadas reales de la malla gruesa (donde se calculó la solución)
x_grueso = np.arange(Nx) * h_real          # 0, 5, 10, ..., 400
y_grueso = np.arange(Ny) * h_real          # 0, 5, ..., 40

# Coordenadas de la malla fina = tamaño ORIGINAL del canal (resolución 1)
x_fino = np.arange(0, Lx + 1, 1.0)         # 0, 1, 2, ..., 400  (401 puntos)
y_fino = np.arange(0, Ly + 1, 1.0)         # 0, 1, 2, ...,  40  ( 41 puntos)

# --- Paso 1: interpolación cúbica natural a lo largo de x, para cada fila j ---
u_paso_x = np.zeros((len(x_fino), Ny))
for j in range(Ny):
    spline_x = CubicSpline(x_grueso, u_grueso[:, j], bc_type="natural")
    u_paso_x[:, j] = spline_x(x_fino)

# --- Paso 2: interpolación cúbica natural a lo largo de y, para cada columna fina de x ---
u_fino = np.zeros((len(x_fino), len(y_fino)))
for i in range(len(x_fino)):
    spline_y = CubicSpline(y_grueso, u_paso_x[i, :], bc_type="natural")
    u_fino[i, :] = spline_y(y_fino)

X_fino, Y_fino = np.meshgrid(x_fino, y_fino, indexing="ij")
obstaculo_fino = np.zeros_like(u_fino, dtype=bool)
obstaculo_fino |= (X_fino <= 45.0) & (Y_fino >= 30.0)
obstaculo_fino |= (X_fino >= 175.0) & (X_fino <= 220.0) & (Y_fino <= 5.0)

u_fino[obstaculo_fino] = 0.0

print(f"  Malla original recuperada: {len(x_fino)} x {len(y_fino)} puntos "
      f"({Lx:.0f} x {Ly:.0f} unidades, h = 1)")
print("\nResultados en la malla de tamaño original (interpolada)")
print("  Velocidad mínima :", np.nanmin(u_fino))
print("  Velocidad máxima :", np.nanmax(u_fino))
print("  Velocidad media  :", np.nanmean(u_fino))

def _mapear_a_identidad(distancia, rango_destino, rango_real, transicion):
    distancia = np.asarray(distancia, dtype=float)
    fuente = np.empty_like(distancia)

    m1 = distancia <= rango_destino
    fuente[m1] = distancia[m1] * (rango_real / rango_destino)

    m2 = (distancia > rango_destino) & (distancia <= rango_destino + transicion)
    t = (distancia[m2] - rango_destino) / transicion
    fuente[m2] = (1 - t) * rango_real + t * distancia[m2]

    m3 = distancia > rango_destino + transicion
    fuente[m3] = distancia[m3]
    return fuente


def _desvanecer(distancia, inicio, fin):
    """1.0 cerca del obstáculo (distancia<=inicio), 0.0 lejos (distancia>=fin)."""
    distancia = np.asarray(distancia, dtype=float)
    t = np.clip((distancia - inicio) / (fin - inicio), 0.0, 1.0)
    return 1.0 - t


def calcular_coordenadas_fuente(X, Y, escala_transicion=1.0):
    X0_1, GS_ANCHO_1 = 100.0, 45.0
    D0_1, GS_ALTO_1 = 15.0, 10.0
    TX_1 = 30.0 * escala_transicion
    TY_1 = 8.0 * escala_transicion

    dist_x1 = X
    prof_out1 = Ly - Y

    fuente_x1 = _mapear_a_identidad(dist_x1.ravel(), X0_1, GS_ANCHO_1, TX_1).reshape(dist_x1.shape)
    fuente_prof1 = _mapear_a_identidad(prof_out1.ravel(), D0_1, GS_ALTO_1, TY_1).reshape(prof_out1.shape)

    alpha1 = _desvanecer(prof_out1, D0_1, D0_1 + TY_1)
    beta1 = _desvanecer(dist_x1, X0_1, X0_1 + TX_1)

    x_src1 = dist_x1 - alpha1 * (dist_x1 - fuente_x1)
    prof_src1 = prof_out1 - beta1 * (prof_out1 - fuente_prof1)
    y_src1 = Ly - prof_src1
    w1 = alpha1 * beta1

    CX2 = 200.0
    X0_2, GS_ANCHO_2 = 50.0, 22.5
    D0_2, GS_ALTO_2 = 15.0, 5.0
    TX_2 = 20.0 * escala_transicion
    TY_2 = 6.0 * escala_transicion

    dist_x2 = np.abs(X - CX2)
    signo2 = np.sign(X - CX2)
    prof_out2 = Y

    fuente_x2 = _mapear_a_identidad(dist_x2.ravel(), X0_2, GS_ANCHO_2, TX_2).reshape(dist_x2.shape)
    fuente_prof2 = _mapear_a_identidad(prof_out2.ravel(), D0_2, GS_ALTO_2, TY_2).reshape(prof_out2.shape)

    alpha2 = _desvanecer(prof_out2, D0_2, D0_2 + TY_2)
    beta2 = _desvanecer(dist_x2, X0_2, X0_2 + TX_2)

    x_src2 = CX2 + signo2 * (dist_x2 - alpha2 * (dist_x2 - fuente_x2))
    y_src2 = prof_out2 - beta2 * (prof_out2 - fuente_prof2)
    w2 = alpha2 * beta2

    w0 = np.clip(1.0 - w1 - w2, 0.0, 1.0)
    suma_pesos = w0 + w1 + w2
    x_fuente = (w0 * X + w1 * x_src1 + w2 * x_src2) / suma_pesos
    y_fuente = (w0 * Y + w1 * y_src1 + w2 * y_src2) / suma_pesos

    x_fuente = np.clip(x_fuente, 0.0, Lx)
    y_fuente = np.clip(y_fuente, 0.0, Ly)
    return x_fuente, y_fuente


x_src_fino, y_src_fino = calcular_coordenadas_fuente(X_fino, Y_fino, escala_transicion=0.7)

_interp_visual_fino = RegularGridInterpolator(
    (x_fino, y_fino), u_fino, bounds_error=False, fill_value=None
)
_puntos_fino = np.column_stack([x_src_fino.ravel(), y_src_fino.ravel()])
u_plot = _interp_visual_fino(_puntos_fino).reshape(X_fino.shape)

X_grueso, Y_grueso = np.meshgrid(x_grueso, y_grueso, indexing="ij")

ESCALA_TRANSICION_GRUESO = 0.7

x_src_grueso, y_src_grueso = calcular_coordenadas_fuente(
    X_grueso, Y_grueso, escala_transicion=ESCALA_TRANSICION_GRUESO
)

_interp_visual_grueso = RegularGridInterpolator(
    (x_grueso, y_grueso), u_grueso, method="nearest", bounds_error=False, fill_value=None
)
_puntos_grueso = np.column_stack([x_src_grueso.ravel(), y_src_grueso.ravel()])
u_grueso_plot = _interp_visual_grueso(_puntos_grueso).reshape(X_grueso.shape)

# ==========================================================================
# 7. VISUALIZACIÓN - MAPAS DE CALOR
# ==========================================================================


def dibujar_obstaculos_nr(ax):
    """Obstáculos dibujados con el tamaño de Newton-Raphson (solo visual)."""
    ax.add_patch(patches.Rectangle((0, 25), 100, 15, facecolor="black"))
    ax.add_patch(patches.Rectangle((150, 0), 100, 15, facecolor="black"))


fig1b, ax1b = plt.subplots(figsize=(18, 4))
im1b = ax1b.imshow(
    u_grueso_plot.T,
    origin="lower",
    extent=[0, Lx, 0, Ly],
    cmap="turbo",
    aspect="auto",
    interpolation="nearest",
    vmin=0,
    vmax=U0,
)
fig1b.colorbar(im1b, ax=ax1b, label="Velocidad u [m/s]")
dibujar_obstaculos_nr(ax1b)
ax1b.set_title("Malla discretizada (h = 5) - Gauss-Seidel")
ax1b.set_xlabel("x [m]")
ax1b.set_ylabel("y [m]")
ax1b.set_xlim(0, Lx)
ax1b.set_ylim(0, Ly)
fig1b.tight_layout()

# --- Mapa de calor: tamaño original recuperado (spline cúbico natural) ---
fig2, ax2 = plt.subplots(figsize=(18, 4))
im2 = ax2.imshow(
    u_plot.T,
    origin="lower",
    extent=[0, Lx, 0, Ly],
    cmap="turbo",
    aspect="auto",
    interpolation="bilinear",
    vmin=0,
    vmax=U0,
)
fig2.colorbar(im2, ax=ax2, label="Velocidad u [m/s]")
dibujar_obstaculos_nr(ax2)
ax2.set_title("Tamaño original recuperado (h = 1) - Spline Cúbico Natural")
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.set_xlim(0, Lx)
ax2.set_ylim(0, Ly)
fig2.tight_layout()

plt.show()