# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Small example models (used by the tests and hand-tests).

``toy_phage_ir`` is written directly as IR; ``toy_phage_model`` is the same model written with
the Python API. The two must give identical IR, which the tests check.
``toy_three_axis_model`` keeps the infected cells' genotype: ``I[a, z, s]`` over three axes.
``plaque_model`` is a phage plaque front, space x infection age.
``drug_tolerance_model`` is a tumour cell population on a continuous drug-resistance level,
with selection, phenotypic diffusion and a drug-induced drift.
"""

from __future__ import annotations

import numpy as np

import tenqsp as tq
from tenqsp.ir import (
    AxisSpec,
    BoundarySpec,
    CompartmentSpec,
    DefinitionSpec,
    DoseSpec,
    FieldSpec,
    InitSpec,
    IntegralSpec,
    ModelIR,
    ObservableSpec,
    ParamSpec,
    TermSpec,
    TransitionSpec,
)


def faces(lower: float, upper: float, n: int) -> tuple[float, ...]:
    return tuple(float(x) for x in np.linspace(lower, upper, n + 1))


def toy_phage_ir(n_z: int = 30, n_a: int = 40) -> ModelIR:
    """The spike toy model: B(z, s), I(a, z), P (see ADR 0002)."""
    return ModelIR(
        name="phage_toy",
        axes=(
            AxisSpec("z", "continuous", faces=faces(0, 1, n_z), kind="bidirectional"),
            AxisSpec("s", "categorical", labels=("S", "R")),
            AxisSpec("a", "continuous", faces=faces(0, 1.5, n_a), kind="one_way", units="h"),
        ),
        params=(
            ParamSpec("g_max", 1.0),
            ParamSpec("K", 1e9),
            ParamSpec("beta", 50.0),
            ParamSpec("delta", 0.1),
            ParamSpec("h_max", 5.0),
            ParamSpec("tau", 0.8),
            ParamSpec("s_h", 0.05),
            ParamSpec("w_amp", 0.5),
            ParamSpec("Dz", 0.01),
            ParamSpec("mu", 1e-6),
            ParamSpec("k", (1e-9, 1e-12), over="s"),
            ParamSpec("cost", (0.0, 0.1), over="s"),
        ),
        fields=(FieldSpec("B", ("z", "s")), FieldSpec("I", ("a", "z"))),
        compartments=(CompartmentSpec("P"),),
        definitions=(
            DefinitionSpec("div", "g_max*(1 - z)*(1 - cost[s])*B*(1 - B_tot/K)"),
            DefinitionSpec("ads", "k[s]*(1 - 0.5*z)*B*P"),
            DefinitionSpec("h", "h_max*sigmoid((a - tau)/s_h)"),
        ),
        integrals=(
            IntegralSpec("B_tot", "B", ("z", "s")),
            IntegralSpec("ads_z", "ads", ("s",)),  # adsorption summed over classes, per z
            IntegralSpec("ads_total", "ads", ("z", "s")),
            IntegralSpec("lysis", "h*I", ("a", "z")),
        ),
        transitions=(TransitionSpec("M", "s", (("S", "R", "mu"), ("R", "S", "mu"))),),
        terms=(
            TermSpec("reaction", "B", "div - ads"),
            TermSpec("mutate", "B", "div", axis="s", transitions="M"),
            TermSpec("advection", "B", "w_amp*(0.3 - z)", axis="z", scheme="erlang"),
            TermSpec("diffusion", "B", "Dz", axis="z"),
            TermSpec("advection", "I", "1", axis="a", scheme="sharp"),
            TermSpec("hazard", "I", "h"),
            TermSpec("diffusion", "I", "Dz", axis="z"),
            TermSpec("rate", "P", "beta*lysis - ads_total - delta*P"),
        ),
        boundaries=(
            BoundarySpec("I", "a", "lower", "inflow", value="ads_z"),
            BoundarySpec("I", "a", "upper", "outflow"),
        ),
        inits=(
            InitSpec("B", where=(("s", "S"),), total="1e7", shape="normal(z, mean=0.1, sd=0.08)"),
            InitSpec("B", where=(("s", "R"),), value="0"),
            InitSpec("I", value="0"),
            InitSpec("P", value="0"),
        ),
        doses=(DoseSpec("P", (0.0,), ("1e9",)),),
        observables=(ObservableSpec("bacteria", "B_tot"), ObservableSpec("phage", "P")),
    )


def toy_phage_model(n_z: int = 30, n_a: int = 40) -> tq.Model:
    """``toy_phage_ir``, written with the Python API."""
    m = tq.Model("phage_toy")
    z = m.axis("z", 0, 1, n_z, kind="bidirectional")
    s = m.axis("s", labels=["S", "R"])
    a = m.axis("a", 0, 1.5, n_a, kind="one_way", units="h")
    g_max, K, beta, delta, h_max, tau, s_h, w_amp, Dz, mu = m.params(
        g_max=1.0, K=1e9, beta=50.0, delta=0.1, h_max=5.0, tau=0.8, s_h=0.05, w_amp=0.5,
        Dz=0.01, mu=1e-6,
    )  # fmt: skip
    k = m.param("k", {"S": 1e-9, "R": 1e-12}, over=s)
    cost = m.param("cost", {"S": 0.0, "R": 0.1}, over=s)

    B = m.field("B", axes=[z, s])  # bacteria by dormancy level z and susceptibility s
    I = m.field("I", axes=[a, z])  # infected cells by infection age a and dormancy z
    P = m.compartment("P")  # free phage

    B_tot = m.integral(B, over=[z, s], name="B_tot")
    div = m.define("div", g_max * (1 - z) * (1 - cost[s]) * B * (1 - B_tot / K))
    ads = m.define("ads", k[s] * (1 - 0.5 * z) * B * P)
    h = m.define("h", h_max * tq.sigmoid((a - tau) / s_h))
    ads_z = m.integral(ads, over=s, name="ads_z")  # adsorption summed over classes, per z
    ads_total = m.integral(ads, over=[z, s], name="ads_total")
    lysis = m.integral(h * I, over=[a, z], name="lysis")
    M = m.transitions("M", axis=s, rates={"S->R": mu, "R->S": mu})

    m.reaction(B, div - ads)
    m.transition(B, axis=s, matrix=M, of=div)
    m.advect(B, axis=z, velocity=w_amp * (0.3 - z))
    m.diffuse(B, axis=z, D=Dz)
    m.advect(I, axis=a, velocity=1, scheme="sharp")
    m.hazard(I, h)
    m.diffuse(I, axis=z, D=Dz)
    m.ode(P, beta * lysis - ads_total - delta * P)
    m.inflow(I, axis=a, at="lower", flux=ads_z)
    m.outflow(I, axis=a, at="upper")

    m.init(B, where={s: "S"}, total=1e7, shape=tq.normal(z, mean=0.1, sd=0.08))
    m.init(B, 0, where={s: "R"})
    m.init(I, 0)
    m.init(P, 0)
    m.dose(P, times=[0.0], amounts=[1e9])
    m.observable("bacteria", B_tot)
    m.observable("phage", P)
    return m


def toy_three_axis_model(n_z: int = 16, n_a: int = 24) -> tq.Model:
    """Toy phage model with a three-axis field: infected cells ``I[a, z, s]`` by infection age
    ``a``, dormancy level ``z`` and susceptibility ``s`` (infected cells keep their genotype).

    Bacteria ``B[z, s]`` grow, mutate on division and drift/diffuse in ``z``; adsorption moves
    them into ``I`` at age 0 in the same ``(z, s)`` cell; lysis and cells reaching the end of
    the age axis are counted in the compartment ``L``, so with ``g_max = 0`` the cells
    ``B + I + L`` are conserved exactly. Values are illustrative."""
    m = tq.Model("three_axis_toy")
    a = m.axis("a", 0, 1.5, n_a, kind="one_way", units="h")
    z = m.axis("z", 0, 1, n_z, kind="bidirectional")
    s = m.axis("s", labels=["S", "R"])
    g_max, K, beta, delta, h_max, tau, s_h, w_amp, Dz, mu = m.params(
        g_max=1.0, K=1e9, beta=50.0, delta=0.1, h_max=5.0, tau=0.8, s_h=0.05, w_amp=0.5,
        Dz=0.01, mu=1e-6,
    )  # fmt: skip
    k = m.param("k", {"S": 1e-9, "R": 1e-11}, over=s)
    cost = m.param("cost", {"S": 0.0, "R": 0.1}, over=s)
    lag = m.param("lag", {"S": 1.0, "R": 1.3}, over=s)  # resistant infections lyse later

    B = m.field("B", axes=[z, s], units="cells")
    I = m.field("I", axes=[a, z, s], units="cells")
    P = m.compartment("P", units="PFU")
    L = m.compartment("L", units="cells")  # lysed (and aged-out) infected cells

    B_tot = m.integral(B, over=[z, s], name="B_tot")
    I_tot = m.integral(I, over=[a, z, s], name="I_tot")
    div = m.define("div", g_max * (1 - z) * (1 - cost[s]) * B * (1 - (B_tot + I_tot) / K))
    ads = m.define("ads", k[s] * (1 - 0.5 * z) * B * P)
    h = m.define("h", h_max * tq.sigmoid((a - tau * lag[s]) / s_h))
    lysis = m.integral(h * I, over=[a, z, s], name="lysis")
    M = m.transitions("M", axis=s, rates={"S->R": mu, "R->S": mu})

    m.reaction(B, div - ads)
    m.transition(B, axis=s, matrix=M, of=div)
    m.advect(B, axis=z, velocity=w_amp * (0.3 - z))
    m.diffuse(B, axis=z, D=Dz)
    m.advect(I, axis=a, velocity=1, scheme="sharp")
    m.hazard(I, h)
    m.diffuse(I, axis=z, D=Dz)
    m.inflow(I, axis=a, at="lower", flux=ads)
    m.outflow(I, axis=a, at="upper", to=L)
    m.ode(P, beta * lysis - m.integral(ads, over=[z, s], name="ads_total") - delta * P)
    m.ode(L, lysis)

    m.init(B, where={s: "S"}, total=1e7, shape=tq.normal(z, mean=0.1, sd=0.08))
    m.init(B, where={s: "R"}, total=1e4, shape=tq.normal(z, mean=0.1, sd=0.08))
    m.init(I, 0)
    m.init(P, 0)
    m.init(L, 0)
    m.dose(P, times=[0.0], amounts=[1e9])
    m.observable("bacteria", B_tot)
    m.observable("infected", I_tot)
    m.observable("phage", P)
    return m


def phage_latency_model(n: int = 40) -> tq.Model:
    """The infection-age example: uninfected B and phage P well mixed, infected cells I by age."""
    m = tq.Model("phage_latency")
    age = m.axis("age", lower=0.0, upper=2.0, n=n, units="h", kind="one_way")
    B = m.compartment("B", init=1e6)  # uninfected bacteria (well-mixed)
    P = m.compartment("P", init=1e7)  # free phage
    I = m.field("I", axes=[age], init=0.0)  # infected cells by infection age

    k, beta, g, Bmax, delta, h_max, tau, s = m.params(
        k=1e-9, beta=50.0, g=1.0, Bmax=1e9, delta=0.1, h_max=5.0, tau=0.8, s=0.05
    )
    h = h_max * tq.sigmoid((age.coord - tau) / s)  # age-dependent lysis hazard

    m.ode(B, g * B * (1 - B / Bmax) - k * B * P)
    m.advect(I, axis=age, velocity=1.0, scheme="sharp")
    m.inflow(I, axis=age, at="lower", flux=k * B * P)
    m.outflow(I, axis=age, at="upper", to=None)  # absorbed, recorded in the ledger
    m.hazard(I, rate=h)
    m.ode(P, beta * m.integral(h * I, over=age) - k * B * P - delta * P)

    m.observable("burst_rate", m.integral(h * I, over=age))
    m.dose(P, times=[0.0], amounts=[1e8])
    return m


def plaque_model(
    n_x: int = 240, n_a: int = 40, *, L: float = 600.0, a_max: float = 1.5, latency: bool = True
) -> tq.Model:
    """Gallery Model 10: a phage plaque spreading through a bacterial lawn (Yin & McCaskill
    1992; Fort & Méndez 2002; Gourley & Kuang 2004), in infection-age form.

    Units: micrometres, hours, per mL. Uninfected bacteria ``B(x)`` do not grow (a stationary
    lawn); free phage ``P(x)`` diffuse and adsorb; infected cells ``I(a, x)`` age at speed 1
    and lyse with hazard ``h(a)`` (a latent period near ``tau``), releasing ``beta`` phage.
    The phage start as ``P0 exp(-lam0 x)``; both ends are closed (zero flux), so the front
    moves from ``x = 0`` to ``x = L``. A start steeper than the front's own decay rate
    (``lam0`` above lambda*, as the default) gives the minimal speed c*; a shallower one
    gives the faster speed c(lam0) of the linear theory. ``latency=False``: lysis at once on
    adsorption (no ``I``; the Yin–McCaskill limit). The values are illustrative.
    """
    m = tq.Model("plaque" if latency else "plaque_no_latency")
    x = m.axis("x", 0.0, L, n_x, kind="bidirectional", units="um")
    D, k, B0, beta, delta, h_max, tau, s, P0, lam0 = m.params(
        D=400.0, k=1e-8, B0=1e9, beta=50.0, delta=0.05, h_max=30.0, tau=0.5, s=0.03,
        P0=1e6, lam0=1.0,
    )  # fmt: skip
    B = m.field("B", axes=[x])
    P = m.field("P", axes=[x])
    ads = m.define("ads", k * B * P)
    if latency:
        a = m.axis("a", 0.0, a_max, n_a, kind="one_way", units="h")
        I = m.field("I", axes=[a, x])
        h = m.define("h", h_max * tq.sigmoid((a - tau) / s))
        lysis = m.integral(h * I, over=a, name="lysis")  # a field over x
        m.advect(I, axis=a, velocity=1, scheme="sharp")
        m.hazard(I, h)
        m.inflow(I, axis=a, at="lower", flux=ads)
        m.outflow(I, axis=a, at="upper")
        m.init(I, 0)
        m.observable("infected", m.integral(I, over=[a, x]))
    else:
        lysis = ads
    m.reaction(B, -ads)
    m.reaction(P, beta * lysis - ads - delta * P)
    m.diffuse(P, axis=x, D=D)
    m.init(B, B0)
    m.init(P, P0 * tq.exp(-lam0 * x))
    m.observable("bacteria", m.integral(B, over=x))
    m.observable("phage", m.integral(P, over=x))
    return m


def oral_absorption_model(
    n_g: int = 160, n_x: int = 50, *, g_max: float = 8.0, scheme: str = "sharp"
) -> tq.Model:
    """A physiological oral absorption model:
    the stomach and the small intestine as fields, the colon and the body as compartments.

    - Stomach ``G(g)``, ``g`` = time since the dose was swallowed (h): no absorption; the
      drug leaves at the gastric-emptying hazard of a Weibull, ``(beta_ge/eta)(g/eta)^(beta_ge
      - 1)`` (``beta_ge = 1``: first order; ``> 1``: a lag before emptying).
    - Small intestine ``S(x)``, ``x`` = fraction of its length (0 proximal, 1 the ileocaecal
      valve): carried along in ``T_si`` hours (Yu & Amidon: 199 min), spreading with
      ``D_x`` (0: plug flow; > 0: the dispersion model of Ni et al. 1980, Danckwerts ends),
      absorbed at ``ka(x) = ka_si * ka_ratio^x``, falling log-linearly from the proximal to the
      distal value (``ka_ratio`` = distal/proximal; ``ka = 2 Peff / R``).
    - Colon ``Col``: absorbed at ``ka_col``, leaves (faeces) at ``k_col``.
    - Body ``Ac`` (amount), one compartment: ``conc = Ac / V``, cleared at ``CL``.

    One dose ``Dose`` at t = 0 enters the stomach at ``g = 0`` (a regimen given to a simulation
    replaces it: ``tq.bolus("G", 100, start=0, every=12, n=7, shape=tq.point("g", at=0))``).
    Drug still in the stomach at ``g_max`` (3e-8 of the dose for the defaults) moves on into
    the small intestine with the emptied drug, so nothing is lost: the intestine's inlet
    condition reads the stomach at its end, ``G(g=g_max)``. Defaults are
    atenolol-like (Dahlgren et al.: Peff jejunum 0.45, ileum 0.15, colon 0.013 x 1e-4 cm/s;
    R = 1.75 cm) with first-order emptying of a fasted liquid (half-time 15 min); CL and V
    are illustrative. Drug in solution only: no dissolution, precipitation or gut-wall
    metabolism (plan §2.3).
    """
    m = tq.Model("oral_absorption")
    g = m.axis("g", 0.0, g_max, n_g, kind="one_way", units="h")
    x = m.axis("x", 0.0, 1.0, n_x, kind="spatial")
    Dose, eta, beta_ge, T_si, D_x, ka_si, ka_ratio, ka_col, k_col, CL, V = m.params(
        Dose=100.0, eta=0.25 / np.log(2), beta_ge=1.0, T_si=199 / 60, D_x=0.0, ka_si=0.185,
        ka_ratio=1 / 3, ka_col=0.0053, k_col=1 / 24, CL=8.0, V=60.0,
    )  # fmt: skip
    G = m.field("G", axes=[g])
    S = m.field("S", axes=[x])
    Col = m.compartment("Col")
    Ac = m.compartment("Ac")

    h_ge = m.define("h_ge", beta_ge / eta * (g / eta) ** (beta_ge - 1))
    ka = m.define("ka", ka_si * ka_ratio**x)
    m.advect(G, axis=g, velocity=1, scheme="sharp")
    m.hazard(G, h_ge)
    m.at(G, g="upper", outflow=True)  # older than g_max: on to the intestine (below)
    emptying = m.integral(h_ge * G, over=g, name="emptying_rate")

    m.advect(S, axis=x, velocity=1 / T_si, scheme=scheme)
    m.diffuse(S, axis=x, D=D_x)
    # inlet: the flux v*S - D_x*dS/dx entering at x = 0 is what the stomach empties, plus
    # what reaches its last age (speed 1 along g, so that flux is G there)
    m.at(S, x=0, flux=emptying + G(g="upper"))
    m.at(S, x=1, outflow=True)  # through the ileocaecal valve
    m.reaction(S, -ka * S)
    absorbed = m.integral(ka * S, over=x, name="absorbed_si_rate")

    m.ode(Col, S(x="upper") / T_si - (ka_col + k_col) * Col)
    m.ode(Ac, absorbed + ka_col * Col - CL / V * Ac)

    m.dose(G, times=[0.0], amounts=Dose, shape=tq.point(g, at=0.0))  # swallowed at t = 0
    m.init(S, 0)
    m.init(Col, 0)
    m.init(Ac, 0)
    m.observable("conc", Ac / V)
    m.observable("stomach", m.integral(G, over=g))
    m.observable("small_intestine", m.integral(S, over=x))
    m.observable("emptying", emptying)  # amount per hour
    m.observable("absorption_si", absorbed)  # amount per hour
    return m


DRUG_TOLERANCE_TEXT = """
model drug_tolerance
axis x in [{lower}, {upper}] n={n}                # resistance level (1: the most resistant)

param gamma=0.66, eta=0.132, beta=0.01, K=1e8     # per day (Lorenzi et al. 2016, Table 1)
param vbar=0.02                                   # drift per unit of drug (illustrative)
param ke=0, c0=0                                  # drug: elimination rate, level at t = 0
param n0=1e6, x0=0, s0=0.5                        # cells at t = 0: number, mean, sd

field n(x)                                        # cancer cells by resistance level
compartment C                                     # drug level (a death rate, per day)

d = gamma/K                                       # death from competition, per cell
rho = int_x(n)                                    # the tumour size

d_t(n) = -d_x(vbar*C*n) + d_x(beta*d_x(n))
         + (gamma - eta*x^2)*n - d*rho*n - C*(x - 1)^2*n      using sharp
    at x=lower: vbar*C*n - beta*d_x(n) = 0                    # no cell leaves the axis
    at x=upper: vbar*C*n - beta*d_x(n) = 0
d_t(C) = -ke*C

n(0) = n0*normal(x, mean=x0, sd=s0)
C(0) = c0

observable size = rho
observable mean_x = int_x(x*n)/rho
observable var_x = int_x(x^2*n)/rho - (int_x(x*n)/rho)^2
"""


def drug_tolerance_model(n: int = 160, *, lower: float = -4.0, upper: float = 4.0) -> tq.Model:
    """Cancer cells on a continuous drug-resistance level ``x``, under a cytotoxic drug that
    selects resistant cells and also pushes cells toward resistance (a drug-induced drift):

        d_t(n) = -d_x(vbar*C*n) + d_x(beta*d_x(n)) + [gamma - eta*x^2 - d*rho - C*(x - 1)^2]*n

    with ``rho = int_x(n)``, ``d = gamma/K`` and sealed ends. Cells at ``x = 0`` proliferate
    fastest (``gamma - eta*x^2``: resistance costs proliferation); the drug kills at
    ``C*(x - 1)^2``, so ``x = 1`` is the most resistant level; ``beta`` is the rate of random
    heritable changes in expression (non-genetic instability); ``vbar*C`` is the speed at
    which the drug moves cells toward resistance (stress-induced adaptation). The text of the
    model is ``DRUG_TOLERANCE_TEXT``. Time in days.

    Sources (each equation checked against the paper):

    - Selection and diffusion, the parameter values and the closed-form equilibria: Lorenzi T,
      Chisholm RH, Clairambault J. Tracking the evolution of cancer cell populations through
      the mathematical lens of phenotype-structured equations. Biol Direct 2016;11:43
      (eqs. 3, 6, 7; Table 1: gamma = 0.66, eta = 0.132, beta = 0.01 per day, d = gamma/K
      with K = 1e8; Theorems 1-2: Gaussian equilibria, mean c/(eta + c), variance
      sqrt(beta/(eta + c)), size (gamma_c - sqrt(beta*(eta + c)))/d with
      gamma_c = gamma - eta*c/(eta + c)). Their model has no drift.
    - The drift term, in conservation form ``d_x(v*n)``, modelling stress-induced
      adaptation: Chisholm RH, Lorenzi T, Lorz A. Effects of an advection term in nonlocal
      Lotka-Volterra equations. Commun Math Sci 2016;14:1181-8 (eq. 1.1); motivated by
      Chisholm RH, Lorenzi T, Lorz A, et al. Cancer Res 2015;75:930-9, where the velocity
      is proportional to the drug, v = -vbar*c(t)*H(...). The Cancer Res model was checked
      only as restated by two of its authors' group (Clairambault J, Pouchol C. Biomath
      2019;8:1905147), not in the original.

    Differences from the papers:

    - One trait (Lorenzi et al. 2016). Chisholm et al. 2015 has two, survival potential and
      proliferation potential, on [0, 1]^2, with the drift along proliferation potential
      and switched by a Heaviside function of survival potential; here the drift acts along
      the single resistance level and is the same at every level.
    - ``vbar`` (0.02 per day per unit of drug) is illustrative; the papers give no value
      for this one-trait form.
    - The drug is a compartment ``C`` (a death rate, per day, as the paper's c(t)) so that
      doses and elimination can drive it; ``ke = 0`` with ``C(0) = c`` is the paper's
      constant dose.
    - The paper poses the equation on the whole line and solves it on [-2L, 2L] = [-4, 4];
      here the ends of [lower, upper] are sealed (zero flux). For the defaults the
      Gaussian's tails at +-4 are below 1e-12 of its peak, so the two agree.

    What the tests use (derived for this example, not in the papers): with constant
    ``vbar*C`` and quadratic fitness the solution stays Gaussian if it starts Gaussian, and
    its size rho, mean m and variance s obey three ODEs, with c = C(t):

        s' = 2*beta - 2*(eta + c)*s^2
        m' = vbar*c + 2*(eta + c)*s*(c/(eta + c) - m)
        rho' = rho*(gamma_c - (eta + c)*((m - c/(eta + c))^2 + s) - d*rho)

    so at a constant drug level the drift moves the equilibrium mean by
    vbar*c/(2*sqrt(beta*(eta + c))) and lowers the equilibrium size by
    (vbar*c)^2/(4*beta*d) (the cost of lagging behind the drift).
    """
    return tq.Model.from_text(DRUG_TOLERANCE_TEXT.format(n=n, lower=lower, upper=upper))
