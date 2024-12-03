"""
sympy derivation of the constants c0, c1, c2, c3, c4 referenced
in the GCRB paper
"""

from spatialmath import UnitQuaternion
from sympy import collect, symbols, Matrix

x, y, z = symbols("x y z")

kappa, lambda_, mu, nu = symbols("kappa lambda mu nu")


# define vector d
d = Matrix([x, y, z])

# define vector pt, the target point
pt_i, pt_j, pt_k = symbols("pt_i pt_j pt_k")
pt = Matrix([pt_i, pt_j, pt_k])

# define all h_i, the ith rows of r_ti respectively
r_ti_11, r_ti_12, r_ti_13 = symbols("r_ti_11 r_ti_12 r_ti_13")
r_ti_21, r_ti_22, r_ti_23 = symbols("r_ti_21 r_ti_22 r_ti_23")
r_ti_31, r_ti_32, r_ti_33 = symbols("r_ti_31 r_ti_32 r_ti_33")

h1 = Matrix([r_ti_11, r_ti_12, r_ti_13])
h2 = Matrix([r_ti_21, r_ti_22, r_ti_23])
h3 = Matrix([r_ti_31, r_ti_32, r_ti_33])

# define the equations
y_expr = -(lambda_ * x + nu * z) / mu

# simplified versions of the equations 22b, 22c
eqn_22b = -(-mu * x + lambda_ * y + kappa * z) * h2.dot(d - pt)
eqn_22c = -(nu * x + kappa * y - lambda_ * z) * h3.dot(d - pt)

sing_expr = eqn_22b - eqn_22c
sing_expr = sing_expr.subs(y, y_expr)

# collect the terms
poly = sing_expr.as_poly().as_expr()


poly = collect(poly, x**2)
C4 = poly.coeff(x**2)
poly = poly - C4 * x**2

poly = collect(poly, z**2)
C3 = poly.coeff(z**2)
poly = poly - C3 * z**2

poly = collect(poly, x * z)
C2 = poly.coeff(x * z)
poly = poly - C2 * x * z

poly = collect(poly, x)
C1 = poly.coeff(x)
poly = poly - C1 * x

poly = collect(poly, z)
C0 = poly.coeff(z)


# extra logic to keep all coeffs in clean form
C0 = collect(C0, nu)

C1 = collect(C1, 1 / mu)
C1 = collect(C1, pt_i)
C1 = collect(C1, pt_j)
C1 = collect(C1, pt_k)


# calculation somewhat different when mu is zero
eqn_22b_alt = eqn_22b.subs(mu, 0)
eqn_22c_alt = eqn_22c.subs(mu, 0)
sing_expr = eqn_22b_alt - eqn_22c_alt
sing_expr = sing_expr.subs(x, -(nu / lambda_) * z)

sing_expr = sing_expr.as_poly().as_expr()

sing_expr = collect(sing_expr, y**2)
C4s = sing_expr.coeff(y**2)
sing_expr = sing_expr - C4s * y**2

sing_expr = collect(sing_expr, z**2)
C3s = sing_expr.coeff(z**2)
sing_expr = sing_expr - C3s * z**2

sing_expr = collect(sing_expr, y * z)
C2s = sing_expr.coeff(y * z)
sing_expr = sing_expr - C2s * y * z

sing_expr = collect(sing_expr, y)
C1s = sing_expr.coeff(y)
sing_expr = sing_expr - C1s * y

sing_expr = collect(sing_expr, z)
C0s = sing_expr.coeff(z)
sing_expr = sing_expr - C0s * z


def params_as_subs_value(
    uq: list[float] | None = None,
    r_ti: list[float] | None = None,
    pt: list[float] | None = None,
) -> list:
    res = []

    if uq is not None:
        kappa_val, lambda_val, mu_val, nu_val = uq
        res.extend(
            [(kappa, kappa_val), (lambda_, lambda_val), (mu, mu_val), (nu, nu_val)]
        )

    if r_ti is not None:
        r_ti_11_val, r_ti_12_val, r_ti_13_val = r_ti[0]
        r_ti_21_val, r_ti_22_val, r_ti_23_val = r_ti[1]
        r_ti_31_val, r_ti_32_val, r_ti_33_val = r_ti[2]
        res.extend(
            [
                (r_ti_11, r_ti_11_val),
                (r_ti_12, r_ti_12_val),
                (r_ti_13, r_ti_13_val),
                (r_ti_21, r_ti_21_val),
                (r_ti_22, r_ti_22_val),
                (r_ti_23, r_ti_23_val),
                (r_ti_31, r_ti_31_val),
                (r_ti_32, r_ti_32_val),
                (r_ti_33, r_ti_33_val),
            ]
        )
    if pt is not None:
        pt_i_val, pt_j_val, pt_k_val = pt
        res.extend([(pt_i, pt_i_val), (pt_j, pt_j_val), (pt_k, pt_k_val)])

    return res


def c0_sub(uq: list, r_ti: list, pt: list) -> float:
    subs = params_as_subs_value(uq, r_ti, pt)
    return float(C0.subs(subs).evalf())


def c1_sub(uq: list, r_ti: list, pt: list) -> float:
    subs = params_as_subs_value(uq, r_ti, pt)
    return float(C1.subs(subs).evalf())


def c2_sub(uq: list, r_ti: list) -> float:
    subs = params_as_subs_value(uq, r_ti)
    return float(C2.subs(subs).evalf())


def c3_sub(uq: list, r_ti: list) -> float:
    subs = params_as_subs_value(uq, r_ti)
    return float(C3.subs(subs).evalf())


def c4_sub(uq: list, r_ti: list, pt: list) -> float:
    subs = params_as_subs_value(uq, r_ti, pt)
    return float(C4.subs(subs).evalf())


if __name__ == "__main__":
    # define R_ti, target position and quaternion
    from common.robot import ConstantCurvatureCR, ConstantCurvatureSegment
    from math import pi

    coeffs = [C0, C1, C2, C3, C4]
    for i, c in enumerate(coeffs):
        print(f"c_{i}\n", c)
    print("\n")

    segment1 = ConstantCurvatureSegment(1 / 0.1, pi / 6, 0.05, is_extensible=True)
    segment2 = ConstantCurvatureSegment(1 / 0.05, pi / 2, 0.05, is_extensible=True)
    target_robot = ConstantCurvatureCR([segment1, segment2])
    target_pose = target_robot.pose_vector()

    target = target_robot.t_matrix().A
    r_ti = target[:3, :3].T
    pt = target[:3, 3]
    uq = UnitQuaternion(target[:3, :3]).A

    print(f"c0: {c0_sub(uq, r_ti, pt)}\n")
    print(f"c1: {c1_sub(uq, r_ti, pt)}\n")
    print(f"c2: {c2_sub(uq, r_ti)}\n")
    print(f"c3: {c3_sub(uq, r_ti)}\n")
    print(f"c4: {c4_sub(uq, r_ti, pt)}\n")
