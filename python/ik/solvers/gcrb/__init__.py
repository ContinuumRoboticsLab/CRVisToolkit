"""
in the 2019 paper 'Kinematics of Continuum Robots With Constant Curvature
Bending and Extension Capabilities" by Garriga-Casanovas and Rodriguez-y-Baena,
where a closed-form IK solution for 2/3-segment robots is proposed, the full
expressions for the solution are only given in the XYZ coorindate formulation,
and only only when Z is the parameterized coordinate.

As such, this module consists of two parts: first, a sympy script used to
determine the closed-form solutions in different formulations, and second,
the actual implementation of the closed-form solver, which makes use of the
output of the sympy script but does not ever use sympy itself at runtime.
"""
