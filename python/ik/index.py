from enum import Enum


class IkSolverType(Enum):
    NR = "Newton-Rhapson Numerical Method"
    NEPPALLI_CLOSED_FORM = "Neppalli Closed Form Solution"
    GCRB_CLOSED_FORM = "Garriga-Casanovas and Rodrigues y Baena Closed Form Solution"
    FABRIKC = "FABRIKc Method"
    MICS = "MICS Method"
