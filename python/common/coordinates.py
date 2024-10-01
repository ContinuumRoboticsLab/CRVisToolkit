"""
The Coordinates utility module provides classes that represent different coordinate
representations of diffferent geometric objects. The module also provides conversion
+ initialization utilities to help keep code legible and make conversion simple.
"""

from enum import Enum
from numbers import Number


class CrConfigurationType(Enum):
    """
    while the traditional configuration of a Constant-Curvate section is done in
    terms of kappa, phi, and length, some solvers benefit from alternate
    representations. This enum defines the different representations that
    the CR object can be initialized with + converted to
    """

    # the default kappa, phi, length representation
    KPL = 0

    # the representation seen in Garriga-Casanovas e.t al. 2019
    # uses an "eta" angle parameter instead of length
    KPE = 1

    def _required_params(self) -> set[str]:
        """
        returns the parameters required for the given configuration type
        """
        match self:
            case CrConfigurationType.KPL:
                return {"kappa", "phi", "length"}
            case CrConfigurationType.KPE:
                return {"kappa", "phi", "eta"}
            case _:
                raise ValueError(f"Invalid configuration type: {self}")

    def validate(self, **kwargs) -> bool:
        """
        make sure the parameters are valid for the given configuration type
        """
        match self:
            case CrConfigurationType.KPL | CrConfigurationType.KPE:
                if not set(kwargs.keys()) == self._required_params():
                    return False
                for value in kwargs.items():
                    if not isinstance(value, Number):
                        return False
                return True
            case _:
                return False


class CrSectionConfiguration:
    """
    an class that represents the configuration of a single Constant-Curvate section
    in literature, this configuration (for either a single section or the entire robot)
    is denoted theta.

    under the hood, the traditional kappa, phi, length representation is used,
    but the class allows for easy instantiation from different representations
    and can output conversions to other types as well, allowing it to be used
    in all solvers.

    The primary purpose of this class is to convert between representations
    """

    def __init__(self, repr: CrConfigurationType = CrConfigurationType.KPL, **kwargs):
        if not repr.validate(**kwargs):
            raise ValueError(f"Invalid configuration for {repr}")

        # instantiate the object with the required parameters
        for key in repr._required_params():
            setattr(self, key, kwargs[key])
