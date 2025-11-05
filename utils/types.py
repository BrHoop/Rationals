from enum import Enum


class BCType(Enum):
    """
    Enum for boundary condition types
    """

    NONE = 0
    RHS = 1
    FUNCTION = 2


class FilterType(Enum):
    """
    Enum for filter types
    """

    NONE = 1
    KREISS_OLIGER_O6 = 2
    KREISS_OLIGER_O8 = 3
    JTT4 = 4
    JTT6 = 5
    JTP6 = 6
    JTT8 = 7
    JTP8 = 8
    KP4 = 9
    F2 = 10


class FilterApply(Enum):
    """
    Enum for filter application
    """

    NONE = 1
    RHS = 2
    APPLY_VARS = 3
    APPLY_DERIVS = 4


class DerivType(Enum):
    NONE = 0
    D1_E44 = 1
    D2_E44 = 2
    D1_E642 = 3
    D2_E642 = 4
    D1_JP6 = 5
    D2_JP6 = 6
    D1_JT4 = 7
    D2_JT4 = 8
    D1_JT6 = 9
    D2_JT6 = 10
    D1_BYUP6 = 11
    D2_BYUP6 = 12
    D1_Wm6 = 13
    D1_Wp6 = 14
    D1_DSQ6A = 15
    D2_DSQ6A = 16
    D1_ME44 = 17
    D2_ME44 = 18
    D1_ME642 = 19
    D2_ME642 = 20
    D1_KP4 = 21
    D1_SP4 = 22
    D1_DE4 = 23
    D1_DSQ6B = 24
    D2_DSQ6B = 25
    D1_DSQ6B_LEFT = 26
    D1_DSQ6B_RIGHT = 27
    D1_BYU_C6_Op10 = 28
    D1_BYU_P4_Op18 = 29
    D1_E42 = 30
    D2_E42 = 31
    D1_E666 = 32
    D2_E666 = 33


class CFDSolve(Enum):
    NONE = 0
    SCIPY = 1
    LUSOLVE = 2
    PENTAPY = 3
    D_INV = 4
    D_LU = 5


filter_type_map = {
    "None": FilterType.NONE,
    "KO6": FilterType.KREISS_OLIGER_O6,
    "KO8": FilterType.KREISS_OLIGER_O8,
    "JTT4": FilterType.JTT4,
    "JTT6": FilterType.JTT6,
    "JTP6": FilterType.JTP6,
    "JTT8": FilterType.JTT8,
    "JTP8": FilterType.JTP6,
    "KP4": FilterType.KP4,
    "F2": FilterType.F2,
}

filter_apply_map = {
    "Vars": FilterApply.APPLY_VARS,
    "Derivs": FilterApply.APPLY_DERIVS,
    "Rhs": FilterApply.RHS,
    "None": FilterApply.NONE,
}

d1_type_map = {
    "E4": DerivType.D1_E44,
    "E42": DerivType.D1_E42,
    "E44": DerivType.D1_E44,
    "E6": DerivType.D1_E642,
    "E642": DerivType.D1_E642,
    "E666": DerivType.D1_E666,
    "JT4": DerivType.D1_JT4,
    "JT6": DerivType.D1_JT6,
    "JP6": DerivType.D1_JP6,
    "KP4": DerivType.D1_KP4,
    "DSQ6A": DerivType.D1_DSQ6A,
    "DSQ6B": DerivType.D1_DSQ6B,
    "DSQ6B": DerivType.D1_DSQ6B,
    "ME4": DerivType.D1_ME44,
    "ME44": DerivType.D1_ME44,
    "ME6": DerivType.D1_ME642,
    "ME642": DerivType.D1_ME642,
    "BYUC6_Op10": DerivType.D1_BYU_C6_Op10,
    "BYUP4_Op18": DerivType.D1_BYU_P4_Op18,
}

d2_type_map = {
    "E4": DerivType.D2_E44,
    "E42": DerivType.D2_E42,
    "E44": DerivType.D2_E44,
    "E6": DerivType.D2_E642,
    "E642": DerivType.D2_E642,
    "E666": DerivType.D2_E666,
    "JT4": DerivType.D2_JT4,
    "JT6": DerivType.D2_JT6,
    "JP6": DerivType.D2_JP6,
    "DSQ6A": DerivType.D2_DSQ6A,
    "DSQ6B": DerivType.D2_DSQ6B,
    "DSQ6B": DerivType.D2_DSQ6B,
    "ME4": DerivType.D2_ME44,
    "ME44": DerivType.D2_ME44,
    "ME6": DerivType.D2_ME642,
    "ME642": DerivType.D2_ME642,
}

cfd_solve_map = {
    "SCIPY": CFDSolve.SCIPY,
    "LUSOLVE": CFDSolve.LUSOLVE,
    "D_INV": CFDSolve.D_INV,
    "D_LU": CFDSolve.D_LU,
    "PENTAPY": CFDSolve.PENTAPY,
}
