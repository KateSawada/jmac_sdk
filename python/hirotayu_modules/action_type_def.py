# 動作モードと狙っている役の種類を定義
from enum import Enum, auto

class ActionModeType(Enum):
    MENZEN = auto()
    KOKUSHI = auto()
    FURO = auto()
    FURO_YAKUHAI = auto()

class TargetYakuType(Enum):
    NO_TARGET = auto()
    CHINITSU_MANZU = auto()
    CHINITSU_PINZU = auto()
    CHINITSU_SOZU = auto()
    HONITSU_MANZU = auto()
    HONITSU_PINZU = auto()
    HONITSU_SOZU = auto()
    TSUISO = auto()