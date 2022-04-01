from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoGaussian, AutoLowRankMultivariateNormal
from pyro.infer.autoguide import AutoStructured
from Guides.PolyDiagonalNormal import PolyDiagNorm
from Guides.SymMultivariateNormal import SymMultiNorm
from Guides.LowRankNormal import LowRankNormal
from Guides.ToeplitzMultivariate import ToeplitzMultivariateNorm
from Guides.BlockDiag import BlockMultivariateNorm
from Guides.CirculantMultivariate import CirculantMultivariateNorm

guide_list = {
    "autonormal": AutoDiagonalNormal,
    "multinormal": AutoMultivariateNormal,
    "lowrank": AutoLowRankMultivariateNormal,
    "polydiag": PolyDiagNorm,
    "symmetric": SymMultiNorm,
    "lowrank2": LowRankNormal,
    "toeplitz": ToeplitzMultivariateNorm,
    "blockdiag": BlockMultivariateNorm,
    "circulant": CirculantMultivariateNorm,
    "structured": AutoStructured
}