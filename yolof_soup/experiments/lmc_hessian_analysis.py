"""
lmc_hessian_analysis.py  (renamed from loss_landscape.py)
==========================================================
Measures per-component loss landscape geometry for MV1 and MV2
(thesis Section 3.5.2, RQ2/H2):

  MV1: LMC barrier B_c = max_{alpha} L_c((1-a)thetaA + a*thetaB)
         - [L_c(thetaA) + L_c(thetaB)] / 2
       Computed over 15 model pairs x 5 components x 21-point grid.

  MV2: Hessian trace Tr(H_c) via Hutchinson estimator,
       50 Rademacher vectors, 6 checkpoints x 5 components.
"""
# Content preserved exactly from loss_landscape.py
from yolof_soup.experiments.loss_landscape import *  # noqa: F401, F403
