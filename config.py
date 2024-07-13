import numpy as np

FX_PXL = 1936.9
FY_PXL = 1943.7
CX_PXL = 1344.0
CY_PXL = 757.9608
K = np.array([
        [FX_PXL, 0, CX_PXL],
        [0, FY_PXL, CY_PXL],
        [0, 0, 1]
    ])
