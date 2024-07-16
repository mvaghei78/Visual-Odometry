import numpy as np

FX_PXL = 1936.9
FY_PXL = 1943.7
CX_PXL = 1344.0
CY_PXL = 757.9608
drone_K = np.array([
        [FX_PXL, 0, CX_PXL],
        [0, FY_PXL, CY_PXL],
        [0, 0, 1]
    ])

gaustav_K = np.array([
        [2393.952166119461, -3.410605131648481e-13, 932.3821770809047],
        [0, 2398.118540286656, 628.2649953288065],
        [0, 0, 1]
    ])

drone_P = drone_K @ (np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1))
gaustav_P = gaustav_K @ (np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1))