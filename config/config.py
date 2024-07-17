import numpy as np

FX_PXL = 1936.9
FY_PXL = 1943.7
CX_PXL = 1344.0
CY_PXL = 757.9608
DRONE_K = np.array([
        [FX_PXL, 0, CX_PXL],
        [0, FY_PXL, CY_PXL],
        [0, 0, 1]
    ])

GAUSTAV_K = np.array([
        [2393.952166119461, -3.410605131648481e-13, 932.3821770809047],
        [0, 2398.118540286656, 628.2649953288065],
        [0, 0, 1]
    ])

GAUSTAV_PATH = './dataset/GustavIIAdolf'
HIGH_ALTITUDA_PATH = './dataset/drone_high_altitude'
PLY_PATH = './result/result.ply'