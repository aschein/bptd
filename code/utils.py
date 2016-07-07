import numpy as np
import numpy.random as rn


MIN_GAMMA_SHAPE = 1e-5
MIN_GAMMA_SCALE = 1e-5
MIN_GAMMA_SAMPLE = 1e-300


def sample_gamma(shp, sca, size=None):
    if isinstance(shp, np.ndarray):
        shp = np.clip(shp, a_min=MIN_GAMMA_SHAPE, a_max=None, out=shp)
    else:
        shp = np.clip(shp, a_min=MIN_GAMMA_SHAPE, a_max=None)

    if isinstance(sca, np.ndarray):
        sca = np.clip(sca, a_min=MIN_GAMMA_SCALE, a_max=None, out=sca)
    else:
        sca = np.clip(sca, a_min=MIN_GAMMA_SCALE, a_max=None)
    return np.clip(rn.gamma(shp, sca, size=size), a_min=MIN_GAMMA_SAMPLE, a_max=None)


if __name__ == '__main__':
    print sample_gamma(np.ones(10) * 1e-10, 1e-1000, size=(2, 10)).mean()
