import numpy as np
from scipy import signal
from astropy.convolution import Gaussian2DKernel
from skimage.transform import resize

def add_noise_to_radiograph(
    direct,
    x_stddevlevel=None,
    y_stddevlevel=None,
    sigma_scatter=10,
    scatter_scaling=0.2,
    siglevel=None,
    x_tiltlevel=None,
    y_tiltlevel=None,
    gammalevel=None,
    photonlevel=None,
    scatter_polynomial_order=1,
    gammascaling=1,
    photonscaling=1,
    gamma_kernel_path='../kernels/gamma_kernel.dat',
    photon_kernel_path='../kernels/photon_kernel.dat',
    detector_kernel_path='../kernels/detector_blur_az.dat'
):
    """
    Adds synthetic noise to a clean radiograph image using multiple physics-inspired components:
    - anisotropic Gaussian source blur
    - detector blur
    - background tilt field
    - correlated scatter
    - gamma and photon noise

    Parameters:
        direct: 2D NumPy array (clean radiograph)
        x_stddevlevel, y_stddevlevel: Gaussian kernel std devs for x and y (sampled if None)
        sigma_scatter: stddev for correlated scatter Gaussian kernel
        scatter_scaling: scaling for scatter field
        siglevel: strength of tilt background field (sampled if None)
        x_tiltlevel, y_tiltlevel: polynomial coefficients for x and y tilt (sampled if None)
        gammalevel: expected gamma photons per pixel (sampled if None)
        photonlevel: expected photon counts (sampled if None)
        scatter_polynomial_order: degree of background polynomial
        gammascaling, photonscaling: scaling of correlated gamma/photon noise
        gamma_kernel_path, photon_kernel_path, detector_kernel_path: file paths for precomputed kernels

    Returns:
        noisy_rad: 2D NumPy array with synthetic noise added
    """

    # ---------------- Load fixed kernels ---------------- #
    gamma = np.genfromtxt(gamma_kernel_path).reshape(301, 301)
    photon = np.genfromtxt(photon_kernel_path).reshape(81, 81)
    detector = np.genfromtxt(detector_kernel_path).reshape(201, 201)

    dim = 650
    direct_zoom = np.copy(direct)

    # ------------------- Default Parameter Sampling ------------------- #
    num_samples = 10
    gaussian_min, gaussian_max = 1.0, 3.1
    tilt_min, tilt_max = -0.000039, 0.0000039
    siglevel_min, siglevel_max = 0.5, 1.6
    gamma_min, gamma_max = 39000, 50000
    photon_min, photon_max = 350, 450

    gaussian_range = np.linspace(gaussian_min, gaussian_max, num_samples)
    tilt_range = np.linspace(tilt_min, tilt_max, num_samples)
    siglevel_range = np.linspace(siglevel_min, siglevel_max, num_samples)
    gamma_range = np.linspace(gamma_min, gamma_max, num_samples)
    photon_range = np.linspace(photon_min, photon_max, num_samples)

    if x_stddevlevel is None:
        x_stddevlevel = np.random.choice(gaussian_range)
    if y_stddevlevel is None:
        y_stddevlevel = np.random.choice(gaussian_range)
    if x_tiltlevel is None:
        x_tiltlevel = np.random.choice(tilt_range)
    if y_tiltlevel is None:
        y_tiltlevel = np.random.choice(tilt_range)
    if siglevel is None:
        siglevel = np.random.choice(siglevel_range)
    if gammalevel is None:
        gammalevel = np.random.choice(gamma_range)
    if photonlevel is None:
        photonlevel = np.random.choice(photon_range)

    # ---------------- Source Blur ---------------- #
    gaussian_angle = np.arange(5., 26., 1.) * np.pi / 180.0
    angle = np.random.choice(gaussian_angle, 1)

    gaussian_2D_kernel = Gaussian2DKernel(x_stddev=x_stddevlevel, y_stddev=y_stddevlevel,
                                          theta=angle, x_size=141, y_size=141)

    imgconlv1 = signal.fftconvolve(direct_zoom, gaussian_2D_kernel, mode='same')
    imgconlv1 -= imgconlv1.min()

    # Detector blur convolution
    imgconlv = signal.fftconvolve(imgconlv1, detector, mode='same')

    # ---------------- Correlated Scatter ---------------- #
    scatter = Gaussian2DKernel(x_stddev=sigma_scatter)
    scatterimage = signal.fftconvolve(direct_zoom, scatter, mode='same') * scatter_scaling

    # ---------------- Background Tilt ---------------- #
    tilt_crop_min, tilt_crop_max = 220, 420
    maxsig = np.mean(imgconlv[tilt_crop_min:tilt_crop_max, tilt_crop_min:tilt_crop_max])
    avsiglevel = maxsig * siglevel

    m, n = imgconlv.shape
    x1, x2 = np.mgrid[:m, :n]

    tilt = 0
    for order in range(1, scatter_polynomial_order + 1):
        tilt += x_tiltlevel * x1**order + y_tiltlevel * x2**order

    tilt = tilt / m**(scatter_polynomial_order - 1)
    tilt = 1 + tilt - np.mean(tilt)
    tilt *= avsiglevel

    # ---------------- Combine Base Signal ---------------- #
    signalblurscatter = imgconlv + tilt + scatterimage
    maxtotalsig = np.max(signalblurscatter)
    normsignal = signalblurscatter / maxtotalsig

    # ---------------- Gamma Noise ---------------- #
    gammasignal = normsignal * gammalevel
    gammanoise = (np.random.poisson(gammasignal) - gammasignal) / gammalevel
    correlatedgamma = gammascaling * signal.fftconvolve(gammanoise, gamma, mode='same')

    # ---------------- Photon Noise ---------------- #
    photonsignal = normsignal * photonlevel
    photonnoise = (np.random.poisson(photonsignal) - photonsignal) / photonlevel
    correlatedphoton = photonscaling * signal.fftconvolve(photonnoise, photon, mode='same')

    # ---------------- Final Output ---------------- #
    signalblurscatternoise = (normsignal + correlatedgamma + correlatedphoton) * maxtotalsig
    noisy_rad = np.copy(signalblurscatternoise)
    noisy_rad[noisy_rad < 0] = 0  # clip negative pixels

    return noisy_rad
