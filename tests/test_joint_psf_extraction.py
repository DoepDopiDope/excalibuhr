import numpy as np
import unittest

from excalibuhr.utils import gaussian_mixture_psf, joint_psf_extraction


class JointPsfExtractionTest(unittest.TestCase):

  def _check_recovery(self, n_components):
    rng = np.random.default_rng(20260909+n_components)
    nwave, nspatial = 192, 61
    spatial = np.arange(nspatial, dtype=float)
    centroid = 36.2
    separation = 0.721/0.056
    widths = np.array([1.35, 2.8, 5.0])[:n_components]
    weights = np.array([1.0]) if n_components == 1 else np.array([0.78, 0.22])
    if n_components == 3:
        weights = np.array([0.72, 0.22, 0.06])
    primary_psf = gaussian_mixture_psf(spatial, centroid, widths, weights)
    companion_psf = gaussian_mixture_psf(spatial, centroid-separation, widths, weights)
    phase = np.linspace(0, 4*np.pi, nwave)
    primary_true = 1.2e5*(1+0.08*np.sin(phase))
    companion_true = 1800.*(1+0.15*np.cos(1.3*phase))
    background = 35.+0.15*(spatial-spatial.mean())
    model = (primary_true[:, None]*primary_psf +
             companion_true[:, None]*companion_psf + background)
    variance = np.full_like(model, 30.**2)
    data = model+rng.normal(scale=np.sqrt(variance))
    bad = rng.random(data.shape) < 0.035
    data[50, 17] += 2500.

    primary, primary_error, companion, companion_error, diagnostics = (
        joint_psf_extraction(data, variance, bad, obj_cen=36.,
                             companion_sep=separation,
                             psf_components=n_components, block_size=64,
                             background="linear")
    )

    valid = np.isfinite(companion)
    self.assertGreater(valid.mean(), 0.98)
    self.assertLess(abs(np.nanmedian((primary-primary_true)/primary_true)), 0.025)
    self.assertLess(abs(np.nanmedian((companion-companion_true)/companion_true)), 0.08)
    self.assertGreater(np.nanmedian(primary_error), 0)
    self.assertGreater(np.nanmedian(companion_error), 0)
    collapsed_residual = np.nanmedian(diagnostics["residual"], axis=0)
    companion_pixel = int(round(centroid-separation))
    local = collapsed_residual[companion_pixel-2:companion_pixel+3]
    self.assertLess(abs(np.nanmean(local)), 4*np.nanstd(collapsed_residual))
    self.assertTrue(diagnostics["mask"][50, 17])

  def test_recovery_k1(self):
    self._check_recovery(1)

  def test_recovery_k2(self):
    self._check_recovery(2)

  def test_recovery_k3(self):
    self._check_recovery(3)

  def test_joint_psf_rejects_invalid_configuration(self):
    arrays = np.ones((16, 20))
    with self.assertRaises(ValueError):
        joint_psf_extraction(arrays, arrays, arrays.astype(bool), 10,
                             psf_components=0)

  def test_polynomial_regularization_recovers_smooth_psf(self):
    rng = np.random.default_rng(42)
    nwave, nspatial = 256, 61
    spatial = np.arange(nspatial, dtype=float)
    coordinate = np.linspace(-1, 1, nwave)
    primary_true = 8.e4*(1.+0.05*np.sin(5.*coordinate))
    companion_true = np.full(nwave, 1400.)
    data = np.empty((nwave, nspatial))
    for index, value in enumerate(coordinate):
        centroid = 36.1+0.3*value+0.1*value**2
        widths = [1.2+0.08*value, 3.2+0.15*value]
        weights = [0.76-0.03*value, 0.24+0.03*value]
        data[index] = (
            primary_true[index]*gaussian_mixture_psf(
                spatial, centroid, widths, weights) +
            companion_true[index]*gaussian_mixture_psf(
                spatial, centroid-0.721/0.056, widths, weights) + 20.)
    variance = np.full_like(data, 25.**2)
    data += rng.normal(scale=np.sqrt(variance))
    result = joint_psf_extraction(
        data, variance, np.zeros_like(data, dtype=bool), 36.,
        psf_components=2, block_size=32, polynomial_degree=2)
    self.assertLess(abs(np.nanmedian(
        (result[0]-primary_true)/primary_true)), 0.03)
    self.assertLess(abs(np.nanmedian(
        (result[2]-companion_true)/companion_true)), 0.10)
    self.assertEqual(result[4]["polynomial_coefficients"].shape, (3, 4))

  def test_profile_interpolation_is_normalized_and_recovers_flux(self):
    rng = np.random.default_rng(7)
    nwave, nspatial = 160, 61
    spatial = np.arange(nspatial, dtype=float)
    coordinate = np.linspace(-1, 1, nwave)
    primary_true = np.full(nwave, 7.e4)
    companion_true = np.full(nwave, 1200.)
    data = np.empty((nwave, nspatial))
    for index, value in enumerate(coordinate):
        centroid = 36.+0.25*value
        widths = [1.1+0.05*value, 3.5-0.1*value]
        data[index] = (
            primary_true[index]*gaussian_mixture_psf(
                spatial, centroid, widths, [0.75, 0.25]) +
            companion_true[index]*gaussian_mixture_psf(
                spatial, centroid-0.721/0.056, widths, [0.75, 0.25]) + 15.)
    variance = np.full_like(data, 20.**2)
    data += rng.normal(scale=np.sqrt(variance))
    result = joint_psf_extraction(
        data, variance, np.zeros_like(data, dtype=bool), 36.,
        psf_components=2, block_size=32, interpolate_psf=True)
    self.assertLess(abs(np.nanmedian(
        (result[0]-primary_true)/primary_true)), 0.03)
    self.assertLess(abs(np.nanmedian(
        (result[2]-companion_true)/companion_true)), 0.10)
    self.assertTrue(result[4]["interpolate_psf"])


if __name__ == "__main__":
    unittest.main()
