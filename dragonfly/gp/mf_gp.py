"""
  Implements the kernel, GP and fitter for multi-fidelity GPs.
  --kandasamy@cs.cmu.edu
"""
from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=abstract-class-not-used

# Local imports
from . import kernel as gp_kernel
from .gp_core import GP, GPFitter, mandatory_gp_args
from ..utils.option_handler import load_options
from ..utils.reporters import get_reporter
from ..utils.ancillary_utils import get_list_of_floats_as_str


def get_ZX_from_ZZ_XX(ZZ, XX):
  """ Get a combined representation for the fidelity and domian data. """
  if hasattr(ZZ, '__iter__') and len(ZZ) == len(XX):
    return [(z, x) for (z, x) in zip(ZZ, XX)]
  else:
    return (ZZ, XX)


class MFGP(GP):
  """ A GP to be used in multi-fidelity settings. """

  def __init__(self, ZZ, XX, YY, mf_kernel,
               mean_func, noise_var, *args, **kwargs):
    """ Constructor. ZZ, XX, YY are the fidelity points, domain points and labels
        respectively.
        mf_kernel is either a combined kernel or a namespace with the following fields:
        {scale, fidel_kernel, domain_kernel}.
    """
    self.ZZ = list(ZZ)
    self.XX = list(XX)
    self.YY = list(YY)
    if not isinstance(mf_kernel, gp_kernel.Kernel):
      kernel = gp_kernel.CartesianProductKernel(mf_kernel.scale,
                 [mf_kernel.fidel_kernel, mf_kernel.domain_kernel])
      self.fidel_kernel = mf_kernel.fidel_kernel
      self.domain_kernel = mf_kernel.domain_kernel
    else:
      kernel = mf_kernel
    ZX = self.get_ZX_from_ZZ_XX(ZZ, XX) # The 'X' data
    # Call super constructor
    super(MFGP, self).__init__(ZX, YY, kernel, mean_func, noise_var, *args, **kwargs)

  @classmethod
  def get_ZX_from_ZZ_XX(cls, ZZ, XX):
    """ Get a combined representation for the fidelity and domian data.
        Can be overridden by a child class if there is a more efficient representation."""
    return get_ZX_from_ZZ_XX(ZZ, XX)

  def eval_at_fidel(self, ZZ_test, XX_test, *args, **kwargs):
    """ Evaluates the GP at [ZZ_test, XX_test]. Read eval in gp_core.GP for more details.
    """
    ZX_test = self.get_ZX_from_ZZ_XX(ZZ_test, XX_test)
    return self.eval(ZX_test, *args, **kwargs)

  def eval_at_fidel_with_hallucinated_observations(self, ZZ_test, XX_test,
                                                   ZZ_halluc, XX_halluc, *args, **kwargs):
    """ Evaluates with hallucinated observations. """
    ZX_test = self.get_ZX_from_ZZ_XX(ZZ_test, XX_test)
    ZX_halluc = self.get_ZX_from_ZZ_XX(ZZ_halluc, XX_halluc)
    return self.eval_with_hallucinated_observations(ZX_test, ZX_halluc, *args, **kwargs)

  def set_mf_data(self, ZZ, XX, YY, build_posterior=True):
    """ Sets the MF data for the GP. """
    self.ZZ = list(ZZ)
    self.XX = list(XX)
    self.YY = list(YY)
    ZX = self.get_ZX_from_ZZ_XX(ZZ, XX) # The 'X' data
    super(MFGP, self).set_data(ZX, YY, build_posterior)

  def add_mf_data_multiple(self, ZZ_new, XX_new, YY_new, *args, **kwargs):
    """ Adds new data to the multi-fidelity GP. """
    ZX_new = self.get_ZX_from_ZZ_XX(ZZ_new, XX_new)
    self.ZZ.extend(ZZ_new)
    self.XX.extend(XX_new)
    self.add_data_multiple(ZX_new, YY_new, *args, **kwargs)

  def add_mf_data_single(self, zz_new, xx_new, yy_new, *args, **kwargs):
    """ Adds a single new data to the multi-fidelity GP. """
    self.add_mf_data_multiple([zz_new], [xx_new], [yy_new], *args, **kwargs)

  def draw_mf_samples(self, num_samples, ZZ_test=None, XX_test=None, *args, **kwargs):
    """ Draws samples from a multi-fidelity GP. """
    ZX_test = None if ZZ_test is None else self.get_ZX_from_ZZ_XX(ZZ_test, XX_test)
    return self.draw_samples(num_samples, ZX_test, *args, **kwargs)

  def get_fidel_kernel(self):
    """ Return the fidel_space kernel. """
    return self.fidel_kernel

  def get_domain_kernel(self):
    """ Return the domain kernel. """
    return self.domain_kernel

  def _child_str(self):
    """ Returns a string representation of the MF-GP. """
    if hasattr(self, 'fidel_kernel') and hasattr(self, 'domain_kernel'):
      fidel_ke_str = self._get_kernel_str(self.fidel_kernel)
      domain_ke_str = self._get_kernel_str(self.domain_kernel)
      kernel_str = 'fid:: %s, dom:: %s'%(fidel_ke_str, domain_ke_str)
    else:
      kernel_str = str(self.kernel)
    ret = 'scale: %0.3f, %s'%(self.kernel.hyperparams['scale'], kernel_str)
    return ret

  @classmethod
  def _get_kernel_str(cls, kern):
    """ Gets a string format of the kernel depending on whether it is SE/Poly. """
    if isinstance(kern, gp_kernel.ExpDecayKernel):
      ret = 'expd: offs=%0.3f, pow=%s'%(kern.hyperparams['offset'],
        get_list_of_floats_as_str(kern.hyperparams['powers']))
    elif isinstance(kern, gp_kernel.SEKernel) or isinstance(kern, gp_kernel.MaternKernel):
      hp_name = 'dim_bandwidths'
      kern_name = 'se' if isinstance(kern, gp_kernel.SEKernel) else \
                  'matern(%0.1f)'%(kern.hyperparams['nu'])
      if kern.dim > 4:
        ret = '%0.4f(avg)'%(kern.hyperparams[hp_name].mean())
      else:
        ret = get_list_of_floats_as_str(kern.hyperparams[hp_name])
      ret = kern_name + ': ' + ret
    elif isinstance(kern, gp_kernel.PolyKernel):
      ret = 'poly: %s'%(get_list_of_floats_as_str(kern.hyperparams['dim_scalings']))
    else:    # Return an empty string.
      ret = str(kern)
    return ret


class MFGPFitter(GPFitter):
  """ A GP Fitter for Multi-fidelity GPs. This is mostly a wrapper for MFGPs that
      want to use GPFitter. All the heavy lifting is happening in GPFitter. """
  # pylint: disable=abstract-method

  def __init__(self, ZZ, XX, YY, options=None, reporter=None):
    """ Constructor. """
    reporter = get_reporter(reporter)
    options = load_options(mandatory_gp_args, partial_options=options)
    self.ZZ = ZZ
    self.XX = XX
    self.YY = YY
    self.num_tr_data = len(self.YY)
    ZX = get_ZX_from_ZZ_XX(ZZ, XX)
    super(MFGPFitter, self).__init__(ZX, YY, options, reporter)

