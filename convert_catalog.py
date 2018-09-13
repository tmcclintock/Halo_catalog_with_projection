"""This file contains the converter object, which takes in a catalog of halo masses (just an array), and spits out new arrays of various richnesses.
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy import interpolate
from scipy.integrate import quad

class Converter(object):
    def __init__(self, intrinsic_scatter, M_min=1.738e11, M_pivot = 4.266e12, alpha=0.77):
        """
        intrinsic_scatter is the percent scatter in the M-lambda_true relation
        M_min is the minimum mass where a halo tends to have a satellite galaxy
        M_pivot is the pivot mass in the M-lambda relation
        alpha is the exponent of the relation

        Note: the default values are from using the SDSS results with the fox cosmology.
        The fiducial intrinsic_scatter for SDSS w/ fox would be 0.26.
        """
        self.sig_int = intrinsic_scatter
        self.M_min = M_min
        self.M_pivot = M_pivot
        self.alpha = alpha
        self.is_setup = False
        self.icdf_is_setup = False
        
    #Functions for various lambda definitions
    def lambda_satellite(self, M):
        """Satellite richness"""
        return ((M-self.M_min)/self.M_pivot)**self.alpha
    
    def lambda_true(self, M):
        """True richness of a halo. 1+satellite richness.
        This is <lambda_true | M>."""
        return 1 + self.lambda_satellite(M)

    def sigma_intrinsic_at_M(self, M):
        """Intrinsic scatter given a mass"""
        return self.sig_int * self.lambda_satellite(M)

    def lambda_true_realization(self, M):
        """Given a mass, return the realization of lambda_true.
        This is <lambda_true | M> + Poisson noise + Gaussian noise."""
        ls = self.lambda_satellite(M)
        sigM = self.sigma_intrinsic_at_M(M)
        poisson_draw = np.random.poisson(lam=ls)
        noise        = np.random.normal(0, sigM)
        return 1. + poisson_draw + noise

    #Functions for evaluating P(lambda_obs| lambda_true)
    def interp_at_x(xbins,param,xout):
        """Dummy function for a InterpolatedUnivariateSpline"""
        dummy=ius(xbins,param,k=1)
        return dummy(xout)

    def setup_splines(self):
        """This is a slow function that should only be run once."""
        if self.is_setup: return
        z_bins    = np.linspace(0.10,0.80,15)
        Nz        = len(z_bins)
        l_in_bins = np.array([1.,3.,5.,7.,9.,12.,15.55555534,20.,24.,26.11111069,30.,36.66666412,40.,47.22222137,57.77777863,68.33332825,78.8888855,89.44444275,100.,120.,140.,160.])
        Nlam      = len(l_in_bins)

        fit_lssmock=np.loadtxt('prj_params_LSSmock DESY1A_v1.1.txt')
        tau_prjmock_fit   = np.reshape(fit_lssmock[0,:], (Nz, Nl))
        mu_prjmock_fit    = np.reshape(fit_lssmock[1,:], (Nz, Nl))
        sig_prjmock_fit   = np.reshape(fit_lssmock[2,:], (Nz, Nl))
        fmask_prjmock_fit = np.reshape(fit_lssmock[3,:], (Nz, Nl))
        fprj_prjmock_fit  = np.reshape(fit_lssmock[4,:], (Nz, Nl))

        # EXTRAPOLATION GRID - setting up splines
        l_in_bins2=np.linspace(1, 300, 300)
        Nl2 = len(l_in_bins2)
        mu_interp  = np.zeros((Nz, Nl2))
        sig_interp = np.zeros((Nz, Nl2))
        tau_interp = np.zeros((Nz, Nl2))
        fm_interp  = np.zeros((Nz, Nl2))
        fp_interp  = np.zeros((Nz, Nl2))
        for i in range(Nz):
            mu_interp[i,:]  = interp_at_x(l_in_bins[:], mu_prjmock_fit[i,:], l_in_bins2)
            sig_interp[i,:] = interp_at_x(l_in_bins[:], sig_prjmock_fit[i,:], l_in_bins2)
            tau_interp[i,:] = interp_at_x(l_in_bins[:], tau_prjmock_fit[i,:], l_in_bins2)
            fm_interp[i,:]  = interp_at_x(l_in_bins[:], fmask_prjmock_fit[i,:], l_in_bins2)
            fp_interp[i,:]  = interp_at_x(l_in_bins[:], fprj_prjmock_fit[i,:], l_in_bins2)
            continue
        fm_interp[fm_interp<0.] = 0.
        fm_interp[fm_interp>1.] = 1.
        fp_interp[fp_interp<0.] = 0.
        fp_interp[fp_interp>1.] = 1.
        self.muprjfit  = interpolate.interp2d(l_in_bins2, z_bins, mu_interp, kind='linear')
        self.sigprjfit = interpolate.interp2d(l_in_bins2, z_bins, sig_interp, kind='linear')
        self.tauprjfit = interpolate.interp2d(l_in_bins2, z_bins, tau_interp, kind='linear')
        self.fmaskfit  = interpolate.interp2d(l_in_bins2, z_bins, fm_interp, kind='linear')
        self.fprjfit   = interpolate.interp2d(l_in_bins2, z_bins, fp_interp, kind='linear')

        #Finished
        self.is_setup = True
        return

    def P_lambda_obs_lambda_true(self, l_in, z_in, l_out):
        """The PDF of a given lambda_obs given a particular lambda_true at some redshift.
        l_in is lambda_true and l_out is lambda_obs.
        This is a Gaussian convolved with a Poissonian."""
        if not self.is_setup: self.setup_splines()

        #Evaluate the quantities for which we have splines for. See setup_splines()
        tau      = self.tauprjfit(l_in, z_in)
        mu       = self.muprjfit(l_in, z_in)
        sig_pure = self.sigprjfit(l_in, z_in)
        f_mask   = self.fmaskfit(l_in, z_in)
        f_prj    = self.fprjfit(l_in, z_in)

        #Calculate some quantities
        sig2_l    = sig_pure**2.
        inv_sq2sig2_l = 1./np.sqrt(2*sig2_l)
        erfc_arg1 = (mu+tau*sig2_l-l_out)*inv_sq2sig2_l #/np.sqrt(2*sig2_l)
        erfc_arg2 = (l_out-mu)*inv_sq2sig2_l #/np.sqrt(2*sig2_l)
        erfc_arg3 = (l_out-mu+l_in)*inv_sq2sig2_l #/np.sqrt(2*sig2_l)
        erfc_arg4 = (mu+tau*sig2_l-l_out-l_in)*inv_sq2sig2_l #/np.sqrt(2*sig2_l)
        exptau = np.exp(0.5*tau*(2.*mu+tau*sig2_l-2.*l_out),dtype='float128')
        gauss  =(1.-f_mask)*(1.-f_prj)*np.exp(-0.5*(l_out-mu)**2./sig2_l)/np.sqrt(2.*np.pi*sig2_l)*2.
        pdf=gauss+(exptau*spc.erfc(erfc_arg1)*((1.-f_mask)*f_prj*tau+f_mask*f_prj/l_in)+
                   f_mask/l_in*(spc.erfc(erfc_arg2)-spc.erfc(erfc_arg3)-f_prj*np.exp(-tau*l_in)*exptau*spc.erfc(erfc_arg4)))
        pdf[pdf<0.]=0. # check for negative probabilities
        return pdf*0.5 # normalize

    #Routines for the CDF(lambda_obs | lambda_true), used for making draws
    #def dummy(self, l_out, l_in, z_in):
    #    """Wrapper for the probability."""
    #    return self.P_lambda_obs_lambda_true(l_in, z_in, l_out)

    def CDF_lambda_obs_lambda_true(self, l_out, l_in, z_in):
        if not self.is_setup: self.setup_splines()
        cdf_temp = np.vectorize(quad)(self.P_lambda_obs_lambda_true, -10., l_out, args=(l_in, z_in), epsrel=1.0e-16, limit=50)
        return cdf_temp[0]

    def setup_iCDF(self, l_in, z_in):
        l_out_grid = np.linspace(-10,lin*3.0) # the grid should be between -inf and +inf
        cdf4interp = self.CDF_lambda_obs_lambda_true(l_out_grid, l_in, z_in)
        # remove numerical error in the integration
        cdf4interp[cdf4interp<0.] = 0.
        cdf4interp[cdf4interp>1.] = 1.
        #inverse of the comulative distribution function used in generate_l_ob()
        self.icdf=ius(cdf4interp,lout_grid, k=1)
        self.current_lambda_true = l_in
        self.current_z_true = z_in
        return

    
    def Draw_from_CDF(self, N_draws, l_in, z_in):
        #Check if the current iCDF spline is setup
        if not self.icdf_is_setup: self.setup_iCDF(l_in, z_in)
        if not np.fabs(l_in - self.current_lambda_true) < 1e-4: self.setup_iCDF(l_in, z_in)
        if not np.fabs(z_in - self.current_z_true) < 1e-4: self.setup_iCDF(l_in, z_in)
        #It's setup, so we are safe
        return self.icdf(np.random.uniform(size=N_draws))
