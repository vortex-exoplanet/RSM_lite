"""
PyRSM_lite is a simplified version of the PyRSM python package for exoplanets detection which applies the Regime Switching Model (RSM) framework on ADI (and ADI+SDI) sequences (see Dahlqvist et al. A&A, 2020, 633, A95).
The RSM map algorithm relies on one or several PSF subtraction techniques to process one or multiple ADI sequences before computing a final probability map. Considering the large
set of parameters needed for the computation of the RSM detection map (parameters for the selected PSF-subtraction techniques as well as the RSM algorithm itself), a parameter selection framework
called auto-RSM (Dahlqvist et al., A&A, 2021, 656, A54) is proposed to automatically select the optimal parametrization. The proposed multi-step parameter optimization framework can be divided into 
three main steps, (i) the selection of the optimal set of parameters for the considered PSF-subtraction techniques, (ii) the optimization of the RSM approach parametrization, and (iii) the 
selection of the optimal set of PSF-subtraction techniques and ADI sequences to be considered when generating the final detection map. 

The add_cube and add_model methods allows to consider several cubes and models to generate
the cube of residuals used to compute the RSM map. The cube should be provided by the same instrument
or rescaled to a unique pixel size. The class can be used with only ADI and ADI+SDI. A specific PSF should 
be provided for each cube. In the case of ADI+SDI a single psf should be provided per cube (typically the PSF
average over the set of frequencies Five different models and two forward model variants are available. 
Each model can be parametrized separately. Five different models and two forward model variants are available. 
The method like_esti allows the estimation of a cube of likelihoods containing for each pixel
and each frame the likelihood of being in the planetary or the speckle regime. These likelihood cubes
are then used by the probmap_esti method to provide the final probability map based on the RSM framework. 

The second set of methods regroups the four main methods used by the auto-RSM/auto-S/N framework.
The opti_model method allows the optimization of the PSF subtraction techniques parameters based on the 
minimisation of the average contrast. The opti_RSM method takes care of the optimization of the parameters 
of the RSM framework (all related to the computation of the likelihood associated to every pixels and frames). The
third method RSM_combination, relies on a greedy selection algorithm to define the optimal set of 
ADI sequences and PSF-subtraction techniques to consider when generating the final detection map using the RSM
approach. Finally, the opti_map method allows to compute the final RSM detection map. The optimization of
the parameters can be done using the reversed parallactic angles, blurring potential planetary signals while
keeping the main characteristics of the speckle noise. An S/N map based code is also proposed and encompasses
the opti_model, the RSM_combination and the opti_map methods. For the last two methods, the SNR 
parameter should be set to True.

The last set of methods regroups the methods allowing the computation of contrast curves and the 
characterization of a detected astrophysical signals. The self.contrast_curve method allows the computation
 of a contrast curve at a pre-defined completeness level (see Dahlqvist et al. 2021 for more details), 
 while the self.contrast_matrix method provided contrast curves for a range of completeness levels defined
 by the number of fake companion injected (completeness level from 1/n_fc to 1-1/n_fc with n_fc the number
 of fake companions). This last method provides a good representation of the contrast/completeness 
 distribution but requires a longer computation time. The self.target_charact() method allows the 
 estimation of the photometry and astrometry of a detected signal (see Dahlqvist et al. 2022 for more details)

Part of the code have been directly inspired by the VIP and PyKLIP packages for the estimation of the cube
of residuas and forward model PSF.
"""

__author__ = 'Carl-Henrik Dahlqvist'

import numpy as np
from scipy.interpolate import Rbf
import pandas as pd
import numpy.linalg as la
from vip_hci.var import cube_filter_highpass
from vip_hci.preproc import check_pa_vector,check_scal_vector
from vip_hci.preproc.derotation import _find_indices_adi
from vip_hci.preproc.rescaling import _find_indices_sdi
import scipy as sp
from multiprocessing import cpu_count
from vip_hci.psfsub.svd import get_eigenvectors
from vip_hci.psfsub.llsg import _patch_rlrps
from vip_hci.preproc import cube_rescaling_wavelengths as scwave
from sklearn.decomposition import NMF
from scipy.optimize import curve_fit
from skimage.draw import disk 
import vip_hci as vip
from vip_hci.var import get_annulus_segments,frame_center,prepare_matrix
from vip_hci.preproc import frame_crop,cube_derotate,cube_crop_frames,cube_collapse
from vip_hci.fm import cube_inject_companions, frame_inject_companion, normalize_psf
from hciplot import plot_frames 
from vip_hci.config.utils_conf import pool_map, iterable
from multiprocessing import Pool
import photutils
from scipy import stats
import pickle
import multiprocessing as mp
import sklearn.gaussian_process as gp
from scipy.stats import norm
import pyswarms.backend as P
from pyswarms.backend.topology import Star  
import scipy as sc
import lmfit
from scipy.optimize import minimize
import math
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
#from utils import (llsg_adisdi,loci_adisdi,do_pca_patch,_decompose_patch,
#annular_pca_adisdi,NMF_patch,nmf_adisdi,LOCI_FM,KLIP_patch,perturb,KLIP,
#get_time_series,poly_fit,interpolation,remove_outliers,check_delta_sep ,rot_scale,
#bayesian_optimization,_define_annuli,Hessian)



var_dict = {}

def init_worker(X, X_shape):
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape
    
def shared_array(name: str, shape=None):
    mp_array = globals()[name]
    return np.frombuffer(mp_array.get_obj(), dtype=np.dtype(mp_array.get_obj()._type_)).reshape(shape)

            
class PyRSM:

    def __init__(self,fwhm,minradius,maxradius,pxscale=0.1,ncore=1,max_r_fm=30,inv_ang=True,trunc=None,imlib='opencv', interpolation='lanczos4'):
        
        """
        Initialization of the PyRSM object on which the add_cube and add_model 
        functions will be applied to parametrize the model. The functions 
        like_esti and prob_esti will then allow the computation of the
        likelihood cubes and the final RSM map respectively.
        
        Parameters
        ----------
        fwhm: int
            Full width at half maximum for the instrument PSF
        minradius : int
            Center radius of the first annulus considered in the RSM probability
            map estimation. The radius should be larger than half 
            the value of the 'crop' parameter 
        maxradius : int
            Center radius of the last annulus considered in the RSM probability
            map estimation. The radius should be smaller or equal to half the
            size of the image minus half the value of the 'crop' parameter 
        pxscale : float
            Value of the pixel in arcsec/px. Only used for printing plots when
            ``showplot=True`` in like_esti. 
        ncore : int, optional
            Number of processes for parallel computing. By default ('ncore=1') 
            the algorithm works in single-process mode. 
        max_r_fm: int, optional
            Largest radius for which the forward model version of KLIP or LOCI
            are used, when relying on forward model versions of RSM. Forward model 
            versions of RSM have a higher performance at close separation, considering
            their computation time, their use should be restricted to small angular distances.
            Default is None, i.e. the foward model version are used for all considered
            angular distance.
        inv_ang: bool, optional
            If True, the sign of the parallactic angles of all ADI sequence is flipped for
            the entire optimization procedure. Default is True.
        trunc: int, optional
            Maximum angular distance considered for the full-frame parameter optimization. Defaullt is None.
        imlib : str, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
        interpolation : str, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
        """
    
        self.ncore = ncore 
        self.minradius = minradius 
        self.maxradius = maxradius 
        self.fwhm = fwhm 
        self.pxscale = pxscale
        self.imlib=imlib
        self.interpolation=interpolation
        
        self.inv_ang=inv_ang
        self.trunc=trunc
        
        if max_r_fm is not None:
            self.max_r=max_r_fm
        else:
            self.max_r=maxradius
        
        self.psf = []         
        self.cube=[]
        self.pa=[]
        self.scale_list=[]
        
        self.model = []    
        self.delta_rot = []
        self.delta_sep = []
        self.nsegments = [] 
        self.ncomp = []    
        self.rank = [] 
        self.tolerance = []
        self.asize=[]
        self.opti_bound=[]
        self.psf_fm=[]

        self.intensity = [] 
        self.distri = []         
        self.var = [] 
        self.interval=[]
        self.crop=[]
        self.crop_range=[]
        
        self.like_fin=[]
        self.distrisel=[]
        self.mixval=[]  
        self.fiterr=[]
        self.probmap=None
        
        self.param=None
        self.opti=False
        self.contrast=[]
        self.ini_esti=[]
        
        self.opti_sel=None
        self.threshold=None
        
        self.final_map=None
        
        
    def add_cube(self,psf, cube, pa, scale_list=None):
     
        """    
        Function used to add an ADI seuqence to the set of cubes considered for
        the RSM map estimation.
        
        Parameters
        ----------

        psf : numpy ndarray 2d
            2d array with the normalized PSF template, with an odd shape.
            The PSF image must be centered wrt to the array! Therefore, it is
            recommended to run the function ``normalize_psf`` to generate a 
            centered and flux-normalized PSF template.
        cube : numpy ndarray, 3d or 4d
            Input cube (ADI sequences), Dim 1 = temporal axis, Dim 2-3 = spatial axis
            Input cube (ADI + SDI sequences), Dim 1 = wavelength, Dim 2=temporal axis
            Dim 3-4 = spatial axis     
        pa : numpy ndarray, 1d
            Parallactic angles for each frame of the ADI sequences. 
        scale_list: numpy ndarray, 1d, optional
            Scaling factors in case of IFS data (ADI+mSDI cube). Usually, the
            scaling factors are the central channel wavelength divided by the
            shortest wavelength in the cube (more thorough approaches can be used
            to get the scaling factors). This scaling factors are used to re-scale
            the spectral channels and align the speckles. Default is None
        """
        if cube.shape[-1]%2==0:
            raise ValueError("Images should have an odd size")

        self.psf.append(psf)         
        self.cube.append(cube)
        self.pa.append(pa)
        self.scale_list.append(scale_list)
        self.like_fin.append([])
        self.distrisel.append([])
        self.mixval.append([])
        self.fiterr.append([])
              

    def add_method(self, model,delta_rot=0.5,delta_sep=0.1,asize=5,nsegments=1,ncomp=20,rank=5,tolerance=1e-2,interval=[5],var='FR',crop_size=5, crop_range=1,opti_bound=None):

        
        """
        Function used to add a model to the set of post-processing techniques used to generate
        the cubes of residuals on which is based the computation of the likelihood of being 
        in either the planetary of the background regime. These likelihood matrices allow
        eventually the definition of the final RSM map.
        
        Parameters
        ----------
    
        model : str
            Selected ADI-based post-processing techniques used to 
            generate the cubes of residuals feeding the Regime Switching model.
            'APCA' for annular PCA, NMF for Non-Negative Matrix Factorization, LLSG
            for Local Low-rank plus Sparse plus Gaussian-noise decomposition, LOCI 
            for locally optimized combination of images and'KLIP' for Karhunen-Loeve
            Image Projection. There exitsts a foward model variant of KLIP and LOCI called 
            respectively 'FM KLIP' and 'FM LOCI'.
        delta_rot : float, optional
            Factor for tunning the parallactic angle threshold, expressed in FWHM.
            Default is 0.5 (excludes 0.5xFHWM on each side of the considered frame).
        delta_sep : float, optional
            The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
            Default is 0.1.
        asize : int, optional
            Width in pixels of each annulus.When a single. Default is 5. 
        n_segments : int, optional
            The number of segments for each annulus. Default is 1 as we work annulus-wise.
        ncomp : int, optional
            Number of components used for the low-rank approximation of the 
            speckle field with 'APCA', 'KLIP' and 'NMF'. Default is 20.
        rank : int, optional        
            Expected rank of the L component of the 'LLSG' decomposition. Default is 5.
        tolerance: float, optional
            Tolerance level for the approximation of the speckle field via a linear 
            combination of the reference images in the LOCI algorithm. Default is 1e-2.
        interval: list of float or int, optional
            List of values taken by the delta parameter defining, when mutliplied by the 
            standard deviation, the strengh of the planetary signal in the Regime Switching model.
            Default is [5]. The different delta paramaters are tested and the optimal value
            is selected via maximum likelmihood.
        var: str, optional
            Model used for the residual noise variance estimation. Five different approaches
            are proposed: 'ST', 'FR', and 'FM'. While all six can be used when 
            intensity='Annulus', only the last three can be used when intensity='Pixel'. 
            When using ADI+SDI dataset only 'FR' and 'FM' can be used. Default is 'ST'.
            
            'ST': consider every frame and pixel in the selected annulus with a 
            width equal to asize (default approach)
            
            'FR': consider the pixels in the selected annulus with a width equal to asize
            but separately for every frame.
            
            'FM': consider the pixels in the selected annulus with a width 
            equal to asize but separately for every frame. Apply a mask one FWHM 
            on the selected pixel and its surrounding.

        modtocube: bool, optional
            Parameter defining if the concatenated cube feeding the RSM model is created
            considering first the model or the different cubes. If 'modtocube=False',
            the function will select the first cube then test all models on it and move 
            to the next one. If 'modtocube=True', the model will select one model and apply
            it to every cubes before moving to the next model. Default is True.
        crop_size: int, optional
            Part of the PSF tempalte considered is the estimation of the RSM map
        crop_range: int, optional
            Range of crop sizes considered in the estimation of the RSM map, starting with crop_size
            and increasing the crop size incrementally by 2 pixels up to a crop size of 
            crop_size + 2 x (crop_range-1).
    
        opti_bound: list, optional
            List of boundaries used for the parameter optimization. 
                - For APCA: [[L_ncomp,U_ncomp],[L_nseg,U_nseg],[L_delta_rot,U_delta_rot]]
                  Default is [[15,45],[1,4],[0.25,1]]
                - For NMF: [[L_ncomp,U_ncomp]]
                  Default is [[2,20]]
                - For LLSG: [[L_ncomp,U_ncomp],[L_nseg,U_nseg]]
                  Default is [[1,10],[1,4]]
                - For LOCI: [[L_tolerance,U_tolerance],[L_delta_rot,U_delta_rot]]
                  Default is [[1e-3,1e-2],[0.25,1]]
                - For FM KLIP: [[L_ncomp,U_ncomp],[L_delta_rot,U_delta_rot]]
                  Default is [[15,45],[0.25,1]]
                - For FM LOCI: [[L_tolerance,U_tolerance],[L_delta_rot,U_delta_rot]]
                  Default is [[1e-3,1e-2],[0.25,1]]
            with L_ the lower bound and U_ the Upper bound.                
        """

        
        if crop_size+2*(crop_range-1)>=2*round(self.fwhm)+1:
            raise ValueError("Maximum cropsize should be lower or equal to two FWHM, please change accordingly either 'crop_size' or 'crop_range'")
            
        if any(var in myvar for myvar in ['ST','SM','FR'])==False:
            raise ValueError("'var' not recognized")

        for c in range(len(self.cube)):
            if self.cube[c].ndim==4:
                if any(var in myvar for myvar in ['ST','SM'])==True:
                    raise ValueError("'var' not recognized for ADI+SDI cube'. 'var' should be 'FR'")   
                check_delta_sep(self.scale_list[c],delta_sep,self.minradius,self.fwhm,c)
        
        self.model.append(model)
        self.delta_rot.append(np.array([np.repeat(delta_rot,(len(self.cube)))]*(self.maxradius+asize)))
        self.delta_sep.append(np.array([np.repeat(delta_sep,(len(self.cube)))]*(self.maxradius+asize)))
        self.nsegments.append(np.array([np.repeat(nsegments,(len(self.cube)))]*(self.maxradius+asize)))
        self.ncomp.append(np.array([np.repeat(ncomp,(len(self.cube)))]*(self.maxradius+asize)))  
        self.rank.append(np.array([np.repeat(rank,(len(self.cube)))]*(self.maxradius+asize)))
        self.tolerance.append(np.array([np.repeat(tolerance,(len(self.cube)))]*(self.maxradius+asize)))
        self.interval.append(np.array([[interval]*(len(self.cube))]*(self.maxradius+asize)))
        self.asize.append(asize)
        self.opti_bound.append(opti_bound)        
        self.var.append(np.array([np.repeat(var,(len(self.cube)))]*(self.maxradius+asize)))
        self.crop.append(np.array([np.repeat(crop_size,(len(self.cube)))]*(self.maxradius+asize)))        

        self.crop_range.append(crop_range)

        for i in range(len(self.cube)):
            
            self.like_fin[i].append(None)
            self.distrisel[i].append(None)
            self.mixval[i].append(None)
            self.fiterr[i].append(None)
            
            
                    
    def save_parameters(self,folder,name):
        
        with open(folder+name+'.pickle', "wb") as save:
            pickle.dump([self.model, self.delta_rot,self.nsegments,self.ncomp,self.rank,self.tolerance,self.asize,self.var,self.crop,self.crop_range,self.opti_sel,self.threshold,self.flux_opti,self.opti_theta,self.interval],save)
        
        
    def load_parameters(self,name):
        


        with open(name+'.pickle', "rb") as read:
            saved_param = pickle.load(read)

        
        self.model= saved_param[0]
        self.delta_rot = saved_param[1]
        self.nsegments = saved_param[2]
        self.ncomp = saved_param[3]   
        self.rank = saved_param[4]
        self.tolerance = saved_param[5]
        self.asize= saved_param[6]       
        self.var = saved_param[7]
        self.crop= saved_param[8]
        self.crop_range= saved_param[9]
        self.opti_sel=saved_param[10]
        self.threshold=saved_param[11]
        self.flux_opti=saved_param[12]
        self.opti_theta=saved_param[13]
        self.interval=saved_param[14]
        
    def likelihood(self,ann_center,cuben,modn,mcube,cube_fc=None,verbose=True):
         
        if type(mcube) is not np.ndarray:
            mcube = shared_array('resi_cube', shape=mcube)
        
        if mcube.ndim==4:
            z,n,y,x=mcube.shape
            
        else:
            n,y,x=mcube.shape
            z=1
        
        range_int=len(self.interval[modn][ann_center,cuben])
        
        likemap=np.zeros(((n*z)+1,x,y,range_int,2,self.crop_range[modn]))
        
        np.random.seed(10) 

        def likfcn(cuben,modn,mean,var,mixval,mcube,ann_center,distrim,probcube=0,var_f=None):

            #np.random.seed(10) 
            
            phi=np.zeros(2)
            if self.cube[cuben].ndim==4:
                n_t=self.cube[cuben].shape[1]  

            ceny, cenx = frame_center(mcube[0])
            indicesy,indicesx=get_time_series(mcube,ann_center)

            range_int=len(self.interval[modn][ann_center,cuben])
                
            if self.model[modn]=='FM KLIP' or self.model[modn]=='FM LOCI':
                
                psf_fm_out=np.zeros((len(indicesx),mcube.shape[0],int(2*round(self.fwhm)+1),int(2*round(self.fwhm)+1)))

            if (self.crop[modn][ann_center,cuben]+2*(self.crop_range[modn]-1))!=self.psf[cuben].shape[-1]:
                if len(self.psf[cuben].shape)==2:
                    psf_temp=frame_crop(self.psf[cuben],int(self.crop[modn][ann_center,cuben]+2*(self.crop_range[modn]-1)),cenxy=[int(self.psf[cuben].shape[1]/2),int(self.psf[cuben].shape[1]/2)],verbose=False)
                else:
                    psf_temp=cube_crop_frames(self.psf[cuben],int(self.crop[modn][ann_center,cuben]+2*(self.crop_range[modn]-1)),xy=(int(self.psf[cuben].shape[1]/2),int(self.psf[cuben].shape[1]/2)),verbose=False)
            else:
                psf_temp=self.psf[cuben]

            for i in range(0,len(indicesy)):

                psfm_temp=None
                cubind=0
                poscenty=indicesy[i]
                poscentx=indicesx[i]
                 
                                #PSF forward model computation for KLIP

                if self.model[modn]=='FM KLIP':
                    
                    an_dist = np.sqrt((poscenty-ceny)**2 + (poscentx-cenx)**2)
                    theta = np.degrees(np.arctan2(poscenty-ceny, poscentx-cenx))    


                    model_matrix=cube_inject_companions(np.zeros_like(mcube), self.psf[cuben], self.pa[cuben], 1,an_dist, plsc=0.1, theta=theta, n_branches=1,verbose=False)

                    pa_threshold = np.rad2deg(2 * np.arctan(self.delta_rot[modn][ann_center,cuben] * self.fwhm / (2 * ann_center)))
                    mid_range = np.abs(np.amax(self.pa[cuben]) - np.amin(self.pa[cuben])) / 2
                    if pa_threshold >= mid_range - mid_range * 0.1:
                        pa_threshold = float(mid_range - mid_range * 0.1)

                    psf_map=np.zeros_like(model_matrix)
                    indices = get_annulus_segments(mcube[0], ann_center-int(self.asize[modn]/2),int(self.asize[modn]),1)


                    for b in range(0,n):
                        psf_map_temp = perturb(b,model_matrix[:, indices[0][0], indices[0][1]], self.ncomp[modn][ann_center,cuben],evals_matrix, evecs_matrix,
                                   KL_basis_matrix,sci_mean_sub_matrix,refs_mean_sub_matrix, self.pa[cuben], self.fwhm, pa_threshold, ann_center)
                        psf_map[b,indices[0][0], indices[0][1]]=psf_map_temp-np.mean(psf_map_temp)


                    psf_map_der = cube_derotate(psf_map, self.pa[cuben], imlib='opencv',interpolation='lanczos4')
                    psfm_temp=cube_crop_frames(psf_map_der,int(2*round(self.fwhm)+1),xy=(poscentx,poscenty),verbose=False)
                    psf_fm_out[i,:,:,:]=psfm_temp

                            
                #PSF forward model computation for LOCI
                            
                if self.model[modn]=='FM LOCI':
                    
                    an_dist = np.sqrt((poscenty-ceny)**2 + (poscentx-cenx)**2)
                    theta = np.degrees(np.arctan2(poscenty-ceny, poscentx-cenx))  
                    
                    model_matrix=cube_inject_companions(np.zeros_like(mcube), self.psf[cuben], self.pa[cuben], 1,an_dist, plsc=0.1, 
                         theta=theta, n_branches=1,verbose=False)
            
                    indices = get_annulus_segments(self.cube[cuben][0], ann_center-int(self.asize[modn]/2),int(self.asize[modn]),1)
            
                    values_fc = model_matrix[:, indices[0][0], indices[0][1]]

                    cube_res_fc=np.zeros_like(model_matrix)
                
                    matrix_res_fc = np.zeros((values_fc.shape[0], indices[0][0].shape[0]))
                
                    for e in range(values_fc.shape[0]):
                    
                        recon_fc = np.dot(coef_list[e], values_fc[ind_ref_list[e]])
                        matrix_res_fc[e] = values_fc[e] - recon_fc

                    cube_res_fc[:, indices[0][0], indices[0][1]] = matrix_res_fc
                    cube_der_fc = cube_derotate(cube_res_fc-np.mean(cube_res_fc), self.pa[cuben], imlib='opencv', interpolation='lanczos4')
                    psfm_temp=cube_crop_frames(cube_der_fc,int(2*round(self.fwhm)+1),xy=(poscentx,poscenty),verbose=False)
                    psf_fm_out[i,:,:,:]=psfm_temp

                            
                # Reverse the temporal direction when moving from one patch to the next one to respect the temporal proximity of the pixels and limit potential non-linearity
                        
                if i%2==1:
                    range_sel=range(n*z)
                else:
                    range_sel=range(n*z-1,-1,-1)
                    
                # Likelihood computation for the patch i
                
                for j in range_sel: 
                    
                    for m in range(range_int):

                        if psfm_temp is not None:
                                psfm_temp2=psfm_temp[j]
                        else:
                                psfm_temp2=psf_temp
                                
                        for v in range(0,self.crop_range[modn]):
                            
                            cropf=int(self.crop[modn][ann_center,cuben]+2*v)
                            if psfm_temp2.shape[1]==cropf:
                                psfm=psfm_temp2
                            else:
                                if len(psfm_temp2.shape)==2:
                                    psfm=frame_crop(psfm_temp2,cropf,cenxy=[int(psfm_temp2.shape[0]/2),int(psfm_temp2.shape[0]/2)],verbose=False)
                                else:
                                    psfm=frame_crop(psfm_temp2[int(j//n_t)],cropf,cenxy=[int(psfm_temp2.shape[0]/2),int(psfm_temp2.shape[0]/2)],verbose=False)
                                

                            if self.var[modn][ann_center,cuben]=='ST':
                                svar=var[v]
                                alpha=mean[v]
                                mv=mixval[v]
                                sel_distri=distrim[v]
                                phi[1]=self.interval[modn][ann_center,cuben][m]*np.sqrt(var[v])          
            
                            elif self.var[modn][ann_center,cuben]=='FR':
                                svar=var[j,v]
                                alpha=mean[j,v]
                                mv=mixval[j,v]
                                sel_distri=distrim[j,v]
                                phi[1]=self.interval[modn][ann_center,cuben][m]*np.sqrt(var[j,v])
            
                            elif self.var[modn][ann_center,cuben]=='SM':
                                svar=var[i,v]
                                alpha=mean[i,v]
                                mv=mixval[i,v]
                                sel_distri=distrim[i,v] 
                                phi[1]=self.interval[modn][ann_center,cuben][m]*np.sqrt(var[i,v])
            
                            if self.model[modn]=='FM KLIP' or self.model[modn]=='FM LOCI':
                                phi[1]=5*phi[1]
                                
                            for l in range(0,2):

                                #Likelihood estimation
                                
                                ff=frame_crop(mcube[j],cropf,cenxy=[poscentx,poscenty],verbose=False)-phi[l]*psfm-alpha
                                if sel_distri==0:
                                        cftemp=(1/np.sqrt(2 * np.pi*svar))*np.exp(-0.5*np.multiply(ff,ff)/svar)
                                elif sel_distri==1:
                                        cftemp=1/(np.sqrt(2*svar))*np.exp(-abs(ff)/np.sqrt(0.5*svar))
                                elif sel_distri==2:
                                        cftemp=(mv*(1/np.sqrt(2 * np.pi*svar))*np.exp(-0.5*np.multiply(ff,ff)/svar)+(1-mv)*1/(np.sqrt(2*svar))*np.exp(-abs(ff)/np.sqrt(0.5*svar)))

                                probcube[int(cubind),int(indicesy[i]),int(indicesx[i]),int(m),l,v]=cftemp.sum()
                
                    cubind+=1
            
            return probcube
        
        
        if verbose==True:
            print("Radial distance: "+"{}".format(ann_center)) 


        
        evals_matrix=[]
        evecs_matrix=[]
        KL_basis_matrix=[]
        refs_mean_sub_matrix=[]
        sci_mean_sub_matrix=[]
        resicube_klip=None
        
        ind_ref_list=None
        coef_list=None
        
        if self.opti==True and cube_fc is not None:
            
            cube_test=cube_fc+self.cube[cuben]
        else:
            cube_test=self.cube[cuben]
                    
        #Estimation of the KLIP/KLIP FM cube of residuals for the selected annulus
        
        if self.model[modn]=='FM KLIP' or (self.opti==True and self.model[modn]=='KLIP'):
                       

            resicube_klip=np.zeros_like(self.cube[cuben])
            

            pa_threshold = np.rad2deg(2 * np.arctan(self.delta_rot[modn][ann_center,cuben] * self.fwhm / (2 * (ann_center))))
            mid_range = np.abs(np.amax(self.pa[cuben]) - np.amin(self.pa[cuben])) / 2
            if pa_threshold >= mid_range - mid_range * 0.1:
                pa_threshold = float(mid_range - mid_range * 0.1)
            
            indices = get_annulus_segments(self.cube[cuben][0], ann_center-int(self.asize[modn]/2),int(self.asize[modn]),1)

            for k in range(0,self.cube[cuben].shape[0]):

                evals_temp,evecs_temp,KL_basis_temp,sub_img_rows_temp,refs_mean_sub_temp,sci_mean_sub_temp =KLIP_patch(k,cube_test[:, indices[0][0], indices[0][1]], self.ncomp[modn][ann_center,cuben], self.pa[cuben], self.asize[modn], pa_threshold, ann_center)
                resicube_klip[k,indices[0][0], indices[0][1]] = sub_img_rows_temp

                evals_matrix.append(evals_temp)
                evecs_matrix.append(evecs_temp)
                KL_basis_matrix.append(KL_basis_temp)
                refs_mean_sub_matrix.append(refs_mean_sub_temp)
                sci_mean_sub_matrix.append(sci_mean_sub_temp)

            mcube=cube_derotate(resicube_klip,self.pa[cuben],imlib=self.imlib, interpolation=self.interpolation)

            
        #Estimation of the LOCI FM cube of residuals for the selected annulus
        
        elif self.model[modn]=='FM LOCI':
            
            
            resicube, ind_ref_list,coef_list=LOCI_FM(cube_test, self.psf[cuben], ann_center, self.pa[cuben],None, self.asize[modn], self.fwhm, self.tolerance[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben],None)
            mcube=cube_derotate(resicube,self.pa[cuben])

        # Computation of the annular LOCI (used during the parameter optimization) 
        
        elif (self.opti==True and self.model[modn]=='LOCI'):
            
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube_test,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            
            if scale_list is not None:
                resicube=np.zeros_like(cube_rot_scale)
                for i in range(int(max(scale_list)*ann_center/self.asize[modn])):
                    indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
                    resicube[:,indices[0][0], indices[0][1]]=LOCI_FM(cube_rot_scale, self.psf[cuben], ann_center,angle_list,scale_list, self.asize[modn], self.fwhm, self.tolerance[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben],self.delta_sep[modn][ann_center,cuben])[0][:,indices[0][0], indices[0][1]]
            
            else:
                resicube, ind_ref_list,coef_list=LOCI_FM(cube_rot_scale, self.psf[cuben], ann_center, self.pa[cuben],None, self.fwhm, self.fwhm, self.tolerance[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben],None)
 
            mcube=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)

        # Computation of the annular APCA (used during the parameter optimization) 

        elif (self.opti==True and self.model[modn]=='APCA'):
            
                  
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube_test,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            resicube=np.zeros_like(cube_rot_scale)
            if scale_list is not None:
                range_adisdi=range(int(max(scale_list)*ann_center/self.asize[modn]))
            else:
                range_adisdi=range(1)
                
            for i in range_adisdi:
                
                pa_threshold = np.rad2deg(2 * np.arctan(self.delta_rot[modn][ann_center,cuben] * self.fwhm / (2 * (ann_center+self.asize[modn]*i))))
                mid_range = np.abs(np.amax(self.pa[cuben]) - np.amin(self.pa[cuben])) / 2
                if pa_threshold >= mid_range - mid_range * 0.1:
                    pa_threshold = float(mid_range - mid_range * 0.1)
                indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
    
    
                for k in range(0,cube_rot_scale.shape[0]):
                    for l in range(self.nsegments[modn][ann_center,cuben]):
                    
                        resicube[k,indices[l][0], indices[l][1]],v_resi,data_shape=do_pca_patch(cube_rot_scale[:, indices[l][0], indices[l][1]], k,  angle_list,scale_list, self.fwhm, pa_threshold,self.delta_sep[modn][ann_center,cuben], ann_center+self.asize[modn]*i,
                           svd_mode='lapack', ncomp=self.ncomp[modn][ann_center,cuben],min_frames_lib=2, max_frames_lib=200, tol=1e-1,matrix_ref=None)
            mcube=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)
            
        # Computation of the annular NMF (used during the parameter optimization) 
                
        elif (self.opti==True and self.model[modn]=='NMF'):
            
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube_test,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            resicube=np.zeros_like(cube_rot_scale)

            matrix = prepare_matrix(cube_rot_scale, None, None, mode='fullfr',
                        verbose=False)
            res= NMF_patch(matrix, ncomp=self.ncomp[modn][ann_center,cuben], max_iter=50000,random_state=None)
            resicube=np.reshape(res,(cube_rot_scale.shape[0],cube_rot_scale.shape[1],cube_rot_scale.shape[2]))
        
            mcube=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)

        # Computation of the annular LLSG (used during the parameter optimization) 
                
        elif (self.opti==True and self.model[modn]=='LLSG'):

            
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube_test,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            resicube=np.zeros_like(cube_rot_scale)
            if scale_list is not None:
                range_adisdi=range(int(max(scale_list)*ann_center/self.asize[modn]))
            else:
                range_adisdi=range(1)
                
            for i in range_adisdi:
                indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
    
                for l in range(self.nsegments[modn][ann_center,cuben]):
                    
                    resicube[:,indices[l][0], indices[l][1]]= _decompose_patch(indices,l, cube_rot_scale,self.nsegments[modn][ann_center,cuben],
                            self.rank[modn][ann_center,cuben], low_rank_ref=False, low_rank_mode='svd', thresh=1,thresh_mode='soft', max_iter=40, auto_rank_mode='noise', cevr=0.9,
                                 residuals_tol=1e-1, random_seed=10, debug=False, full_output=False).T

            mcube=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)
          

        zero_test=abs(mcube.sum(axis=1).sum(axis=1))
        if np.min(zero_test)==0:
            mcube[np.argmin(zero_test),:,:]=mcube.mean(axis=0)
            
        # Fitness error computation for the noise distribution(s),
        # if automated probability distribution selection is activated (var='A')
        # the fitness errors allow the determination of the optimal distribution


        def vm_esti(modn,arr,var_e,mean_e):
            
            def gaus(x,x0,var):
                return 1/np.sqrt(2 * np.pi*var)*np.exp(-(x-x0)**2/(2*var))
    
            def lap(x,x0,var):
                bap=np.sqrt(var/2)
                return (1/(2*bap))*np.exp(-np.abs(x-x0)/bap)
    
            def mix(x,x0,var,a):
                bap=np.sqrt(var/2)
                return a*(1/(2*bap))*np.exp(-np.abs(x-x0)/bap)+(1-a)*1/np.sqrt(2 * np.pi*var)*np.exp(-(x-x0)**2/(2*var))
            
            
            def te_f_mh(func1,func2,bin_m,hist,p0_1,bounds_1,p0_2,bounds_2,mean,var,distri):
               try: 
                   popt,pcov = curve_fit(func1,bin_m,hist,p0=p0_1,bounds=bounds_1)
                   fiter=sum(abs(func1(bin_m,*popt)-hist))
                   mean,var,a=popt
               except (RuntimeError, ValueError, RuntimeWarning):
                   try:
                       popt,pcov = curve_fit(func2,bin_m,hist,p0=p0_2,bounds=bounds_2)
                       fiter=sum(abs(func2(bin_m,*popt)-hist))
                       a=popt
                   except (RuntimeError, ValueError, RuntimeWarning):
                       a=0.01/max(abs(bin_m))
                       fiter=sum(abs(func2(bin_m,a)-hist))
               return mean,a,var,fiter,distri
                   
            def te_f_gl(func,bin_m,hist,p0_1,bounds_1,mean,var,distri):
               try: 
                   popt,pcov = curve_fit(func,bin_m,hist,p0=p0_1,bounds=bounds_1)
                   mean,var=popt
                   fiter=sum(abs(func(bin_m,*popt)-hist))
               except (RuntimeError, ValueError, RuntimeWarning): 
                   mean,var=[mean,var]
                   fiter=sum(abs(func(bin_m,mean,var)-hist))
               return mean,None,var,fiter,distri   
           
            def te_m(func,bin_m,hist,p0,bounds,mean,var,distri):
                try:
                   popt,pcov = curve_fit(func,bin_m,hist,p0=p0,bounds=bounds)
                   mixval=popt
                   return mean,mixval,var,sum(abs(func(bin_m,*popt)-hist)),distri
               
                except (RuntimeError, ValueError, RuntimeWarning):
                    return mean,0.5,var,sum(abs(func(bin_m,0.5)-hist)),distri
                
                   
            def te_gl(func,bin_m,hist,mean,var,distri):
               return mean,None,var,sum(abs(func(bin_m,mean,var)-hist)),distri 
           

            mixval_temp=None

            try:
                hist, bin_edge =np.histogram(arr,bins='auto',density=True)
            except (IndexError, TypeError, ValueError,MemoryError):
                hist, bin_edge =np.histogram(arr,bins=np.max([20,int(len(arr)/20)]),density=True)
            if len(hist)>100:
                hist, bin_edge =np.histogram(arr,bins=np.max([100,int(len(arr)/100)]),density=True) 
                
            bin_mid=(bin_edge[0:(len(bin_edge)-1)]+bin_edge[1:len(bin_edge)])/2
                     
            fixmix = lambda binm, mv: mix(binm,mean_e,var_e,mv)    

            res=[]
            res.append(te_gl(gaus,bin_mid,hist,mean_e,var_e,0))
            res.append(te_gl(lap,bin_mid,hist,mean_e,var_e,1))
            res.append(te_m(fixmix,bin_mid,hist,[0.5],[(0),(1)],mean_e,var_e,2))
            
            fiterr=list([res[0][3],res[1][3],res[2][3]])
            distrim_temp=fiterr.index(min(fiterr))
            fiterr_temp=min(fiterr)
           
            mean_temp=res[distrim_temp][0]
            mixval_temp=res[distrim_temp][1]
            var_temp=res[distrim_temp][2]
            fiterr_temp=res[distrim_temp][3]

            
            return mean_temp,var_temp,fiterr_temp,mixval_temp,distrim_temp

        # Noise distribution parameter estimation considering the selected region (ST, F, FM, SM or T)
                                    
        var_f=None
        
        
        if self.var[modn][ann_center,cuben]=='ST':
            
            var=np.zeros(self.crop_range[modn])
            var_f=np.zeros(self.crop_range[modn])
            mean=np.zeros(self.crop_range[modn])
            mixval=np.zeros(self.crop_range[modn])
            fiterr=np.zeros(self.crop_range[modn])
            distrim=np.zeros(self.crop_range[modn])
            max_hist=np.zeros(self.crop_range[modn])
            
            
            for v in range(0,self.crop_range[modn]):
                cropf=int(self.crop[modn][ann_center,cuben]+2*v)
                indices = get_annulus_segments(mcube[0], ann_center-int(cropf/2),cropf,1)

                poscentx=indices[0][1]
                poscenty=indices[0][0]
                
                arr = np.ndarray.flatten(mcube[:,poscenty,poscentx])
            
                mean[v],var[v],fiterr[v],mixval[v],distrim[v]=vm_esti(modn,arr,np.var(mcube[:,poscenty,poscentx]),np.mean(mcube[:,poscenty,poscentx]))
                
                var_f[v]=np.var(mcube[:,poscenty,poscentx])
            
        elif self.var[modn][ann_center,cuben]=='FR':
            
            var=np.zeros(((n*z),self.crop_range[modn]))
            var_f=np.zeros(((n*z),self.crop_range[modn]))
            mean=np.zeros(((n*z),self.crop_range[modn]))
            mixval=np.zeros(((n*z),self.crop_range[modn]))
            fiterr=np.zeros(((n*z),self.crop_range[modn]))
            distrim=np.zeros(((n*z),self.crop_range[modn]))
            max_hist=np.zeros(((n*z),self.crop_range[modn]))
            
            for v in range(0,self.crop_range[modn]):
                cropf=int(self.crop[modn][ann_center,cuben]+2*v)
                indices = get_annulus_segments(mcube[0], ann_center-int(cropf/2),cropf,1)

                poscentx=indices[0][1]
                poscenty=indices[0][0]
            
                for a in range((n*z)):
           
                    arr = np.ndarray.flatten(mcube[a,poscenty,poscentx])
            
                    var_f[a,v]=np.var(mcube[a,poscenty,poscentx])
                
                    if var_f[a,v]==0:
                        mean[a,v]=mean[a-1,v]
                        var[a,v]=var[a-1,v]
                        fiterr[a,v]=fiterr[a-1,v]
                        mixval[a,v]=mixval[a-1,v]
                        distrim[a,v]=distrim[a-1,v]
                        max_hist[a,v]=max_hist[a-1,v]
                        var_f[a,v]=var_f[a-1,v]
                    else:    
                    
                        mean[a,v],var[a,v],fiterr[a,v],mixval[a,v],distrim[a,v]=vm_esti(modn,arr,np.var(mcube[a,poscenty,poscentx]),np.mean(mcube[a,poscenty,poscentx]))

        elif self.var[modn][ann_center,cuben]=='SM':
            
            indicesy,indicesx=get_time_series(mcube,ann_center)
            
            var=np.zeros((len(indicesy),self.crop_range[modn]))
            var_f=np.zeros((len(indicesy),self.crop_range[modn]))
            mean=np.zeros((len(indicesy),self.crop_range[modn]))
            mixval=np.zeros((len(indicesy),self.crop_range[modn]))
            fiterr=np.zeros((len(indicesy),self.crop_range[modn]))
            distrim=np.zeros((len(indicesy),self.crop_range[modn]))
            max_hist=np.zeros((len(indicesy),self.crop_range[modn]))
            size_seg=2
            
            for v in range(0,self.crop_range[modn]):
                cropf=int(self.crop[modn][ann_center,cuben]+2*v)
            
                for a in range(len(indicesy)):
            
                    if (a+int(cropf*3/2)+size_seg)>(len(indicesy)-1):
                        posup= a+int(cropf*3/2)+size_seg-len(indicesy)-1
                    else:
                        posup=a+int(cropf*3/2)+size_seg
               
                    indc=disk((indicesy[a], indicesx[a]),cropf/2)
           
                    radist_b=np.sqrt((indicesx[a-int(cropf*3/2)-size_seg-1]-int(x/2))**2+(indicesy[a-int(cropf*3/2)-size_seg-1]-int(y/2))**2)
           
                    if (indicesx[a-int(cropf*3/2)-size_seg-1]-int(x/2))>=0:
                        ang_b= np.arccos((indicesy[a-int(cropf*3/2)-size_seg-1]-int(y/2))/radist_b)/np.pi*180                        
                    else:
                        ang_b= 360-np.arccos((indicesy[a-int(cropf*3/2)-size_seg-1]-int(y/2))/radist_b)/np.pi*180
          
                    radist_e=np.sqrt((indicesx[posup]-int(x/2))**2+(indicesy[posup]-int(y/2))**2)
           
                    if (indicesx[posup]-int(x/2))>=0:
                        ang_e= np.arccos((indicesy[posup]-int(y/2))/radist_e)/np.pi*180
                    else:
                        ang_e= 360-np.arccos((indicesy[posup]-int(y/2))/radist_e)/np.pi*180
                                     
                    if ang_e<ang_b:
                        diffang=(360-ang_b)+ang_e
                    else:
                        diffang=ang_e-ang_b

            
                    indices = get_annulus_segments(mcube[0], ann_center-int(cropf/2),cropf,int(360/diffang),ang_b)
                    positionx=[]
                    positiony=[]
           
                    for k in range(0,len(indices[0][1])):
                        if len(set(np.where(indices[0][1][k]==indc[1])[0]) & set(np.where(indices[0][0][k]==indc[0])[0]))==0:
                            positionx.append(indices[0][1][k])
                            positiony.append(indices[0][0][k])

        
                    arr = np.ndarray.flatten(mcube[:,positiony,positionx])
                    var_f[a,v]=np.var(mcube[:,positiony,positionx])
                
                    if var_f[a,v]==0:
                        mean[a,v]=mean[a-1,v]
                        var[a,v]=var[a-1,v]
                        fiterr[a,v]=fiterr[a-1,v]
                        mixval[a,v]=mixval[a-1,v]
                        distrim[a,v]=distrim[a-1,v]
                        max_hist[a,v]=max_hist[a-1,v]
                        var_f[a,v]=var_f[a-1,v]
                    else:    
                        mean[a,v],var[a,v],fiterr[a,v],mixval[a,v],distrim[a,v]=vm_esti(modn,arr,np.var(mcube[:,positiony,positionx]),np.mean(mcube[:,positiony,positionx]))

        #Estimation of the cube of likelihoods
        #print(np.bincount(distrim.flatten(order='C').astype(int)))
        res=likfcn(cuben,modn,mean,var,mixval,mcube,ann_center,distrim,likemap,var_f)
        indicesy,indicesx=get_time_series(mcube,ann_center)

        return ann_center,res[:,indicesy,indicesx]
                       

        
    def lik_esti(self, sel_cube=None,showplot=False,verbose=True):
        
        """
        Function allowing the estimation of the likelihood of being in either the planetary regime 
        or the background regime for the different cubes. The likelihood computation is based on 
        the residual cubes generated  with the considered set of models.
        
        Parameters
        ----------
    
        showplot: bool, optional
            If True, provides the plots of the final residual frames for the selected 
            ADI-based post-processing techniques along with the final RSM map. Default is False.
        fulloutput: bool, optional
            If True, provides the selected distribution, the fitness erros and the mixval 
            (for distri='mix') for every annulus in respectively obj.distrisel, obj.fiterr
            and obj.mixval (the length of these lists are equall to maxradius - minradius, the
            size of the matrix for each annulus depends on the approach selected for the variance
            estimation, see var in add_model)
        verbose : bool, optional
            If True prints intermediate info. Default is True.
        """
        

        def init(probi):
            global probcube
            probcube = probi
            
            
        for j in range(len(self.cube)):
            
            if self.ncore!=1:
                if self.cube[j].ndim==4:
                    residuals_cube_=np.zeros((self.cube[j].shape[0]*self.cube[j].shape[1],self.cube[j].shape[-1],self.cube[j].shape[-1]))
                else:
                    residuals_cube_=np.zeros((self.cube[j].shape[0],self.cube[j].shape[-1],self.cube[j].shape[-1]))
            
                resi_cube = mp.Array(typecode_or_type='f', size_or_initializer=int(np.prod(residuals_cube_.shape)))
                globals()['resi_cube'] = resi_cube
                ini_resi_cube = shared_array('resi_cube', shape=residuals_cube_.shape) 
            
            for i in range(len(self.model)):
                        
                #Computation of the cube of residuals
                if sel_cube is None or [j,i] in sel_cube:

                    if self.opti==False:
                
                        if self.model[i]=='APCA':
                            if verbose:
                                print("Annular PCA estimation") 
        
                            residuals_cube_, frame_fin = annular_pca_adisdi(self.cube[j], self.pa[j], self.scale_list[j], fwhm=self.fwhm, ncomp=self.ncomp[i][0,j], asize=self.asize[i], radius_int=self.minradius, radius_out=self.maxradius,
                                      delta_rot=self.delta_rot[i][0,j],delta_sep=self.delta_sep[i][0,j], svd_mode='lapack', n_segments=int(self.nsegments[i][0,j]), nproc=self.ncore,full_output=True,verbose=False)
                            if showplot:
                                plot_frames(frame_fin,title='APCA', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)
        
                        elif self.model[i]=='NMF':
                            if verbose:
                                print("NMF estimation") 
                            residuals_cube_, frame_fin = nmf_adisdi(self.cube[j], self.pa[j], self.scale_list[j], ncomp=self.ncomp[i][0,j], max_iter=5000, random_state=0, mask_center_px=None,full_output=True,verbose=False)
                            if showplot:
                                plot_frames(frame_fin,title='NMF', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)
        
                        elif self.model[i]=='LLSG':
                            if verbose:
                                print("LLSGestimation") 
        
                            residuals_cube_, frame_fin = llsg_adisdi(self.cube[j], self.pa[j],self.scale_list[j], self.fwhm, rank=self.rank[i][0,j],asize=self.asize[i], thresh=1,n_segments=int(self.nsegments[i][0,j]),radius_int=self.minradius, radius_out=self.maxradius, max_iter=40, random_seed=10, nproc=self.ncore,full_output=True,verbose=False)
                           
                            if showplot:
                                plot_frames(frame_fin,title='LLSG', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)
        
                        elif self.model[i]=='LOCI':
                            if verbose:
                                print("LOCI estimation") 
                            residuals_cube_,frame_fin=loci_adisdi(self.cube[j], self.pa[j],self.scale_list[j], fwhm=self.fwhm,asize=self.asize[i],radius_int=self.minradius, radius_out=self.maxradius, n_segments=int(self.nsegments[i][0,j]),tol=self.tolerance[i][0,j], nproc=self.ncore, optim_scale_fact=2,delta_rot=self.delta_rot[i][0,j],delta_sep=self.delta_sep[i][0,j],verbose=False,full_output=True)
                            if showplot:
                                plot_frames(frame_fin,title='LOCI', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)
                
                        elif self.model[i]=='KLIP':
                            if verbose:
                                print("KLIP estimation") 
                            cube_out, residuals_cube_, frame_fin = KLIP(self.cube[j], self.pa[j], ncomp=self.ncomp[i][0,j], fwhm=self.fwhm, asize=self.asize[i],radius_int=self.minradius, radius_out=self.maxradius, 
                                      delta_rot=self.delta_rot[i][0,j],full_output=True,verbose=False)
                            if showplot:
                                plot_frames(frame_fin,title='KLIP', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)
                        elif self.model[i]=='FM LOCI' or self.model[i]=='FM KLIP':
                            residuals_cube_=np.zeros_like(self.cube[j])
                            frame_fin=np.zeros_like(self.cube[j][0])
                            
                        zero_test=abs(residuals_cube_.sum(axis=1).sum(axis=1))
                        if np.min(zero_test)==0:
                            residuals_cube_[np.argmin(zero_test),:,:]=residuals_cube_.mean(axis=0)
                    else:
                        residuals_cube_=np.zeros_like(rot_scale('ini',self.cube[j],None,self.pa[j],self.scale_list[j],self.imlib, self.interpolation)[0])
                        frame_fin=np.zeros_like(residuals_cube_[0])
                        
                    #Likelihood computation for the different models and cubes
                    
                    if self.model[i]=='FM KLIP' or self.model[i]=='FM LOCI':
                        max_rad=np.min([self.max_r+1,self.maxradius+1])
                    else:
                        max_rad=self.maxradius+1

                    like_temp=np.zeros(((residuals_cube_.shape[0]+1),residuals_cube_.shape[1],residuals_cube_.shape[2],len(self.interval[i][0,j]),2,self.crop_range[i]))    

                    if self.ncore==1:
            
                        for e in range(self.minradius,max_rad):
                            res=self.likelihood(e,j,i,residuals_cube_,None,verbose)
                            indicesy,indicesx=get_time_series(self.cube[0],res[0])
                            like_temp[:,indicesy,indicesx,:,:,:]=res[1] 
                    else:
                        
                        ini_resi_cube[:] = residuals_cube_
                        #Sometimes the parallel computation get stuck, to avoid that we impose a maximum computation
                        #time of 0.005 x number of frame x number of pixels iin the last annulus x 2 x pi (0.005 second
                        #per treated pixel)
                        time_out=0.03*residuals_cube_.shape[0]*2*self.maxradius**np.pi
                        results=[]    
                        pool=Pool(processes=self.ncore)           
                        for e in range(self.minradius,max_rad):
                            results.append(pool.apply_async(self.likelihood,args=(e,j,i,residuals_cube_.shape,None,verbose)))
                        #[result.wait(timeout=time_out) for result in results]
                        
                        it=self.minradius
                        for result in results:
                            try:
                                res=result.get(timeout=time_out)
                                indicesy,indicesx=get_time_series(self.cube[0],res[0])
                                like_temp[:,indicesy,indicesx,:,:,:]=res[1]  
                                del result
                            except mp.TimeoutError:
                                time_out=0.1
                                pool.terminate()
                                pool.join()
                                res=self.likelihood(it,j,i,residuals_cube_,None,False)
                                indicesy,indicesx=get_time_series(self.cube[0],res[0])
                                like_temp[:,indicesy,indicesx,:,:,:]=res[1] 
                            it+=1 
                        pool.close()
                        pool.join()
                    
                    like=[]

                    for k in range(self.crop_range[i]):
                        like.append(like_temp[0:residuals_cube_.shape[0],:,:,:,:,k])
                    
                    self.like_fin[j][i]=like
                like=[]
                like_temp=[]
                
                
    def likfcn(self,ann_center,like_cube,estimator,ns):
        
        if type(like_cube) is not np.ndarray:
            like_cube = shared_array('like_cube', shape=like_cube)

        from vip_hci.var import frame_center  
        
        def forback(obs,Trpr,prob_ini):
            
            #Forward backward model relying on past and future observation to 
            #compute the probability based on a two-states Markov chain
    
            scalefact_fw=np.zeros(obs.shape[1])
            scalefact_bw=np.zeros(obs.shape[1])
            prob_fw=np.zeros((2,obs.shape[1]))
            prob_bw=np.zeros((2,obs.shape[1]))
            prob_fin=np.zeros((2,obs.shape[1]))
            prob_pre_fw=0
            prob_pre_bw=0
            lik=0
            
            for i in range(obs.shape[1]):
                if obs[:,i].sum()!=0:
                    j=obs.shape[1]-1-i
                    if i==0:
                        prob_cur_fw=np.dot(np.diag(obs[:,i]),Trpr).dot(prob_ini)
                        prob_cur_bw=np.dot(Trpr,np.diag(obs[:,j])).dot(prob_ini)
                    else:
                        prob_cur_fw=np.dot(np.diag(obs[:,i]),Trpr).dot(prob_pre_fw)
                        prob_cur_bw=np.dot(Trpr,np.diag(obs[:,j])).dot(prob_pre_bw)
        
                    scalefact_fw[i]=prob_cur_fw.sum()
                    if scalefact_fw[i]==0:
                        prob_fw[:,i]=0
                    else:
                        prob_fw[:,i]=prob_cur_fw/scalefact_fw[i]
                    prob_pre_fw=prob_fw[:,i]
        
                    scalefact_bw[j]=prob_cur_bw.sum()
                    if scalefact_bw[j]==0:
                        prob_bw[:,j]=0
                    else:
                        prob_bw[:,j]=prob_cur_bw/scalefact_bw[j]
                        
                    prob_pre_bw=prob_bw[:,j]
    
            scalefact_fw_tot=(scalefact_fw).sum()                
            scalefact_bw_tot=(scalefact_bw).sum()
    
    
            for k in range(obs.shape[1]):
                if (prob_fw[:,k]*prob_bw[:,k]).sum()==0:
                    prob_fin[:,k]=0
                else:
                    prob_fin[:,k]=(prob_fw[:,k]*prob_bw[:,k])/(prob_fw[:,k]*prob_bw[:,k]).sum()
    
            lik = scalefact_fw_tot+scalefact_bw_tot
    
            return prob_fin, lik
    
    
        def RSM_esti(obs,Trpr,prob_ini):
            
            #Original RSM approach involving a forward two-states Markov chain to compute the probabilities
    
            prob_fin=np.zeros((2,obs.shape[1]))
            prob_pre=0
            lik=0
    
            for i in range(obs.shape[1]):
                if obs[:,i].sum()!=0:
                    if i==0:
                        cf=obs[:,i]*np.dot(Trpr,prob_ini)
                    else:
                        cf=obs[:,i]*np.dot(Trpr,prob_pre)
        
                    
                    f=sum(cf) 
                    if f==0:
                       prob_fin[:,i]=0
                    else:
                        lik+=np.log(f)
                        prob_fin[:,i]=cf/f
                    prob_pre=prob_fin[:,i]
                else:
                    prob_fin[:,i]=[1,0]
    
            return prob_fin, lik
    
        probmap = np.zeros((like_cube.shape[0],like_cube.shape[1],like_cube.shape[2]))
        ceny, cenx = frame_center(like_cube[0,:,:,0,0])
        
        indicesy,indicesx=get_time_series(like_cube[:,:,:,0,0],ann_center)

        npix = len(indicesy)
        pini=[1-ns/(like_cube.shape[0]*(npix)),1/(like_cube.shape[0]*ns),ns/(like_cube.shape[0]*(npix)),1-1/(like_cube.shape[0]*ns)]
        prob=np.reshape([pini],(2, 2)) 

        Trpr= prob

        #Initialization of the Regime Switching model
        #I-prob
        mA=np.concatenate((np.diag(np.repeat(1,2))-prob,[np.repeat(1,2)]))
        #sol
        vE=np.repeat([0,1],[2,1])
        #mA*a=vE -> mA'mA*a=mA'*vE -> a=mA'/(mA'mA)*vE
            
        prob_ini=np.dot(np.dot(np.linalg.inv(np.dot(mA.T,mA)),mA.T),vE)

        cf=np.zeros((2,len(indicesy)*like_cube.shape[0],like_cube.shape[3]))
        totind=0
        for i in range(0,len(indicesy)):

            poscenty=indicesy[i]
            poscentx=indicesx[i]
                
            for j in range(0,like_cube.shape[0]):        

                    for m in range(0,like_cube.shape[3]):
                        
                        cf[0,totind,m]=like_cube[j,poscenty,poscentx,m,0]
                        cf[1,totind,m]=like_cube[j,poscenty,poscentx,m,1]
                    totind+=1
                    
        #Computation of the probability cube via the regime switching framework
        
        prob_fin=[] 
        lik_fin=[]
        for n in range(like_cube.shape[3]):
            if estimator=='Forward':
                prob_fin_temp,lik_fin_temp=RSM_esti(cf[:,:,n],Trpr,prob_ini)
            elif estimator=='Forward-Backward':
                prob_fin_temp,lik_fin_temp=forback(cf[:,:,n],Trpr,prob_ini)


            prob_fin.append(prob_fin_temp)
            lik_fin.append(lik_fin_temp)
        
        cub_id1=0   
        for i in range(0,len(indicesy)):
            cub_id2=0
            for j in range(like_cube.shape[0]):
                    probmap[cub_id2,indicesy[i],indicesx[i]]=prob_fin[lik_fin.index(max(lik_fin))][1,cub_id1]
                    cub_id1+=1
                    cub_id2+=1
                    
        return probmap[:,indicesy,indicesx],ann_center

    def probmap_esti(self,modthencube=True,ns=1,sel_crop=None, estimator='Forward',colmode='median',ann_center=None,sel_cube=None):
        
        """
        Function allowing the estimation of the final RSM map based on the likelihood computed with 
        the lik_esti function for the different cubes and different post-processing techniques 
        used to generate the speckle field model. The RSM map estimation may be based on a forward
        or forward-backward approach.
        
        Parameters
        ----------
        
        modtocube: bool, optional
            Parameter defining if the concatenated cube feeding the RSM model is created
            considering first the model or the different cubes. If 'modtocube=False',
            the function will select the first cube then test all models on it and move 
            to the next one. If 'modtocube=True', the model will select one model and apply
            it to every cubes before moving to the next model. Default is True.
        ns: float , optional
             Number of regime switches. Default is one regime switch per annulus but 
             smaller values may be used to reduce the impact of noise or disk structures
             on the final RSM probablity map.
        sel_crop: list of int or None, optional
            Selected crop sizes from proposed crop_range (crop size = crop_size + 2 x (sel_crop)).
            A specific sel_crop should be provided for each mode. Default is crop size = [crop_size]
        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
        ann_center:int, optional
            Selected annulus if the probabilities are computed for a single annulus 
            (Used by the optimization framework). Default is None
        sel_cube: list of arrays,optional
            List of selected PSF-subtraction techniques and ADI sequences used to generate 
            the final probability map. [[i1,j1],[i2,j2],...] with i1 the first considered PSF-subtraction
            technique and j1 the first considered ADI sequence, i2 the second considered PSF-subtraction
            technique, etc. Default is None whih implies that all PSF-subtraction techniques and all
            ADI sequences are used to compute the final probability map.
        """
            
        import numpy as np 
     

        if type(sel_crop)!=np.ndarray:
            sel_crop=np.zeros(len(self.model)*len(self.cube))
        if sel_cube==None:    
            if modthencube==True:
                for i in range(len(self.model)):
                    for j in range(len(self.cube)):
                        if (i+j)==0:
                            if self.like_fin[j][i][int(int(sel_crop[i]))].shape[3]==1:
                                like_cube=np.repeat(self.like_fin[j][i][int(sel_crop[i])],len(self.interval[i][0,j]),axis=3)
                            else:
                                like_cube=self.like_fin[j][i][int(sel_crop[i])]
                        else:
                            if self.like_fin[j][i][int(sel_crop[i])].shape[3]==1:
                                like_cube=np.append(like_cube,np.repeat(self.like_fin[j][i][int(sel_crop[i])],len(self.interval[i][0,j]),axis=3),axis=0)
                            else:
                                like_cube=np.append(like_cube,self.like_fin[j][i][int(sel_crop[i])],axis=0) 
            else:
                for i in range(len(self.cube)):
                    for j in range(len(self.model)):
                        if (i+j)==0:
                            if self.like_fin[i][j][int(sel_crop[j])].shape[3]==1:
                                like_cube=np.repeat(self.like_fin[i][j][int(sel_crop[j])],len(self.interval[i][0,j]),axis=3)
                            else:
                                like_cube=self.like_fin[i][j][int(sel_crop[j])]
                        else:
                            if self.like_fin[i][j][int(sel_crop[j])].shape[3]==1:
                                like_cube=np.append(like_cube,np.repeat(self.like_fin[i][j][int(sel_crop[j])],len(self.interval[i][0,j]),axis=3),axis=0)
                            else:
                                like_cube=np.append(like_cube,self.like_fin[i][j][int(sel_crop[j])],axis=0) 
        else:
             for i in range(len(sel_cube)):
                 if i==0:
                    if self.like_fin[sel_cube[i][0]][sel_cube[i][1]][int(sel_crop[i])].shape[3]==1:
                        like_cube=np.repeat(self.like_fin[sel_cube[i][0]][sel_cube[i][1]][int(sel_crop[i])],len(self.interval[sel_cube[i][1]][0,sel_cube[i][0]]),axis=3)
                    else:
                        like_cube=self.like_fin[sel_cube[i][0]][sel_cube[i][1]][int(sel_crop[i])]
                 else:
                    if self.like_fin[sel_cube[i][0]][sel_cube[i][1]][int(sel_crop[i])].shape[3]==1:
                        like_cube=np.append(like_cube,np.repeat(self.like_fin[sel_cube[i][0]][sel_cube[i][1]][int(sel_crop[i])],len(self.interval[sel_cube[i][1]][0,sel_cube[i][0]]),axis=3),axis=0)
                    else:
                        like_cube=np.append(like_cube,self.like_fin[sel_cube[i][0]][sel_cube[i][1]][int(sel_crop[i])],axis=0) 

        
        n,y,x,l_int,r_n =like_cube.shape 


        probmap = np.zeros((like_cube.shape[0],like_cube.shape[1],like_cube.shape[2]))
        if ann_center is not None:
            indicesy,indicesx=get_time_series(self.cube[0],ann_center)
            probmap[:,indicesy,indicesx]=self.likfcn(ann_center,like_cube,estimator,ns)[0]
        else:
            
            if self.ncore!=1:
                like_array = mp.Array(typecode_or_type='f', size_or_initializer=int(np.prod(like_cube.shape)))
                globals()['like_cube'] = like_array
                ini_like_array = shared_array('like_cube', shape=like_cube.shape)
                ini_like_array[:] = like_cube
                #Sometimes the parallel computation get stuck, to avoid that we impose a maximum computation
                #time of 0.005 x number of frame x number of pixels iin the last annulus x 2 x pi (0.005 second
                #per treated pixel)
                time_out=0.005*like_cube.shape[0]*2*self.maxradius*np.pi
                results=[]    
                pool=Pool(processes=self.ncore)           
                for e in range(self.minradius,self.maxradius+1):
                    results.append(pool.apply_async(self.likfcn,args=(e,like_cube.shape,estimator,ns)))
                #[result.wait(timeout=time_out) for result in results]
                
                it=self.minradius
                for result in results:
                    try:
                        res=result.get(timeout=time_out)
                        indicesy,indicesx=get_time_series(self.cube[0],res[1])
                        probmap[:,indicesy,indicesx]=res[0]
                        print('Annulus: {}'.format(res[1]))
                        del result
                    except mp.TimeoutError:
                        time_out=0.1
                        pool.terminate()
                        pool.join()
                        res=self.likfcn(it,like_cube,estimator,ns)
                        indicesy,indicesx=get_time_series(self.cube[0],res[1])
                        probmap[:,indicesy,indicesx]=res[0]
                        print('Time out: {}'.format(res[1]))
                    it+=1
                pool.close()
                pool.join()
            else:
                 for e in range(self.minradius,self.maxradius+1):
                     indicesy,indicesx=get_time_series(self.cube[0],e)
                     probmap[:,indicesy,indicesx]=self.likfcn(e,like_cube,estimator,ns)[0]
            
            
        if colmode == 'mean':
            self.probmap= np.nanmean(probmap, axis=0)
        elif colmode == 'median':
            self.probmap= np.nanmedian(probmap, axis=0)
        elif colmode == 'sum':
            self.probmap= np.sum(probmap, axis=0)
        elif colmode == 'max':
            self.probmap= np.max(probmap, axis=0)
        like_cube=[]
        probmap=[]
            
                                  
        
    def model_esti(self,modn,cuben,ann_center,cube): 
        
        """
        Function used during the optimization process to compute the cube of residuals
        for a given annulus whose center is defined by ann_center. The PSF-subtraction
        techniques index is given by modn, the ADI sequence index by cuben
        """  
               
        if self.model[modn]=='FM KLIP' or self.model[modn]=='KLIP':
                
            resicube=np.zeros_like(self.cube[cuben])

            pa_threshold = np.rad2deg(2 * np.arctan(self.delta_rot[modn][ann_center,cuben] * self.fwhm / (2 * ann_center)))
            mid_range = np.abs(np.amax(self.pa[cuben]) - np.amin(self.pa[cuben])) / 2
            if pa_threshold >= mid_range - mid_range * 0.1:
                pa_threshold = float(mid_range - mid_range * 0.1)


            indices = get_annulus_segments(self.cube[cuben][0], ann_center-int(self.asize[modn]/2),int(self.asize[modn]),1)


            for k in range(0,self.cube[cuben].shape[0]):

                evals_temp,evecs_temp,KL_basis_temp,sub_img_rows_temp,refs_mean_sub_temp,sci_mean_sub_temp =KLIP_patch(k,cube[:, indices[0][0], indices[0][1]], self.ncomp[modn][ann_center,cuben], self.pa[cuben], self.crop[modn][ann_center,cuben], pa_threshold, ann_center)
                resicube[k,indices[0][0], indices[0][1]] = sub_img_rows_temp

            resicube_der=cube_derotate(resicube,self.pa[cuben])
            frame_fin=cube_collapse(resicube_der, mode='median')


        elif self.model[modn]=='FM LOCI':
            
            
            resicube, ind_ref_list,coef_list=LOCI_FM(cube, self.psf[cuben], ann_center, self.pa[cuben],None, self.asize[modn], self.fwhm, self.tolerance[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben],None)
            resicube_der=cube_derotate(resicube,self.pa[cuben])
            frame_fin=cube_collapse(resicube_der, mode='median')
            
        
        elif self.model[modn]=='LOCI':
            
            
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            
            if scale_list is not None:
                resicube=np.zeros_like(cube_rot_scale)
                for i in range(int(max(scale_list)*ann_center/self.asize[modn])):
                    indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
                    resicube[:,indices[0][0], indices[0][1]]=LOCI_FM(cube_rot_scale, self.psf[cuben], ann_center,angle_list,scale_list, self.asize[modn], self.fwhm, self.tolerance[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben],self.delta_sep[modn][ann_center,cuben])[0][:,indices[0][0], indices[0][1]]
            
            else:
                resicube, ind_ref_list,coef_list=LOCI_FM(cube_rot_scale, self.psf[cuben], ann_center, self.pa[cuben],None, self.fwhm, self.fwhm, self.tolerance[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben],None)
 
            resicube_der=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)
            frame_fin=cube_collapse(resicube_der, mode='median')


        elif self.model[modn]=='APCA':
            
                  
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            resicube=np.zeros_like(cube_rot_scale)
            if scale_list is not None:
                range_adisdi=range(int(max(scale_list)*ann_center/self.asize[modn]))
            else:
                range_adisdi=range(1)
                
            for i in range_adisdi:
                
                pa_threshold = np.rad2deg(2 * np.arctan(self.delta_rot[modn][ann_center,cuben] * self.fwhm / (2 * (ann_center+self.asize[modn]*i))))
                mid_range = np.abs(np.amax(self.pa[cuben]) - np.amin(self.pa[cuben])) / 2
                if pa_threshold >= mid_range - mid_range * 0.1:
                    pa_threshold = float(mid_range - mid_range * 0.1)
                indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
    
    
                for k in range(0,cube_rot_scale.shape[0]):
                    for l in range(self.nsegments[modn][ann_center,cuben]):
                    
                        resicube[k,indices[l][0], indices[l][1]],v_resi,data_shape=do_pca_patch(cube_rot_scale[:, indices[l][0], indices[l][1]], k,  angle_list,scale_list, self.fwhm, pa_threshold,self.delta_sep[modn][ann_center,cuben], ann_center+self.asize[modn]*i,
                           svd_mode='lapack', ncomp=self.ncomp[modn][ann_center,cuben],min_frames_lib=2, max_frames_lib=200, tol=1e-1,matrix_ref=None)
            resicube_der=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)
            frame_fin=cube_collapse(resicube_der, mode='median')

                
        elif self.model[modn]=='NMF':
            
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            resicube=np.zeros_like(cube_rot_scale)
            
            nfr = cube_rot_scale.shape[0]
            matrix = np.reshape(cube_rot_scale, (nfr, -1))
            res= NMF_patch(matrix, ncomp=self.ncomp[modn][ann_center,cuben], max_iter=5000,random_state=None)
            resicube=np.reshape(res,(cube_rot_scale.shape[0],cube_rot_scale.shape[1],cube_rot_scale.shape[2]))
        
            resicube_der=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)
            frame_fin=cube_collapse(resicube_der, mode='median')
            
                
        elif self.model[modn]=='LLSG':

            
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            resicube=np.zeros_like(cube_rot_scale)
            if scale_list is not None:
                range_adisdi=range(int(max(scale_list)*ann_center/self.asize[modn]))
            else:
                range_adisdi=range(1)
                
            for i in range_adisdi:
                indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
    
                for l in range(self.nsegments[modn][ann_center,cuben]):
                    
                    resicube[:,indices[l][0], indices[l][1]]= _decompose_patch(indices,l, cube_rot_scale,self.nsegments[modn][ann_center,cuben],
                            self.rank[modn][ann_center,cuben], low_rank_ref=False, low_rank_mode='svd', thresh=1,thresh_mode='soft', max_iter=40, auto_rank_mode='noise', cevr=0.9,
                                 residuals_tol=1e-1, random_seed=10, debug=False, full_output=False).T
                        
            resicube_der=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)
            frame_fin=cube_collapse(resicube_der, mode='median')
        
        return frame_fin,resicube_der      
    
        
    def contrast_esti(self,param): 
        
        """
        Function used during the PSF-subtraction techniques optimization process 
        to compute the average contrast for a given annulus via multiple injection 
        of fake companions, relying on the approach developed by Mawet et al. (2014), 
        Gonzales et al. (2017) and Dahlqvist et Al. (2021). The PSF-subtraction
        techniques index is given by self.param[1], the ADI sequence index by self.param[0]
        and the annulus center by self.param[2]. The parameters for the PSF-subtraction 
        technique are contained in param.
        """  

        
        cuben=self.param[0]
        modn=self.param[1]
        ann_center=self.param[2]
        

        if self.model[modn]=='APCA':

            self.ncomp[modn][ann_center,cuben]=int(param[0])
            self.nsegments[modn][ann_center,cuben]=int(param[1])
            self.delta_rot[modn][ann_center,cuben]=abs(param[2])

        elif self.model[modn]=='NMF':
            
            self.ncomp[modn][ann_center,cuben]=abs(int(param))
            

        elif self.model[modn]=='LLSG':
            
            self.rank[modn][ann_center,cuben]=int(param[0])
            self.nsegments[modn][ann_center,cuben]=int(param[1])

        elif self.model[modn]=='LOCI' or self.model[modn]=='FM LOCI':

            self.tolerance[modn][ann_center,cuben]=abs(param[0])
            self.delta_rot[modn][ann_center,cuben]=abs(param[1])              
            
        elif self.model[modn]=='KLIP' or self.model[modn]=='FM KLIP':

            self.ncomp[modn][ann_center,cuben]=abs(int(param[0]))
            self.delta_rot[modn][ann_center,cuben]=abs(param[1])
                        
        ceny, cenx = frame_center(self.cube[cuben])
        
        frame_nofc=self.model_esti(modn,cuben,ann_center,self.cube[cuben])[0]
            
        psf_template = normalize_psf(self.psf[cuben], fwhm=self.fwhm, verbose=False,size=self.psf[cuben].shape[1])


        
        # Noise computation using the approach proposed by Mawet et al. (2014)
        
        ang_step=360/((np.deg2rad(360)*ann_center)/self.fwhm)
        
        tempx=[]
        tempy=[]
        
        for l in range(int(((np.deg2rad(360)*ann_center)/self.fwhm))):
            newx = ann_center * np.cos(np.deg2rad(ang_step * l+self.opti_theta[cuben,ann_center]))
            newy = ann_center * np.sin(np.deg2rad(ang_step * l+self.opti_theta[cuben,ann_center]))
            tempx.append(newx)
            tempy.append(newy)
        
        tempx=np.array(tempx)
        
        tempy = np.array(tempy) +int(ceny)
        tempx = np.array(tempx) + int(cenx)
  
        apertures = photutils.CircularAperture(np.array((tempx, tempy)).T, round(self.fwhm/2))
        fluxes = photutils.aperture_photometry(frame_nofc, apertures)
        fluxes = np.array(fluxes['aperture_sum'])
         
        n_aper = len(fluxes)
        ss_corr = np.sqrt(1 + 1/(n_aper-1))
        sigma_corr = stats.t.ppf(stats.norm.cdf(5), n_aper)*ss_corr
        noise = np.std(fluxes)
        
        flux = sigma_corr*noise*2
        fc_map = np.ones((self.cube[cuben].shape[-1],self.cube[cuben].shape[-1])) * 1e-6
        fcy=[]
        fcx=[]
        cube_fc =self.cube[cuben]
        
        # Average contrast computation via multiple injections of fake companions
        
        ang_fc=range(int(self.opti_theta[cuben,ann_center]),int(360+self.opti_theta[cuben,ann_center]),int(360//min((len(fluxes)/2),8)))
        for i in range(len(ang_fc)):
            cube_fc = cube_inject_companions(cube_fc, psf_template,
                                 self.pa[cuben], flux,ann_center, self.pxscale,
                                 theta=ang_fc[i],verbose=False)
            y = int(ceny) + ann_center * np.sin(np.deg2rad(
                                                   ang_fc[i]))
            x = int(cenx) + ann_center * np.cos(np.deg2rad(
                                                   ang_fc[i]))
            if self.cube[cuben].ndim==4:
                fc_map = frame_inject_companion(fc_map, np.mean(psf_template,axis=0), y, x,
                                            flux)
            else:
                fc_map = frame_inject_companion(fc_map, psf_template, y, x,
                                            flux)
            fcy.append(y)
            fcx.append(x)

        
        frame_fc=self.model_esti(modn,cuben,ann_center,cube_fc)[0]
    
        contrast=[]
        
        for j in range(len(ang_fc)):
            apertures = photutils.CircularAperture(np.array(([fcx[j],fcy[j]])), round(self.fwhm/2))
            injected_flux = photutils.aperture_photometry(fc_map, apertures)['aperture_sum']
            recovered_flux = photutils.aperture_photometry((frame_fc - frame_nofc), apertures)['aperture_sum']
            throughput = float(recovered_flux / injected_flux)
    
            if throughput>0:
                contrast.append(flux / throughput)
                
        if len(contrast)!=0:
            contrast_mean=np.mean(contrast)
        else:
            contrast_mean=-1
            
            
        return np.where(contrast_mean<0,0,1/contrast_mean), contrast_mean,param
        
        
        

    
    
    def bayesian_optimisation(self, loss_function, bounds,param_type, prev_res=None, opti_param={'opti_iter':15,'ini_esti':10, 'random_search':100} ,ncore=1):

        """ bayesian_optimisation
        Uses Gaussian Processes to optimise the loss function `loss_function`.
        
        Parameters
        ----------
        
            loss_loss: function.
                Function to be optimised, in our case the contrast_esti function.
            bounds: numpy ndarray, 2d
                Lower and upper bounds of the parameters used to compute the cube of
                residuals allowing the estimation of the average contrast.
            param_type: list
                Type of parameters used by the PSF-subtraction technique to compute the cube
                of residuals allowing the estimation of the average contrast ('int' or 'float')
            prev_res: None or numpy ndarray, 2d
                Parmater sets and corresponding average contrasts generated at the previous 
                angular distance. Allow to smooth the transition from one annulus to another 
                during the optimization process for the annular mode
            param_optimisation: dict, optional
                Dictionnary regrouping the parameters used by the Bayesian. 'opti_iter' is the number of
                iterations,'ini_esti' number of sets of parameters for which the loss function is computed
                to initialize the Gaussian process, random_search the number of random searches for the
                selection of the next set of parameters to sample based on the maximisation of the 
                expected immprovement. Default is {'opti_iter':15,'ini_esti':10, 'random_search':100}
            ncore : int, optional
                Number of processes for parallel computing. By default ('ncore=1') 
                the algorithm works in single-process mode. 
        """

        
        def expected_improvement(x, gauss_proc, eval_loss, space_dim):
        
            x_p = x.reshape(-1, space_dim)
        
            mu, sigma = gauss_proc.predict(x_p, return_std=True)
        
            opti_loss = np.max(eval_loss)
        
            # In case sigma equals zero
            with np.errstate(divide='ignore'):
                Z = (mu - opti_loss) / sigma
                expected_improvement = (mu - opti_loss) * norm.cdf(Z) + sigma * norm.pdf(Z)
                expected_improvement[sigma == 0.0] == 0.0
        
            return -1* expected_improvement
            
        if prev_res==None:
            x_ini = []
            y_ini = []
        else:
            x_ini = prev_res[0]
            y_ini = prev_res[1]
            

        flux_fin=[]
    
        space_dim = bounds.shape[0]
        ann_center=self.param[2]
        modn=self.param[1]
        cuben=self.param[0]
        
        params_m=[]
        
        for i in range(len(param_type)):
            if param_type[i]=='int':
                params_m.append(np.random.random_integers(bounds[i, 0], bounds[i, 1], (opti_param['ini_esti'])))
            else:
                params_m.append(np.random.uniform(bounds[i, 0], bounds[i, 1], (opti_param['ini_esti'])))
                
        if self.model[modn]=='FM KLIP' or self.model[modn]=='FM LOCI':
            max_rad=self.max_r+1
        else:
            max_rad=self.maxradius+1
            
        # Determination of the considered angular distances for the optimization process
        
        if self.trunc is not None:
            max_rad=min(self.trunc*self.asize[modn],max_rad)
        if max_rad>self.minradius+3*self.asize[modn]+self.asize[modn]//2:
            range_sel = list(range(self.minradius+self.asize[modn]//2,self.minradius+3*self.asize[modn]+self.asize[modn]//2,self.asize[modn]))
            if max_rad>self.minradius+7*self.asize[modn]:
                range_sel.extend(list(range(self.minradius+3*self.asize[modn]+self.asize[modn]//2,self.minradius+7*self.asize[modn],2*self.asize[modn])))
                range_sel.extend(list(range(self.minradius+7*self.asize[modn]+self.asize[modn]//2,max_rad-3*self.asize[modn]//2-1,4*self.asize[modn])))
                range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[modn]*self.asize[modn]-self.asize[modn]//2-1)
            else:
                range_sel.extend(list(range(self.minradius+3*self.asize[modn]+self.asize[modn]//2,max_rad-self.asize[modn]//2,2*self.asize[modn])))
                if max_rad==self.minradius+7*self.asize[modn]:
                    range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[modn]*self.asize[modn]-self.asize[modn]//2-1)
        else:
            range_sel=list(range(self.minradius+self.asize[modn]//2,max_rad-self.asize[modn]//2,self.asize[modn]))
    
        it=0
        
        for j in range_sel:
            
            self.param[2]=j       
            res_param = pool_map(ncore, loss_function, iterable(np.array(params_m).T))

            res_mean_temp=[]
            for res_temp in res_param:
                res_mean_temp.append(res_temp[0])
            self.mean_opti[cuben,modn,j]=np.mean(np.asarray(res_mean_temp))
            
            y_ini_temp=[]
            for res_temp in res_param:
                if j==self.minradius+self.asize[modn]//2:
                    x_ini.append(res_temp[2])
                y_ini_temp.append(res_temp[0]/self.mean_opti[cuben,modn,j])


            y_ini.append([y_ini_temp])
            it+=1
            print(self.model[modn]+' Gaussian process initialization: annulus {} done!'.format(j))                  
        y_ini=list(np.asarray(y_ini).sum(axis=0))
             

        # Creation the Gaussian process
        
        kernel = gp.kernels.RBF(1.0, length_scale_bounds=(0.5,5))
    
        model = gp.GaussianProcessRegressor( kernel,
                                           alpha=1e-2,                               
                                        n_restarts_optimizer=0,
                                        normalize_y=False)
        param_f=[]
        flux_f=[]
        optires=[]
        crop_r=1

        for j in range(crop_r):
            x_fin = []
            y_fin = []
        
            params_m=[]
            self.param[2]=ann_center
            
            y_i=y_ini[0]
            x_cop=x_ini.copy()
            y_cop=list(y_i.copy())
                     
            it2=0
            flux_fin=np.zeros((len(range_sel),opti_param['opti_iter']))
            
            for n in range(opti_param['opti_iter']):
                if len(x_fin)>0:
                    x_cop.append(x_fin[-1])
                    y_cop.append(y_fin[-1])
                    model.fit(np.array(x_cop), np.asarray(y_cop))
                else:
                    model.fit(np.array(x_ini),y_i)
        
                # Selection of next parameter set via maximisation of the expected improvement
                
                if opti_param['random_search']:
                    x_random=[]
                    for i in range(len(param_type)):
                        if param_type[i]=='int':
                            x_random.append(np.random.random_integers(bounds[i, 0], bounds[i, 1], (opti_param['random_search'])))
                        else:
                            x_random.append(np.random.uniform(bounds[i, 0], bounds[i, 1], (opti_param['random_search'])))
                            
                    x_random=np.array(x_random).T
                    ei = -1 * expected_improvement(x_random, model, y_i, space_dim=space_dim) 
                    params_m = x_random[np.argmax(ei), :]
                    
                    # Duplicates break the GP
                    if any(np.abs(params_m - x_ini).sum(axis=1) > 1e-10):
            
                        for i in range(len(param_type)):
                            if param_type[i]=='int':
                                params_m[i]=np.random.random_integers(bounds[i, 0], bounds[i, 1],1)
                            else:
                                params_m[i]=np.random.uniform(bounds[i, 0], bounds[i, 1], 1)
                                
                it1=0
                y_fin_temp=[]

                for k in range_sel:
                    self.param[2]=k
                    res_temp =loss_function(params_m)
                    flux_fin[it1,it2]=res_temp[1]
                    if k==self.minradius+self.asize[modn]//2:
                        x_fin.append(res_temp[2])
                    y_fin_temp.append((res_temp[0]/self.mean_opti[cuben,modn,self.param[2]]))
                    it1+=1


                y_fin.append(np.asarray(y_fin_temp).sum(axis=0))  
                it2+=1


            param_f.append(x_fin[np.argmax(np.array(y_fin))])
            flux_f.append(flux_fin[:,np.argmax(np.array(y_fin))])
            optires.append(max(y_fin))    

        
        print(self.model[modn]+' Bayesian optimization done!')  
            
 
        return param_f[optires.index(max(optires))],max(optires), [x_ini, y_ini],flux_f[optires.index(max(optires))]
    
    def pso_optimisation(self, loss_function, bounds,param_type, prev_res=None, opti_param={'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10,'opti_iter':15,'ini_esti':10} ,ncore=1):

        """ bayesian_optimisation
        Uses Gaussian Processes to optimise the loss function `loss_function`.
        
        Parameters
        ----------
        
            loss_loss: function.
                Function to be optimised, in our case the contrast_esti function.
            bounds: numpy ndarray, 2d
                Lower and upper bounds of the parameters used to compute the cube of
                residuals allowing the estimation of the average contrast.
            param_type: list
                Type of parameters used by the PSF-subtraction technique to compute the cube
                of residuals allowing the estimation of the average contrast ('int' or 'float')
            prev_res: None or numpy ndarray, 2d
                Parmater sets and corresponding average contrasts generated at the previous 
                angular distance. Allow to smooth the transition from one annulus to another 
                during the optimization process for the annular mode
            param_optimisation: dict, optional
                Dictionnary regrouping the parameters used by the PSO optimization
                framework. 'w' is the inertia factor, 'c1' is the cognitive factor and
                'c2' is the social factor, 'n_particles' is the number of particles in the
                swarm (number of point tested at each iteration), 'opti_iter' the number of
                iterations,'ini_esti' number of sets of parameters for which the loss
                function is computed to initialize the PSO. {'c1': 1, 'c2': 1, 'w':0.5,
                'n_particles':10,'opti_iter':15,'ini_esti':10}
            ncore : int, optional
                Number of processes for parallel computing. By default ('ncore=1') 
                the algorithm works in single-process mode. 
        """

        if prev_res==None:
            x_ini = []
            y_ini = []
        else:
            x_ini = prev_res[0]
            y_ini = prev_res[1]
            
        ann_center=self.param[2]
        modn=self.param[1]
        cuben=self.param[0]

        
        
        params_m=[]
        for i in range(len(param_type)):
            if param_type[i]=='int':
                params_m.append(np.random.random_integers(bounds[i, 0], bounds[i, 1], (opti_param['ini_esti'])))
            else:
                params_m.append(np.random.uniform(bounds[i, 0], bounds[i, 1], (opti_param['ini_esti'])))
                
        if self.model[modn]=='FM KLIP' or self.model[modn]=='FM LOCI':
            max_rad=self.max_r+1
        else:
            max_rad=self.maxradius+1
            
        # Determination of the considered angular distances for the optimization process
        
        if self.trunc is not None:
            max_rad=min(self.trunc*self.asize[modn],max_rad)
        if max_rad>self.minradius+3*self.asize[modn]+self.asize[modn]//2:
            range_sel = list(range(self.minradius+self.asize[modn]//2,self.minradius+3*self.asize[modn]+self.asize[modn]//2,self.asize[modn]))
            if max_rad>self.minradius+7*self.asize[modn]:
                range_sel.extend(list(range(self.minradius+3*self.asize[modn]+self.asize[modn]//2,self.minradius+7*self.asize[modn],2*self.asize[modn])))
                range_sel.extend(list(range(self.minradius+7*self.asize[modn]+self.asize[modn]//2,max_rad-3*self.asize[modn]//2-1,4*self.asize[modn])))
                range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[modn]*self.asize[modn]-self.asize[modn]//2-1)
            else:
                range_sel.extend(list(range(self.minradius+3*self.asize[modn]+self.asize[modn]//2,max_rad-self.asize[modn]//2,2*self.asize[modn])))
                if max_rad==self.minradius+7*self.asize[modn]:
                    range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[modn]*self.asize[modn]-self.asize[modn]//2-1)
        else:
            range_sel=list(range(self.minradius+self.asize[modn]//2,max_rad-self.asize[modn]//2,self.asize[modn]))
    
        
        for j in range_sel:
            
            self.param[2]=j       
            res_param = pool_map(ncore, loss_function, iterable(np.array(params_m).T))

            res_mean_temp=[]
            for res_temp in res_param:
                res_mean_temp.append(1/res_temp[0])
            self.mean_opti[cuben,modn,j]=np.mean(np.asarray(res_mean_temp))
            
            y_ini_temp=[]
            for res_temp in res_param:
                if j==self.minradius+self.asize[modn]//2:
                    x_ini.append(res_temp[2])
                y_ini_temp.append((1/res_temp[0])/self.mean_opti[cuben,modn,j])


            y_ini.append([y_ini_temp])
            print(self.model[modn]+' PSO process initialization: annulus {} done!'.format(j))                  
        y_ini=list(np.asarray(y_ini).sum(axis=0))


        # Initialization of the PSO algorithm
        x_i=np.array(x_ini)
        
        my_topology = Star() # The Topology Class
        my_options_start={'c1': opti_param['c1'], 'c2': opti_param['c2'], 'w':opti_param['w']}
        my_options_end = {'c1': opti_param['c1'], 'c2': opti_param['c2'], 'w': opti_param['w']}
                
        param_f=[]
        flux_f=[]
        optires=[]
        crop_r=1
        
        for j in range(crop_r):
        
            params_m=[]
            self.param[2]=ann_center            
            y_i=y_ini[0]

            ini_position=x_i[y_i<=np.sort(y_i)[opti_param['n_particles']-1],:]
            my_swarm = P.create_swarm(n_particles=opti_param['n_particles'], dimensions=bounds.shape[0], options=my_options_start, bounds=(list(bounds.T[0]),list(bounds.T[1])),init_pos=ini_position)
            my_swarm.bounds=(list(bounds.T[0]),list(bounds.T[1]))
            my_swarm.velocity_clamp=None
            my_swarm.vh=P.handlers.VelocityHandler(strategy="unmodified")
            my_swarm.bh=P.handlers.BoundaryHandler(strategy="periodic")
            my_swarm.current_cost=y_i[y_i<=np.sort(y_i)[opti_param['n_particles']-1]]
            my_swarm.pbest_cost = my_swarm.current_cost  
            
            iterations = opti_param['opti_iter']
            
            for n in range(iterations):

                y_i=[]
                # Part 1: Update personal best
                if n!=0:
                        
                    for j in range_sel:
                        
                        self.param[2]=j       
                        res_param = pool_map(ncore, loss_function, iterable(np.array(my_swarm.position)))
                        
                        res_mean_temp=[]
                        for res_temp in res_param:
                            res_mean_temp.append(1/res_temp[0])
                        
                        self.mean_opti[cuben,modn,j]=np.mean(np.asarray(res_mean_temp))
                        
                        y_ini_temp=[]
                        for res_temp in res_param:
                            y_ini_temp.append((1/res_temp[0])/self.mean_opti[cuben,modn,j])
                        
                        y_i.append([y_ini_temp])
                        
                    my_swarm.current_cost=np.asarray(y_i).sum(axis=0)[0]

                
               
                my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm) # Update and store
    
                my_swarm.options={'c1': my_options_start['c1']+n*(my_options_end['c1']-my_options_start['c1'])/(iterations-1), 'c2': my_options_start['c2']+n*(my_options_end['c2']-my_options_start['c2'])/(iterations-1), 'w': my_options_start['w']+n*(my_options_end['w']-my_options_start['w'])/(iterations-1)}
                # Part 2: Update global best
                # Note that gbest computation is dependent on your topology
                if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
                    my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)
    
                # Let's print our output
    
                print('Iteration: {} | Best cost: {}'.format(n+1, my_swarm.best_cost))
    
            # Part 3: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
                my_swarm.velocity = my_topology.compute_velocity(my_swarm, my_swarm.velocity_clamp, my_swarm.vh, my_swarm.bounds
                )
                my_swarm.position = my_topology.compute_position(my_swarm, my_swarm.bounds, my_swarm.bh
                )
                
            param_f.append(my_swarm.best_pos)
            flux_f.append(1/my_swarm.best_cost)
            optires.append(my_swarm.best_cost)    


        print(self.model[modn]+' PSO optimization done!')          
 
        return param_f[optires.index(max(optires))],max(optires), [x_ini, y_ini],flux_f[optires.index(max(optires))]


    def opti_model(self,optimisation_model='PSO', param_optimisation={'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10,'opti_iter':15,'ini_esti':10, 'random_search':100}):
             
        """
        Function allowing the optimization of the PSF-subtraction techniques parameters 
        (for more details see Dahlqvist et al. 2021)
        
        Parameters
        ----------
        optimisation_model: str, optional
            Approach used for the paramters optimal selection via the maximization of the 
            contrast for APCA, LOCI, KLIP and KLIP FM. The optimization is done either
            via a Bayesian approach ('Bayesian') or using Particle Swarm optimization
            ('PSO'). Default is PSO.
        param_optimisation: dict, optional
            dictionnary regrouping the parameters used by the Bayesian or the PSO optimization
            framework. For the Bayesian optimization we have 'opti_iter' the number of iterations,
            ,'ini_esti' number of sets of parameters for which the loss function is computed to
            initialize the Gaussian process, random_search the number of random searches for the
            selection of the next set of parameters to sample based on the maximisation of the 
            expected immprovement. For the PSO optimization, 'w' is the inertia factor, 'c1' is
            the cognitive factor and 'c2' is the social factor, 'n_particles' is the number of
            particles in the swarm (number of point tested at each iteration), 'opti_iter' the 
            number of iterations,'ini_esti' number of sets of parameters for which the loss
            function is computed to initialize the PSO. {'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10
            ,'opti_iter':15,'ini_esti':10, 'random_search':100}
        filt: True, optional
            If True, a Hampel Filter is applied on the set of parameters for the annular mode
            in order to avoid outliers due to potential bright artefacts.
        """
            
        self.opti=True
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
            
        self.opti_theta=np.zeros((len(self.cube),self.maxradius+5))
        self.contrast=np.zeros((len(self.cube),len(self.model),self.maxradius+5)) 
        self.flux_opti=np.zeros((len(self.cube),len(self.model),self.maxradius+5))
        self.mean_opti=np.zeros((len(self.cube),len(self.model),self.maxradius+5))
    
                           
        for k in range(len(self.cube)):
            
            for j in range(len(self.model)):
                simures=None
                self.ini=True         
                if self.model[j]=='FM KLIP' or self.model[j]=='FM LOCI':
                    max_rad=self.max_r+1
                else:
                    max_rad=self.maxradius+1
                    
                # Determination of the considered angular distances for the optimization process

                if self.trunc is not None:
                    max_rad=min(self.trunc*self.asize[j],max_rad)
                if max_rad>self.minradius+3*self.asize[j]+self.asize[j]//2:
                    range_sel = list(range(self.minradius+self.asize[j]//2,self.minradius+3*self.asize[j]+self.asize[j]//2,self.asize[j]))
                    if max_rad>self.minradius+7*self.asize[j]:
                        range_sel.extend(list(range(self.minradius+3*self.asize[j]+self.asize[j]//2,self.minradius+7*self.asize[j],2*self.asize[j])))
                        range_sel.extend(list(range(self.minradius+7*self.asize[j]+self.asize[j]//2,max_rad-3*self.asize[j]//2-1,4*self.asize[j])))
                        range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[j]*self.asize[j]-self.asize[j]//2-1)
                    else:
                        range_sel.extend(list(range(self.minradius+3*self.asize[j]+self.asize[j]//2,max_rad-self.asize[j]//2,2*self.asize[j])))
                        if max_rad==self.minradius+7*self.asize[j]:
                            range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[j]*self.asize[j]-self.asize[j]//2-1)
                else:
                    range_sel=list(range(self.minradius+self.asize[j]//2,max_rad-self.asize[j]//2,self.asize[j]))

                for i in range_sel:  
                    
                    indicesy,indicesx=get_time_series(self.cube[k],i)
                    cube_derot,angle_list,scale_list=rot_scale('ini',self.cube[k],None,self.pa[k],self.scale_list[k], self.imlib, self.interpolation)  
                    cube_derot=rot_scale('fin',self.cube[k],cube_derot,angle_list,scale_list, self.imlib, self.interpolation)
                    apertures = photutils.CircularAperture(np.array((indicesx, indicesy)).T, round(self.fwhm/2))
                    fluxes = photutils.aperture_photometry(cube_derot.sum(axis=0), apertures)
                    fluxes = np.array(fluxes['aperture_sum'])
                    x_sel=indicesx[np.argsort(fluxes)[len(fluxes)//2]]
                    y_sel=indicesy[np.argsort(fluxes)[len(fluxes)//2]]
                    
                    ceny, cenx = frame_center(cube_derot[0])
                    
                    self.opti_theta[k,i]=np.degrees(np.arctan2(y_sel-ceny, x_sel-cenx))
                    
                for i in range_sel:                    
                        
                    if self.model[j]=='APCA':
                        
                        self.param=[k,j,i]
                        
                        if self.opti_bound[j] is None:
                            bounds=np.array([[15,45],[1,4],[0.25,1]])
                        else:
                            bounds=np.array(self.opti_bound[j])
                            
                        param_type=['int','int','float']

                        if optimisation_model=='Bayesian':
                            opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                        else:
                            opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.pso_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                        self.ncomp[j][:,k]=int(opti_param[0])
                        self.nsegments[j][:,k]=int(opti_param[1])
                        self.delta_rot[j][:,k]=opti_param[2] 
                        break
                 
                       
                    elif self.model[j]=='NMF':
                        
                        self.param=[k,j,i]
                        
                        optires=[]
                        flux=[]
                        test_param=[]
                        sel_param=[]
                        
                        if self.opti_bound[j] is None:
                            bounds=[2,20]
                        else:
                            bounds=self.opti_bound[j][0]
                        
                        param_range=range(bounds[0],bounds[1]+1)
                        flux=np.zeros((len(range_sel),len(param_range)))
                        it1=0
                        for h in range_sel:
                            
                            self.param=[k,j,h]
                            res_param=[]
                            
                            res_param = pool_map(self.ncore, self.contrast_esti, iterable(param_range))
                            
                            res_mean=[]
                            for res_temp in res_param:
                                res_mean.append(res_temp[0])
                            res_mean=np.mean(np.asarray(res_mean))

                            it2=0
                            for res_temp in res_param:
                                flux[it1,it2]=res_temp[1]
                                if h==self.minradius+self.asize[j]//2:
                                    sel_param.append(res_temp[2])
                                    optires.append(np.asarray(res_temp[0])/np.asarray(res_mean))
                                else:
                                    optires[it2]+=res_temp[0]/res_mean
                                   
                                it2+=1
                            it1+=1
                        print('NMF optimization done!')
                        optires=np.array(optires)
                        
                        self.ncomp[j][:,k]=sel_param[np.unravel_index(optires.argmax(), optires.shape)[0]]
                        self.contrast[k,j,range_sel]=optires.max()
                        self.flux_opti[k,j,range_sel]=flux[:,np.unravel_index(optires.argmax(), optires.shape)[0]]
                        break
                                    
                    
                    elif self.model[j]=='LLSG':
                        
                        self.param=[k,j,i]
                        
                        optires=[]
                        flux=[]
                        test_param=[]
                        sel_param=[]
                        
                        if self.opti_bound[j] is None:
                                bounds=[[1,10],[1,4]]
                        else:
                                bounds=self.opti_bound[j]
                        
                        for l in range(bounds[0][0],bounds[0][1]+1):
                            for m in range(bounds[1][0],bounds[1][1]+1):
                                test_param.append([l,m])
                                
                        flux=np.zeros((len(range_sel),len(test_param)))
                        it1=0
                        for h in range_sel:
                            self.param=[k,j,h]
                            res_param = pool_map(self.ncore, self.contrast_esti, iterable(np.array(test_param)))
                            
                            res_mean=[]
                            for res_temp in res_param:
                                res_mean.append(res_temp[0])
                            res_mean=np.mean(np.asarray(res_mean))
                            
                            it2=0
                            for res_temp in res_param:
                                flux[it1,it2]=res_temp[1]
                                if h==self.minradius+self.asize[j]//2:
                                    sel_param.append(res_temp[2])
                                    optires.append(np.asarray(res_temp[0])/np.asarray(res_mean))
                                else:
                                    optires[it2]+=res_temp[0]/res_mean
                                    
                                it2+=1
                            it1+=1
                        print('LLSG optimization done!')
                        optires=np.array(optires)
                        
                        self.rank[j][:,k]=sel_param[np.unravel_index(optires.argmax(), optires.shape)[0]][0]
                        self.nsegments[j][:,k]=sel_param[np.unravel_index(optires.argmax(), optires.shape)[0]][1]
                        self.contrast[k,j,range_sel]=optires.max()
                        self.flux_opti[k,j,range_sel]=flux[:,np.unravel_index(optires.argmax(), optires.shape)[0]]
                        break
                      
                    elif self.model[j]=='LOCI':
                        
                        self.param=[k,j,i]
                        
                        if self.opti_bound[j] is None:
                                bounds=np.array([[1e-3,1e-2],[0.25,1]])
                        else:
                                bounds=np.array(self.opti_bound[j])

                        param_type=['float','float']
                        

                        if optimisation_model=='Bayesian':
                            opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                        else:
                            opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.pso_optimisation(self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                        self.tolerance[j][:,k]=opti_param[0]
                        self.delta_rot[j][:,k]=opti_param[1]
                        break
   
                    elif self.model[j]=='KLIP':
        
                        self.param=[k,j,i]
                        
                        if self.opti_bound[j] is None:
                                bounds=np.array([[15,45],[0.25,1]])
                        else:
                                bounds=np.array(self.opti_bound[j])

                        param_type=['int','float']

                        
                        if optimisation_model=='Bayesian':
                            opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                        else:
                            opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.pso_optimisation(self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                        self.ncomp[j][:,k]=int(opti_param[0])
                        self.delta_rot[j][:,k]=opti_param[1]

                        break
                    
                    elif self.model[j]=='FM LOCI':

                        self.param=[k,j,i]
                        
                        if self.opti_bound[j] is None:
                                bounds=np.array([[1e-3,1e-2],[0.25,1]])
                        else:
                                bounds=np.array(self.opti_bound[j])

                        param_type=['float','float']
                        
                        if optimisation_model=='Bayesian':
                            opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                        else:
                            opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.pso_optimisation(self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                        self.tolerance[j][:,k]=opti_param[0]
                        self.delta_rot[j][:,k]=opti_param[1]

                        break
                    
                    elif self.model[j]=='FM KLIP':

                        self.param=[k,j,i]
                        
                        if self.opti_bound[j] is None:
                                bounds=np.array([[15,45],[0.25,1]])
                        else:
                                bounds=np.array(self.opti_bound[j])
                                
                        param_type=['int','float']

                        if optimisation_model=='Bayesian':
                            opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                        else:
                            opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.pso_optimisation(self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                        self.ncomp[j][:,k]=int(opti_param[0])
                        self.delta_rot[j][:,k]=opti_param[1]

                        break
                    
                    print('Model parameters selection: Cube {} : Model {} : Radius {} done!'.format(k, j,i))
                
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
                    
                    
    def RSM_test(self,cuben,modn,ann_center,sel_theta,sel_flux,thresh=False):
        
        """
        Function computing the cube of likelihoods for a given PSF-subtraction 
        techniques 'modn', a given ADI sequence 'cuben' and a given angular distance 'ann_center'
        with or without the injection of a fake companion (respc. thresh=False and thresh=False). 
        Sel_theta indicate the azimuth of the injected fake companion and sel_flux the flux 
        associated to the fake companion. This function is used by the RSM optimization function (opti_RSM).
        """

        if thresh==False:
            
            psf_template = normalize_psf(self.psf[cuben], fwhm=self.fwhm, verbose=False,size=self.psf[cuben].shape[1])
            
            cube_fc= cube_inject_companions(np.zeros_like(self.cube[cuben]), psf_template,
                                 self.pa[cuben], sel_flux,ann_center, self.pxscale,
                                 theta=sel_theta,verbose=False)
               
            result = self.likelihood(ann_center,cuben,modn,np.zeros_like(self.cube[cuben]),cube_fc,False)
                
        else:
                
            result = self.likelihood(ann_center,cuben,modn,np.zeros_like(self.cube[cuben]),None,False)
        
        like_temp=np.zeros(((rot_scale('ini',self.cube[cuben],None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)[0].shape[0]+1),self.cube[cuben].shape[-2],self.cube[cuben].shape[-1],len(self.interval[modn][ann_center,cuben]),2,self.crop_range[modn]))    
        
        indicesy,indicesx=get_time_series(self.cube[cuben],ann_center)          
        like_temp[:,indicesy,indicesx,:,:,:]=result[1]
        
        
        like=[]

        for k in range(self.crop_range[modn]):
            like.append(like_temp[0:(like_temp.shape[0]-1),:,:,:,:,k])
        
        self.like_fin[cuben][modn]=like 

                
                
    def perf_esti(self,cuben,modn,ann_center,sel_theta):
        
        """
        Function computing the performance index used for the RSM optimization 
        based on the cube of likelihoods generated by a PSF-subtraction 
        techniques 'modn', relying on the ADI sequence 'cuben'. The performance 
        index is defined as the ratio of the peak probability of
        an injected fake companion (injected at an angular distance 'ann_center'
        and an azimuth 'sel_theta') on the peak (noise) probability in the 
        remaining of the considered annulus. 
        This function is used by the RSM optimization function (opti_RSM).
        """

        ceny, cenx = frame_center(self.cube[cuben])
        twopi=2*np.pi
        sigposy=int(ceny + np.sin(sel_theta/360*twopi)*ann_center)
        sigposx=int(cenx+ np.cos(sel_theta/360*twopi)*ann_center)

        indc = disk((sigposy, sigposx),int(self.fwhm/2)+1)
        
        max_detect=self.probmap[indc[0],indc[1]].max()
        
        self.probmap[indc[0],indc[1]]=0
        
        indicesy,indicesx=get_time_series(self.cube[cuben],ann_center)
        
        if self.inv_ang==True:
            bg_noise=np.max(self.probmap[indicesy,indicesx])
        else:
            bg_noise=np.mean(self.probmap[indicesy,indicesx])
        
        return max_detect/bg_noise
    

        
    def opti_RSM_crop(self,ann_center,cuben,modn,estimator,colmode):
        
        """
        Function computing the performance index of the RSM detection map for a given range
        of crop sizes self.crop_range for the annulus ann_center on the cube of likelihoods generated 
        by a PSF-subtraction techniques 'modn', relying on the ADI sequence 'cuben'. 
        The detection map on which the perforance index is computed is using the
        estimator ('Forward' or 'Forward-Bakward') probability computation mode and
        the colmode ('mean', 'median' or 'max') the sum the obtained probabilities along the time axis.
        This function is used by the RSM optimization function (opti_RSM).
        """
                            
        opti_res=[]
        self.RSM_test(cuben,modn,ann_center,self.opti_theta[cuben,ann_center],self.flux_opti[cuben,modn,ann_center])
        for l in range(self.crop_range[modn]):
            
            self.probmap_esti(modthencube=True,ns=1,sel_crop=[l], estimator=estimator,colmode=colmode,ann_center=ann_center,sel_cube=[[cuben,modn]])
            opti_res.append(self.perf_esti(cuben,modn,ann_center,self.opti_theta[cuben,ann_center]))

        return np.asarray(opti_res),ann_center
    
    
    def opti_RSM_var_full(self,ann_center,cuben,modn,estimator,colmode):
        
        """
        Function computing the performance index of the RSM detection map for the different
        possible regions used to compute the noise mean and variance ('ST','FR','FM','SM','TE')
        and the two estimation mode (Gaussian maximum likelihood or variance base estimator, resp. 
        flux= True or False) for the annulus ann_center on the cube of likelihoods generated 
        by a PSF-subtraction techniques 'modn', relying on the ADI sequence 'cuben' and the full-frame 
        optimization mode. The detection map on which the performance index is computed uses the
        estimator ('Forward' or 'Forward-Bakward') probability computation mode and
        the colmode ('mean', 'median' or 'max') to sum the obtained probabilities along the time axis.
        This function is used by the RSM optimization function (opti_RSM).
        """
        
        self.RSM_test(cuben,modn,ann_center,self.opti_theta[cuben,ann_center],self.flux_opti[cuben,modn,ann_center])
        self.probmap_esti(modthencube=True,ns=1,sel_crop=[0], estimator=estimator,colmode=colmode,ann_center=ann_center,sel_cube=[[cuben,modn]])
        optires=self.perf_esti(cuben,modn,ann_center,self.opti_theta[cuben,ann_center])
        

        return optires,ann_center
    
    
                  
    def opti_RSM(self,estimator='Forward',colmode='median'):
        
        
        """
        Function optimizing five parameters of the RSM algorithm, the crop size, the method used 
        to compute the intensity parameter (pixel-wise Gaussian maximum likelihood or annulus-wise, 
        variance based estimation), the region used for the computation of the noise mean and variance 
        ('ST', 'FR', 'FM', 'SM', 'TE', see Dalqvist et al. 2021 for the definition), and define
        if the noise mean and variance estimation should be performed empiricaly or via best fit.
        For the variance based estimation of the intensity parameter, an additional parameter, 
        the multiplicative factor, is also optimized. The detection map on which the performance index
        is computed uses the estimator ('Forward' or 'Forward-Bakward') probability computation mode and
        the colmode ('mean', 'median' or 'max') to sum the obtained probabilities along the time axis.
        
        Parameters
        ----------
        
        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
        """
        
        if (any('FM KLIP'in mymodel for mymodel in self.model) or any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==1:
            self.max_r=self.maxradius
        elif(any('FM KLIP'in mymodel for mymodel in self.model) and any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==2:
            self.max_r=self.maxradius  
        
        self.opti=True
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
        
        self.crop_noflux=self.crop.copy()
        self.crop_flux=self.crop.copy()
        self.opti_theta=np.zeros((len(self.cube),self.maxradius+5))


        for j in range(len(self.model)):
            for i in range(len(self.cube)):
                
                # Determination of the considered angular distances for the optimization process
                
                if self.model[j]=='FM KLIP' or self.model[j]=='FM LOCI':
                    max_rad=self.max_r+1
                else:
                    max_rad=self.maxradius+1
                               
                interval=int(self.interval[j][0,i])
                res_interval_crop=np.zeros((max_rad,interval,self.crop_range[j]))
                
                if self.trunc is not None:
                    max_rad=min(self.trunc*self.asize[j],max_rad)
                if max_rad>self.minradius+3*self.asize[j]+self.asize[j]//2:
                    range_sel = list(range(self.minradius+self.asize[j]//2,self.minradius+3*self.asize[j]+self.asize[j]//2,self.asize[j]))
                    if max_rad>self.minradius+7*self.asize[j]:
                        range_sel.extend(list(range(self.minradius+3*self.asize[j]+self.asize[j]//2,self.minradius+7*self.asize[j],2*self.asize[j])))
                        range_sel.extend(list(range(self.minradius+7*self.asize[j]+self.asize[j]//2,max_rad-3*self.asize[j]//2-1,4*self.asize[j])))
                        range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[j]*self.asize[j]-self.asize[j]//2-1)
                    else:
                        range_sel.extend(list(range(self.minradius+3*self.asize[j]+self.asize[j]//2,max_rad-self.asize[j]//2,2*self.asize[j])))
                        if max_rad==self.minradius+7*self.asize[j]:
                            range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[j]*self.asize[j]-self.asize[j]//2-1)
                else:
                    range_sel=list(range(self.minradius+self.asize[j]//2,max_rad-self.asize[j]//2,self.asize[j]))

                    
                # Computation of the median flux position in the original ADI sequence which will be used during the RSM optimization
                for k in range_sel:  
                    
                    indicesy,indicesx=get_time_series(self.cube[i],k)
                    cube_derot,angle_list,scale_list=rot_scale('ini',self.cube[i],None,self.pa[i],self.scale_list[i], self.imlib, self.interpolation)  
                    cube_derot=rot_scale('fin',self.cube[i],cube_derot,angle_list,scale_list, self.imlib, self.interpolation)
                    apertures = photutils.CircularAperture(np.array((indicesx, indicesy)).T, round(self.fwhm/2))
                    fluxes = photutils.aperture_photometry(cube_derot.sum(axis=0), apertures)
                    fluxes = np.array(fluxes['aperture_sum'])
                    x_sel=indicesx[np.argsort(fluxes)[len(fluxes)//2]]
                    y_sel=indicesy[np.argsort(fluxes)[len(fluxes)//2]]
                    
                    ceny, cenx = frame_center(cube_derot[0])
                    
                    self.opti_theta[i,k]=np.degrees(np.arctan2(y_sel-ceny, x_sel-cenx))

                # Step-1: Selection of the optimal crop size, the intensity parameter estimator, and the multiplicative factor for the variance based estimator   
                
                
                self.var[j][:,i]='FR'

                for k in range(1,interval+1):
                    self.interval[j][:,i]=[k]
            
                    res_param=pool_map(self.ncore, self.opti_RSM_crop, iterable(range_sel),i,j,estimator,colmode)
                    
                    for res_temp in res_param:
                        res_interval_crop[res_temp[1],k-1,:]=res_temp[0]
                    
                interval_crop_sum=np.asarray(res_interval_crop).sum(axis=0) 
                self.interval[j][:,i]=np.unravel_index(interval_crop_sum.argmax(), interval_crop_sum.shape)[0]+1
                self.crop[j][:,i]=self.crop[j][0,i]+2*np.unravel_index(interval_crop_sum.argmax(), interval_crop_sum.shape)[1]
            
                print('Interval and crop size selected: Cube {}, Model {}'.format(i,j))

                optires=np.zeros((3))
                optires[0]=interval_crop_sum.max()
                 # Step-2: selection of the optimal region to compute the noise mean and variance

                if self.cube[i].ndim==3:
                    
                    crop_range_temp=np.copy(self.crop_range[j])
                    self.crop_range[j]=1
                    
                    fit_sel=np.zeros((max_rad,2))
                        
                    var_list=['FR','ST','SM']
                    
                    for m in range(1,len(var_list)):
                        
                            self.var[j][:,i]=var_list[m]
                            res_param=pool_map(self.ncore, self.opti_RSM_var_full, iterable(range_sel),i,j,estimator,colmode)
                            opti_temp=0
                            for res_temp in res_param:
                                opti_temp+=res_temp[0]
                            optires[m]=opti_temp
                                  
                    self.var[j][:,i]=['FR','ST','SM'][optires.argmax()]
                    fit_sel[0,0]=optires.max()
    
                    print('Variance estimation method selected: Cube {} : Model {}'.format(i,j))
                
                self.crop_range[j]=crop_range_temp
                                
            self.crop_range[j]=1
            
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
     
        
    def opti_combi_full(self,l,estimator,colmode,op_sel,SNR=False,ini=False):
        
        """
        Function computing the performance index of the RSM detection map for the tested sets of
        likelihhod cubes/residuals cubes used by the optimal subset selection function in the case of the full-frame 
        optimization mode. The performance index is computed based on the set of likelihood cubes op_seli
        for the radial distance l, relying either on the RSM algorithm or S/N map to compute the detection map.
        When using the RSM algorithm to gennerated the final probability map, the detection map is computed 
        using the estimator ('Forward' or 'Forward-Bakward') probability computation mode and the colmode
        ('mean', 'median' or 'max') to sum the obtained probabilities along the time axis. 
        This function is used by the optimal subset of likelihood cubes selection function (RSM_combination).
        """
        
        op_seli=op_sel.copy()
        if SNR==True:
            
            mod_sel=[]
            
            for k in range(len(op_seli)):
                i=op_seli[k][0]
                j=op_seli[k][1]
                if (self.model[j]!='FM KLIP' and self.model[j]!='FM LOCI') or (self.model[j]=='FM KLIP' and l<self.max_r) or (self.model[j]=='FM LOCI' and l<self.max_r):
                    mod_sel.append(op_seli[k])
            if len(mod_sel)>0:
                return self.contrast_multi_snr(l,mod_sel=mod_sel)
            else:
                return 0
        else:
                
            mod_deli=[]
            
            if self.contrast_sel=='Max':
                opti_pos=np.unravel_index(self.flux_opti[:,:,l].argmax(), self.flux_opti[:,:,l].shape)
            elif self.contrast_sel=='Median':
                opti_pos=np.unravel_index(np.argmin(abs(self.flux_opti[:,:,l]-np.median(self.flux_opti[:,:,l]))), self.flux_opti[:,:,l].shape)
            elif self.contrast_sel=='Min':
                opti_pos=np.unravel_index(self.flux_opti[:,:,l].argmin(), self.flux_opti[:,:,l].shape)
                
            opti_theta=self.opti_theta[opti_pos[0],l]        
            for k in range(len(op_seli)):
                i=op_seli[k][0]
                j=op_seli[k][1]
                if (self.model[j]=='FM KLIP' and l>=self.max_r) or (self.model[j]=='FM LOCI' and l>=self.max_r) or (self.model[j]=='FM KLIP' and ini==True) or (self.model[j]=='FM LOCI' and ini==True):
                    mod_deli.append(k)

            
            if len(mod_deli)>0:
                for index in sorted(mod_deli, reverse=True): 
                    del op_seli[index]  
                
            if len(op_seli)>0:
                self.probmap_esti(ns=1, estimator=estimator,colmode=colmode,ann_center=l,sel_cube=op_seli)
                return self.perf_esti(i,j,l,opti_theta)
            else:
                return 0
    
    
    
    def RSM_test_multi(self,cuben,modn,range_sel):
        
        """
        Function computing the cube of likelihoods for a given PSF-subtraction 
        techniques 'modn', a given ADI sequence 'cuben' and a given angular distance 'ann_center'
        with or without the injection of a fake companion (respc. thresh=False and thresh=False). 
        Sel_theta indicate the azimuth of the injected fake companion and sel_flux the flux 
        associated to the fake companion. This function is used by the RSM optimization function (opti_RSM).
        """
        
        like_temp=np.zeros(((rot_scale('ini',self.cube[cuben],None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)[0].shape[0]+1),self.cube[cuben].shape[-2],self.cube[cuben].shape[-1],len(self.interval[modn][np.argmax(self.interval[modn][:,cuben]),cuben]),2,self.crop_range[modn]))    
            
        for l in range_sel:
            
            if self.contrast_sel=='Max':
                opti_pos=np.unravel_index(self.flux_opti[:,:,l].argmax(), self.flux_opti[:,:,l].shape)
            elif self.contrast_sel=='Median':
                opti_pos=np.unravel_index(np.argmin(abs(self.flux_opti[:,:,l]-np.median(self.flux_opti[:,:,l]))), self.flux_opti[:,:,l].shape)
            elif self.contrast_sel=='Min':
                opti_pos=np.unravel_index(self.flux_opti[:,:,l].argmin(), self.flux_opti[:,:,l].shape)
                
            flux_opti=self.flux_opti[opti_pos[0],opti_pos[1],l]
            opti_theta=self.opti_theta[opti_pos[0],l]    
                
            psf_template = normalize_psf(self.psf[cuben], fwhm=self.fwhm, verbose=False,size=self.psf[cuben].shape[1])

            
            cube_fc= cube_inject_companions(np.zeros_like(self.cube[cuben]), psf_template,
                                 self.pa[cuben], flux_opti,l, self.pxscale,
                                 theta=opti_theta,verbose=False)
               
            result = self.likelihood(l,cuben,modn,np.zeros_like(self.cube[cuben]),cube_fc,False)
                

            indicesy,indicesx=get_time_series(self.cube[cuben],l)          
            like_temp[:,indicesy,indicesx,:,:,:]=result[1]
        
        
        like=[]

        for k in range(self.crop_range[modn]):
            like.append(like_temp[0:(like_temp.shape[0]-1),:,:,:,:,k])
        
        self.like_fin[cuben][modn]=like 
    
                                         
    def opti_combination(self,estimator='Forward',colmode='median',threshold=True,contrast_sel='Max',combination='Bottom-Up',SNR=False):  

        """
        Function selecting the sets of likelihhod cubes/residuals cubes maximizing the annulus-wise 
        perfomance index for the annular or the global performance index for full-frame optimization
        mode, for respectively the auto-RSM and auto-S/N frameworks.
        
        Parameters
        ----------
        

        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
        threshold: bool, optional
            When True a radial treshold is computed on the final detection map with parallactic angles reversed.
            For a given angular separation, the radial threshold is defined as the maximum probability observed 
            within the annulus. The radia thresholds are checked for outliers and smoothed via a Hampel filter.
            Only used when relying on the auto-RSM framework. Default is True.
        contrast_sel: str,optional
            Contrast and azimuth definition for the optimal likelihood cubes/ residuall cubes selection.
            If 'Max' ('Min' or 'Median'), the largest (smallest or median) contrast obtained during the 
            PSF-subtraction techniques optimization will be chosen along the corresponding 
            azimuthal position for the likelihood cubes selection. Default is 'Max'.
        combination: str,optional
            Type of greedy selection algorithm used for the selection of the optimal set of cubes 
            of likelihoods/cubes of residuals (either 'Bottom-Up' or 'Top-Down'). For more details
            see Dahlqvist et al. (2021). Default is 'Bottom-Up'.
        SNR: bool,optional
            If True, the auto-S/N framework is used, resulting in an optimizated final S/N map when using 
            subsequently the opti_map. If False the auto-RSM framework is used, providing an optimized
            probability map when using subsequently the opti_map.
        """
        
        if (any('FM KLIP'in mymodel for mymodel in self.model) or any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==1:
            self.maxradius=self.max_r
        elif(any('FM KLIP'in mymodel for mymodel in self.model) and any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==2:
            self.maxradius=self.max_r  
        
        self.opti=True
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
                
        self.opti_sel=list(np.zeros((self.maxradius+1)))
        self.threshold=np.zeros((self.maxradius+1))            
        self.contrast_sel=contrast_sel
        self.combination=combination
        
                
        for j in range(len(self.model)):
            for i in range(len(self.cube)):
                
                # Determination of the considered angular distances for the optimization process
                
                if self.model[j]=='FM KLIP' or self.model[j]=='FM LOCI':
                    max_rad=self.max_r+1
                else:
                    max_rad=self.maxradius+1
                
                if self.trunc is not None:
                    max_rad=min(self.trunc*self.asize[j],max_rad)
                if max_rad>self.minradius+3*self.asize[j]+self.asize[j]//2:
                    range_sel = list(range(self.minradius+self.asize[j]//2,self.minradius+3*self.asize[j]+self.asize[j]//2,self.asize[j]))
                    if max_rad>self.minradius+7*self.asize[j]:
                        range_sel.extend(list(range(self.minradius+3*self.asize[j]+self.asize[j]//2,self.minradius+7*self.asize[j],2*self.asize[j])))
                        range_sel.extend(list(range(self.minradius+7*self.asize[j]+self.asize[j]//2,max_rad-3*self.asize[j]//2-1,4*self.asize[j])))
                        range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[j]*self.asize[j]-self.asize[j]//2-1)
                    else:
                        range_sel.extend(list(range(self.minradius+3*self.asize[j]+self.asize[j]//2,max_rad-self.asize[j]//2,2*self.asize[j])))
                        if max_rad==self.minradius+7*self.asize[j]:
                            range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[j]*self.asize[j]-self.asize[j]//2-1)
                else:
                    range_sel=list(range(self.minradius+self.asize[j]//2,max_rad-self.asize[j]//2,self.asize[j]))
                    
                    
                # Computation of the median flux position in the original ADI sequence which will be used during the RSM optimization
                for k in range_sel:  
                    
                    indicesy,indicesx=get_time_series(self.cube[i],k)
                    cube_derot,angle_list,scale_list=rot_scale('ini',self.cube[i],None,self.pa[i],self.scale_list[i], self.imlib, self.interpolation)  
                    cube_derot=rot_scale('fin',self.cube[i],cube_derot,angle_list,scale_list, self.imlib, self.interpolation)
                    apertures = photutils.CircularAperture(np.array((indicesx, indicesy)).T, round(self.fwhm/2))
                    fluxes = photutils.aperture_photometry(cube_derot.sum(axis=0), apertures)
                    fluxes = np.array(fluxes['aperture_sum'])
                    x_sel=indicesx[np.argsort(fluxes)[len(fluxes)//2]]
                    y_sel=indicesy[np.argsort(fluxes)[len(fluxes)//2]]
                    
                    ceny, cenx = frame_center(cube_derot[0])
                    
                    self.opti_theta[i,k]=np.degrees(np.arctan2(y_sel-ceny, x_sel-cenx))
                    
                if SNR==False:   
                    self.RSM_test_multi(i,j,range_sel)
        
        # Selection of the optimal set of cubes of likelihoods / cubes of residuals via top-down or bottom-up greedy selection
        ncore_temp=np.copy(self.ncore)
        self.ncore=1
            
        # Determination of the considered angular distances for the optimization process

        if self.trunc is not None:
            max_rad=min(self.trunc*self.asize[0],self.maxradius+1)
        else:
            max_rad=self.maxradius+1
        if max_rad>self.minradius+3*self.asize[0]+self.asize[0]//2:
            range_sel = list(range(self.minradius+self.asize[0]//2,self.minradius+3*self.asize[0]+self.asize[0]//2,self.asize[0]))
            if max_rad>self.minradius+7*self.asize[0]:
                range_sel.extend(list(range(self.minradius+3*self.asize[0]+self.asize[0]//2,self.minradius+7*self.asize[0],2*self.asize[0])))
                range_sel.extend(list(range(self.minradius+7*self.asize[0]+self.asize[0]//2,max_rad-3*self.asize[0]//2-1,4*self.asize[0])))
                range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[0]*self.asize[0]-self.asize[0]//2-1)
            else:
                range_sel.extend(list(range(self.minradius+3*self.asize[0]+self.asize[0]//2,max_rad-self.asize[0]//2,2*self.asize[0])))
                if max_rad==self.minradius+7*self.asize[0]:
                    range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[0]*self.asize[0]-self.asize[0]//2-1)
        else:
            range_sel=list(range(self.minradius+self.asize[0]//2,max_rad-self.asize[0]//2,self.asize[0]))

        if self.combination=='Top-Down':
            mod_sel=[[0,0]]*(len(self.cube)*len(self.model))
            it=0
            for i in range(len(self.cube)):
                for j in range(len(self.model)):
                    mod_sel[it]=[i,j]
                    it+=1

            results=pool_map(self.ncore, self.opti_combi_full,iterable(range_sel),estimator,colmode,mod_sel,SNR)
            res_opti=sum(results) 
            prev_res_opti=0
            print('Initialization done!')
            while res_opti>prev_res_opti:
                prev_res_opti=res_opti
                res_temp=[]
                for i in range(len(mod_sel)):
                    temp_sel=mod_sel.copy()
                    del temp_sel[i]
                    
                    results=pool_map(self.ncore, self.opti_combi_full,iterable(range_sel),estimator,colmode,temp_sel,SNR)
                    res_temp.append(sum(results))
                res_opti=max(res_temp)
                if res_opti>prev_res_opti: 
                    del mod_sel[np.argmax(res_temp)]
                
                print('Round done!') 

        
            self.opti_sel=list([mod_sel]*(self.maxradius+1))
                 
            print('Greedy selection done!') 
            
        elif self.combination=='Bottom-Up':
            op_sel=[]
            res_sep=[]
            sel_cube=[]
            for i in range(len(self.cube)):
                for j in range(len(self.model)):

                            results=pool_map(self.ncore, self.opti_combi_full,iterable(range_sel),estimator,colmode,[[i,j]],SNR,True)
                            
                            res_sep.append(sum(results))
                            sel_cube.append([i,j])
            
            print('Initialization done!')
            op_sel.append(sel_cube[np.argmax(np.array(res_sep))])
            opti_res=max(res_sep)
            del sel_cube[np.argmax(np.array(res_sep))]
            prev_opti_res=0
            while opti_res>prev_opti_res and len(sel_cube)>0:
                res_temp=[]
                mod_del=[]
                for l in range(len(sel_cube)):
                    op_sel.append(sel_cube[l])
                    
                    results=pool_map(self.ncore, self.opti_combi_full,iterable(range_sel),estimator,colmode,op_sel,SNR)
                    
                    res_temp.append(sum(results))
                    if sum(results)<opti_res:
                        mod_del.append(l)

                    del op_sel[len(op_sel)-1]
                    
                if max(res_temp)>opti_res:
                    prev_opti_res=opti_res
                    opti_res=max(res_temp)
                    op_sel.append(sel_cube[np.argmax(np.array(res_temp))])
                    mod_del.append(np.argmax(np.array(res_temp)))
                else:
                    prev_opti_res=opti_res
                if len(mod_del)>0:
                    for index in sorted(mod_del, reverse=True): 
                        del sel_cube[index]
                    
                print('Round done!')
            
            self.opti_sel=list([op_sel]*(self.maxradius+1))
            
            #if self.contrast_sel!='Min':
            #    self.contrast_sel='Min'
            #    for n in range(len(op_sel)):
            #        self.RSM_test_multi(op_sel[n][0],op_sel[n][1],range_sel)
            #    results=pool_map(self.ncore, self.opti_combi_full,iterable(range_sel),estimator,colmode,op_sel,SNR)
            #    self.opti_combi_res=np.sum(results)
            #else:    
            #     self.opti_combi_res=opti_res
            print('Greedy selection done!') 
                
        # Computation of the radial thresholds

        if threshold==True and SNR==False: 
           # if sum(self.threshold[(self.max_r+1):(self.maxradius+1)])==0:
            #    range_sel= range(self.minradius,self.max_r+1)
            #else:

            self.threshold_esti(estimator=estimator,colmode=colmode,Full=False)            
                
        self.ncore=ncore_temp
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
                
                

    def threshold_esti(self,estimator='Forward',colmode='median',Full=False): 
        
        if self.opti_sel==None:
            self.opti_sel=list([[[0,0]]]*(self.maxradius+1))  
            
        if Full==True:
            mod_sel=[[0,0]]*(len(self.cube)*len(self.model))
            it=0
            for i in range(len(self.cube)):
                for j in range(len(self.model)):
                    mod_sel[it]=[i,j]
                    it+=1
            self.opti_sel=list([mod_sel]*(self.maxradius+1))
        
        range_sel= range(self.minradius,self.maxradius+1)
        self.opti=False
        
        if self.inv_ang==False:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
                
        self.opti_map(estimator=estimator,colmode=colmode,threshold=False,Full=False)
        
        self.final_map_inv=np.copy(self.final_map)
        
                
        for k in range_sel:
            indicesy,indicesx=get_time_series(self.cube[0],k)
            self.threshold[k]=np.max(self.final_map[indicesy,indicesx])
        
        self.threshold=poly_fit(self.threshold,range_sel,3)
            
        if self.inv_ang==False:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
                
        print('Threshold determination done!')
        
        
    def opti_map(self,estimator='Forward',colmode='median',ns=1,threshold=True,Full=False,SNR=False,verbose=True): 
        
        
        """
        Function computing the final detection map using the optimal set of parameters for the
        PSF-subtraction techniques (and for the RSM algorithm in the case of auto-RSM) and the
        optimal set of cubes of likelihoods/ cubes of residuals for respectively the auto-RSM 
        and auto-S/N frameworks.
        
        Parameters
        ----------
        
        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
        ns: float , optional
            Number of regime switches. Default is one regime switch per annulus but 
            smaller values may be used to reduce the impact of noise or disk structures
            on the final RSM probablity map.
        threshold: bool, optional
            When True the radial treshold is computed during the RSM_combination is applied on the
            final detection map with the original parallactic angles. Only used when relying on the auto-RSM
            framework. Default is True.
        Full: bool,optional
            If True, the entire set of ADI-sequences and PSF-subtraction techniques are used to 
            generate the final detection map. If performed after RSM_combination, the obtained optimal set
            is repkaced by the entire set of cubes. Please make ure you have saved the optimal set
            via the save_parameter function. Default is 'False'.
        SNR: bool,optional
            If True, the auto-S/N framework is used, resulting in an optimizated final S/N map when using 
            subsequently the opti_map. If False the auto-RSM framework is used, providing an optimized
            probability map when using subsequently the opti_map.
        """
            
        self.final_map=np.zeros((self.cube[0].shape[-2],self.cube[0].shape[-1]))
        
        if self.opti_sel==None:
            self.opti_sel=list([[[0,0]]]*(self.maxradius+1))
          
        if type(self.threshold)!=np.ndarray:
            self.threshold=np.zeros((self.maxradius+1))   
            
        if Full==True:
            mod_sel=[[0,0]]*(len(self.cube)*len(self.model))
            it=0
            for i in range(len(self.cube)):
                for j in range(len(self.model)):
                    mod_sel[it]=[i,j]
                    it+=1
            self.opti_sel=list([mod_sel]*(self.maxradius+1))
           
        # Computation of the final detection map for the auto-RSM or auto-S/N for the full-frame case  
    
        if SNR==True:
            self.final_map=self.SNR_esti_full(sel_cube=self.opti_sel[0],verbose=verbose)
        else:
        
            range_sel= range(self.minradius,self.maxradius+1)
            self.opti=False
            mod_sel=self.opti_sel[0].copy()
            self.lik_esti(sel_cube=mod_sel,verbose=verbose)
            
            
            if 'FM KLIP' not in self.model and 'FM LOCI' not in self.model:
                self.probmap_esti(ns=ns, estimator=estimator,colmode=colmode,ann_center=None,sel_cube=mod_sel) 
                for k in range_sel:
                    indicesy,indicesx=get_time_series(self.cube[0],k) 
                    if threshold==True:
                        self.final_map[indicesy,indicesx]=self.probmap[indicesy,indicesx]-self.threshold[k]
                    else:
                        self.final_map[indicesy,indicesx]=self.probmap[indicesy,indicesx]       
            else:
                for k in range_sel:
                    
                    if k>=self.max_r:
                        
                        try:
                            del mod_sel[list(np.asarray(mod_sel)[:,1]).index(self.model.index('FM KLIP'))]
                        except (ValueError,IndexError):
                            pass
                        try:
                            del mod_sel[list(np.asarray(mod_sel)[:,1]).index(self.model.index('FM LOCI'))] 
                        except (ValueError,IndexError):
                            pass
    
                        if len(mod_sel)==0:
                            break
                        self.probmap_esti(ns=ns, estimator=estimator,colmode=colmode,ann_center=k,sel_cube=mod_sel) 
                            
                    else:
                        self.probmap_esti(ns=ns, estimator=estimator,colmode=colmode,ann_center=k,sel_cube=mod_sel) 
    
    
                    indicesy,indicesx=get_time_series(self.cube[0],k)                    
                    if threshold==True:
                        self.final_map[indicesy,indicesx]=self.probmap[indicesy,indicesx]-self.threshold[k]
                    else:
                        self.final_map[indicesy,indicesx]=self.probmap[indicesy,indicesx]
                    
        if verbose:
            print('Final RSM map computation done!')

        
        
        
    def contrast_multi_snr(self,ann_center,mod_sel=[[0,0]]): 
        
        """
        Function computing the performance index of the S/N detection map (the contrast) when using of multiple cubes
        of residuals to generate the final S/N map (auto-S/N framweork). The final S/N map is obtained 
        by averaging the S/N maps of the selected cubes of residuals. The performance index is computed for a
        radial distane ann_center using the set of cubes of residuals mod_sel. [[i1,j1],[i2,j2],...] with i1
        the first considered PSF-subtraction technique and j1 the first considered ADI sequence, i2 the
        second considered PSF-subtraction technique, etc. This function is used by the optimal subset of
        likelihood cubes selection function (RSM_combination).
        """
        
        ceny, cenx = frame_center(self.cube[0])
        
        init_angle = 0
        ang_step=360/((np.deg2rad(360)*ann_center)/self.fwhm)
        
        tempx=[]
        tempy=[]
        
        for l in range(int(((np.deg2rad(360)*ann_center)/self.fwhm))):
            newx = ann_center * np.cos(np.deg2rad(ang_step * l+init_angle))
            newy = ann_center * np.sin(np.deg2rad(ang_step * l+init_angle))
            tempx.append(newx)
            tempy.append(newy)
        
        tempx=np.array(tempx)
        
        tempy = np.array(tempy) +int(ceny)
        tempx = np.array(tempx) + int(cenx)
      
        apertures = photutils.CircularAperture(np.array((tempx, tempy)).T, round(self.fwhm/2))
        
        flux=np.zeros(len(mod_sel))
        injected_flux=np.zeros((len(mod_sel),min(len(apertures)//2,8)))
        recovered_flux=np.zeros((len(mod_sel),min(len(apertures)//2,8)))

        for k in range(len(mod_sel)):
            cuben=mod_sel[k][0]
            modn=mod_sel[k][1]
        
            frame_nofc=self.model_esti(modn,cuben,ann_center,self.cube[cuben])[0]
                
            apertures = photutils.CircularAperture(np.array((tempx, tempy)).T, round(self.fwhm/2))

            fluxes = photutils.aperture_photometry(frame_nofc, apertures)
            fluxes = np.array(fluxes['aperture_sum'])

            n_aper = len(fluxes)
            ss_corr = np.sqrt(1 + 1/(n_aper-1))
            sigma_corr = stats.t.ppf(stats.norm.cdf(5), n_aper)*ss_corr

            noise = np.std(fluxes)
        
            flux[k] = sigma_corr*noise

            psf_template = normalize_psf(self.psf[cuben], fwhm=self.fwhm, verbose=False,size=self.psf[cuben].shape[1])
            
            fc_map = np.ones((self.cube[cuben].shape[-2],self.cube[cuben].shape[-1])) * 1e-6
            fcy=[]
            fcx=[]
            cube_fc =self.cube[cuben]
            ang_fc=range(int(init_angle),int(360+init_angle),int(360//min((len(fluxes)//2),8)))
            for i in range(len(ang_fc)):
                cube_fc = cube_inject_companions(cube_fc, psf_template,
                                     self.pa[cuben], flux[k],ann_center, self.pxscale,
                                     theta=ang_fc[i],
                                     verbose=False)
                y = int(ceny) + ann_center * np.sin(np.deg2rad(
                                                       ang_fc[i]))
                x = int(cenx) + ann_center * np.cos(np.deg2rad(
                                                       ang_fc[i]))
                if self.cube[cuben].ndim==4:
                    fc_map = frame_inject_companion(fc_map, np.mean(psf_template,axis=0), y, x,
                                            flux[k])
                else:
                    fc_map = frame_inject_companion(fc_map, psf_template, y, x,
                                            flux[k])
                fcy.append(y)
                fcx.append(x)
    
            
            frame_fc=self.model_esti(modn,cuben,ann_center,cube_fc)[0]
            
            
            for j in range(len(ang_fc)):
                apertures = photutils.CircularAperture(np.array(([fcx[j],fcy[j]])), round(self.fwhm/2))
                injected_flux[k,j] = photutils.aperture_photometry(fc_map, apertures)['aperture_sum']
                recovered_flux[k,j] = photutils.aperture_photometry((frame_fc - frame_nofc), apertures)['aperture_sum']
                
        contrast=[]        
        for j in range(len(ang_fc)):
            
            recovered_flux_conso=0
            injected_flux_conso=0
            if len(mod_sel)==1:
                recovered_flux_conso=recovered_flux[0,j]
                injected_flux_conso=injected_flux[0,j]
            else:
                for k in range(len(mod_sel)):
                    temp_list=np.array(range(len(mod_sel)))
                    temp_list=np.delete(temp_list,k)
                    recovered_flux_conso+=recovered_flux[k,j]*np.prod(flux[temp_list])
                    injected_flux_conso+=injected_flux[k,j]*np.prod(flux[temp_list])
            
            throughput = float(recovered_flux_conso / injected_flux_conso)
            
            if np.prod(flux)/throughput>0:
                contrast.append(np.mean(flux) / throughput)
                
        if len(contrast)!=0:
            contrast_mean=np.mean(contrast)
        else:
            contrast_mean=-1
            
        return np.where(contrast_mean<0,0,1/contrast_mean)
    
    
    
    
    def SNR_esti_full(self, sel_cube=[[0,0]],verbose=True):
        
        """
        Function computing the final S/N detection map, in the case of the full-frame optimization mode, 
        after optimization of the PSF-subtraction techniques and the optimal selection of the residual cubes.
        The final S/N map is obtained by averaging the S/N maps of the selected cubes of residuals provided
        by  mod_sel, [[i1,j1],[i2,j2],...] with i1 the first considered PSF-subtraction technique and j1 the
        first considered ADI sequence, i2 the second considered PSF-subtraction technique, etc.
        This function is used by the final detection map computation function (opti_map).
        """
        
        snr_temp=[]
         
        for k in range(len(sel_cube)):
            
                j=sel_cube[k][0]
                i=sel_cube[k][1]
                #Computation of the SNR maps

                if self.model[i]=='APCA':
                    print("Annular PCA estimation") 
                    residuals_cube_, frame_fin = annular_pca_adisdi(self.cube[j], self.pa[j], self.scale_list[j], fwhm=self.fwhm, ncomp=self.ncomp[i][0,j], asize=self.asize[i], 
                              delta_rot=self.delta_rot[i][0,j],delta_sep=self.delta_sep[i][0,j], svd_mode='lapack', n_segments=int(self.nsegments[i][0,j]), nproc=self.ncore,full_output=True,verbose=False)
                    snr_temp.append(vip.metrics.snrmap(frame_fin, fwhm=self.fwhm, approximated=False, plot=False,nproc=self.ncore, verbose=False))
            
                elif self.model[i]=='NMF':
                    print("NMF estimation") 
                    residuals_cube_, frame_fin = nmf_adisdi(self.cube[j], self.pa[j], self.scale_list[j], ncomp=self.ncomp[i][0,j], max_iter=100, random_state=0, mask_center_px=None,full_output=True,verbose=False)
                    snr_temp.append(vip.metrics.snrmap(frame_fin, fwhm=self.fwhm, approximated=False, plot=False,nproc=self.ncore, verbose=False))
            
                elif self.model[i]=='LLSG':
                    print("LLSGestimation") 

                    residuals_cube_, frame_fin = llsg_adisdi(self.cube[j], self.pa[j],self.scale_list[j], self.fwhm, rank=self.rank[i][0,j],asize=self.asize[i], thresh=1,n_segments=int(self.nsegments[i][0,j]), max_iter=40, random_seed=10, nproc=self.ncore,full_output=True,verbose=False)
                    snr_temp.append(vip.metrics.snrmap(frame_fin, fwhm=self.fwhm, approximated=False, plot=False,nproc=self.ncore, verbose=False))
            
                elif self.model[i]=='LOCI':
                    print("LOCI estimation") 
                    residuals_cube_,frame_fin=loci_adisdi(self.cube[j], self.pa[j],self.scale_list[j], fwhm=self.fwhm,asize=self.asize[i], n_segments=int(self.nsegments[i][0,j]),tol=self.tolerance[i][0,j], nproc=self.ncore, optim_scale_fact=2,delta_rot=self.delta_rot[i][0,j],delta_sep=self.delta_sep[i][0,j],verbose=False,full_output=True)
                    snr_temp.append(vip.metrics.snrmap(frame_fin, fwhm=self.fwhm, approximated=False, plot=False,nproc=self.ncore, verbose=False))
            
                elif self.model[i]=='KLIP':
                    print("KLIP estimation") 
                    cube_out, residuals_cube_, frame_fin = KLIP(self.cube[j], self.pa[j], ncomp=self.ncomp[i][0,j], fwhm=self.fwhm, asize=self.asize[i], 
                              delta_rot=self.delta_rot[i][0,j],full_output=True,verbose=False)
                    snr_temp.append(vip.metrics.snrmap(frame_fin, fwhm=self.fwhm, approximated=False, plot=False,nproc=self.ncore, verbose=False))
                
        return np.array(snr_temp).mean(axis=0)

    def estimate_probmax_fc(self,a,b,level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor=[1],estimator='Forward', colmode='median'):
        
        ncore_temp=np.copy(self.ncore)
        self.ncore=1
        self.minradius=a-1
        self.maxradius=a+1
        
        for j in range(len(cube)):
    
            cubefc = cube_inject_companions(cube[j], self.psf[j], self.pa[j], level*scaling_factor[j],a, plsc=self.pxscale, 
                                     theta=b/n_fc*360, n_branches=1,verbose=False)
            self.cube[j]=cubefc
        
        self.opti_map(estimator=estimator,colmode=colmode,threshold=threshold,verbose=False) 

        rsm_fin = np.where(abs(np.nan_to_num(self.final_map))>0.000001,0,probmap)+np.nan_to_num(self.final_map)

        rsm_fin=rsm_fin+abs(rsm_fin.min())

        n,y,x=cubefc.shape
        twopi=2*np.pi
        sigposy=int(y/2 + np.sin(b/n_fc*twopi)*a)
        sigposx=int(x/2+ np.cos(b/n_fc*twopi)*a)
    
        indc = disk((sigposy, sigposx),int(self.fwhm))
        max_target=np.nan_to_num(rsm_fin[indc[0],indc[1]]).max()
        max_empty=np.nan_to_num(probmap[indc[0],indc[1]]).max()
        rsm_fin[indc[0],indc[1]]=0
        
        max_map=np.nan_to_num(rsm_fin).max()
        
        print("Distance: "+"{}".format(a)+" Position: "+"{}".format(b)+" Level: "+"{}".format(level)+" Probability difference: "+"{}".format(max_target-max_map))
        self.ncore=ncore_temp
        if max_empty>=max_target:
            return -0.001,b
        else:    
            return max_target-max_map,b
    

    def contrast_curve(self,an_dist,ini_contrast,probmap,inv_ang=False,threshold=False,psf_oa=None,estimator='Forward',colmode='median',n_fc=20,completeness=0.9):
        
        """
        Function allowing the computation of contrast curves using the RSM framework
        (for more details see Dahlqvist et al. 2021)
        
        Parameters
        ----------
        an_dist: list or ndarray
            List of angular separations for which a contrast has to be estimated.
        ini_contrast: list or ndarray
            Initial contrast for the range of angular separations included in an_dist.
            The number of initial contrasts shoul be equivalent to the number of angular
            separations.
        probmap: numpy 2d ndarray
            Detection map provided by the RSM algorithm via opti_map or probmap_esti.
        inv_ang: bool, optional
            If True, the sign of the parallactic angles of all ADI sequence is flipped for
            the computation of the contrast. Default is False.
        threshold: bool, optional 
            If an angular separation based threshold has been used when generating the
            detection map, the same set of thresholds should be considered as well during
            the contrast computation. Default is False.
        psf_oa: bool, optional, optional
            Saturated PSF of the host star used to compute the scaling factor allowing the 
            conversion between contrasts and fluxes for the injection of fake companions
            during the computation of the contrast. If no Saturated PSF is provided, the 
            ini_contrast should be provided in terms of flux instead of contrast. 
            Default is None.
        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
        n_fc: int, optional
            Number of azimuths considered for the computation of the True positive rate/completeness,
            (number of fake companions injected separately). The number of azimuths is defined 
            such that the selected completeness is reachable (e.g. 95% of completeness requires at least
            20 fake companion injections). Default 20.
        completeness: float, optional
            The completeness level to be achieved when computing the contrasts, i.e. the True positive
            rate reached at the threshold associated to the first false positive (the first false 
            positive is defined as the brightest speckle present in the entire detection map). Default 95.
            
        Returns
        ----------
        1D numpy ndarray containg the contrasts for the considered angular distance at the selected
        completeness level.            
        """
        
        if type(probmap) is not np.ndarray:
            if type(self.final_map) is not np.ndarray:
                raise ValueError("A detection map should be provided")
            else:
                probmap=np.copy(self.final_map)
        if threshold==True and type(self.threshold) is not np.ndarray:
            raise ValueError("If threshold is true a threshold should be provided via self.threshold")
            
        if (100*completeness)%(100/n_fc)>0:
            n_fc=int(100/math.gcd(int(100*completeness), 100))
        
        minradius_temp=np.copy(self.minradius)
        maxradius_temp=np.copy(self.maxradius)
        cube=self.cube.copy()
        scaling_factor=[]
        
        for i in range(len(self.psf)):
       
            if psf_oa is not None:
                indices = get_annulus_segments(self.psf[i],0,round((self.fwhm)/2),1) 
                indices_oa = get_annulus_segments(psf_oa[i],0,round((self.fwhm)/2),1) 
                scaling_factor.append(psf_oa[i][indices_oa[0][0], indices_oa[0][1]].sum()/self.psf[i][indices[0][0], indices[0][1]].sum())
                
            else:
                scaling_factor.append(1)
        
        contrast_curve=np.zeros((len(an_dist)))
        ini_contrast_in=np.copy(ini_contrast)
        
        for k in range(0,len(an_dist)):
        
            a=an_dist[k]
            level=ini_contrast[k]
            pos_detect=[]
            
            detect_bound=[None,None]
            level_bound=[None,None]
            
            while len(pos_detect)==0:
                pos_detect=[] 
                pos_non_detect=[]
                val_detect=[] 
                val_non_detect=[] 
                
                res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(range(0,n_fc)),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
                
                for res_i in res:
                    
                   if res_i[0]>0:
                       pos_detect.append(res_i[1])
                       val_detect.append(res_i[0])
                   else:
                       pos_non_detect.append(res_i[1])
                       val_non_detect.append(res_i[0])
                        
                        
                if len(pos_detect)==0:
                    level=level*1.5
                    
            if len(pos_detect)>round(completeness*n_fc):
                detect_bound[1]=len(pos_detect)
                level_bound[1]=level
            elif len(pos_detect)<round(completeness*n_fc):
                detect_bound[0]=len(pos_detect)
                level_bound[0]=level            
                pos_non_detect_temp=pos_non_detect.copy()
                val_non_detect_temp=val_non_detect.copy()
                pos_detect_temp=pos_detect.copy()
                val_detect_temp=val_detect.copy()
            
            perc=len(pos_detect)/n_fc  
            print("Initialization done: current precentage "+"{}".format(perc))    
                
            while (detect_bound[0]==None or detect_bound[1]==None) and len(pos_detect)!=round(completeness*n_fc):
                
                if detect_bound[0]==None:
                    
                    level=level*0.5
                    pos_detect=[] 
                    pos_non_detect=[]
                    val_detect=[] 
                    val_non_detect=[] 
                    
                    res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(range(0,n_fc)),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
                
                    for res_i in res:
                        
                       if res_i[0]>0:
                           pos_detect.append(res_i[1])
                           val_detect.append(res_i[0])
                       else:
                           pos_non_detect.append(res_i[1])
                           val_non_detect.append(res_i[0])
                        
    
                    if len(pos_detect)>round(completeness*n_fc) and level_bound[1]>level:
                        detect_bound[1]=len(pos_detect)
                        level_bound[1]=level
                    elif len(pos_detect)<round(completeness*n_fc):
                        detect_bound[0]=len(pos_detect)
                        level_bound[0]=level 
                        pos_non_detect_temp=pos_non_detect.copy()
                        val_non_detect_temp=val_non_detect.copy()
                        pos_detect_temp=pos_detect.copy()
                        val_detect_temp=val_detect.copy()
                        
                elif detect_bound[1]==None:
                    
                    level=level*1.5
                    
                    res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(-np.sort(-np.array(pos_non_detect))),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
                
                    it=len(pos_non_detect)-1        
                    for res_i in res:
                        
                        if res_i[0]>0:
                           pos_detect.append(res_i[1])
                           val_detect.append(res_i[0])
                           del pos_non_detect[it]
                           del val_non_detect[it]
                        it-=1
                           
                    if len(pos_detect)>round(completeness*n_fc):
                        detect_bound[1]=len(pos_detect)
                        level_bound[1]=level
                    elif len(pos_detect)<round(completeness*n_fc)  and level_bound[0]<level:
                        detect_bound[0]=len(pos_detect)
                        level_bound[0]=level
                        pos_non_detect_temp=pos_non_detect.copy()
                        val_non_detect_temp=val_non_detect.copy()
                        pos_detect_temp=pos_detect.copy()
                        val_detect_temp=val_detect.copy()
                        
            if len(pos_detect)!=round(completeness*n_fc):
                print("Boundaries defined "+"{}".format(detect_bound[0]/n_fc)+" {}".format(detect_bound[1]/n_fc)) 
            
                pos_non_detect=pos_non_detect_temp.copy()
                val_non_detect=val_non_detect_temp.copy()
                pos_detect=pos_detect_temp.copy()
                val_detect=val_detect_temp.copy()
            
            while len(pos_detect)!=round(completeness*n_fc) and (level_bound[1]-level_bound[0])/level_bound[0]>1e-4:
                                
                level=level_bound[0]+(level_bound[1]-level_bound[0])/(detect_bound[1]-detect_bound[0])*(completeness*n_fc-detect_bound[0])
            
                res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(-np.sort(-np.array(pos_non_detect))),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
            
                it=len(pos_non_detect)-1      
                for res_i in res:
                    
                    if res_i[0]>0:
                       pos_detect.append(res_i[1])
                       val_detect.append(res_i[0])
                       del pos_non_detect[it]
                       del val_non_detect[it]
                    it-=1
                       
                if len(pos_detect)>round(completeness*n_fc):
                    detect_bound[1]=len(pos_detect)
                    level_bound[1]=level
                elif len(pos_detect)<round(completeness*n_fc)  and level_bound[0]<level:
                    detect_bound[0]=len(pos_detect)
                    level_bound[0]=level
                    pos_non_detect_temp=pos_non_detect.copy()
                    val_non_detect_temp=val_non_detect.copy()
                    pos_detect_temp=pos_detect.copy()
                    val_detect_temp=val_detect.copy()               
                
                if len(pos_detect)!=round(completeness*n_fc):
                    
                    pos_non_detect=pos_non_detect_temp.copy()
                    val_non_detect=val_non_detect_temp.copy()
                    pos_detect=pos_detect_temp.copy()
                    val_detect=val_detect_temp.copy()
    
                    print("Current boundaries "+"{}".format(detect_bound[0]/n_fc)+" {}".format(detect_bound[1]/n_fc)) 
    
          
            print("Distance: "+"{}".format(a)+" Final contrast "+"{}".format(level))  
            contrast_curve[k]=level           
            ini_contrast=ini_contrast_in*(contrast_curve[0:(k+1)]/ini_contrast_in[0:(k+1)]).sum()/(k+1)
                
        self.minradius=minradius_temp
        self.maxradius=maxradius_temp
        self.cube=cube
        
        return contrast_curve                              
                                                   
    def contrast_matrix(self,an_dist,ini_contrast,probmap=None,inv_ang=False,threshold=False,psf_oa=None,estimator='Forward',colmode='median',n_fc=20):
    
        """
        Function allowing the computation of three dimensional contrast curves using the RSM framework,
        with contrasts computed for multiple completeness level, allowing the reconstruction of the
        contrast/completeness distribution for every considered angular separations.
        (for more details see Dahlqvist et al. 2021)
        
        Parameters
        ----------
        an_dist: list or ndarray
            List of angular separations for which a contrast has to be estimated.
        ini_contrast: list or ndarray
            Initial contrast for the range of angular separations included in an_dist.
            The number of initial contrasts shoul be equivalent to the number of angular
            separations.
        probmap: numpy 2d ndarray
            Detection map provided by the RSM algorithm via opti_map or probmap_esti.
        inv_ang: bool, optional
            If True, the sign of the parallactic angles of all ADI sequence is flipped for
            the computation of the contrast. Default is False.
        threshold: bool, optional 
            If an angular separation based threshold has been used when generating the
            detection map, the same set of thresholds should be considered as well during
            the contrast computation. Default is False.
        psf_oa: bool, optional, optional
            Saturated PSF of the host star used to compute the scaling factor allowing the 
            conversion between contrasts and fluxes for the injection of fake companions
            during the computation of the contrast. If no Saturated PSF is provided, the 
            ini_contrast should be provided in terms of flux instead of contrast. 
            Default is None.
        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
        n_fc: int, optional
            Number of azimuths considered for the computation of the True positive rate/completeness,
            (number of fake companions injected separately). The range of achievable completenness 
            depends on the number of considered azimuths (the minimum completeness is defined as
            1/n_fc an the maximum is 1-1/n_fc). Default 20.
        
        Returns
        ----------
        2D numpy ndarray providing the contrast with the first axis associated to the angular distance
        and the second axis associated to the completeness level.
        """
        
        if type(probmap) is not np.ndarray:
            if type(self.final_map) is not np.ndarray:
                raise ValueError("A detection map should be provided")
            else:
                probmap=np.copy(self.final_map)
        if threshold==True and type(self.threshold) is not np.ndarray:
            raise ValueError("If threshold is true a threshold should be provided via self.threshold")
        
        minradius_temp=np.copy(self.minradius)
        maxradius_temp=np.copy(self.maxradius)
        cube=self.cube.copy()
        scaling_factor=[]
        
        for i in range(len(self.psf)):
       
            if psf_oa is not None:
                indices = get_annulus_segments(self.psf[i],0,round((self.fwhm)/2),1) 
                indices_oa = get_annulus_segments(psf_oa[i],0,round((self.fwhm)/2),1) 
                scaling_factor.append(psf_oa[i][indices_oa[0][0], indices_oa[0][1]].sum()/self.psf[i][indices[0][0], indices[0][1]].sum())
                
            else:
                scaling_factor.append(1)
        
        contrast_matrix=np.zeros((len(an_dist),n_fc+1))
        detect_pos_matrix=[[]]*(n_fc+1)
        ini_contrast_in=np.copy(ini_contrast)
        
        for k in range(0,len(an_dist)):
        
            a=an_dist[k]
            level=ini_contrast[k]
            pos_detect=[] 
            detect_bound=[None,None]
            level_bound=[None,None]
            
            while len(pos_detect)==0:
                pos_detect=[] 
                pos_non_detect=[]
                
                res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(range(0,n_fc)),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
                
                for res_i in res:
                    
                   if res_i[0]>0:
                       pos_detect.append(res_i[1])
                   else:
                       pos_non_detect.append(res_i[1])
                        
                contrast_matrix[k,len(pos_detect)]=level
                detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),list(pos_non_detect.copy())]
                if len(pos_detect)==0:
                    level=level*1.5
    
            perc=len(pos_detect)/n_fc  
            
            print("Initialization done: current data point "+"{}".format(perc)) 
           
            while contrast_matrix[k,0]==0:
                
                level=level*0.5
                
                res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(-np.sort(-np.array(pos_detect))),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
            
                it=len(pos_detect)-1        
                for res_i in res:
                    
                    if res_i[0]<0:
                       pos_non_detect.append(res_i[1])
                       del pos_detect[it]
                    it-=1
    
                contrast_matrix[k,len(pos_detect)]=level
                detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),list(pos_non_detect.copy())]
    
            print("Lower boundary found") 
            
            level=contrast_matrix[k,np.where(contrast_matrix[k,:]>0)[0][-1]]
            
            pos_detect=[] 
            pos_non_detect=list(np.arange(0,n_fc))
            
            while contrast_matrix[k,n_fc]==0:
                
                level=level*1.5
                
                res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(-np.sort(-np.array(pos_non_detect))),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
            
                it=len(pos_non_detect)-1        
                for res_i in res:
                    
                    if res_i[0]>0:
                       pos_detect.append(res_i[1])
                       del pos_non_detect[it]
                    it-=1
    
                contrast_matrix[k,len(pos_detect)]=level
                detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),list(pos_non_detect.copy())]
    
            print("Upper boundary found") 
    
            missing=np.where(contrast_matrix[k,:]==0)[0]
            computed=np.where(contrast_matrix[k,:]>0)[0]
            while len(missing)>0:
                
                detect_bound[0]= computed[np.argmax((computed-missing[0])[computed<missing[0]])]
                level_bound[0]=contrast_matrix[k,detect_bound[0]]
                detect_bound[1]= -np.sort(-computed)[np.argmax(np.sort((missing[0]-computed))[np.sort((missing[0]-computed))<0])]
                level_bound[1]=contrast_matrix[k,detect_bound[1]]
                it=0    
                while len(pos_detect)!=missing[0] and (level_bound[1]-level_bound[0])/level_bound[0]>1e-4:
                    
                    if np.argmin([len(detect_pos_matrix[detect_bound[1]][0]),len(detect_pos_matrix[detect_bound[0]][1])])==0:
                        

                        pos_detect=list(np.sort(detect_pos_matrix[detect_bound[1]][0]))
                        pos_non_detect=list(np.sort(detect_pos_matrix[detect_bound[1]][1]))
                        
                        level=level_bound[1]+(level_bound[1]-level_bound[0])/(detect_bound[1]-detect_bound[0])*(missing[0]-detect_bound[1])
                        
                        res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(-np.sort(-np.array(pos_detect))),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
                    
                        it=len(pos_detect)-1      
                        for res_i in res:
                            
                            if res_i[0]<0:
                               pos_non_detect.append(res_i[1])
                               del pos_detect[it]
                            it-=1   
    
                    else:
                        
                        pos_detect=list(np.sort(detect_pos_matrix[detect_bound[0]][0]))
                        pos_non_detect=list(np.sort(detect_pos_matrix[detect_bound[0]][1]))
                                   
                        level=level_bound[0]+(level_bound[1]-level_bound[0])/(detect_bound[1]-detect_bound[0])*(missing[0]-detect_bound[0])
                        
                        res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(-np.sort(-np.array(pos_non_detect))),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
                    
                        it=len(pos_non_detect)-1      
                        for res_i in res:
                            
                            if res_i[0]>0:
                               pos_detect.append(res_i[1])
                               del pos_non_detect[it]
                            it-=1
                        
                    if len(pos_detect)>missing[0]:
                        detect_bound[1]=len(pos_detect)
                        level_bound[1]=level
                    elif len(pos_detect)<missing[0]  and level_bound[0]<level:
                        detect_bound[0]=len(pos_detect)
                        level_bound[0]=level
        
                    contrast_matrix[k,len(pos_detect)]=level
                    detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),list(pos_non_detect.copy())]
                    
                    if len(pos_detect)==missing[0]:
    
                        print("Data point "+"{}".format(len(pos_detect)/n_fc)+" found. Still "+"{}".format(len(missing)-i-1)+" data point(s) missing") 
                
                if (level_bound[1]-level_bound[0])/level_bound[0]<1e-4:
                    
                    res=np.array(res)
                    if len(pos_detect)>missing[0]:
                        non_dist=[[x,res[list(res[:,1]).index(x),0]] for x in detect_pos_matrix[detect_bound[1]][0] if x not in detect_pos_matrix[detect_bound[0]][0] ]
                        for h in range(len(pos_detect)-missing[0]):
                            detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),list(pos_non_detect.copy())] 
                            pos_non_detect.append(pos_detect.pop(pos_detect.index(non_dist[np.argmin(np.array(non_dist)[:,1])][0])))
                            contrast_matrix[k,len(pos_detect)]=level
                            non_dist.pop(np.argmin(np.array(non_dist)[:,1]))
                    elif len(pos_detect)<missing[0]:
                        non_dist=[[x,res[list(res[:,1]).index(x),0]] for x in detect_pos_matrix[detect_bound[1]][0] if x not in detect_pos_matrix[detect_bound[0]][0] ]
                        for h in range(missing[0]-len(pos_detect)):
                            pos_detect.append(pos_non_detect.pop(pos_non_detect.index(non_dist[np.argmax(np.array(non_dist)[:,1])][0])))
                            non_dist.pop(np.argmax(np.array(non_dist)[:,1]))
                            detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),list(pos_non_detect.copy())]  
                            contrast_matrix[k,len(pos_detect)]=level
                    
                computed=np.where(contrast_matrix[k,:]>0)[0]
                missing=np.where(contrast_matrix[k,:]==0)[0]
                
            ini_contrast=ini_contrast_in*(contrast_matrix[k,int(n_fc/2)]/ini_contrast_in[0:(k+1)]).sum()/(k+1)
           
        self.minradius=minradius_temp
        self.maxradius=maxradius_temp
        self.cube=cube
        
        return contrast_matrix 
    

    def multi_estimate_flux_negfc(self,param,level=None,expected_pos=None,cube=None,ns=1,loss_func='value',scaling_factor=[1],opti='photo',norma_factor=[1,1]):
        
        args=(level,expected_pos,cube,ns,loss_func,scaling_factor,'interval',norma_factor)  
        res_param = pool_map(self.ncore, self.estimate_flux_negfc, iterable(np.array(param)),*args)
        return np.array(res_param)
        
    def estimate_flux_negfc(self,param,level,expected_pos,cube,ns,loss_func='value',scaling_factor=[1],opti='photo',norma_factor=[1,1]):
    
        ncore_temp=np.copy(self.ncore)
        self.ncore=1
        
        if opti=='photo':
            level=param
        elif opti=='bayesian':
            level=param[0]
            expected_pos=[param[1],param[2]]
        elif opti=='interval':
            level=param[0]*norma_factor[0]
            expected_pos=[param[1],param[2]]
    
        ceny, cenx = frame_center(cube[0])
    
        an_center=np.sqrt((expected_pos[0]-ceny)**2+(expected_pos[1]-cenx)**2)
    
        a=int(an_center)
    
        expected_theta=np.degrees(np.arctan2(expected_pos[0]-ceny, expected_pos[1]-cenx))
    
        self.minradius=a-round(self.fwhm)//2
        self.maxraddius=a+round(self.fwhm)//2
    
        for j in range(len(cube)):
    
            cubefc = cube_inject_companions(cube[j], self.psf[j],self.pa[j], -level*scaling_factor[j],an_center, plsc=self.pxscale, 
                                    theta=expected_theta, n_branches=1,verbose=False)
            self.cube[j]=cubefc    

        self.opti_map(estimator='Forward-Backward',ns=ns,colmode='median',threshold=False,verbose=False) 
        #plot_frames(self.final_map)
    
        indices=get_annulus_segments(cube[0][0],a-int(self.fwhm)//2,int(self.fwhm)+1,1) 
    
        indc = disk((expected_pos[0], expected_pos[1]),2*int(self.fwhm)+1)
        indc2 = disk((expected_pos[0], expected_pos[1]),3*int(self.fwhm)+1)
        
        ind_circle=np.array([x for x in set(tuple(x) for x in np.array(indices[0]).T) & set(tuple(x) for x in np.array(indc).T)]).T
        
        ind_circle2=np.array([x for x in set(tuple(x) for x in np.array(indices[0]).T) & set(tuple(x) for x in np.array(indc2).T)]).T
    
        del_list=-np.sort(-np.array([np.intersect1d(np.where(ind_circle2[0]==ind_circle[0][x])[0],np.where(ind_circle2[1]==ind_circle[1][x])[0]) for x in range(len(ind_circle[1]))]).reshape(1,-1))[0]
      
        indicey=  list(ind_circle2[0]) 
        indicex= list(ind_circle2[1])
    
        for k in range(len(del_list)):
    
            del indicex[del_list[k]]
            del indicey[del_list[k]]        
    
        values_t=self.final_map[ind_circle[0],ind_circle[1]]
        values=self.final_map[indicey,indicex]
    
        mean_values = np.mean(values)
        #values_t=abs(values_t-mean_values)
        values=abs(values-mean_values)
        
        self.ncore=ncore_temp
        
        if loss_func=='value':
            if opti=='bayesian':
                return param,1/np.mean(values_t)
            elif opti=='interval':
                return np.mean(values_t)*norma_factor[1]
            else:
                return np.mean(values_t)
        elif loss_func=='prob':
            mean_values = np.mean(values)
            bap=np.sqrt((np.std(values)**2)/2)
            if opti=='bayesian':
                return param,1/(np.sum(np.abs(values_t-mean_values)/bap)) 
            else:               
                return np.sum(np.abs(values_t-mean_values)/bap)
    
        
    def target_charact(self,expected_pos,psf_oa=None, ns=1, loss_func='value',param_optimisation={'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10,'opti_iter':15,'ini_esti':10},photo_bound=[1e-5,1e-4,10],ci_esti=None,first_guess=False):
    
        
        """
        Planet characterization algorithm relying on the RSM framework and the negative
        fake companion approach. The function provides both the estimated astrometry
        and photometry of a planetary candidate as well as the associated error.
        (for more details see Dahlqvist et al. 2021)
        
        Parameters
        ----------
        expected_pos: list or ndarray
            (Y,X) position of the detected planetary candidate.
        psf_oa: bool, optional
            Saturated PSF of the host star used to compute the scaling factor allowing the 
            conversion between contrasts and fluxes. If no Saturated PSF is provided, the 
            photometry will be provided in terms of flux not contrast. 
            Default is None.
        ns: float , optional
            Number of regime switches. Default is one regime switch per annulus but 
            smaller values may be used to reduce the impact of noise or disk structures
            on the final RSM probablity map. The number of regime switches my be increase
            in the case of faint sources to ease their characterization. Default is 1.
        loss_func: str, optional
            Loss function used for the computation of the source astrometry and photometry.
            If 'value', it relies on the minimization of the average probability within a 2
            FWHM aperture centered on the expected position of the source. If 'prob', it
            considers the entire annulus for the computation of the background noise statistics
            and use a Gaussian distribution to determine the probability that the probabilities
            associated with the planetary candidates in the detection map belongs to the 
            background noise distribution. Default is 'value'.
        param_optimisation: dict, optional
            dictionnary regrouping the parameters used by the Bayesian or the PSO optimization
            framework. For the Bayesian optimization we have 'opti_iter' the number of iterations,
            ,'ini_esti' number of sets of parameters for which the loss function is computed to
            initialize the Gaussian process, random_search the number of random searches for the
            selection of the next set of parameters to sample based on the maximisation of the 
            expected immprovement. For the PSO optimization, 'w' is the inertia factor, 'c1' is
            the cognitive factor and 'c2' is the social factor, 'n_particles' is the number of
            particles in the swarm (number of point tested at each iteration), 'opti_iter' the 
            number of iterations,'ini_esti' number of sets of parameters for which the loss
            function is computed to initialize the PSO. {'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10
            ,'opti_iter':15,'ini_esti':10,'random_search':100}
        photo_bound: list, optional
            Photometry range considered during the estimation. The range, expressed in terms of
            contrast (or flux if psf_oa is not provided), is given by the first two number, while
            the third number gives the number of tested values within this range. Default [1e-5,1e-4,10].
        ci_esti: str, optional
            Parameters determining if a confidence interval should be computed for the photometry
            and astrometry.The erros associated with the photometry and astrometry can be estimated
            via the inversion of the hessian matrix ('hessian') or via the BFGS minimisation approach 
            ('BFGS') which allows to further improve the precision of the estimates but requires 
            more computation time. Default is None, implying no computation of confidence intervals.
        first_guess: boolean, optional
            Define if an initialisation of the algorrithm is done via a standard negfc (using the VIP
            function firstguess) before applying the PSO or Bayesian optimisation. This initialisation
            is useful when the target is very bright. It relies on PCA approach, SNR ratio maps and
            negative fake companion injection to estimate the photometry and astrometry. Default is
            False.
        
        Returns
        ----------
        List containing, the photometry, the astrometry expressed in pixel and in angular distance 
        (mas) and angular position (°), as well as the errors associated with the photometry and
        astrometry at 68% (one standard deviation) and at 95% if ci_esti is 'hessian' or 'BFGS'.
        In the first case the order is:
        [photometry,centroid position y,centroid position x, radial distance, angular position]
        and in the second case:
        [photometry,centroid position y,centroid position x, photometry standar error, position y
        standard error,position x standard error, photometry 95% error, position y 95% error,
        position x 95% error  radial distance, angular position]
        
        """ 
        
        if self.opti_sel==None:
            mod_sel=[[0,0]]*(len(self.cube)*len(self.model))
            it=0
            for i in range(len(self.cube)):
                for j in range(len(self.model)):
                    mod_sel[it]=[i,j]
                    it+=1
            self.opti_sel=list([mod_sel]*(self.maxradius+1))
        
        minradius_temp=np.copy(self.minradius)
        maxradius_temp=np.copy(self.maxradius)
        cube=self.cube.copy()
        scaling_factor=[]
    
        for i in range(len(self.psf)):
    
            if psf_oa is not None:
                indices = get_annulus_segments(self.psf[i],0,round((self.fwhm)/2),1) 
                indices_oa = get_annulus_segments(psf_oa[i],0,round((self.fwhm)/2),1) 
                scaling_factor.append(psf_oa[i][indices_oa[0][0], indices_oa[0][1]].sum()/self.psf[i][indices[0][0], indices[0][1]].sum())
    
            else:
                scaling_factor.append(1)
    
        ceny, cenx = frame_center(cube[0])
    
        an_center=np.sqrt((expected_pos[0]-ceny)**2+(expected_pos[1]-cenx)**2)
    
        a=int(an_center)
    
        self.minradius=a-round(self.fwhm)
        self.maxradius=a+round(self.fwhm)
    
        self.opti_map(estimator='Forward-Backward',ns=ns,colmode='median',threshold=False,verbose=False) 
        #plot_frames(self.final_map)
        indc = disk((expected_pos[0], expected_pos[1]),2*int(self.fwhm)+1)
        centroid_y,centroid_x=np.where(self.final_map == np.amax(self.final_map[indc[0],indc[1]]))
        print("Peak value: "+"cartesian Y {}".format(centroid_y[0])+", X {}".format(centroid_x[0]))
    
        psf_subimage, suby, subx = vip.var.get_square(self.final_map, 19,
                                                  centroid_y[0],centroid_x[0], position=True)       
        y, x = np.indices(psf_subimage.shape)
    
        y=y.flatten()
        x=x.flatten()
        psf_subimage=psf_subimage.flatten()
    
        model_fit = lmfit.models.Gaussian2dModel()
        params = model_fit.guess(psf_subimage, x, y)
        result = model_fit.fit(psf_subimage, x=x, y=y, params=params)
    
        centroid_x=result.params['centerx'].value+ subx        
        centroid_y=result.params['centery'].value+ suby
        erry=result.params['centery'].stderr
        errx=result.params['centerx'].stderr
    
        print("First guess astrometry: "+"cartesian Y {}".format(centroid_y)+", X {}".format(centroid_x)+", Error Y {}".format(erry)+", Error X {}".format(errx))  
    
        expected_pos=np.copy([centroid_y,centroid_x]) 
    
        from vip_hci.fm import firstguess
        #from vip_hci.negfc import firstguess_from_coord 
        if first_guess==True:
            ini_flux=[]
            ini_posy=[]
            ini_posx=[]
            for i in range(len(self.psf)):
                step=(np.log10(photo_bound[1])-np.log10(photo_bound[0]))/(photo_bound[2]*2)
                flux_list=np.power(np.repeat(10,photo_bound[2]*2+1),
            np.array(np.arange(np.log10(photo_bound[0]),np.log10(photo_bound[1])+step,step))[:photo_bound[2]*2+1])
                temp_res = firstguess(cube[i], self.pa[i], self.psf[i], ncomp=20,
                                       planets_xy_coord=[(centroid_x,centroid_y)], fwhm=self.fwhm, 
                                      f_range=flux_list*scaling_factor[i], annulus_width=5, aperture_radius=2,
                                           imlib = 'opencv',interpolation= 'lanczos4', simplex=True, 
                                       plot=False, verbose=False)
    
                ini_flux.append(temp_res[2]/scaling_factor[i])
                posy_fc= ceny+temp_res[0] * np.sin(np.deg2rad(temp_res[1]))
                posx_fc = cenx+temp_res[0] * np.cos(np.deg2rad(temp_res[1]))
                ini_posy.append(posy_fc)
                ini_posx.append(posx_fc)    
            curr_flux=np.mean(ini_flux)
            centroid_y=np.mean(ini_posy)
            centroid_x=np.mean(ini_posx)
            lower_b_flux=10**(np.log10(curr_flux)-step)
            upper_b_flux=10**(np.log10(curr_flux)+step)
            
            print("Initial astrometry: "+"cartesian Y {}".format(centroid_y)+", X {}".format(centroid_x))
    
        if first_guess==False or abs(expected_pos[0]-centroid_y)>2 or abs(expected_pos[1]-centroid_x)>2:
            step=(np.log10(photo_bound[1])-np.log10(photo_bound[0]))/photo_bound[2]
            flux_list=np.power(np.repeat(10,photo_bound[2]+1),
        np.array(np.arange(np.log10(photo_bound[0]),np.log10(photo_bound[1])+step,step))[:photo_bound[2]+1])
            args=(0,expected_pos,cube,ns, 'value',scaling_factor,'photo')   
            res_list=pool_map(self.ncore, self.estimate_flux_negfc, iterable(flux_list.T),*args)
    
            curr_flux=flux_list[np.argmin(np.array(res_list))]
            if np.argmin(np.array(res_list))==(len(flux_list)-1):
                print('Upper bound reached, consider an extension of the upper bound.')
                upper_b_flux=10**(np.log10(photo_bound[1])+step)
            else:
                upper_b_flux=flux_list[np.argmin(np.array(res_list))+1]
            if np.argmin(np.array(res_list))==0:
                print('Lower bound reached, consider an extension of the lower bound.')
                lower_b_flux=10**(np.log10(photo_bound[0])-step)
            else:
                lower_b_flux=flux_list[np.argmin(np.array(res_list))-1]

    
        print("Inital estimated photometry {}".format(curr_flux))
        
        if abs(expected_pos[0]-centroid_y)>2 or abs(expected_pos[1]-centroid_x)>2:
            centroid_y,centroid_x=expected_pos
        else:
            expected_pos=[centroid_y,centroid_x]

        my_topology = Star() # The Topology Class
        my_options_start={'c1': param_optimisation['c1'], 'c2': param_optimisation['c2'], 'w':param_optimisation['w']}
        my_options_end = {'c1': param_optimisation['c1'], 'c2': param_optimisation['c2'], 'w': param_optimisation['w']}

         # The Swarm Class
        kwargs=dict(level=curr_flux/10**(int(np.log10(curr_flux))-1),expected_pos=expected_pos,cube=cube,ns=ns,loss_func=loss_func,scaling_factor=scaling_factor,opti='interval',norma_factor=[10**(int(np.log10(curr_flux))-1),1])

        bounds=([lower_b_flux/10**(int(np.log10(curr_flux))-1),expected_pos[0]-1,expected_pos[1]-1],[upper_b_flux/10**(int(np.log10(curr_flux))-1),expected_pos[0]+1,expected_pos[1]+1])
        params=[]
        for i in range(len(bounds[0])):

            params.append(np.random.uniform(bounds[0][i], bounds[1][i], (param_optimisation['ini_esti'])))

        params=np.append(np.array(params).T,[[curr_flux/10**(int(np.log10(curr_flux))-1),expected_pos[0],expected_pos[1]]],0)


        ini_cost=self.multi_estimate_flux_negfc(params,**kwargs)

        ini_position=params[ini_cost<np.sort(ini_cost)[param_optimisation['n_particles']],:]

        my_swarm = P.create_swarm(n_particles=param_optimisation['n_particles'], dimensions=3, options=my_options_start, bounds=([lower_b_flux/10**(int(np.log10(curr_flux))-1),centroid_y-1,centroid_x-1],[upper_b_flux/10**(int(np.log10(curr_flux))-1),centroid_y+1,centroid_x+1]),init_pos=ini_position)
        my_swarm.bounds=([lower_b_flux/10**(int(np.log10(curr_flux))-1),centroid_y-1,centroid_x-1],[upper_b_flux/10**(int(np.log10(curr_flux))-1),centroid_y+1,centroid_x+1])
        my_swarm.velocity_clamp=None
        my_swarm.vh=P.handlers.VelocityHandler(strategy="unmodified")
        my_swarm.bh=P.handlers.BoundaryHandler(strategy="periodic")
        my_swarm.current_cost=ini_cost[ini_cost<np.sort(ini_cost)[param_optimisation['n_particles']]]
        my_swarm.pbest_cost = my_swarm.current_cost

        iterations = param_optimisation['opti_iter'] # Set 100 iterations

        for i in range(iterations):
            # Part 1: Update personal best
            if i!=0:
                my_swarm.current_cost = self.multi_estimate_flux_negfc(my_swarm.position,**kwargs) # Compute current c        ost

            my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm) # Update and store

            my_swarm.options={'c1': my_options_start['c1']+i*(my_options_end['c1']-my_options_start['c1'])/(iterations-1), 'c2': my_options_start['c2']+i*(my_options_end['c2']-my_options_start['c2'])/(iterations-1), 'w': my_options_start['w']+i*(my_options_end['w']-my_options_start['w'])/(iterations-1)}
            # Part 2: Update global best
            # Note that gbest computation is dependent on your topology
            if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
                my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)

            # Let's print our output

            print('Iteration: {} | my_swarm.best_cost: {}'.format(i+1, my_swarm.best_cost))

        # Part 3: Update position and velocity matrices
    # Note that position and velocity updates are dependent on your topology
            my_swarm.velocity = my_topology.compute_velocity(my_swarm, my_swarm.velocity_clamp, my_swarm.vh, my_swarm.bounds
            )
            my_swarm.position = my_topology.compute_position(my_swarm, my_swarm.bounds, my_swarm.bh
            )

        expected_pos=my_swarm.best_pos[1:3]
        centroid_y,centroid_x=expected_pos
        curr_flux=my_swarm.best_pos[0]*10**(int(np.log10(curr_flux))-1)
        max_val=my_swarm.best_cost
    
    
        print("Optimal astrometry {}".format(i+1)+" {}".format(expected_pos))
        print("Optimal photometry {}".format(i+1)+" {}".format(curr_flux))
    
        if ci_esti is None:
            
            self.minradius=minradius_temp
            self.maxradius=maxradius_temp
            self.cube=cube
            
            ceny, cenx = frame_center(cube[0])
            rad_dist= np.sqrt((centroid_y-ceny)**2+(centroid_x-cenx)**2)*self.pxscale*1000
            theta=np.degrees(np.arctan2(centroid_y-ceny, centroid_x-cenx))
            print("Estimated astrometry: "+"radial distance {}".format(rad_dist)+" mas, Angle {}".format(theta))
    
            
            return curr_flux,centroid_y,centroid_x,rad_dist,theta,my_swarm.best_cost
        else:
            print("Confidence interval computation")  
    
            if ci_esti=='hessian':
                power_curr_flux=10**(int(np.log10(curr_flux))-1)
                fun = lambda param: self.estimate_flux_negfc(param,curr_flux/power_curr_flux,expected_pos,cube,ns,loss_func,scaling_factor,'interval',[power_curr_flux,10**(-round(np.log10(max_val)))])   
                opti_param, hessian = Hessian([curr_flux/power_curr_flux,expected_pos[0],expected_pos[1]],fun,step=0.05)   
                curr_flux,expected_pos[0],expected_pos[1]=opti_param
                curr_flux*=power_curr_flux
                try:
                    std=np.nan_to_num(np.sqrt(np.diag(np.linalg.inv(hessian))))
                    int95=sc.stats.norm.ppf(0.95)*std
                except (RuntimeError, ValueError, RuntimeWarning):
                    std=[0,0,0]
                    int95=[0,0,0]
            elif ci_esti=='BFGS' or any(int95==0):
                res=minimize(self.estimate_flux_negfc,x0=[opti_param[0]/10**(int(np.log10(curr_flux))-1),opti_param[1],opti_param[2]], method='L-BFGS-B',bounds=((lower_b_flux/10**(int(np.log10(curr_flux))-1),upper_b_flux/10**(int(np.log10(curr_flux))-1)),(opti_param[1]-0.2,opti_param[1]+0.2),(opti_param[2]-0.2,opti_param[2]+0.2)),args=(curr_flux/10**(int(np.log10(curr_flux))-1),expected_pos,cube,ns,loss_func,scaling_factor,'interval',[10**(int(np.log10(curr_flux))-1),10**(-round(np.log10(max_val)))]),options={'ftol':5e-3,'gtol':5e-4, 'eps': 1e-2, 'maxiter': 10, 'finite_diff_rel_step': [1e-2,1e-2,1e-2]})
                try:
                    std=np.nan_to_num(np.sqrt(np.diag(res['hess_inv'].todense())))
                    int95=sc.stats.norm.ppf(0.95)*std
                    curr_flux=res.x[0]*10**(int(np.log10(curr_flux))-1)
                    expected_pos=res.x[1:3]
                except (RuntimeError, ValueError, RuntimeWarning):
                    std=[0,0,0]
                    int95=[0,0,0]
    
            err_phot=std[0]*10**(int(np.log10(curr_flux))-1)
            int95[0]*=10**(int(np.log10(curr_flux))-1)
            erry=std[1]
            errx=std[2]
            centroid_y,centroid_x=expected_pos
    
            print("Estimated photometry: "+" {}".format(curr_flux)+", Error {}".format(err_phot))
    
    
            print("Estimated astrometry: "+"Cartesian Y {}".format(centroid_y)+", X {}".format(centroid_x)+", Error Y {}".format(erry)+", Error X {}".format(errx))  
    
            ceny, cenx = frame_center(cube[0])
            rad_dist= np.sqrt((centroid_y-ceny)**2+(centroid_x-cenx)**2)*self.pxscale*1000
            theta=np.degrees(np.arctan2(centroid_y-ceny, centroid_x-cenx))
            print("Estimated astrometry: "+"radial distance {}".format(rad_dist)+" mas, Angle {}".format(theta))
    
            self.minradius=minradius_temp
            self.maxradius=maxradius_temp
            self.cube=cube
            
            return curr_flux,centroid_y,centroid_x,err_phot,erry,errx,int95[0],int95[1],int95[2],rad_dist,theta,my_swarm.best_cost
            


"""
Set of functions used by the PyRSM class to compute detection maps and optimize the parameters
of the RSM algorithm and PSF-subtraction techniques via the auto-RSM and auto-S/N frameworks
"""

def check_delta_sep(scale_list,delta_sep,minradius,fwhm,c):
    wl = np.asarray(scale_list)
    wl_ref = wl[len(wl)//2]
    sep_lft = (wl_ref - wl) / wl_ref * ((minradius + fwhm * delta_sep) / fwhm)
    sep_rgt = (wl - wl_ref) / wl_ref * ((minradius - fwhm * delta_sep) / fwhm)
    map_lft = sep_lft >= delta_sep
    map_rgt = sep_rgt >= delta_sep
    indices = np.nonzero(map_lft | map_rgt)[0]

    if indices.size == 0:
        raise RuntimeError(("No frames left after radial motion threshold for cube {}. Try "
                           "decreasing the value of `delta_sep`").format(c))   
                                        
def rot_scale(step,cube,cube_scaled,angle_list,scale_list, imlib, interpolation):
    
    """
    Function used to rescale the frames when relying on ADI+SDI before the computation the reference PSF
    (step='ini') and rescale and derotate the frames to generate the cube of residuals used by the RSM 
    algorithm (step='fin').
        
        Parameters
        ----------

        step: str
            'ini' before the reference PSF computation and 'fin' after PSF subtraction.
        cube: numpy ndarray, 3d or 4d
            Original cube
        cube_scaled: numpy ndarray, 3d
            Cube of residuals to be rescaled and derotated (None for the step='ini')
        angle_list : numpy ndarray, 1d
            Parallactic angles for each frame of the ADI sequences. 
        scale_list: numpy ndarray, 1d, optional
            Scaling factors in case of IFS data (ADI+mSDI cube). Usually, the
            scaling factors are the central channel wavelength divided by the
            shortest wavelength in the cube (more thorough approaches can be used
            to get the scaling factors). This scaling factors are used to re-scale
            the spectral channels and align the speckles. Default is None
        imlib : str, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
        interpolation : str, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    """
    
    if cube.ndim == 4:
        
        z, n, y_in, x_in = cube.shape
        scale_list = check_scal_vector(scale_list)
        
        if step=='ini':
        # rescaled cube, aligning speckles for SDI
            for i in range(n):
                if i==0:
                    fin_cube = scwave(cube[:, i, :, :], scale_list,
                                      imlib=imlib, interpolation=interpolation)[0]
                    fin_pa=np.repeat(angle_list[i],z)
                    fin_scale=scale_list
                else: 
                    
                    fin_cube = np.append(fin_cube,scwave(cube[:, i, :, :], scale_list,
                                      imlib=imlib, interpolation=interpolation)[0],axis=0)
                    fin_pa=np.append(fin_pa,np.repeat(angle_list[i],z),axis=0)
                    fin_scale=np.append(fin_scale,scale_list,axis=0)
                    
            return fin_cube,fin_pa,fin_scale
                

        elif step=='fin':  
            
                                
                cube_rescaled = scwave(cube_scaled, scale_list, 
                                full_output=True, inverse=True,
                                y_in=y_in, x_in=x_in, imlib=imlib,
                                interpolation=interpolation)[0]
                
                cube_derotated=cube_derotate(cube_rescaled,angle_list, interpolation=interpolation,imlib=imlib)

                 
                                    
                return  cube_derotated

                
    if cube.ndim == 3:
        
        if step=='ini':
     
            return cube,angle_list,None
        
        elif step=='fin':    

            cube_derotated=cube_derotate(cube_scaled,angle_list, interpolation=interpolation,imlib=imlib)
            
            return cube_derotated

def _define_annuli(angle_list, ann, n_annuli, fwhm, radius_int, annulus_width,
                   delta_rot, n_segments, verbose, strict=False):
    """ Function that defines the annuli geometry using the input parameters.
    Returns the parallactic angle threshold, the inner radius and the annulus
    center for each annulus. Function taken from python package VIP_HCI (Gomez et al. 2017).
    """
    if ann == n_annuli - 1:
        inner_radius = radius_int + (ann * annulus_width - 1)
    else:
        inner_radius = radius_int + ann * annulus_width
    ann_center = inner_radius + (annulus_width / 2)
    pa_threshold = np.rad2deg(2 * np.arctan(delta_rot * fwhm / (2 * ann_center)))
    mid_range = np.abs(np.amax(angle_list) - np.amin(angle_list)) / 2
    if pa_threshold >= mid_range - mid_range * 0.1:
        pa_threshold = float(mid_range - mid_range * 0.1)

    return pa_threshold, inner_radius, ann_center

    
    
def remove_outliers(time_s, range_sel, k=5, t0=3):
    """
    Hampel Filter to remove potential outliers in the set of selected parameters 
    for the annular mode of the auto-RSM framework
    """
    vals=pd.DataFrame(data=time_s[range_sel])
    L= 1.4826
    rolling_median=vals.rolling(k).median()
    difference=np.abs(rolling_median-vals)
    median_abs_deviation=difference.rolling(k).median()
    threshold= t0 *L * median_abs_deviation
    outlier_idx=difference>threshold
    vals[outlier_idx]=threshold[outlier_idx]
    return(vals.to_numpy().reshape(-1))
       
def interpolation(time_s,range_sel): 
    
    """
    Interpolation algorithm for the RSM parameters 
    for the annular mode of the auto-RSM framework
    """
    
    time_series=time_s.copy()
    time_series[range_sel]=remove_outliers(time_series,range_sel)
    fit = Rbf(range_sel,time_s[range_sel])
    inter_point = np.linspace(range_sel[0],range_sel[-1]+1, num=(range_sel[-1]-range_sel[0]+1), endpoint=True)
    return fit(inter_point)

def poly_fit(time_s,range_sel,poly_n):
    
    """
    Smoothing procedure for the computation of the final radial thresholds
    which are subtracted from the final RSM detection map in the final step
    of the auto-RSM framework
    """
    
    time_series=time_s.copy()
    time_series[range_sel]=remove_outliers(time_series,range_sel)
    fit_p=np.poly1d(np.polyfit(range_sel,time_series[range_sel], poly_n))
    time_series=fit_p(range(len(time_series)))
    return time_series

def get_time_series(mcube,ann_center):
    
        """
        Function defining and ordering (anti-clockwise) the pixels composing
        an annulus at a radial distance of ann_center for an ADI sequence mcube
        """
        if mcube.ndim == 4:
            indices = get_annulus_segments(mcube[0,0,:,:], ann_center,1,4,90)
        else:
            indices = get_annulus_segments(mcube[0], ann_center,1,4,90)

        tempind=np.vstack((indices[0][0],indices[0][1]))
        ind = np.lexsort((tempind[0], tempind[1]))

        indicesy=tempind[0,ind[::-1]]
        indicesx=tempind[1,ind[::-1]] 

        tempind=np.vstack((indices[1][0],indices[1][1]))
        ind = np.lexsort((-tempind[0], tempind[1]))

        indicesy=np.hstack((indicesy,tempind[0,ind[::-1]]))
        indicesx=np.hstack((indicesx,tempind[1,ind[::-1]]))

        tempind=np.vstack((indices[2][0],indices[2][1]))
        ind = np.lexsort((tempind[0], tempind[1]))

        indicesy=np.hstack((indicesy,tempind[0,ind]))
        indicesx=np.hstack((indicesx,tempind[1,ind])) 

        tempind=np.vstack((indices[3][0],indices[3][1]))
        ind = np.lexsort((-tempind[0], tempind[1]))

        indicesy=np.hstack((indicesy,tempind[0,ind]))
        indicesx=np.hstack((indicesx,tempind[1,ind]))
            
        return indicesy,indicesx
 
def perturb(frame,model_matrix,numbasis,evals_matrix, evecs_matrix, KL_basis_matrix,sci_mean_sub_matrix,refs_mean_sub_matrix, angle_list, fwhm, pa_threshold, ann_center):
    

    """
    Function allowing the estimation of the PSF forward model when relying on KLIP
    for the computation of the speckle field. The code is based on the PyKLIP library
     considering only the ADI case with a singlle number of principal components considered.
    For more details about the code, consider the PyKLIP library or the originall articles
    (Pueyo, L. 2016, ApJ, 824, 117 or
     Ruffio, J.-B., Macintosh, B., Wang, J. J., & Pueyo, L. 2017, ApJ, 842)
    """
    
    #Selection of the reference library based on the given parralactic angle threshold

    if pa_threshold != 0:
        indices_left = _find_indices_adi(angle_list, frame,
                                             pa_threshold, truncate=False)

        models_ref = model_matrix[indices_left]

    else:
        models_ref = model_matrix


    #Computation of the self-subtraction and over-subtraction for the current frame
    
    model_sci = model_matrix[frame]  
    KL_basis=KL_basis_matrix[frame]
    sci_mean_sub=sci_mean_sub_matrix[frame]
    refs_mean_sub=refs_mean_sub_matrix[frame]
    evals=evals_matrix[frame]
    evecs=evecs_matrix[frame]

    max_basis = KL_basis.shape[0]
    N_pix = KL_basis.shape[1]

    models_mean_sub = models_ref - np.nanmean(models_ref, axis=1)[:,None] 
    models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0
    
    model_sci_mean_sub = model_sci- np.nanmean(model_sci)
    model_sci_mean_sub[np.where(np.isnan(model_sci_mean_sub))] = 0
    model_sci_mean_sub_rows = np.reshape(model_sci_mean_sub,(1,N_pix))
    sci_mean_sub_rows = np.reshape(sci_mean_sub,(1,N_pix))
    
    delta_KL = np.zeros([max_basis, N_pix])

    models_mean_sub_X_refs_mean_sub_T = models_mean_sub.dot(refs_mean_sub.transpose())

    for k in range(max_basis):
        Zk = np.reshape(KL_basis[k,:],(1,KL_basis[k,:].size))
        Vk = (evecs[:,k])[:,None]


        diagVk_X_models_mean_sub_X_refs_mean_sub_T = (Vk.T).dot(models_mean_sub_X_refs_mean_sub_T)
        models_mean_sub_X_refs_mean_sub_T_X_Vk = models_mean_sub_X_refs_mean_sub_T.dot(Vk)
        DeltaZk = -(1/(2*np.sqrt(evals[k])))*(diagVk_X_models_mean_sub_X_refs_mean_sub_T.dot(Vk) + ((Vk.T).dot(models_mean_sub_X_refs_mean_sub_T_X_Vk))).dot(Zk)+(Vk.T).dot(models_mean_sub)


        for j in range(k):
            Zj = KL_basis[j, :][None,:]
            Vj = evecs[:, j][:,None]
            DeltaZk += np.sqrt(evals[j])/(evals[k]-evals[j])*(diagVk_X_models_mean_sub_X_refs_mean_sub_T.dot(Vj) + ((Vj.T).dot(models_mean_sub_X_refs_mean_sub_T_X_Vk))).dot(Zj)
        for j in range(k+1, max_basis):
            Zj = KL_basis[j, :][None,:]
            Vj = evecs[:, j][:,None]
            DeltaZk += np.sqrt(evals[j])/(evals[k]-evals[j])*(diagVk_X_models_mean_sub_X_refs_mean_sub_T.dot(Vj) + ((Vj.T).dot(models_mean_sub_X_refs_mean_sub_T_X_Vk))).dot(Zj)

        delta_KL[k] = DeltaZk/np.sqrt(evals[k])
        
    oversubtraction_inner_products = np.dot(model_sci_mean_sub_rows, KL_basis.T)  
    
    selfsubtraction_1_inner_products = np.dot(sci_mean_sub_rows, delta_KL.T)
    selfsubtraction_2_inner_products = np.dot(sci_mean_sub_rows, KL_basis.T)

    oversubtraction_inner_products[max_basis::] = 0
    klipped_oversub = np.dot(oversubtraction_inner_products, KL_basis)
    
    selfsubtraction_1_inner_products[0,max_basis::] = 0
    selfsubtraction_2_inner_products[0,max_basis::] = 0
    klipped_selfsub = np.dot(selfsubtraction_1_inner_products, KL_basis) + \
                          np.dot(selfsubtraction_2_inner_products, delta_KL)

    return model_sci[None,:] - klipped_oversub - klipped_selfsub   
        



def KLIP(cube, angle_list, nann=None, local=False,radius_int=None,radius_out=None, fwhm=4, asize=2, n_segments=1,delta_rot=1, ncomp=1,min_frames_lib=2, max_frames_lib=200,imlib='opencv',nframes=None, interpolation='lanczos4', collapse='median',full_output=False, verbose=1):

    """
    Function allowing the estimation of the cube of residuals after
    the subtraction of the speckle field modeled via the KLIP framework 
    """
    
    array = cube
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array')
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length')

    n, y, _ = array.shape
    
    angle_list = check_pa_vector(angle_list)
    
    if asize is None:
        annulus_width = int(np.ceil(2 * fwhm))
    elif isinstance(asize, int):
        annulus_width = asize
        
    # Annulus parametrization 
    

    if local==True:
            if nann> 2*annulus_width:
                n_annuli = 5
                radius_int=(nann//annulus_width-2)*annulus_width 
            else:
                n_annuli = 4 
                radius_int=(nann//annulus_width-1)*annulus_width
    else:
        if radius_int%asize>int(asize/2):
            radius_int=(radius_int//asize)*asize
        elif radius_int>=asize:
            radius_int=(radius_int//asize-1)*asize
        else:
            radius_int=0
            
        if radius_out is not None:
            
            if radius_out%asize>int(asize/2):
                radius_out=(radius_out//asize+2)*asize
            else:
                radius_out=(radius_out//asize+1)*asize
            
            n_annuli = int((radius_out - radius_int) / asize)
            
        elif radius_out==None:
            n_annuli = int((y / 2 - radius_int) / asize)
        
        elif radius_out>int(y / 2): 
            
            n_annuli = int((y / 2 - radius_int) / asize)
        
            
    # Definition of the number of segment for the diifferent annuli

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == 'auto':
        n_segments = list()
        n_segments.append(2)  
        n_segments.append(3)  
        ld = 2 * np.tan(360 / 4 / 2) * asize
        for i in range(2, n_annuli):  
            radius = i * asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    if verbose:
        msg = '# annuli = {}, Ann width = {}, FWHM = {:.3f}'
        print(msg.format(n_annuli, asize, fwhm))
        print('PCA per annulus (or annular sectors):')


    # Definition of the annuli and the corresmponding parralactic angle threshold 
    
    cube_out = np.zeros_like(array)
    for ann in range(n_annuli):
        if isinstance(ncomp, list) or isinstance(ncomp, np.ndarray):
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                msge = 'If ncomp is a list, it must match the number of annuli'
                raise TypeError(msge)
        else:
            ncompann = ncomp

        
        inner_radius = radius_int + ann * annulus_width
        n_segments_ann = n_segments[ann]


        if verbose:
            print('{} : in_rad={}, n_segm={}'.format(ann+1, inner_radius,
                                                     n_segments_ann))


        theta_init = 90
        res_ann_par = _define_annuli(angle_list, ann, int((y / 2 - radius_int) / asize), fwhm,radius_int, annulus_width, delta_rot,n_segments_ann, verbose)
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(array[0], inner_radius, annulus_width,n_segments_ann,theta_init)
        
        # Computation of the speckle field for the different frames and estimation of the cube of residuals
        
        for j in range(n_segments_ann):

            for k in range(array.shape[0]):
                
                res =KLIP_patch(k,array[:, indices[j][0], indices[j][1]], ncompann, angle_list, fwhm, pa_thr, ann_center,nframes=nframes)
                cube_out[k,indices[j][0], indices[j][1]] = res[3]


    # Cube is derotated according to the parallactic angle and collapsed
    
    cube_der = cube_derotate(cube_out, angle_list, imlib=imlib,interpolation=interpolation)
    frame = cube_collapse(cube_der, mode=collapse)

    if full_output:
        return cube_out, cube_der, frame
    else:
        return frame
    
def KLIP_patch(frame, matrix, numbasis, angle_list, fwhm, pa_threshold, ann_center,nframes=None):

    """          
    Function allowing the computation via KLIP of the speckle field for a 
    given sub-region of the original ADI sequence. Code inspired by the PyKLIP librabry
    """
    
    max_frames_lib=200
    
    if pa_threshold != 0:
        if ann_center > fwhm*20:
            indices_left = _find_indices_adi(angle_list,frame,pa_threshold, truncate=True,max_frames=max_frames_lib)
        else:
            indices_left = _find_indices_adi(angle_list, frame,pa_threshold, truncate=False,nframes=nframes)

        refs = matrix[indices_left]
        
    else:
        refs = matrix

    sci = matrix[frame]
    sci_mean_sub = sci - np.nanmean(sci)
    #sci_mean_sub[np.where(np.isnan(sci_mean_sub))] = 0
    refs_mean_sub = refs- np.nanmean(refs, axis=1)[:, None]
    #refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    # Covariance matrix definition
    covar_psfs = np.cov(refs_mean_sub)
    covar_psfs *= (np.size(sci)-1)

    tot_basis = covar_psfs.shape[0]

    numbasis = np.clip(numbasis - 1, 0, tot_basis-1)
    max_basis = np.max(numbasis) + 1

    #Computation of the eigenvectors/values of the covariance matrix
    evals, evecs = la.eigh(covar_psfs)
    evals = np.copy(evals[int(tot_basis-max_basis):int(tot_basis)])
    evecs = np.copy(evecs[:,int(tot_basis-max_basis):int(tot_basis)])
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1])

    # Computation of the principal components
    
    KL_basis = np.dot(refs_mean_sub.T,evecs)
    KL_basis = KL_basis * (1. / np.sqrt(evals))[None,:]
    KL_basis = KL_basis.T 

    N_pix = np.size(sci_mean_sub)
    sci_rows = np.reshape(sci_mean_sub, (1,N_pix))
    
    inner_products = np.dot(sci_rows, KL_basis.T)
    inner_products[0,int(max_basis)::]=0

    #Projection of the science image on the selected prinicpal component
    #to generate the speckle field model

    klip_reconstruction = np.dot(inner_products, KL_basis)

    # Subtraction of the speckle field model from the riginal science image
    #to obtain the residual frame
    
    sub_img_rows = sci_rows - klip_reconstruction 

    return evals,evecs,KL_basis,np.reshape(sub_img_rows, (N_pix)),refs_mean_sub,sci_mean_sub


def LOCI_FM(cube, psf, ann_center, angle_list,scale_list, asize,fwhm, Tol,delta_rot,delta_sep):


    """
    Computation of the optimal factors weigthing the linear combination of reference
    frames used to obtain the modeled speckle field for each frame and allowing the 
    determination of the forward modeled PSF. Estimation of the cube 
    of residuals based on the modeled speckle field.
    """


    cube_res = np.zeros_like(cube)
    ceny, cenx = frame_center(cube[0])
    radius_int=ann_center-int(1.5*asize)
    if radius_int<=0:
        radius_int=1
            
    for ann in range(3):
        n_segments_ann = 1
        inner_radius_ann = radius_int + ann*asize
        pa_threshold = _define_annuli(angle_list, ann, 3, asize,
                                      radius_int, asize, delta_rot,
                                      n_segments_ann, verbose=False)[0]
        
        indices = get_annulus_segments(cube[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann)
        ind_opt = get_annulus_segments(cube[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann,
                                       optim_scale_fact=2)
        
        ayxyx = [inner_radius_ann,pa_threshold, indices[0][0], indices[0][1],
                   ind_opt[0][0], ind_opt[0][1]]
                
        matrix_res, ind_ref, coef, yy, xx = _leastsq_patch(ayxyx,
                         angle_list,scale_list,fwhm,cube,ann_center,'manhattan', 100,delta_sep,
                         'lstsq', Tol,formod=True,psf=psf)
        
        if ann==1:
            ind_ref_list=ind_ref
            coef_list=coef
        
        cube_res[:, yy, xx] = matrix_res
    
    return cube_res, ind_ref_list,coef_list





def nmf_adisdi(cube, angle_list,scale_list=None, cube_ref=None, ncomp=1, scaling=None, max_iter=100,
        random_state=None, mask_center_px=None, imlib='opencv',
        interpolation='lanczos4', collapse='median', full_output=False,
        verbose=True, **kwargs):
    """ Non Negative Matrix Factorization for ADI or ADI+SDI sequences.This function embeds the 
    scikit-learn NMF algorithm solved through coordinate descent method.     
    """
    
    array,angle_list_t,scale_list_t=rot_scale('ini',cube,None,angle_list,scale_list,imlib, interpolation)
            


    n, y, x = array.shape
    
    matrix_ref = prepare_matrix(array, scaling, mask_center_px, mode='fullfr',
                            verbose=verbose)
    matrix_ref += np.abs(matrix_ref.min())
    if cube_ref is not None:
        matrix_ref = prepare_matrix(cube_ref, scaling, mask_center_px,
                                    mode='fullfr', verbose=verbose)
        matrix_ref += np.abs(matrix_ref.min())
 
      
    mod = NMF(n_components=ncomp, solver='cd', init='nndsvd', 
              max_iter=max_iter, random_state=100,tol=1e-3)  

    W = mod.fit_transform(matrix_ref)
    H = mod.components_

    
    reconstructed = np.dot(W, H)
    residuals = matrix_ref - reconstructed
               
    array_out = np.zeros_like(array)
    for i in range(n):
        array_out[i] = residuals[i].reshape(y,x)
            
    cube_der=rot_scale('fin',cube,array_out,angle_list_t,scale_list_t, imlib, interpolation)
    frame_fin = cube_collapse(cube_der, mode=collapse)
    
    return cube_der,frame_fin



def NMF_patch(matrix, ncomp, max_iter,random_state):

    """
    Function allowing the computation via NMF of the speckle field for a 
    given sub-region of the original ADI sequence. The code is a partial reproduction of
    the VIP function NMF_patch (Gonzalez et al. AJ, 154:7,2017)
    """

    refs = matrix+ np.abs(matrix.min())
    

    
    mod = NMF(n_components=ncomp, solver='cd', init='nndsvd', 
              max_iter=max_iter, random_state=100,tol=1e-3)  

    W = mod.fit_transform(refs)
    H = mod.components_

    
    reconstructed = np.dot(W, H)

    residuals = refs - reconstructed

    return residuals



def annular_pca_adisdi(cube, angle_list,scale_list=None, radius_int=0,radius_out=None, fwhm=4, asize=2, n_segments=1,
                 delta_rot=1,delta_sep=0.1, ncomp=1, svd_mode='lapack', nproc=None,
                 min_frames_lib=2, max_frames_lib=200, tol=1e-1, scaling=None,
                 imlib='opencv', interpolation='lanczos4', collapse='median',
                 full_output=False, verbose=False, cube_ref=None, weights=None):
    """ PCA exploiting angular and spectral variability (ADI or ADI+SDI fashion).
    """

    array,angle_list_t,scale_list_t=rot_scale('ini',cube,None,angle_list,scale_list,imlib, interpolation)
            
    n, y, _ = array.shape

    angle_list_t = check_pa_vector(angle_list_t)
    
    if radius_int%asize>int(asize/2):
        radius_int=(radius_int//asize)*asize
    elif radius_int>=asize:
        radius_int=(radius_int//asize-1)*asize
    else:
        radius_int=0
        
    if radius_out is not None:
        
        if radius_out%asize>int(asize/2):
            radius_out=(radius_out//asize+2)*asize
        else:
            radius_out=(radius_out//asize+1)*asize
        
        n_annuli = int((radius_out - radius_int) / asize)
        
    elif radius_out==None:
        n_annuli = int((y / 2 - radius_int) / asize)
    
    elif radius_out>int(y / 2): 
        
        n_annuli = int((y / 2 - radius_int) / asize)
 
    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif isinstance(delta_rot, (int, float)):
        delta_rot = [delta_rot] * n_annuli

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == 'auto':
        n_segments = list()
        n_segments.append(2)  # for first annulus
        n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * asize
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    if verbose:
        msg = 'N annuli = {}, FWHM = {:.3f}'
        print(msg.format(n_annuli, fwhm))
        print('PCA per annulus (or annular sectors):')

    if nproc is None:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = cpu_count() // 2

    # The annuli are built, and the corresponding PA thresholds for frame
    # rejection are calculated (at the center of the annulus)
    cube_out = np.zeros_like(array)
    
    for ann in range(n_annuli):
        if isinstance(ncomp, tuple) or isinstance(ncomp, np.ndarray):
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                raise TypeError('If `ncomp` is a tuple, it must match the '
                                'number of annuli')
        else:
            ncompann = ncomp

        n_segments_ann = n_segments[ann]
        res_ann_par = _define_annuli(angle_list_t, ann, n_annuli, fwhm,
                                     radius_int, asize, delta_rot[ann],
                                     n_segments_ann, verbose)
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(array[0], inner_radius, asize,
                                       n_segments_ann)
        # Library matrix is created for each segment and scaled if needed
        for j in range(n_segments_ann):
            yy = indices[j][0]
            xx = indices[j][1]
            matrix_segm = array[:, yy, xx]  # shape [nframes x npx_segment]
            if cube_ref is not None:
                matrix_segm_ref = cube_ref[:, yy, xx]
            else:
                matrix_segm_ref = None

            res = pool_map(nproc, do_pca_patch, matrix_segm, iterable(range(n)),
                           angle_list_t,scale_list_t, fwhm, pa_thr,delta_sep, ann_center, svd_mode,
                           ncompann, min_frames_lib, max_frames_lib, tol,
                           matrix_segm_ref)
            res = np.array(res)
            residuals = np.array(res[:, 0])

            for fr in range(n):
                cube_out[fr][yy, xx] = residuals[fr]

    # Cube is derotated according to the parallactic angle and collapsed
    cube_der=rot_scale('fin',cube,cube_out,angle_list_t,scale_list_t, imlib, interpolation)
    
    frame = cube_collapse(cube_der, mode=collapse)

    return cube_der, frame
   
def do_pca_patch(matrix, frame, angle_list,scale_list, fwhm, pa_threshold, delta_sep, ann_center,
                 svd_mode, ncomp, min_frames_lib, max_frames_lib, tol,
                 matrix_ref):
    
    """ 
    Function  doing the SVD/PCA for each frame patch. The code is a partial reproduction of
    the VIP function do_pca_patch (Gonzalez et al. AJ, 154:7,2017)  
    """


    if scale_list is not None:
    
        indices_left = np.intersect1d(_find_indices_adi(angle_list, frame,
                                             pa_threshold, truncate=False),_find_indices_sdi(scale_list, ann_center, frame,
                                             fwhm, delta_sep))
    else:
        indices_left = _find_indices_adi(angle_list, frame,
                                             pa_threshold, truncate=False)
    

    data_ref = matrix[indices_left]
    if matrix_ref is not None:
        # Stacking the ref and the target ref (pa thresh) libraries
        data_ref = np.vstack((matrix_ref, data_ref))

    curr_frame = matrix[frame]  # current frame
    V = get_eigenvectors(ncomp, data_ref, svd_mode, noise_error=tol)
    transformed = np.dot(curr_frame, V.T)
    reconstructed = np.dot(transformed.T, V)
    residuals = curr_frame - reconstructed
    return residuals, V.shape[0], data_ref.shape[0]


def do_pca_patch_range(matrix, frame, angle_list,scale_list, fwhm, pa_threshold,delta_sep, ann_center,
                 svd_mode, ncomp_range, min_frames_lib, max_frames_lib, tol,
                 matrix_ref):
    """ 
    Function  doing the SVD/PCA for each frame patch for a range of principal
    component ncomp_range. The code is a partial reproduction of
    the VIP function do_pca_patch (Gonzalez et al. AJ, 154:7,2017)  
    """

    if scale_list is not None:
    
        indices_left = np.intersect1d(_find_indices_adi(angle_list, frame,
                                             pa_threshold, truncate=False),_find_indices_sdi(scale_list, ann_center, frame,
                                             fwhm, delta_sep))
    else:
        indices_left = _find_indices_adi(angle_list, frame,
                                             pa_threshold, truncate=False)

    data_ref = matrix[indices_left]
    if matrix_ref is not None:
        # Stacking the ref and the target ref (pa thresh) libraries
        data_ref = np.vstack((matrix_ref, data_ref))

    curr_frame = matrix[frame]  # current frame
    V = get_eigenvectors(ncomp_range[len(ncomp_range)-1], data_ref, svd_mode, noise_error=tol)
    residuals=[]
    for i in ncomp_range:
        V_trunc=V[ncomp_range[0]:i,:]
        transformed = np.dot(curr_frame, V_trunc.T)
        reconstructed = np.dot(transformed.T, V_trunc)
        residuals.append(curr_frame - reconstructed)
        
    return residuals, V.shape[0], data_ref.shape[0]

         
def loci_adisdi(cube, angle_list,scale_list=None, fwhm=4, metric='manhattan',
                 dist_threshold=50, delta_rot=0.5,delta_sep=0.1, radius_int=0,radius_out=None, asize=4,
                 n_segments=1, nproc=1, solver='lstsq', tol=1e-3,
                 optim_scale_fact=1, imlib='opencv', interpolation='lanczos4',
                 collapse='median', nann=None,local=False, verbose=True, full_output=False):
    """ Least-squares model PSF subtraction for ADI or ADI+SDI. This code is an adaptation of the VIP
    xloci function to provide, if required, the residuals after speckle field subtraction
    for a given annulus.
    """
    
    cube_rot_scale,angle_list_t,scale_list_t=rot_scale('ini',cube,None,angle_list,scale_list,imlib, interpolation)
            
            
    y = cube_rot_scale.shape[1]
    if not asize < y // 2:
        raise ValueError("asize is too large")

    angle_list = check_pa_vector(angle_list)
    if local==True:
            n_annuli = 3 
            radius_int=nann-asize
    else:
        if radius_int%asize>int(asize/2):
            radius_int=(radius_int//asize)*asize
        elif radius_int>=asize:
            radius_int=(radius_int//asize-1)*asize
        else:
            radius_int=0
            
        if radius_out is not None:
            
            if radius_out%asize>int(asize/2):
                radius_out=(radius_out//asize+2)*asize
            else:
                radius_out=(radius_out//asize+1)*asize
            
            n_annuli = int((radius_out - radius_int) / asize)
        
        elif radius_out==None:
            n_annuli = int((y / 2 - radius_int) / asize)
        
        elif radius_out>int(y / 2): 
            
            n_annuli = int((y / 2 - radius_int) / asize)
        
            
    if verbose:
        print("Building {} annuli:".format(n_annuli))

    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif isinstance(delta_rot, (int, float)):
        delta_rot = [delta_rot] * n_annuli

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    annulus_width = asize
    if isinstance(n_segments, int):
        n_segments = [n_segments]*n_annuli
    elif n_segments == 'auto':
        n_segments = list()
        n_segments.append(2)    # for first annulus
        n_segments.append(3)    # for second annulus
        ld = 2 * np.tan(360/4/2) * annulus_width
        for i in range(2, n_annuli):    # rest of annuli
            radius = i * annulus_width
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360/ang)))

    # annulus-wise least-squares combination and subtraction
    cube_res = np.zeros_like(cube_rot_scale)

    ayxyx = []  # contains per-segment data

    for ann in range(n_annuli):
        n_segments_ann = n_segments[ann]
        inner_radius_ann = radius_int + ann*annulus_width

        # angles
        pa_threshold = _define_annuli(angle_list, ann, n_annuli, fwhm,
                                      radius_int, asize, delta_rot[ann],
                                      n_segments_ann, verbose)[0]

        # indices
        indices = get_annulus_segments(cube_rot_scale[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann)
        ind_opt = get_annulus_segments(cube_rot_scale[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann,
                                       optim_scale_fact=optim_scale_fact)

        # store segment data for multiprocessing
        ayxyx += [(inner_radius_ann+asize//2,pa_threshold, indices[nseg][0], indices[nseg][1],
                   ind_opt[nseg][0], ind_opt[nseg][1]) for nseg in
                  range(n_segments_ann)]



    msg = 'Patch-wise least-square combination and subtraction:'
    # reverse order of processing, as outer segments take longer
    res_patch = pool_map(nproc, _leastsq_patch, iterable(ayxyx[::-1]),
                         angle_list_t,scale_list_t,fwhm,cube_rot_scale, None, metric, dist_threshold,delta_sep,
                         solver, tol, verbose=verbose, msg=msg,
                         progressbar_single=True)

    for patch in res_patch:
        matrix_res, yy, xx = patch
        cube_res[:, yy, xx] = matrix_res
        
    cube_der=rot_scale('fin',cube,cube_res,angle_list_t,scale_list_t, imlib, interpolation)
    frame_der_median = cube_collapse(cube_der, collapse)

    if verbose:
        print('Done processing annuli')

    return cube_der, frame_der_median


def _leastsq_patch(ayxyx, angle_list,scale_list,fwhm,cube, nann,metric, dist_threshold,delta_sep,
                   solver, tol,formod=False,psf=None):

    """
    Function allowing th estimation of the optimal factors for the modeled speckle field
    estimation via the LOCI framework. The code has been developped based on the VIP 
    python function _leastsq_patch, but return additionnaly the set of coefficients used for
    the speckle field computation.
    """
    
    ann_center,pa_threshold, yy, xx, yy_opti, xx_opti = ayxyx
    
    ind_ref_list=[]
    coef_list=[]
    
    yy_opt=[]
    xx_opt=[]
        
    for j in range(0,len(yy_opti)):
        if not any(x in np.where(yy==yy_opti[j])[0] for x in np.where(xx==xx_opti[j])[0]):
            xx_opt.append(xx_opti[j])
            yy_opt.append(yy_opti[j])
    

    values = cube[:, yy, xx]  
    matrix_res = np.zeros((values.shape[0], yy.shape[0]))
    values_opt = cube[:, yy_opti, xx_opti]
    n_frames = cube.shape[0]


    for i in range(n_frames):
        
        if scale_list is not None:
    
            ind_fr_i = np.intersect1d(_find_indices_adi(angle_list, i,
                                                 pa_threshold, truncate=False),_find_indices_sdi(scale_list, ann_center, i,
                                                 fwhm, delta_sep))
        else:
            ind_fr_i = _find_indices_adi(angle_list, i,
                                                 pa_threshold, truncate=False)
        if len(ind_fr_i) > 0:
            A = values_opt[ind_fr_i]
            b = values_opt[i]
            if solver == 'lstsq':
                coef = np.linalg.lstsq(A.T, b, rcond=tol)[0]     # SVD method
            elif solver == 'nnls':
                coef = sp.optimize.nnls(A.T, b)[0]
            elif solver == 'lsq':   
                coef = sp.optimize.lsq_linear(A.T, b, bounds=(0, 1),
                                              method='trf',
                                              lsq_solver='lsmr')['x']
            else:
                raise ValueError("`solver` not recognized")
        else:
            msg = "No frames left in the reference set. Try increasing "
            msg += "`dist_threshold` or decreasing `delta_rot`."
            raise RuntimeError(msg)


        if formod==True:
            ind_ref_list.append(ind_fr_i)
            coef_list.append(coef)       
            
        recon = np.dot(coef, values[ind_fr_i])
        matrix_res[i] = values[i] - recon
    
    if formod==True:
        return matrix_res,ind_ref_list,coef_list, yy, xx,
    else:
        return matrix_res, yy,xx
    
      
def llsg_adisdi(cube, angle_list,scale_list, fwhm, rank=10, thresh=1, max_iter=10,
         low_rank_ref=False, low_rank_mode='svd', auto_rank_mode='noise',
         residuals_tol=1e-1, cevr=0.9, thresh_mode='soft', nproc=1,
         asize=None, n_segments=4, azimuth_overlap=None, radius_int=None,radius_out=None,
         random_seed=None, imlib='opencv', interpolation='lanczos4',
         high_pass=None, collapse='median', full_output=True, verbose=True,
         debug=False):
    
    """ Local low rank plus Gaussian PSF subtraction for ADI or ADI+SDI. This 
    code is an adaptation of the VIP llsg function.
    """
    
    cube_rot_scale,angle_list_t,scale_list_t=rot_scale('ini',cube,None,angle_list,scale_list,imlib, interpolation)
    
    list_l, list_s, list_g, f_l, frame_fin, f_g = llsg(cube_rot_scale, angle_list_t, fwhm, rank=rank,asize=asize,radius_int=radius_int,radius_out=radius_out,thresh=1,n_segments=n_segments, max_iter=40, random_seed=10, nproc=nproc,full_output=True,verbose=False)
    res_s=np.array(list_s)
    residuals_cube_=cube_derotate(res_s[0],-angle_list_t)
    cube_der=rot_scale('fin',cube,residuals_cube_,angle_list_t,scale_list_t, imlib, interpolation)
    frame_fin=cube_collapse(cube_der, collapse)
    return cube_der,frame_fin




def llsg(cube, angle_list, fwhm, rank=10, thresh=1, max_iter=10,
         low_rank_ref=False, low_rank_mode='svd', auto_rank_mode='noise',
         residuals_tol=1e-1, cevr=0.9, thresh_mode='soft', nproc=1,
         asize=None, n_segments=4, azimuth_overlap=None, radius_int=None, radius_out=None,
         random_seed=None, imlib='opencv', interpolation='lanczos4',
         high_pass=None, collapse='median', full_output=False, verbose=True,
         debug=False):
    """ Local Low-rank plus Sparse plus Gaussian-noise decomposition (LLSG) as
    described in Gomez Gonzalez et al. 2016. This first version of our algorithm
    aims at decomposing ADI cubes into three terms L+S+G (low-rank, sparse and
    Gaussian noise). Separating the noise from the S component (where the moving
    planet should stay) allow us to increase the SNR of potential planets.
    The three tunable parameters are the *rank* or expected rank of the L
    component, the ``thresh`` or threshold for encouraging sparsity in the S
    component and ``max_iter`` which sets the number of iterations. The rest of
    parameters can be tuned at the users own risk (do it if you know what you're
    doing).
    Parameters
    ----------
    cube : numpy ndarray, 3d
        Input ADI cube.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    fwhm : float
        Known size of the FHWM in pixels to be used.
    rank : int, optional
        Expected rank of the L component.
    thresh : float, optional
        Factor that scales the thresholding step in the algorithm.
    max_iter : int, optional
        Sets the number of iterations.
    low_rank_ref :
        If True the first estimation of the L component is obtained from the
        remaining segments in the same annulus.
    low_rank_mode : {'svd', 'brp'}, optional
        Sets the method of solving the L update.
    auto_rank_mode : {'noise', 'cevr'}, str optional
        If ``rank`` is None, then ``auto_rank_mode`` sets the way that the
        ``rank`` is determined: the noise minimization or the cumulative
        explained variance ratio (when 'svd' is used).
    residuals_tol : float, optional
        The value of the noise decay to be used when ``rank`` is None and
        ``auto_rank_mode`` is set to ``noise``.
    cevr : float, optional
        Float value in the range [0,1] for selecting the cumulative explained
        variance ratio to choose the rank automatically (if ``rank`` is None).
    thresh_mode : {'soft', 'hard'}, optional
        Sets the type of thresholding.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to cpu_count()/2. By default the algorithm works
        in single-process mode.
    asize : int or None, optional
        If ``asize`` is None then each annulus will have a width of ``2*asize``.
        If an integer then it is the width in pixels of each annulus.
    n_segments : int or list of ints, optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli.
    azimuth_overlap : int or None, optional
        Sets the amount of azimuthal averaging.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    random_seed : int or None, optional
        Controls the seed for the Pseudo Random Number generator.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    high_pass : odd int or None, optional
        If set to an odd integer <=7, a high-pass filter is applied to the
        frames. The ``vip_hci.var.frame_filter_highpass`` is applied twice,
        first with the mode ``median-subt`` and a large window, and then with
        ``laplacian-conv`` and a kernel size equal to ``high_pass``. 5 is an
        optimal value when ``fwhm`` is ~4.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints to stdout intermediate info.
    debug : bool, optional
        Whether to output some intermediate information.
    Returns
    -------
    frame_s : numpy ndarray, 2d
        Final frame (from the S component) after rotation and median-combination.
    If ``full_output`` is True, the following intermediate arrays are returned:
    list_l_array_der, list_s_array_der, list_g_array_der, frame_l, frame_s,
    frame_g
    """
    if cube.ndim != 3:
        raise TypeError("Input array is not a cube (3d array)")
    if not cube.shape[0] == angle_list.shape[0]:
        msg = "Angle list vector has wrong length. It must equal the number"
        msg += " frames in the cube"
        raise TypeError(msg)

    if low_rank_mode == 'brp':
        if rank is None:
            msg = "Auto rank only works with SVD low_rank_mode."
            msg += " Set a value for the rank parameter"
            raise ValueError(msg)
        if low_rank_ref:
            msg = "Low_rank_ref only works with SVD low_rank_mode"
            raise ValueError(msg)

    if high_pass is not None:
        cube_init = cube_filter_highpass(cube, 'median-subt', median_size=19,
                                         verbose=False)
        cube_init = cube_filter_highpass(cube_init, 'laplacian-conv',
                                         kernel_size=high_pass, verbose=False)
    else:
        cube_init = cube

    n, y, x = cube.shape

    if azimuth_overlap == 0:
        azimuth_overlap = None

    if radius_int is None:
        radius_int = 0

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    # Same number of pixels per annulus
    if asize is None:
        annulus_width = int(np.ceil(2 * fwhm))  # as in the paper
    elif isinstance(asize, int):
        annulus_width = asize
        
    if radius_int%asize>int(asize/2):
        radius_int=(radius_int//asize)*asize
    elif radius_int>=asize:
        radius_int=(radius_int//asize-1)*asize
    else:
        radius_int=0
        
    if radius_out is not None:
        
        if radius_out%asize>int(asize/2):
            radius_out=(radius_out//asize+2)*asize
        else:
            radius_out=(radius_out//asize+1)*asize
        
        if radius_out>int(y / 2):
            n_annuli = int((y / 2 - radius_int) / asize)
        else:
            n_annuli = int((radius_out - radius_int) / asize)
        
    else :
        
        n_annuli = int((y / 2 - radius_int) / asize)

        

    if n_segments is None:
        n_segments = [4 for _ in range(n_annuli)]   # as in the paper
    elif isinstance(n_segments, int):
        n_segments = [n_segments]*n_annuli
    elif n_segments == 'auto':
        n_segments = []
        n_segments.append(2)    # for first annulus
        n_segments.append(3)    # for second annulus
        ld = 2 * np.tan(360/4/2) * annulus_width
        for i in range(2, n_annuli):    # rest of annuli
            radius = i * annulus_width
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360/ang)))

    if verbose:
        print('Annuli = {}'.format(n_annuli))

    # Azimuthal averaging of residuals
    if azimuth_overlap is None:
        azimuth_overlap = 360   # no overlapping, single config of segments
    n_rots = int(360 / azimuth_overlap)

    matrix_s = np.zeros((n_rots, n, y, x))
    if full_output:
        matrix_l = np.zeros((n_rots, n, y, x))
        matrix_g = np.zeros((n_rots, n, y, x))

    # Looping the he annuli
    if verbose:
        print('Processing annulus: ')
    for ann in range(n_annuli):
        inner_radius = radius_int + ann * annulus_width
        n_segments_ann = n_segments[ann]
        if verbose:
            print('{} : in_rad={}, n_segm={}'.format(ann+1, inner_radius,
                                                     n_segments_ann))

        
        for i in range(n_rots):
            theta_init = i * azimuth_overlap
            indices = get_annulus_segments(cube[0], inner_radius,
                                           annulus_width, n_segments_ann,
                                           theta_init)

            patches = pool_map(nproc, _decompose_patch, indices,
                               iterable(range(n_segments_ann)),cube_init, n_segments_ann,
                               rank, low_rank_ref, low_rank_mode, thresh,
                               thresh_mode, max_iter, auto_rank_mode, cevr,
                               residuals_tol, random_seed, debug, full_output)

            for j in range(n_segments_ann):
                yy = indices[j][0]
                xx = indices[j][1]

                if full_output:
                    matrix_l[i, :, yy, xx] = patches[j][0]
                    matrix_s[i, :, yy, xx] = patches[j][1]
                    matrix_g[i, :, yy, xx] = patches[j][2]
                else:
                    matrix_s[i, :, yy, xx] = patches[j]

    if full_output:
        list_s_array_der = [cube_derotate(matrix_s[k], angle_list, imlib=imlib,
                                          interpolation=interpolation)
                            for k in range(n_rots)]
        list_frame_s = [cube_collapse(list_s_array_der[k], mode=collapse)
                        for k in range(n_rots)]
        frame_s = cube_collapse(np.array(list_frame_s), mode=collapse)

        list_l_array_der = [cube_derotate(matrix_l[k], angle_list, imlib=imlib,
                                          interpolation=interpolation)
                            for k in range(n_rots)]
        list_frame_l = [cube_collapse(list_l_array_der[k], mode=collapse)
                        for k in range(n_rots)]
        frame_l = cube_collapse(np.array(list_frame_l), mode=collapse)

        list_g_array_der = [cube_derotate(matrix_g[k], angle_list, imlib=imlib,
                                          interpolation=interpolation)
                            for k in range(n_rots)]
        list_frame_g = [cube_collapse(list_g_array_der[k], mode=collapse)
                        for k in range(n_rots)]
        frame_g = cube_collapse(np.array(list_frame_g), mode=collapse)

    else:
        list_s_array_der = [cube_derotate(matrix_s[k], angle_list, imlib=imlib,
                                          interpolation=interpolation)
                            for k in range(n_rots)]
        list_frame_s = [cube_collapse(list_s_array_der[k], mode=collapse)
                        for k in range(n_rots)]

        frame_s = cube_collapse(np.array(list_frame_s), mode=collapse)


    if full_output:
        return(list_l_array_der, list_s_array_der, list_g_array_der,
               frame_l, frame_s, frame_g)
    else:
        return frame_s
    

def _decompose_patch(indices, i_patch, cube_init, n_segments_ann, rank, low_rank_ref,
                     low_rank_mode, thresh, thresh_mode, max_iter,
                     auto_rank_mode, cevr, residuals_tol, random_seed,
                     debug=False, full_output=False):
    """ Patch decomposition.
    """
    j = i_patch
    yy = indices[j][0]
    xx = indices[j][1]
    data_segm = cube_init[:, yy, xx]

    if low_rank_ref:
        ref_segments = list(range(n_segments_ann))
        ref_segments.pop(j)
        for m, n in enumerate(ref_segments):
            if m == 0:
                yy_ref = indices[n][0]
                xx_ref = indices[n][1]
            else:
                yy_ref = np.hstack((yy_ref, indices[n][0]))
                xx_ref = np.hstack((xx_ref, indices[n][1]))
        data_ref = cube_init[:, yy_ref, xx_ref]
    else:
        data_ref = data_segm

    patch = _patch_rlrps(data_segm, data_ref, rank, low_rank_ref,
                         low_rank_mode, thresh, thresh_mode,
                         max_iter, auto_rank_mode, cevr,
                         residuals_tol, random_seed, debug=debug,
                         full_output=full_output)
    return patch


def Hessian(x, f, step=0.05):
    
    def matrix_indices(n):
        for i in range(n):
            for j in range(i, n):
                yield i, j
    
    n = len(x)

    h = np.empty(n)
    h.fill(step)

    ee = np.diag(h)

    f0 = f(x)
    g = np.zeros(n)
    #print(x,f0)
    
    new_opti =True
    while new_opti ==True:
        new_opti =False
        for i in range(n):
            g[i] = f(x+ee[i, :])
            #print(x + ee[i, :],g[i])
            if g[i]<f0:
                f0=g[i]
                x+= ee[i, :]
                new_opti ==True 
                break

    hess = np.outer(h, h)
    new_opti =True
    while new_opti ==True:
        new_opti =False
        for i, j in matrix_indices(n):
            f_esti=f(x + ee[i, :] + ee[j, :])
            if f_esti>=f0:
            
                hess[i, j] = ( f_esti -
                          g[i] - g[j] + f0)/hess[i, j]
                hess[j, i] = hess[i, j]
            else:
                f0=f_esti
                x+= ee[i, :] + ee[j, :]
                new_opti =True
                break
            
        return x,hess
    
import os
from vip_hci.metrics import contrast_curve
os.chdir('/...')

#Load datset

pxscale = vip.config.VLT_SPHERE_IRDIS['plsc']

psf = './psf_cube.fits'
angle = './parallactic_angles.fits'
cube = './image_cube.fits'

angs = vip.fits.open_fits(angle)
cube = vip.fits.open_fits(cube)
psf = vip.fits.open_fits(psf)     
  
# Parameters 
scal_fact=1 #scaling factor between off-axis PSF image and science image
min_radius=10 
max_radius=90
ncore=10 #number of cores used for multiprocessing
an_dist=[10,20,30,40,50,60,70,80] #annular distance considered for the contrast matrix computation

ceny, cenx = frame_center(cube[0])

fit = vip.var.fit_2dgaussian(psf, crop=True, cropsize=9, debug=False)
fwhm=float((fit.fwhm_y+fit.fwhm_x)/2)
psfn = vip.preproc.frame_crop(psf, 19)   
psfn = vip.fm.normalize_psf(psfn, float((fit.fwhm_y+fit.fwhm_x)/2))
psfn =  vip.preproc.frame_crop(psfn,11)
psf_oa =  vip.preproc.frame_crop(psf,11)*scal_fact

indices = get_annulus_segments(psfn,0,round((fwhm)/2),1) 
indices_oa = get_annulus_segments(psf_oa,0,round((fwhm)/2),1) 
scaling_factor=psf_oa[indices_oa[0][0], indices_oa[0][1]].sum()/psfn[indices[0][0], indices[0][1]].sum()

# Parameters optimization and computation of RSM detection map
       
f=PyRSM(fwhm,minradius=min_radius,maxradius=max_radius,pxscale=pxscale,ncore=ncore,inv_ang=True,trunc=10)

f.add_cube(psfn,cube, angs)

f.add_method('APCA',interval=[5],crop_size=3,crop_range=1,asize=5,opti_bound=[[5,15],[1,4],[0.25,0.75]])
f.add_method('APCA',interval=[5],crop_size=3,crop_range=1,asize=5,opti_bound=[[16,25],[1,4],[0.25,0.75]])
f.add_method('NMF',interval=[5],crop_size=3,crop_range=1,asize=5,opti_bound=[[5,10]])
f.add_method('NMF',interval=[5],crop_size=3,crop_range=1,asize=5,opti_bound=[[10,20]])
f.add_method('LLSG',interval=[5],crop_size=3,crop_range=1,asize=5,opti_bound=[[1,4],[1,4]])
f.add_method('LLSG',interval=[5],crop_size=3,crop_range=1,asize=5,opti_bound=[[4,8],[1,4]])
f.add_method('LOCI',interval=[5],crop_size=3,crop_range=1,asize=5,opti_bound=[[1e-3,1e-2],[0.25,0.75]])


f.opti_model(optimisation_model='PSO', param_optimisation={'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10,'opti_iter':10,'ini_esti':50, 'random_search':100}) 
f.save_parameters('/.../','Opti_param')

f.opti_RSM(estimator='Forward',colmode='median')
f.save_parameters('/.../','Opti_param')

f.opti_combination(estimator='Forward',colmode='median',threshold=True,contrast_sel='Max',combination='Bottom-Up') 
f.save_parameters('/.../','Opti_param')

vip.fits.write_fits('Opti_map_invang', f.probmap, header=None, precision=np.float32,verbose=True)

f.opti_map(estimator='Forward',colmode='median',threshold=False)

vip.fits.write_fits('Opti_map', f.final_map, header=None, precision=np.float32,verbose=True)

opti_map=np.copy(f.final_map)

from skimage.feature import peak_local_max

#Detection of potential planetary signals (signal above a probability threshold of 0.05)

coords=peak_local_max(f.final_map, threshold_abs=0.05,min_distance=int(np.ceil(fwhm)))
pscy=coords[:,0]
pscx=coords[:,1]


flux=np.zeros_like(pscy)
centroid_y=np.zeros_like(pscy)
centroid_x=np.zeros_like(pscy)
rad_dist=np.zeros_like(pscy)
theta=np.zeros_like(pscy)
best_cost=np.zeros_like(pscy)
err_phot=np.zeros_like(pscy)
erry=np.zeros_like(pscy)
errx=np.zeros_like(pscy)
err_phot95=np.zeros_like(pscy)
erry95=np.zeros_like(pscy)
errx95=np.zeros_like(pscy)

cube_empty=cube

# Characterization of the detected signals

if len(pscy)>4:
    print('Too many detections, detection map analysis required. Consider using the Full=True option to generate the final detection map')
else:
    if len(pscy)>0:
        for k in range(len(pscy)):
            
            
            indices = get_annulus_segments(psfn,0,round((fwhm)/2),1) 
            indices_oa = get_annulus_segments(psf_oa,0,round((fwhm)/2),1) 
            scaling_factor=psf_oa[indices_oa[0][0], indices_oa[0][1]].sum()/psfn[indices[0][0], indices[0][1]].sum()
            indc = disk(((pscy[k],pscx[k])),5)
            posy_fc, posx_fc=np.where(opti_map==np.max(opti_map[indc[0],indc[1]]))
            
            #Definition of the initial photometry via the computation of a S/N based contrast curve
            
            ini_cc = contrast_curve(cube, angs, psfn, fwhm, pxscale,
                            scaling_factor, vip.psfsub.pca, sigma=3, nbranch=1, theta=0,
                            inner_rad=1, wedge=(0, 360), fc_snr=100,
                            plot=False, **{'imlib':'opencv','ncomp':int(f.ncomp[0][12,0])})
            
            sep=np.sqrt((posy_fc[0]-ceny)**2+(posx_fc[0]-cenx)**2)
            ini_flux=np.log10(ini_cc['sensitivity_student'][int(sep)])
    
            # Characterization of the detected signal via target_charact function of the PyRSM framework
            
            indc = disk(((posy_fc[0],posx_fc[0])),int(fwhm))
            if np.max(opti_map[indc[0],indc[1]])>0.9:
                flux[k],centroid_y[k],centroid_x[k],err_phot[k],erry[k],errx[k],err_phot95[k],erry95[k],errx95[k],rad_dist[k],theta[k],best_cost[k]=f.target_charact([posy_fc[0],posx_fc[0]],psf_oa=[psf_oa], ns=1, loss_func='value',param_optimisation={'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10,'opti_iter':7,'ini_esti':50},photo_bound=[np.power(10,ini_flux-1),np.power(10,ini_flux+1),10],ci_esti=True,first_guess=True)
            else:
                flux[k],centroid_y[k],centroid_x[k],err_phot[k],erry[k],errx[k],err_phot95[k],erry95[k],errx95[k],rad_dist[k],theta[k],best_cost[k]=f.target_charact([posy_fc[0],posx_fc[0]],psf_oa=[psf_oa], ns=1, loss_func='value',param_optimisation={'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10,'opti_iter':7,'ini_esti':50},photo_bound=[np.power(10,ini_flux-1),np.power(10,ini_flux+1),10],ci_esti=True,first_guess=False)
    
            cube_empty = cube_inject_companions(cube, psf, angs, flevel=-flux[k]*scaling_factor, plsc=pxscale, 
                                    rad_dists=rad_dist/(1000*pxscale), theta=theta, n_branches=1,verbose=True)

        vip.fits.write_fits('Targets_characterization', np.array([flux,centroid_y,centroid_x,err_phot,erry,errx,err_phot95,erry95,errx95,rad_dist,theta,best_cost]), header=None, precision=np.float32,
                       verbose=True)  
       
    #Computation of the completeness matrix
    
    #Initialization of the contrast using the completeness_curve function based on S/N ratio map
    
    an_dist, comp_curve = vip.metrics.completeness_curve(cube_empty, angs, psfn, fwhm, vip.psfsub.pca_annular, an_dist=an_dist,
                                                         starphot=scaling_factor,nproc=f.ncore,completeness=0.5,n_fc=10,
                                                         algo_dict={'n_segments':f.nsegments[1][12,0], 'delta_rot':f.delta_rot[1][12,0],'ncomp':f.ncomp[0][12,0],'imlib':'opencv'})
    
    # Computation of the completeness matrix using the PyRSM framework
    
    f=PyRSM(fwhm,minradius=min_radius,maxradius=max_radius,pxscale=pxscale,ncore=ncore,inv_ang=True,trunc=10)

    f.add_cube(psfn,cube_empty, angs)
    
    f.add_method('APCA')
    f.add_method('APCA')
    f.add_method('NMF')
    f.add_method('NMF')
    f.add_method('LLSG')
    f.add_method('LLSG')
    f.add_method('LOCI')
    
    f.load_parameters('/.../'+'Opti_param')

    f.opti_map(estimator='Forward',colmode='median',threshold=False)
    
    contrast_mat=f.contrast_matrix(an_dist,comp_curve,probmap=None,psf_oa=[psf_oa],n_fc=20)

    vip.fits.write_fits('Contrast_matrix', np.array(contrast_mat), header=None, precision=np.float32,
                       verbose=True)