# -*- coding: utf-8 -*-
"""
Author:         Jacob Søgaard Larsen <jasla@dtu.dk>

Last revision:  18-01-2019

"""
import numpy as np
from scipy.io import loadmat

def IDRC_2002(filename="data/nir_shootout_2002.mat",NIR_range = None):
    """
    Importing the IDRC 2002 Shootout data. The function assumes a .mat file as the one downloaded from 
    http://eigenvector.com/data/tablets/nir_shootout_2002.mat
    
    Input:
        filename:       The path and filename to the data.
                        The default is "data/nir_shootout_2002.mat"
                        
        NIR_range:      Minimum and maximum wavelength desired, e.g. NIR_range = [0,1750].
                        The data contain wavelength in the range 600nm to 1898nm in steps of 2nm.
                        If all wavelength are desired, then pass None (default).
                        
    Output:
        Dictionary with fields:
                    "Training":     Fields holding the training data. We have combined the original calibration and test data into one. This is done in order
                                    to get more data such that both labelled and unlabelled data is available for training. The field holds a dictionary with
                                    the fields "Instrument 1", "Instrument 2" and "References", holding respectively the measurements from Instrument 1 and
                                    Instrument 2 where the rows correspond to the same tablet. The References field hold the weight percentage of active ingredient (API).
                    
                    "Validation":   The original validation data. The field holds a dictionary with the same fields as "Training".
                    
                    "Wavelength":   The wavelength corresponding to selected region.
    """
    
    dat = loadmat(filename,mat_dtype=False)
    wavelength = dat["calibrate_1"][0,0][7][1,0].flatten()
    
    if isinstance(NIR_range,type(None)):
        idxNIR = np.where(wavelength)[0]
    else:
        idxNIR = (wavelength > min(NIR_range)) * (wavelength < max(NIR_range))
        wavelength = wavelength[idxNIR]
    
    XCal1 = dat["calibrate_1"][0,0][5][:,idxNIR]
    XCal2 = dat["calibrate_2"][0,0][5][:,idxNIR]
    YCal = dat["calibrate_Y"][0,0][5]
    YCal = YCal[:,2]/YCal[:,0] * 100
        
    XVal1 = np.array(dat["validate_1"][0,0][5][:,idxNIR],dtype=np.float64)
    XVal2 = np.array(dat["validate_2"][0,0][5][:,idxNIR],dtype=np.float64)
    YVal = dat["validate_Y"][0,0][5]
    YVal = YVal[:,2]/YVal[:,0] * 100
        
    XTest1 = dat["test_1"][0,0][5][:,idxNIR]
    XTest2 = dat["test_2"][0,0][5][:,idxNIR]
    YTest = dat["test_Y"][0,0][5]
    YTest = YTest[:,2]/YTest[:,0] * 100
        
    XTrain1 = np.array(np.vstack((XCal1,XTest1)),dtype=np.float64)
    XTrain2 = np.array(np.vstack((XCal2,XTest2)),dtype=np.float64)
    YTrain = np.hstack((YCal,YTest))
        
    out = {"Training":{"Instrument 1":XTrain1, "Instrument 2":XTrain2, "References": YTrain},
           "Validation":{"Instrument 1":XVal1, "Instrument 2":XVal2, "References": YVal},
           "Wavelength":wavelength}
    
    return out


def get_simulated_data(driftType=0, NL=40, NU=200, NTest=100, beta=1, seed=None):
    """
    The simulated dataset consist of three overlapping independent bell shaped signals S[0], S[1] and S[2].
    
    Input:
        NL      :       The number of labelled examples from the source domain.
        NU      :       The number of unlabelled examples from the target domain.
        NTest   :       The number of labelled examples from the target domain.
    
        driftType:  0 (default):    The source data consist of two independent signals (S[0] and S[1]), while the target and unlabelled data consist of three independent signals. 
                                    The size of the extra signal is parameterized by beta.
                    
                    1:              The source and target data both consist three independent signals. In the target data, a level shift of size beta has occurred
                                    in the signal of interest parameterized by beta.
                    
                    2:              The source and target data both consist three independent signals. In the target data, a level shift of size beta has occurred
                                    in S[2] parameterized by beta.
    
    Output:
        Dictionary with fields
                "Training":         Dictionary containing the fields "Spectra", "Noisy References" and "True References", 
                                    holding respectively the spectral measurements, the noisy references and the noise free references.
                                    
                "Unlabelled":       Dictionary containing the field "Spectra" with the spectral measurements.
                
                "Test Source":      Dictionary containing the fields "Spectra", "Noisy References" and "True References", 
                                    holding respectively the spectral measurements, the noisy references and the noise free references.
                                    
                "Test Target":      Dictionary containing the fields "Spectra", "Noisy References" and "True References", 
                                    holding respectively the spectral measurements, the noisy references and the noise free references.
                                    
                "Wavelength":       Field holding the wavelength corresponding to the spectral measurements.
                
                "S":                The true observed spectral signals.
    
    
    """
    if not seed is None:
        np.random.seed(seed)
        
    nW = 100 # Number of wavelengths
    x = np.linspace(1,100,nW)
    f = [40,50,60]
    psi = [0.05]*3
    zMin = [3,3,3]
    zMax = [6,6,6]
    stdX = 0.1
    stdY = 0.1**(1/2)
    
    S = np.zeros((x.shape[0],3)) # True signals
    for i in range(3):
        S[:,i] = f[i] * x / ( (f[i]**2 - x**2)**2 + (2 * psi[i] * f[i] * x)**2 )**(1/2)
    
    
    if driftType == 0:
        zTrain = np.random.uniform(low = zMin, high = zMax,size = (NL,3)); zTrain[:,-1] = zMin[-1]
        zU = np.random.uniform(low = zMin, high = zMax[:2] + [zMin[2]+beta],size = (NU,3));
        zTest1 = np.random.uniform(low=zMin, high=zMax, size=(NTest,3) ); zTest1[:,-1] = zMin[-1]
        zTest2 = np.random.uniform(low=zMin, high=zMax[:2] + [zMin[2] + beta],size = (NTest,3));
        
    elif driftType == 1:
        zTrain = np.random.uniform(low = zMin, high = zMax,size = (NL,3))
        zU = np.random.uniform(low = zMin, high = zMax,size = (NU,3)); zU[:,1] += beta
        zTest1 = np.random.uniform(low = zMin, high = zMax,size = (NTest,3))
        zTest2 = np.random.uniform(low = zMin, high = zMax,size = (NTest,3)); zTest2[:,1] += beta
        
    elif driftType == 2:
        zTrain = np.random.uniform(low = zMin, high = zMax,size = (NL,3))
        zU = np.random.uniform(low = zMin, high = zMax,size = (NU,3)); zU[:,2] += beta
        zTest1 = np.random.uniform(low = zMin, high = zMax,size = (NTest,3))
        zTest2 = np.random.uniform(low = zMin, high = zMax,size = (NTest,3)); zTest2[:,2] += beta
    else:
        raise NotImplementedError
    
    # Generate noise to the spectra
    vTrain = np.random.normal(loc=0, scale=stdX, size=(NL,nW) )
    vU = np.random.normal(loc=0, scale=stdX, size=(NU,nW) )
    vTest1 = np.random.normal(loc=0, scale=stdX, size=(NTest,nW) )
    vTest2 = np.random.normal(loc=0, scale=stdX, size=(NTest,nW) )

    # Add noise to reference values
    yTrain = zTrain[:,1] + np.random.normal(scale=stdY, size=(NL,) )
    yTest1 = zTest1[:,1] + np.random.normal(scale=stdY, size=(NTest,) )
    yTest2 = zTest2[:,1] + np.random.normal(scale=stdY, size=(NTest,) )
    
    # Add noise to spectra
    XTrain = np.matmul(zTrain,S.T) + vTrain
    XU = np.matmul(zU,S.T) + vU
    XTest1 = np.matmul(zTest1,S.T) + vTest1
    XTest2 = np.matmul(zTest2,S.T) + vTest2
    
    out = {"Training":  {"Spectra": XTrain, "Noisy References": yTrain, "True References": zTrain[:,1]},
           "Unlabelled": {"Spectra": XU},
           "Test Source": {"Spectra": XTest1, "Noisy References": yTest1, "True References": zTest1[:,1]},
           "Test Target": {"Spectra": XTest2, "Noisy References": yTest2, "True References": zTest2[:,1]},
           "Wavelength": x, "S": S}
    
    return out
