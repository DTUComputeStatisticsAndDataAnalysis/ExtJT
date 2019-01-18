# -*- coding: utf-8 -*-
"""
Author:         Jacob Søgaard Larsen <jasla@dtu.dk>

Last revision:  18-01-2019

"""
import numpy as np
from sklearn import preprocessing as pre
from sklearn.decomposition import PCA
from numpy.random import permutation

#%%
class ExtJTPLS():
    """
    Class implementing the Extended Linear Joint Trained Framework using a Probabilistic PCA structure to model the scatter matrix.
    
    """
    def __init__(self, gamma1, gamma2, gamma3, ncomp=2, scale=False):
        """
        Input:
            gamma1:                         Non-negative float controlling the amount of regularization with respect to the covariance structure.
            
            gamma2:                         Non-negative float constrolling the parameterization of the singular values.
            
            gamma3                          Non-negative float controlling the amount of regularization with respect to the difference between 
                                            the mean of the labelled and unlabelle data.
            
            ncomp:                          Number of PLS components used.
            
            scale:                          If True, the data is standardized with respect to the labelled data before fitting.
                                            If False, the data is only centered with respect to the labelled data before fitting.
        """
        self.scale = scale
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self._gamma = (gamma1,gamma2,gamma3)
        self.ncomp = ncomp

    def fit(self,XL,YL,XU): 
        """
        Function for computing the partial least squares solution to the Extended Linear Joint Trained Framework.
        
        Input:
            XL:                             Matrix of size NL x p with measurements of the labelled data.
            YL:                             Vector of length NL holding the references for the corresponding rows of XL.
            XU:                             Matrix of size NU x p holding the measurements of the unlabelled data.
        
        """
        
        XL = XL.copy()
        XU = XU.copy()
        YL = YL.copy().flatten()
        
        gamma1,gamma2,gamma3 = self._gamma
        ncomp = self.ncomp
        
        nL,p = XL.shape
        nU = XU.shape[0]
        
        # Get centers of labelled data
        muL = XL.mean(axis=0)
        muY = YL.mean(axis=0)
        
        
        # Centering and scaling according to Ryan and Culp 2015
        XLc = XL-muL
        XUc = XU-muL
        YLc = YL-muY
        
        if self.scale:
            x_std_ = XLc.std(axis=0,ddof=1)
            y_std_ = YLc.std(axis=0,ddof=1)
            
            # Scaling according to Ryan and Culp, 2015
            XLc = XLc/x_std_
            XUc = XUc/x_std_
            YLc = YLc/y_std_
        else:
            x_std_ = np.ones((p))
            y_std_ = 1
        
        muU = XUc.mean(axis=0)
        mu = gamma3**(1/2)*nU/(nL+nU) * muU + muL
        XUc = XUc-muU
        
        _,Su,Vu = np.linalg.svd(XUc,full_matrices=False)
        Vu = Vu.T

        # Parapmeterization of the unlabelled data
        XUgamma2 = (Su * (Su**2+gamma2)**(-1/2) * Vu).T
        
        W = np.zeros((p,0)) # X-weights
        P = np.zeros((p,0)) # X-loadings
        Q = np.zeros((1,0)) # Y-loadings
        R = np.zeros((p,0)) # X-rotations = W (P.T W)^(-1)
        TT = np.zeros((1,0)) # Sum of squared x-scores
        
        XY = np.matmul(XLc.T,YLc)
        
        # Main loop
        # Based on the improved kernel algorithm for PLS (Dayal, MacGregor, 1997)
        for i in range(ncomp):
            w = XY
            w = w/((w**2).sum(axis = 0)**(1/2))
            
            r = w
            for j in range(i):
                r = r - (np.dot(P[:,j].T,w)*R[:,j])#.reshape(-1,1)
            
            rTXX = np.matmul(np.matmul(r.T,XLc.T),XLc) + gamma1*gamma2*np.matmul(np.matmul(r.T,XUgamma2.T),XUgamma2) + \
                    gamma3*nU*nL/(nU+nL) * np.dot(r.T,muU)*muU
            
            tt = np.matmul(rTXX,r)
            p =  rTXX.T/tt
            q = np.matmul(r.T,XY).T/tt
            
            XY = XY - p*tt*q
            
            W = np.concatenate((W,w.reshape(-1,1)),axis=1)
            P = np.concatenate((P,p.reshape(-1,1)),axis=1)
            Q = np.concatenate((Q,q.reshape(-1,1)),axis=1)
            R = np.concatenate((R,r.reshape(-1,1)),axis=1)
            TT = np.concatenate((TT,tt.reshape(-1,1)),axis=1)
            
        beta = np.matmul(R,Q.T)*y_std_
        self.beta = beta
        self.W = W
        self.P = P
        self.Q = Q
        self.R = R
        self.TT = TT
        self.x_mean_ = mu
        self.x_std_ = x_std_
        self.y_mean_ = muY
        self.y_std_ = y_std_
        
    def transform(self,X):
        """
        Function used for computing the PLS scores.
        
        Input:
            X:                              Matrix og size N x p
            
        Output:
            x_scores:                       Matrix of size N x ncomp with the PLS scores in the columns
        """
        X = X.copy()
        X -= self.x_mean_
        X /= self.x_std_
        # Apply rotation
        x_scores = np.dot(X, self.R)
        
        return x_scores
    
    def predict(self,X,A=None):
        """
        Function used for prediction the response from new observations.
        
        Input:
            X:                              Matrix of size N x p
            
            A:                              Integer in the interval 0 <= A <= ncomp.
            
        Output:
            ypred:                          Predictions for each row of X corresponding to A PLS components
            
            YPred:                          Predictions for each row of X with the i'th column corresponding to i PLS components
        """
        p = self.x_mean_.shape[0]
        BETA = np.zeros((p,self.ncomp+1))
        
        for i in range(1,self.ncomp+1):
            BETA[:,i] = BETA[:,i-1] + self.R[:,i-1]*self.Q[:,i-1]
        
        YPred = np.matmul((X-self.x_mean_)/self.x_std_,BETA) + self.y_mean_
    
        if not isinstance(A,type(None)):
            ypred = YPred[:,A]
        else:
            ypred = YPred[:,-1]
        
        return (ypred,YPred)


#%%
class PpcaJTPLS():
    """
    Class implementing the Extended Linear Joint Trained Framework using a Probabilistic PCA (Tipping, Bishop 1999) structure to model the scatter matrix.
    
    Scaling is not implemented and has to be done outside of function
    E.g. using sklearn.preprocessing.StandardScaler
    """

    def __init__(self,gamma1, gamma2, gamma3, ncomp=2):
        """
        Input:
            gamma1:                         Non-negative float controlling the amount of regularization with respect to the covariance structure.
            
            gamma2:                         Non-negative float constrolling the parameterization of the singular values.
            
            gamma3                          Non-negative float controlling the amount of regularization with respect to the difference between 
                                            the mean of the labelled and unlabelle data.
                                            
            ncomp:                          Number of PLS components used.
        """
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self._gamma = (gamma1,gamma2,gamma3)
        self.ncomp = ncomp

    def fit(self,XL,YL,muU,Su,Vu,sigma2,nU,scatter_matrix = True): 
        """
        Function for computing the partial least squares solution to the Extended Linear Joint Trained Framework with a PPCA model of the scatter matrix.
        
        Input:
            XL:                             Matrix of size NL x p with measurements of the labelled data.
            
            YL:                             Vector of length NL holding the references for the corresponding rows of XL.
            
            muU:                            Vector of length p holding the column mean of the unlabelled data.
            
            Su:                             Vector of length k holding the k largest singular values of either the covariance matrix or the
                                            scatter matrix of the unlabelled data (see scatter_matrix).
                                            
            Vu:                             Matrix of size p x k holding the eigenvectors corresponding to the k largest singular values of
                                            either the covariance matrix or the scatter matrix of the unlabelled data (see scatter_matrix).
                                            
            sigma2:                         Estimated noise level of the remaining p-k dimensions. Usually calculated as (Su[k:]**2).mean()
            
            nU:                             Number of unlabelled data used for estimating muU, Su, Vu and sigma2
                
            scatter_matrix:     True:       Su are calculated from the scatter matrix of the unlabelled data (e.g. np.matmul(XU.T,XU)).
                                False:      Su are calculated from the covariance matrix of the unlabelled data (e.g. np.cov(XU,rowvar=False)).
                                            In this case the covariance matrix is scaled by a factor of (nU-1).
        """
        XL = XL.copy()
        Su = Su.copy().flatten()
        Vu = Vu.copy()
        YL = YL.copy().flatten()
        muU = muU.copy().flatten()
        gamma1,gamma2,gamma3 = self._gamma
        ncomp = self.ncomp
        
        nL,m = XL.shape
        
        muL = XL.mean(axis=0)
        muY = YL.mean(axis=0)
        mu = gamma3**(1/2)*nU/(nL+nU) * (muU-muL) + muL
        
        # Centering
        XLc = XL-muL
        YLc = YL-muY
        
        # Calculate the eigenvalues of the PPCA
        LambdaU = Su**2 - sigma2
        
        # Parapmeterization of the unlabelled data
        XUgamma2 = (LambdaU**(1/2) * (LambdaU+gamma2)**(-1/2) * Vu).T
                
        x_std_ = np.ones((m))
        y_std_ = 1
        
        
        
        W = np.zeros((m,0))     # X-weights
        P = np.zeros((m,0))     # X-loadings
        Q = np.zeros((1,0))     # Y-loadings
        R = np.zeros((m,0))     # X-rotations = W (P.T W)^(-1)
        TT = np.zeros((1,0))    # Sum of squared x-scores
        
        XY = np.matmul(XLc.T,YLc)
        
        # Main loop
        # Based on the improved kernel algorithm for PLS (Dayal, MacGregor, 1997)
        for i in range(ncomp):
            w = XY
            w = w/((w**2).sum(axis = 0)**(1/2))
            
            r = w
            for j in range(i):
                r = r - (np.dot(P[:,j].T,w)*R[:,j])
                
            if scatter_matrix:
                rTXX = np.matmul(np.matmul(r.T,XLc.T),XLc) + gamma1*gamma2*np.matmul(np.matmul(r.T,XUgamma2.T),XUgamma2) + \
                        gamma1*gamma2 * r.T * sigma2/(sigma2+gamma2) + gamma3*nU*nL/(nU+nL) * np.dot(r.T,muU-muL)*(muU-muL)
            
            else:
                rTXX = np.matmul(np.matmul(r.T,XLc.T),XLc) + gamma1*gamma2*(nU-1)*np.matmul(np.matmul(r.T,XUgamma2.T),XUgamma2) + \
                        gamma1*gamma2 * (nU-1) * r.T * sigma2/(sigma2+gamma2) + gamma3*nU*nL/(nU+nL) * np.dot(r.T,muU-muL)*(muU-muL)
                        
            tt = np.matmul(rTXX,r)
            p =  rTXX.T/tt
            q = np.matmul(r.T,XY).T/tt
            
            XY = XY - p*tt*q
            
            W = np.concatenate((W,w.reshape(-1,1)),axis=1)
            P = np.concatenate((P,p.reshape(-1,1)),axis=1)
            Q = np.concatenate((Q,q.reshape(-1,1)),axis=1)
            R = np.concatenate((R,r.reshape(-1,1)),axis=1)
            TT = np.concatenate((TT,tt.reshape(-1,1)),axis=1)
            
        beta = np.matmul(R,Q.T)*y_std_
        self.beta = beta
        self.W = W
        self.P = P
        self.Q = Q
        self.R = R
        self.TT = TT
        self.x_mean_ = mu
        self.x_std_ = x_std_
        self.y_mean_ = muY
        self.y_std_ = y_std_
        self.sigma2 = sigma2
        
    def transform(self,X):
        """
        Function used for computing the PLS scores.
        
        Input:
            X:                              Matrix og size N x p
            
        Output:
            x_scores:                       Matrix of size N x ncomp with the PLS scores in the columns
        """
        X = X.copy()
            
        X -= self.x_mean_
        X /= self.x_std_
        
        # Apply rotation
        x_scores = np.dot(X, self.R)
        
        return x_scores
    
    def predict(self,X,A=None):
        """
        Function used for prediction the response from new observations.
        
        Input:
            X:                              Matrix of size N x p of observations
            
            A:                              Integer in the interval 0 <= A <= ncomp.
            
        Output:
            ypred:                          Predictions for each row of X corresponding to A PLS components
            
            YPred:                          Predictions for each row of X with the i'th column corresponding to i PLS components
        """
        X = X.copy()
            
        p = self.x_mean_.shape[0]
        BETA = np.zeros((p,self.ncomp+1))
        
        for i in range(1,self.ncomp+1):
            BETA[:,i] = BETA[:,i-1] + self.R[:,i-1]*self.Q[:,i-1]
        
        YPred = np.matmul((X-self.x_mean_)/self.x_std_,BETA) + self.y_mean_
    
        if not isinstance(A,type(None)):
            ypred = YPred[:,A]
        else:
            ypred = YPred[:,-1]
        
        return (ypred,YPred)
    
#%%
def PCANULL(x, scale = False, nperm = 99):
    """
    Function implementing Horns method for estimating the number of signification eigenvalues of the covariance or correlation matrix.
    The implementation is heavily inspired by prcompNull from the R-package sinkr version 0.6 (https://github.com/marchtaylor/sinkr)
    
    References:
        Horn 1965 - A rationale and test for the number of factors in factor analysis
        
    Input:
        x:                                  The data matrix holding the observations in the rows.
        
        scale:                              Whether to standardize the data prior to fitting a PCA.
        
        nperm:                              Number of random permutations used to estimate the distribution of the eigenvalues.
        
    Output:
        Dictionary with the fields
            Lambda:                         The eigenvalues estimated for each of the nperm permutations.
            
            Lambda_orig:                    The original eigenvalues of the centered (and possibly standardized) x.
            
            n_sig:                          Number of significant eigenvalues using the 95% quantile of Lambda as threshold.
    
    """
    # Heavily inspired by prcompNULL from R-package sinkr
    nrow, ncol = x.shape
    n_pca = min(nrow-1,ncol)
    
    if scale:
        x = pre.scale(x)
    
    pca = PCA(n_components=n_pca).fit(x)
    Lambda_orig = pca.explained_variance_
    Lambda = np.zeros(shape=(nperm,n_pca))
    idx = [i for i in range(nrow)]
    for p in range(nperm):
        xtmp = np.zeros(shape = (nrow,ncol))
        for i in range(ncol):
            idx = permutation(idx)
            xtmp[:, i] = x[idx, i]
            
        pcaTmp = PCA(n_components=n_pca).fit(xtmp)
        
        Lambda[p,:] = pcaTmp.explained_variance_
        
    quantile = np.percentile(Lambda,q = 95,axis = 0)
    n_sig = 0
    for i in range(n_pca):
        if Lambda_orig[i] > quantile[i]:
            n_sig += 1
        else:
            break
    
    result = {"Lambda": Lambda, "Lambda_orig": Lambda_orig, "n_sig": n_sig}
    return(result)
    
    
#%%
def plsPREDICT(X,pls,A=None):
    """
    Wrapper function for an object of class sklearn.cross_decomposition.PLSRegression for performing predictions with up to n_components
    
    Input:
        X:                                  Matrix og size N x p
        
        pls:                                Object of class sklearn.cross_decomposition.PLSRegression
        
        A:                                  Integer between 0 and n_components fed to pls
        
    Output:
        ypred:                          Predictions for each row of X corresponding to A PLS components
            
        YPred:                          Predictions for each row of X with the i'th column corresponding to i PLS components
    """
    
    p = pls.coef_.shape[0]
    BETA = np.zeros((p,pls.n_components+1))
    R = pls.x_rotations_
    Q = pls.y_loadings_
    
    for i in range(1,pls.n_components+1):
        BETA[:,i] = BETA[:,i-1] + R[:,i-1]*Q[:,i-1]
    
    YPred = np.matmul(np.multiply(X-pls.x_mean_,1/pls.x_std_),BETA*pls.y_std_) + pls.y_mean_
    if not isinstance(A,type(None)) and A <= pls.n_components:
        ypred = YPred[:,A]
    else:
        ypred = YPred[:,-1]
    
    return (ypred,YPred)


