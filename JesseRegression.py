# -*- coding: utf-8 -*-
"""
@author: jesse
Notes:
    This will be my standard regression package from now on.

    The average math joke is mean.

History:
    Created on Thu Oct 20 09:16:33 2016

"""
from scipy import stats
import numpy as np

def OLS(X, Y):
    '''  ORDINARY LEAST SQUARES
    Assumes Y may have error, while X does not
    uses scipy.stats.linregress
    returns slope, intercept, r, p, sterr    
    '''
    return stats.linregress(X,Y)
    
def RMA(Xin, Yin, alpha = 0.05):
    ''' REDUCED MAJOR AXIS (or GEOMETRIC MEAN REGRESSION)
    
    converted from http://au.mathworks.com/matlabcentral/fileexchange/27918-gmregress/content//gmregress.m
    
    This is a Model II procedure. It standardize variables before the 
    slope is computed. Each of the two variables are transformed to have a 
    mean of zero and a standard deviation of one. The resulting slope is
    the geometric mean of the linear regression coefficient of Y on X. 
    Ricker (1973) coined this term and gives an extensive review of Model
    II regression. It is also known as Standard Major Axis.
    
    Inputs should be numpy arrays of matching size
    Returns: v, u, r, CIr, CIjm
        v = RMA slope
        u = RMA intercept
        CIr = CI for slope, intercept (Ricker method)
        CIjm= CI for slope, intercept (Jolicoeur-Mosimann method)
    '''
    # remove all nan entries
    ynan=np.isnan(Yin)    
    xnan=np.isnan(Xin)
    nans=xnan+ynan
    Y,X=Yin[~nans],Xin[~nans]
    
    n=len(Y)
    
    r=np.corrcoef(X,Y)[0,1]     # correlation coeff
    s=r/abs(r)                  # sign of r (+ or - 1)
    S=np.cov(X,Y)               # covariance array
    SCX=S[0,0]; SCY=S[1,1]; SCP=S[0,1]
    v=s*np.sqrt(SCY/SCX)        # slope
    u=np.mean(Y)-np.mean(X)*v;  # intercept
    
    # Ricker procedure for confidence interval:
    SCv = SCY-(SCP**2)/SCX
    N   = SCv/(n-2)
    sv  = np.sqrt(N/SCX)
    t   = stats.t.ppf(1-alpha/2.0, n-2) # critical t value
    vi  = v-t*sv                # CI lower limit of slope
    vs  = v+t*sv                # CI upper limit of slope
    ui  = np.mean(Y)-np.mean(X)*vs  # lower CI limit of intercept
    us  = np.mean(Y)-np.mean(X)*vi  # upper CI limit of intercept
    
    if ui>us:
        tmp=ui; ui=us; us=tmp
    if vi>vs:
        tmp=vi; vi=vs; vs=tmp
    
    CIr=[[vi,vs],[ui,us]] # CI limits for slope, intercept
    
    # Jolicoeur and Mosimann procedure for confidence interval
    F   = stats.f.ppf(1-alpha, 1, n-2) # critical F statistic
    B   = F*(1-r**2)/(n-2.0)
    a   = np.sqrt(B+1)
    c   = np.sqrt(B)
    qi = v*(a-c)                    # confidence lower limit of slope
    qs = v*(a+c)                    # confidence upper limit of slope
    pi = np.mean(Y)-np.mean(X)*qs   # confidence lower limit of intercept
    ps = np.mean(Y)-np.mean(X)*qi   # confidence upper limit of intercept
    if pi > ps:
        tmp=pi; pi=ps; ps=tmp
    if qi > qs:
        tmp=qi; qi=qs; qs=tmp
        
    CIjm = [[qi,qs],[pi,ps]]
    
    return v, u, r, CIr, CIjm
    
# If you run the script directly you will see this example
if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    Y=np.array([61,37,65,69,54,93,87,89,100,90,97],float)
    X=np.array([14,17,24,25,27,33,34,37,40,41,42],float)
    m1,b1,r1,ci1,ci2=RMA(X,Y)
    m2,b2,r2,p,sterr=OLS(X,Y)
    x=np.array([min(X),max(X)])
    plt.scatter(X,Y)
    plt.plot(x,m1*x+b1,'k',linewidth=2,label='Reduced Major Axis')
    plt.plot(x,m2*x+b2,'fuchsia',linewidth=2,label='Ordinary Least Squares')
    plt.legend(loc='best')
    plt.title('Random example of regression')
    plt.show()
    
