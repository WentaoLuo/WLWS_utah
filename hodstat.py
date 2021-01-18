
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erf
import mpmath
pi = np.pi

# HOD framework-----------------------------------------------------------------
class HODSTAT(object):
  def __init__(self,Mrmin=None,Mrmax=None):
    self.Mrmin= Mrmin
    self.Mrmax= Mrmax
  def paramtable(self):
    """
    from Yang et al ApJ 2008 676:248-261, 2009 695:900-916 table p11
    logMh ranges from 12.16 to 14.58.
    """
    logMh  = np.array([12.16,12.45,12.75,13.05,13.35,13.64,13.94,\
                      14.23,14.58]) 
    phis   = np.array([0.67,1.12,1.87,3.27,5.72,9.60,15.27,23.68,35.51])
    sigc   = np.array([0.107,0.107,0.128,0.137,0.144,0.149,0.157,\
                      0.146,0.141])
    lgLc   = np.array([10.074,10.224,10.350,10.442,10.513,10.584,\
                      10.649,10.714,10.799])
    alpha  = np.array([-1.10,-1.07,-1.09,-1.08,-1.11,-1.20,-1.33,\
                      -1.44,-1.66])

    params = {'logMh':logMh,'phi':phis,'alpha':alpha,'lgLc':lgLc,'sigs':sigc}
    return params 
  def clumfunc(self,logMh):

    params = self.paramtable()
    Mhost  = params['logMh']
    phi0   = params['phi']
    alpha0 = params['alpha']
    lgLc0  = params['lgLc']
    sigs0  = params['sigs']
    phi    = np.interp(logMh,Mhost,phi0)
    alpha  = np.interp(logMh,Mhost,alpha0)
    lgLc   = np.interp(logMh,Mhost,lgMc0)
    sigs   = np.interp(logMh,Mhost,sigs0)
    Lmx    = np.linspace(7.8,12.0,100)

    clumfcen = (1.0/np.sqrt(2.0*pi)/sigs)*np.exp(-0.5*(Lmx-lgLc)**2.0/sigs/sigs) 
    clumfsat = phi*((10.0**Lmx/10.0**(lgLc-0.25))**(alpha+1.0))*np.exp(-((10.0**Lmx/10.0**(lgLc-0.25)))**2.0)
    return {'lgLrange':Lmx,'Central':clumfcen,'Satellite':clumfsat}

  def occupation(self,lgMh):
    lgLmin = 0.4*(4.8-self.Mrmin)
    lgLmax = 0.4*(4.8-self.Mrmax)
    lgLum  = np.linspace(lgLmin,lgMax,100)
    values = self.clumfunc(lgMh)
    Lmx    = values['lgLrange']
    cenlf  = values['Central']
    satlf  = values['Satellite']
    newcen = np,interp(lgLum,Lmx,cenlf)
    newsat = np,interp(lgLum,Lmx,satlf)
    occcen = np.sum(newcen*(lgLmax-lgLmin)/100.0)
    occsat = np.sum(newsat*(lgLmax-lgLmin)/100.0)
  return {'fCen':occcen,'fSat':occsat}
#-------------------------------------------------------------------
