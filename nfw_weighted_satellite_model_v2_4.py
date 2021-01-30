#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
#import halotools
#from halotools.empirical_models.occupation_models import ZuMandelbaum15Cens,ZuMandelbaum15Sats
import emcee
import corner
from scipy import interpolate
import sys
from colossus.lss import peaks
from colossus.lss import bias
from colossus.cosmology import cosmology
from colossus.lss import mass_function
cosmology.setCosmology('planck18')


# Basic Parameters---------------------------
h       = 0.674
w       = -1.0
omega_m = 0.315
omega_l = 0.685
omega_k = 1.0-omega_m-omega_l
rho_crt0= 2.78e11                # M_sun Mpc^-3 *h*h 
rho_bar0= rho_crt0*omega_m       # M_sun Mpc^-3 *h*h
pi      = np.pi
ns      = 0.965
alphas  = -0.04
sigma8  = 0.811
rrp,esdth= np.loadtxt('twohaloesd.dat',unpack=True)

fname,Rc1,Rc2 = np.loadtxt('r_c-range.txt',\
                dtype=np.str,usecols=(0,1,2),\
		                unpack=True)
nx  = len(fname)
ny  = 3077
rc1 = np.zeros(nx)
rc2 = np.zeros(nx)
tabs= np.zeros((nx,ny,2))
for i in range(nx):
  rc1[i] = float(Rc1[i])
  rc2[i] = float(Rc2[i])
  fsub   = 'mean/'+fname[i]
  Rps,dsig = np.loadtxt(fsub,dtype=np.str,usecols=(0,1),unpack=True)
  for j in range(len(Rps)):
     tabs[i,j,0] = Rps[j]
     tabs[i,j,1] = dsig[j]
#---Input galaxy properties----------------------------------
#--- QSO arrays-------------------------------------
#mss  = np.array([10.64,10.75,10.56,10.57,10.66,10.96,10.28,10.72,10.55])
#zzz  = np.array([0.77,0.78,0.80,0.44,0.90,0.81,0.74,0.97,0.57])
#--- GALs arrays-------------------------------------
mss  = np.array([9.18,9.64,10.12,10.55,10.97,11.39])
zzz  = np.array([0.52,0.67,0.63,0.64,0.66,0.73])
idnn = int(sys.argv[1])

M0   = 10.0**(mss[idnn])/1e+12
zl   = zzz[idnn]
#-------------------------------------------------------------
def galaxybias(logM):
  Mh  = 10.0**logM
  bs   = bias.haloBias(Mh,model='tinker10',z=zl,mdef = 'vir')
  return bs

def haloparams(logM,con):
   efunc     = 1.0/np.sqrt(omega_m*(1.0+zl)**3+\
              omega_l*(1.0+zl)**(3*(1.0+w))+\
              omega_k*(1.0+zl)**2)
   rhoc      = rho_crt0/efunc/efunc
   omegmz    = omega_m*(1.0+zl)**3*efunc**2
   ov        = 1.0/omegmz-1.0
   dv        = 18.8*pi*pi*(1.0+0.4093*ov**0.9052)
   rhom      = rhoc*omegmz

   r200 = (10.0**logM*3.0/200./rhom/pi)**(1./3.)
   rs   = r200/con
   delta= (200./3.0)*(con**3)\
               /(np.log(1.0+con)-con/(1.0+con))

   amp  = 2.0*rs*delta*rhoc*10e-14
   res  = np.array([amp,rs,r200])

   return res

def nfwfuncs(Rp,rsub):
     r   = Rp
     x   = r/rsub
     x1  = x*x-1.0
     x2  = 1.0/np.sqrt(np.abs(1.0-x*x))
     x3  = np.sqrt(np.abs(1.0-x*x))
     x4  = np.log((1.0+x3)/(x))
     s1  = r*0.0
     s2  = r*0.0
     ixa = x>0.
     ixb = x<1.0
     ix1 = ixa&ixb
     s1[ix1] = 1.0/x1[ix1]*(1.0-x2[ix1]*x4[ix1])
     s2[ix1] = 2.0/(x1[ix1]+1.0)*(np.log(0.5*x[ix1])\
               +x2[ix1]*x4[ix1])

     ix2 = x==1.0
     s1[ix2] = 1.0/3.0
     s2[ix2] = 2.0+2.0*np.log(0.5)

     ix3 = x>1.0
     s1[ix3] = 1.0/x1[ix3]*(1.0-x2[ix3]*np.arctan(x3[ix3]))
     s2[ix3] = 2.0/(x1[ix3]+1.0)*(np.log(0.5*x[ix3])+\
              x2[ix3]*np.arctan(x3[ix3]))

     res = {'DelSig':s2-s1,'SigmaRp':s1}
     return res

def NFWcen(theta,Rp):
  logM,con     = theta
  #con    = 10.0*(10.0**logM/1.5/1e+13)**(-0.13)
  amp,rs,r200  = haloparams(logM,con)
  struc        = nfwfuncs(Rp,rs)
  res          = amp*struc['DelSig'] 
  return res

def ESDsat(theta,Rp):
  logM,Mprm     = theta
  #amp,rs,r200  = haloparams(logM,con)
  #sumESDsat    = np.zeros(3077)

  #modelsat = ZuMandelbaum15Sats()
  nbins    = 50
  mbins    = np.linspace((np.log10(3.0)+logM),16.0,nbins)
  nsat     = np.zeros(nbins)
  mfunc_so = np.zeros(nbins)
  totalESDsat   = np.zeros(3077)
  for j in range(nbins):
      Mhalo       = 10.0**(mbins[j])
      mfunc_so[j] =mass_function.massFunction(mbins[j],zl,q_in='M',q_out='dndlnM',mdef='vir',model='tinker08')
      nsat[j]     = ((Mhalo-3.0*10.0**(logM))/10.0**(Mprm))
      con    = 4.67*(10.0**logM/1.0e+14)**(-0.11)
      amp,rs,r200 = haloparams(mbins[j],con)
      sumESDsat   = np.zeros(3077)

      for i in range(nx):
         rpp      = tabs[i,:,0]*rs
         struc    = nfwfuncs(rpp,rs)
         wt       = struc['SigmaRp'] 
         ix       = rpp>=r200
         wt[ix]   = 0.0
         dsig     = tabs[i,:,1]
         sumESDsat= sumESDsat+wt*dsig*amp
      totalESDsat = totalESDsat+mfunc_so[j]*nsat[j]*sumESDsat/wt.sum()
  return totalESDsat/(mfunc_so*nsat).sum()

#--------------------------------------------------------------
def lnlike(theta,Rp,esd,err):
  logM,con,Mprm,frac= theta
  #con    = 10.0*(10.0**logM/1.5/1e+13)**(-0.13)
  amp,rs,r200  = haloparams(logM,con)
  stellar= M0/pi/Rp/Rp
  bias   = galaxybias(logM)
  nfwcen = NFWcen([logM,con],Rp)
  res    = ESDsat([logM,Mprm],Rp)
  nfwsat = np.interp(Rp,tabs[0,:,0],res)
  twoesd = bias*np.interp(Rp,rrp,esdth)
  irvir  = Rp<=r200
  twoesd[irvir] = 0.0
  model  = (stellar+(1.0-frac)*nfwcen+frac*nfwsat+twoesd)/h
  invers = 1.0/err/err
  diff   = -0.5*((esd-model)**2*invers-np.log(invers))
  return diff.sum()

def lnprior(theta):
  logM,con,Mprm,frac= theta
  if 11.00<logM<14.0 and 0.1<con<20.0  and 10.0<Mprm<15.0 \
         and 0.0<frac<1.0:# and 9.0<logMsub<13.0:
     return 0.0
  return -np.inf

def lnprob(theta,Rp,esd,err):
  lp = lnprior(theta)
  if not np.isfinite(lp):
     return -np.inf
  return lp+lnlike(theta,Rp,esd,err)

def chi2(esd,error,model):
  chisq=np.sum((esd-model)*(esd-model)/error/error)
  return chisq
#----------------------------------------------------------------
def main():
  prefix = 'esd_gals/' 
  #data   = np.loadtxt(prefix+'DelSig_gals_msbinwt'+str(idnn+3)+'_0.02_2Mpc.dat',unpack=True)
  #data   = np.loadtxt(prefix+'DelSig_gals_msbinwt'+str(idnn+3)+'_0.02_2Mpc.dat',unpack=True)
  data   = np.loadtxt(prefix+'DelSig_gals_msbinwt'+str(idnn+3)+'_0.02_2Mpc.dat',unpack=True)
  ff=open('posteriors_galmsbinwt_'+str(idnn+3)+'.dat','w')
  """
  if idnn ==0:
    data   = np.loadtxt(prefix+'DelSig_qso_all.dat',unpack=True)
    ff=open('posteriors_galmsbin_1.dat','w')
  if idnn ==1:
    data   = np.loadtxt(prefix+'DelSig_typei.dat',unpack=True)
    ff=open('posteriors_qso_typei.dat','w')
  if idnn ==2:
    data   = np.loadtxt(prefix+'DelSig_typeii.dat',unpack=True)
    ff=open('posteriors_qso_typeii.dat','w')
  if idnn ==3:
    data   = np.loadtxt(prefix+'DelSig_lowz.dat',unpack=True)
    ff=open('posteriors_qso_lowz.dat','w')
  if idnn ==4:
    data   = np.loadtxt(prefix+'DelSig_highz.dat',unpack=True)
    ff=open('posteriors_qso_highz.dat','w')
  if idnn ==5:
    data   = np.loadtxt(prefix+'DelSig_higms.dat',unpack=True)
    ff=open('posteriors_qso_higms.dat','w')
  if idnn ==6:
    data   = np.loadtxt(prefix+'DelSig_lowms.dat',unpack=True)
    ff=open('posteriors_qso_lowms.dat','w')
  if idnn ==7:
    data   = np.loadtxt(prefix+'DelSig_higlum.dat',unpack=True)
    ff=open('posteriors_qso_higlum.dat','w')
  if idnn ==8:
    data   = np.loadtxt(prefix+'DelSig_lowlum.dat',unpack=True)
    ff=open('posteriors_qso_lowlum.dat','w')
  """
  rp     = data[7,:]*h
  esda   = data[5,:]/h
  err    = data[11,:]/h
  esd    = esda

  logM   = 12.31
  con    = 7.0
  Mprm   = 11.0
  frac   = 0.15
  
  pars   = np.array([logM,con,Mprm,frac])
  ndim,nwalkers = 4,90
  pos    = [pars+1e-4*np.random.randn(ndim) for i in range(nwalkers)]
  sampler= emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(rp,esd,err),threads=2) 
  sampler.run_mcmc(pos,2000)

  burnin = 100
  samples=sampler.chain[:,burnin:,:].reshape((-1,ndim))
  Mh,cn,Mpm,fc= map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]),zip(*np.percentile(samples,[16,50,84],axis=0)))


  ff.write("%10.6f  %10.6f  %10.6f\n"%(Mh[0],Mh[1],Mh[2]))
  ff.write("%10.6f  %10.6f  %10.6f\n"%(cn[0],cn[1],cn[2]))
  ff.write("%10.6f  %10.6f  %10.6f\n"%(Mpm[0],Mpm[1],Mpm[2]))
  ff.write("%10.6f  %10.6f  %10.6f\n"%(fc[0],fc[1],fc[2]))
  ff.close()

if __name__=='__main__': 
  main()
