#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Basic Parameters---------------------------
#h = 0.673
h = 0.7
Om0 = 0.3
Ol0 = 1.0-Om0
Otot = Om0+Ol0
Ok0 = 1.0-Otot
w = -1
rho_crit0 = 2.78e11 #M_sun Mpc^-3 *h*h
rho_bar0 = rho_crit0*Om0
sigma8 = 0.8
apr =  206269.43                #1/1^{''}
vc = 2.9970e5                   #km/s
G = 4.3e-9                      #(Mpc/h)^1 (Msun/h)^-1 (km/s)^2 
H0 = 70.0                      #km/s/(Mpc/h)
pc = 3.085677e16                #m
kpc = 3.085677e19               #m
Mpc = 3.085677e22               #m
Msun = 1.98892e30               #kg
yr = 31536000.0/365.0*365.25    #second
Gyr = yr*1e9                    #second
pi  = np.pi
omega_m = Om0
omega_l = Ol0
omega_k = Ok0
Gg    =6.67408e-11              
ckm   =3.240779e-17            
ckg   =5.027854e-31           
fac   =(vc*vc*ckg)/(4.*pi*Gg*ckm) 
resp,errorp = np.loadtxt('parents_dist.dat',unpack=True) 

Nboot = 500
#--Basic cosmology calculations--------------------
def efunclcdm(x):
   res = 1.0/np.sqrt(Om0*(1.0+x)**3+Ok0*(1.0+x)**2+Ol0*(1.0+x)**(3*(1.0+w)))
   return res
def Hz(x):
   res = H0/efunclcdm(x)
   return res
#-----------------------------------------------------------------------------
def a(x):
   res = 1.0/(1.0+x)
   return res
#-----------------------------------------------------------------------------
def Dh():
   res = vc/H0
   return res
def Da(x):
   res = Dh()*integrate.romberg(efunclcdm, 0, x)
   return res

#--using NFW profile---- 
def funcs(Rp,rs):
  x   = Rp/rs
  x1  = x*x-1.0
  x2  = 1.0/np.sqrt(np.abs(1.0-x*x))
  x3  = np.sqrt(np.abs(1.0-x*x))
  x4  = np.log((1.0+x3)/(x))
  s1  = Rp*0.0
  s2  = Rp*0.0

  if x >0.0 and x<1.0:
    s1 = 1.0/x1*(1.0-x2*x4)
    s2 = 2.0/(x1+1.0)*(np.log(0.5*x)\
           +x2*x4)
  if x ==1.0:  
    x2 = x==1.0
    s1 = 1.0/3.0
    s2 = 2.0+2.0*np.log(0.5)

  if x>1.0:
    s1 = 1.0/x1*(1.0-x2*np.arctan(x3))
    s2 = 2.0/(x1+1.0)*(np.log(0.5*x)+\
             x2*np.arctan(x3))

  res = {'funcf':s1,'funcg':s2}
  return res
def nfwesd(theta,z,Rp):
  Mh,c      = theta
  efunc     = 1.0/np.sqrt(omega_m*(1.0+z)**3+\
              omega_l*(1.0+z)**(3*(1.0+w))+\
              omega_k*(1.0+z)**2)
  rhoc      = rho_crit0/efunc/efunc
  omegmz    = omega_m*(1.0+z)**3*efunc**2
  ov        = 1.0/omegmz-1.0
  dv        = 18.8*pi*pi*(1.0+0.4093*ov**0.9052)
  rhom      = rhoc*omegmz

  r200 = (10.0**Mh*3.0/200./rhom/pi)**(1./3.)
  rs   = r200/c
  delta= (200./3.0)*(c**3)\
          /(np.log(1.0+c)-c/(1.0+c))
  amp  = 2.0*rs*delta*rhoc*10e-14
  functions = funcs(Rp,rs)
  funcf     = functions['funcf']
  funcg     = functions['funcg']
  esd       = amp*(funcg-funcf)

  return esd

def Rpbins(theta,Nbin,z):
  Rmax  = 2.0
  Rmin  = 0.02
  dl    = Da(z)
  zs    = 0.45
  ds    = Da(zs)
  Sig   =fac*ds/(dl*(ds-dl))/(1.0+z)/(1.0+z)
  rbin  = np.zeros(Nbin+1)
  r     = np.zeros(Nbin)
  xtmp  = (np.log10(Rmax)-np.log10(Rmin))/Nbin
  area  = np.zeros(Nbin)
  ngals = np.zeros(Nbin)
  esd   = np.zeros(Nbin)
  esdnfw= np.zeros(Nbin)
  #err   = np.zeros(Nbin)
  #wts   = np.zeros(Nbin)
  the1  = np.zeros(Nbin)
  the2  = np.zeros(Nbin)
  for i in range(Nbin):
    ytmp1 = np.log10(Rmin)+float(i)*xtmp
    ytmp2 = np.log10(Rmin)+float(i+1)*xtmp
    rbin[i] = 10.0**ytmp1
    rbin[i+1] = 10.0**ytmp2
    the1  = np.arctan(rbin[i]/dl)*180.0*60.0/pi
    the2  = np.arctan(rbin[i+1]/dl)*180.0*60.0/pi
    area[i]= 20.0*pi*(the2**2-the1**2)
    r[i] =(rbin[i])*1./2.+(rbin[i+1])*1./2.
    ngals[i]= np.random.poisson(lam=area[i]) 
    if ngals[i]>0:
      esd[i]= nfwesd(theta,z,r[i])+Sig*np.random.normal(loc=0.0,scale=0.4)/np.sqrt(float(ngals[i]))
      esdnfw[i]= nfwesd(theta,z,r[i])/Sig
      #err[i]= Sig*np.random.normal(loc=0.0,scale=0.4)/np.sqrt(float(ngals[i]))
      
  #return {'radius':r,'NUM':ngals,'ESD':esd,'NFW':esdnfw,'error':err}
  return {'radius':r,'NUM':ngals,'ESD':esd,'NFW':esdnfw}

#----------------------------------------------------------------
def main():
   Nboot= 500
   import sys
   fmock= '../mock/mock_f0.10_s02.dat' 
   Mr,zl,zmax,Vmax,lgMh,R,icen = np.loadtxt(fmock,unpack=True,comments='#')
   icena= icen ==1
   icenb= Mr<= float(sys.argv[1])
   icenc= Mr>= float(sys.argv[2])
   idcen= icena&icenb&icenc

   idsat= icen ==0
   Mrcen= Mr[idcen]
   Mrsat= Mr[idsat]
   zlcen= zl[idcen]
   zlsat= zl[idsat]
   Vcen = Vmax[idcen]
   Vsat = Vmax[idsat]
   Mhcen= lgMh[idcen]
   Nlens= len(Mhcen)
#------------------------------------------------------------------
   zs    = 0.45
   ds    = Da(zs)
   sige  = 0.4
   Nbin  = 10
   shear = np.zeros((Nlens,Nbin))
   Ngax  = np.zeros((Nlens,Nbin))

   sumshrI = np.zeros(Nbin)
   sumwhtI = np.zeros(Nbin)
   sumshrII= np.zeros(Nbin)
   sumwhtII= np.zeros(Nbin)
   for i in range(Nlens):
     Mh   = Mhcen[i]
     c    = 4.67*(10.0**(Mh-14)*h)**(-0.11) # Neto et al 2007 
     info = Rpbins([Mh,c],Nbin,zl[i])
     Rp   = info['radius']
     shear[i,:] = info['NFW']
     Ngax[i,:]  = info['NUM']
     dl   = Da(zl[i])*(1.0+zl[i])
     Sig  =fac*ds/(dl*(ds-dl))/(1.0+zl[i])/(1.0+zl[i])
     for j in range(Nbin):
          ngal  = int(np.around(Ngax[i,j]))
	  #print ngal
	  isx   = np.random.randint(low=0,high=len(errorp),size=ngal)
	  #print j,shear[i,j]
          gmt   = Sig*np.random.normal(loc=shear[i,j],scale=0.4,size=ngal)
	  wht   = 1.0/(sige*sige+errorp[isx]**2)/Sig/Sig
	  #print gmt.sum(),wht.sum()
          sumshrI[j] = sumshrI[j]+(gmt*wht).sum()
          sumwhtI[j] = sumwhtI[j]+wht.sum()
          sumshrII[j]= sumshrII[j]+(gmt*wht).sum()/Vcen[i]
          sumwhtII[j]= sumwhtII[j]+wht.sum()/Vcen[i]

   gammaI  = sumshrI/sumwhtI
   gammaII = sumshrII/sumwhtII
   Mhm     = np.mean(Mhcen)
   zm      = np.mean(zl)
   con     = 4.67*(10.0**(Mhm-14)*h)**(-0.11) # Neto et al 2007 
   esdnfw  = np.zeros(Nbin)
   for i in range(Nbin):
     esdnfw[i]  = nfwesd([Mhm,con],zm,Rp[i])
     print Rp[i], gammaI[i]/h,gammaII[i]/h,esdnfw[i]/h
     #print gammaII
   """ 
   plt.figure(figsize=[9,6])
   #plt.errorbar(Rp,gammaI,yerr=esderr,fmt='k.',\
   #             ms=20,elinewidth=3,label='weight I')
   plt.plot(Rp,gammaII/h,'k-',linewidth=3,label='weight volum limited')
   plt.plot(Rp,gammaI/h,'k--',linewidth=3,label='weight traditional')
   plt.plot(Rp,esdnfw/h,'k:',linewidth=3,label='<Mh> NFW model')
   plt.xlabel('R ($h^{-1}kpc$)',fontsize=20)
   plt.ylabel('ESD ($M_{\odot}/pc^2)$',fontsize=20)
   plt.legend()
   plt.xscale('log')
   plt.yscale('log',nonposy='clip')
   plt.xlim(0.02,2.0)
   plt.ylim(0.01,1000)
   plt.show()
   """ 
if __name__=='__main__':
   main()
