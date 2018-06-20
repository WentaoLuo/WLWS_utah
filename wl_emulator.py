#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Basic Parameters---------------------------
#h = 0.673
h = 1.0
Om0 = 0.3156
Ol0 = 1.0-Om0
Otot = Om0+Ol0
Ok0 = 1.0-Otot
w = -1
rho_crit0 = 2.78e11 #M_sun Mpc^-3 *h*h
rho_bar0 = rho_crit0*Om0
sigma8 = 0.9
apr =  206269.43                #1/1^{''}
vc = 2.9970e5                   #km/s
G = 4.3e-9                      #(Mpc/h)^1 (Msun/h)^-1 (km/s)^2 
H0 = 67.3                      #km/s/(Mpc/h)
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
Gg    =6.754e-11              
ckm   =3.240779e-17            
ckg   =5.027854e-31           
fac   =(vc*vc*ckg)/(4.*pi*Gg*ckm) 


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

#  ixa = x>0.
#  ixb = x<1.0
#  ix1 = ixa&ixb
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
  Rmax = 2.0
  Rmin = 0.1
  dl   = Da(z)
  zs   = 0.45
  ds   = Da(zs)
  Sig  =fac*ds/(dl*(ds-dl))/(1.0+z)/(1.0+z)
  rbin = np.zeros(Nbin+1)
  r    = np.zeros(Nbin)
  xtmp = (np.log10(Rmax)-np.log10(Rmin))/Nbin
  area = np.zeros(Nbin)
  ngals= np.zeros(Nbin)
  esd  = np.zeros(Nbin)
  esdnfw= np.zeros(Nbin)
  err  = np.zeros(Nbin)
  the1 = np.zeros(Nbin)
  the2 = np.zeros(Nbin)
  for i in range(Nbin):
    ytmp1 = np.log10(Rmin)+float(i)*xtmp
    ytmp2 = np.log10(Rmin)+float(i+1)*xtmp
    rbin[i] = 10.0**ytmp1
    rbin[i+1] = 10.0**ytmp2
    the1  = np.arctan(rbin[i]/dl)*180.0*60.0/pi
    the2  = np.arctan(rbin[i+1]/dl)*180.0*60.0/pi
    area[i]= pi*(the2**2-the1**2)
    r[i] =(rbin[i])*1./2.+(rbin[i+1])*1./2.
    ngals[i]= np.random.poisson(lam=area[i]) 
    if ngals[i]>0:
      esd[i]= nfwesd(theta,z,r[i])+Sig*np.random.normal(loc=0.0,scale=0.4)/np.sqrt(float(ngals[i]))
      esdnfw[i]= nfwesd(theta,z,r[i])
      err[i]= Sig*np.random.normal(loc=0.0,scale=0.4)/np.sqrt(float(ngals[i]))
  return {'radius':r,'NUM':ngals,'ESD':esd,'NFW':esdnfw}
#----------------------------------------------------------------
def main():
   Nbin = 10
   Nlens= 5000
   Nboot= 500
   esdall= np.zeros((Nlens,Nbin))
   esdmdl= np.zeros((Nlens,Nbin))
   esdmean= np.zeros(Nbin)
   esdmodel= np.zeros(Nbin)
   esderr = np.zeros(Nbin)
   Ngax   = np.zeros((Nlens,Nbin))
   for i in range(Nlens):
     Mh   = 14.0
     c    = 5.0
     zl   = 0.1
     info = Rpbins([Mh,c],Nbin,zl)
     rr   = info['radius']
     esdall[i,:]  = info['ESD']
     esdmdl[i,:]  = info['NFW']
     Ngax[i,:]  = info['NUM']
     #err  = info['ERR']
   for j in range(Nbin):
     temp  = np.zeros((Nbin,Nboot))
     for iboot in range(Nboot):
	idx = np.random.randint(low=0,high=Nlens,size=Nlens)
#	print idx
        temp[j,iboot]= np.nanmean(esdall[idx,j])
     esderr[j]= np.nanstd(temp[j,:])
     esdmean[j]= np.nanmean(temp[j,:])
     esdmodel[j]= np.nanmean(esdmdl[:,j])
   #print esdmean,esderr
   plt.errorbar(rr,esdmean,yerr=esderr,fmt='k.',ms=20,elinewidth=3,label='Mock')
   plt.plot(rr,esdmodel,'k-',linewidth=3,label='NFW')
   plt.xlabel('R ($h^{-1}kpc$)',fontsize=20)
   plt.ylabel('ESD ($M_{\odot}/pc^2)$',fontsize=20)
   plt.legend()
   plt.xscale('log')
   plt.yscale('log',nonposy='clip')
   plt.xlim(0.1,1.5)
   plt.ylim(1,200)
   plt.show()

if __name__=='__main__':
   main()
