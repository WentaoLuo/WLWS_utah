#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import hodstat

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
sige  = 0.3
zs    = 0.6
Nboot = 200

fname,Rc1,Rc2 = np.loadtxt('r_c-range.tsv',\
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
rtmp = tabs[0,:,0]

Rmax  = 2.0  # largest scale to measure in unit of Mpc/h
Rmin  = 0.01 # smallest scale to measure in unit of Mpc/h
Nbin  = 10   # number of equal logrithm radial bins
sige  = 0.36 # observational shape noise from SDSS DR7
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
def nfwfuncs(rs,Rp):
   r   = Rp
   x   = r/rs
   x1  = x*x-1.0
   x2  = 1.0/np.sqrt(np.abs(1.0-x*x))
   x3  = np.sqrt(np.abs(1.0-x*x))
   x4  = np.log((1.0+x3)/(x))
   s1  = r*0.0
   s2  = r*0.0
   ix1 = (x>0.0) & (x<1.0)

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

   res = s2-s1
   return res

def haloparams(logM,con,zl):
   efunc     = 1.0/np.sqrt(omega_m*(1.0+zl)**3+\
               omega_l*(1.0+zl)**(3*(1.0+w))+\
               omega_k*(1.0+zl)**2)
   rhoc      = rho_crit0/efunc/efunc
   omegmz    = omega_m*(1.0+zl)**3*efunc**2
   ov        = 1.0/omegmz-1.0
   dv        = 18.8*pi*pi*(1.0+0.4093*ov**0.9052)
   rhom      = rhoc*omegmz
   r200 = (10.0**logM*3.0/200./4.0/rhom/pi)**(1./3.)
   rs   = r200/con

   delta= (200./3.0)*(con**3)\
          /(np.log(1.0+con)-con/(1.0+con))

   amp  = 2.0*rs*delta*rhom*10e-14
   res  = np.array([amp,rs,r200])

   return res

def nfwesd(theta,z,Rp):
  Mh,c      = theta
  amp,rs,r200 = haloparams(Mh,c,z)
  functions = nfwfuncs(rs,Rp)
  esd       = amp*(functions)

  return esd

def ESDsat(theta,Rp):
  logM,con,roff,z     = theta
  amp,rs,r200  = haloparams(logM,con,z)
  rr      = roff/rs
  res     = np.zeros(Nbin)
  xx      = np.linspace(1,nx,nx)-1
  idx     = np.abs(rc2-rr)==np.min(np.abs(rc2-rr))
  inx     = int(xx[idx])
  summ    = np.zeros(ny)
  tmp     = 34.0*amp*tabs[inx,:,1]*(rc2[inx]-rc1[inx])/rc2[inx]
  res     = np.interp(Rp,rs*rtmp,tmp)
  return res

def Rpbins(theta,Nbin,z,icen,roff):
  Rmax  = 2.0
  Rmin  = 0.01
  dl    = Da(z)
  ds    = Da(zs)
  Sig   =fac*ds/(dl*(ds-dl))/(1.0+z)/(1.0+z)
  rbin  = np.zeros(Nbin+1)
  r     = np.zeros(Nbin)
  xtmp  = (np.log10(Rmax)-np.log10(Rmin))/Nbin
  area  = np.zeros(Nbin)
  ngals = np.zeros(Nbin)
  esd   = np.zeros(Nbin)
  esdnfw= np.zeros(Nbin)
  the1  = np.zeros(Nbin)
  the2  = np.zeros(Nbin)
  for i in range(Nbin):
    ytmp1 = np.log10(Rmin)+float(i)*xtmp
    ytmp2 = np.log10(Rmin)+float(i+1)*xtmp
    rbin[i] = 10.0**ytmp1
    rbin[i+1] = 10.0**ytmp2
    the1  = np.arctan(rbin[i]/dl)*180.0*60.0/pi
    the2  = np.arctan(rbin[i+1]/dl)*180.0*60.0/pi
    area[i]= 50.0*pi*(the2**2-the1**2)
    ngals[i]= np.random.poisson(lam=area[i])
    r[i] =(rbin[i])*0.5+(rbin[i+1])*0.5
  if  icen==1:
      esdnfw  = nfwesd(theta,z,r)/Sig
  if icen==0:
      logM,c  = theta
      esdnfw  = ESDsat([logM,c,roff,z],r)/Sig
  return {'radius':r,'NUM':ngals,'NFW':esdnfw}

#----------------------------------------------------------------
def main():
   Nboot= 500
   import sys

   fmock= '../../../mock/mock_f0.10_s01.dat' 
   Mr,zl,zmax,Vmax,lgMh,R,icen = np.loadtxt(fmock,unpack=True,comments='#')
   idx= (Mr<=float(sys.argv[1])) & (Mr>=float(sys.argv[2])) & (zl<=float(sys.argv[3]))
   #Nlens=  len(Mr[idx])
   Nlens= 10000 

#------------------------------------------------------------------
   ds    = Da(zs)
   shear = np.zeros((Nlens,Nbin))
   Ngax  = np.zeros((Nlens,Nbin))

   sumshrI  = np.zeros(Nbin)
   sumwhtI  = np.zeros(Nbin)
   sumshrII = np.zeros(Nbin)
   sumwhtII = np.zeros(Nbin)
   sumerrI  = np.zeros(Nbin)
   sumerrII = np.zeros(Nbin)
   sumnum   = np.zeros(Nbin)
   sumesd   = np.zeros(Nbin)
   for i in range(Nlens):
     Mh   = lgMh[idx][i]   #test of utah project
     c    = 4.67*(10.0**(Mh-14)*h)**(-0.11) # Neto et al 2007 
     info = Rpbins([Mh,c],Nbin,zl[idx][i],icen[idx][i],R[idx][i])
     Rp   = info['radius']
     shear[i,:] = info['NFW']
     Ngax[i,:]  = info['NUM']
     dl   = Da(zl[idx][i])*(1.0+zl[idx][i])
     Sig  =fac*ds/(dl*(ds-dl))/(1.0+zl[idx][i])/(1.0+zl[idx][i])
     for j in range(Nbin):
          #sumesd[j]  = sumesd[j]+nfwesd([Mh,c],zl[idx][i],Rp[j])
          #sumesd[j]  = sumesd[j]+nfwesd([Mh,c],0.26,Rp[j])
          ngal  = int(np.around(Ngax[i,j]))
	  sumnum[j]= sumnum[j]+Ngax[i,j]
	  isx   = np.random.randint(low=0,high=len(errorp),size=ngal)
          gmt   = Sig*np.random.normal(loc=shear[i,j],scale=sige,size=ngal)
          #-- Normal weighting-----------------------------
	  wht   = 1.0/(sige*sige+errorp[isx]**2)/Sig/Sig
          #-- No Sig_c weighting-----------------------------
	  #wht   = 1.0/(sige*sige+errorp[isx]**2)
	  #-------- weight I--------------------------------------
          sumshrI[j] = sumshrI[j]+(gmt*wht).sum()
          sumwhtI[j] = sumwhtI[j]+wht.sum()
          #sumwhtI[j] = sumwhtI[j]+wht
          sumerrI[j] = sumerrI[j]+(gmt*gmt*wht*wht).sum()

	  #-------- weight II--------------------------------------
          sumshrII[j]= sumshrII[j]+(gmt*wht).sum()/Vmax[idx][i]
          sumwhtII[j]= sumwhtII[j]+wht.sum()/Vmax[idx][i]
          sumwhtII[j]= sumwhtII[j]+wht.sum()/Vmax[idx][i]
          sumerrII[j]= sumerrII[j]+(gmt*gmt*wht*wht).sum()/Vmax[idx][i]/Vmax[idx][i]

   print '# '+str(Nlens) 
   print '# '+str(sumnum.sum())
   gammaI  = sumshrI/sumwhtI
   errorI  = np.sqrt(sumerrI/(sumwhtI*sumwhtI))/2.0
   gammaII = sumshrII/sumwhtII
   errorII = np.sqrt(sumerrII/sumwhtII/sumwhtII)/2.0
   esdnfw  = np.zeros(Nbin)
   #Mhm     = np.mean(Mhcen)
   #Mhm     = 10.5
   #zm      = 0.26
   #zm      = np.mean(zlcen)
   #con     = 4.67*(10.0**(Mhm-14)*h)**(-0.11) # Neto et al 2007 
    
   #if int(sys.argv[4])==1:
   #   print "# volum limited"
   #   print "# Rp   ESD_wtI    Error_wtI     ESD_wtII    Error_wtII    NFW<Mh>"
   #if int(sys.argv[4])==2:
   #   print "# flux limited"
   #   print "# Rp   ESD_wtI    Error_wtI     ESD_wtII    Error_wtII    NFW<Mh>"

   esdnfw = (sumesd/float(Nlens))
   #for ie in range(Nbin):
      #print Rp[ie],gammaI[ie],errorI[ie],gammaII[ie],errorII[ie],esdnfw[ie]
   rbrp   = np.array([0.026,0.041,0.064,0.103,0.163,0.258,0.410,0.649,1.029,1.631])
   rbesd  = np.array([-5.04,0.18,4.64,1.34,0.25,-0.40,0.72,0.34,0.16,0.30])
   rberr  = np.array([5.69,2.62,1.50,0.92,0.59,0.36,0.23,0.15,0.10,0.06])
   plt.figure(figsize=[9,6])
   plt.errorbar(Rp,gammaI/h,yerr=errorI*1.414/h,fmt='g.',\
                ms=20,elinewidth=3,label='weight I')
   #plt.errorbar(rbrp,rbesd,yerr=rberr,fmt='k.',\
   #             ms=20,elinewidth=3,label='VOICE')
   plt.errorbar(Rp,gammaII/h,yerr=errorII*1.414/h,fmt='r.',\
                ms=20,elinewidth=3,label='weight II')
   #plt.plot(Rp,gammaII/h,'r--',linewidth=3,label='weight volum limited')
   #plt.plot(Rp,gammaI/h,'g--',linewidth=3,label='weight traditional')
   #plt.plot(Rp,gammaI/h,'g--',linewidth=3,label='weight traditional')
   #plt.plot(Rp,esdnfw/h,'b-',linewidth=3,label='<Mh> NFW model')
   plt.xlabel('R ($h^{-1}kpc$)',fontsize=20)
   plt.ylabel('ESD ($M_{\odot}/pc^2)$',fontsize=20)
   plt.legend()
   plt.xscale('log')
   plt.yscale('log',nonposy='clip')
   plt.xlim(0.01,2.0)
   plt.ylim(0.01,500.0)
   plt.show()
    
if __name__=='__main__':
   main()
