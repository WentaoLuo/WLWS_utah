#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import hodstat

def mockhod(lgMh,Vmax,icen):
  lgMmin = np.min(lgMh)
  lgMmax = np.max(lgMh)
  
  nmh    = 50
  step   = (lgMmax-lgMmin)/float(nmh)
  Mh     = np.zeros(nmh)
  cen    = np.zeros(nmh)
  sat    = np.zeros(nmh)
  for i in range(nmh):
     ixc = (lgMh>=lgMmin+step*float(i)) & (lgMh<lgMmin+step*float(i+1)) & \
           (icen==1)
     ixs = (lgMh>=lgMmin+step*float(i)) & (lgMh<lgMmin+step*float(i+1)) & \
           (icen==0)
     if len(Vmax[ixc])>0:
       ctmp   = float(len(lgMh[ixc]))/float(len(Vmax[ixc]))
       stmp   = np.sum(float(len(lgMh[ixs]))/(Vmax[ixs]))
       #stmp   = float(len(lgMh[ixs]))
       #print float(len(lgMh[ixs])),float(len(lgMh[ixc]))
     cen[i] = ctmp
     sat[i] = stmp
     Mh[i]  = lgMmin+step*float(i)+step*0.5
  return {'logMh':Mh,'CenHOD':cen,'SatHOD':sat}
#-----------------------------------------
fmock = '../../../mock/mock_f0.10_s02.dat'
Mr,zl,zmax,Vmax,lgMh,R,icen = np.loadtxt(fmock,unpack=True,comments='#')
Mrlim = np.linspace(-19.0,-22.0,50)

hodm      = mockhod(lgMh,Vmax,icen)
mmk       = hodm['logMh']
mcen      = hodm['CenHOD']
msat      = hodm['SatHOD']
hods      = hodstat.HODSTAT(-18,-23) 
struc     = hods.hodfunc()
mass      = struc['logMh']
cen       = struc['CenHOD']
sat       = struc['SatHOD']
#print mmk
#print mcen
#print msat
plt.plot(mass,cen,'k--',lw=3,label='Central')
plt.plot(mass,sat,'k-.',lw=3,label='Satellite')
plt.plot(mass,cen+sat,'k-',lw=3,label='Total')
plt.plot(mmk,mcen,'b--',lw=3,label='mCentral')
plt.plot(mmk,msat*100,'b-.',lw=3,label='mSatellite')
plt.plot(mmk,mcen+msat*100,'b-',lw=3,label='mTotal')
plt.xlim(11.0,15.0)
plt.ylim(0.0,1000.0)
plt.yscale('log')
plt.xlabel('logMh')
plt.ylabel('Number')
plt.show()

fsatmk= np.zeros(49)
fsatsm= np.zeros(49)
for i in range(49):
  ixcen     = (Mr<=Mrlim[i]) & (Mr>=Mrlim[i+1]) & (lgMh>=12.16) & (lgMh<=14.1) & (icen==1)
  ixsat     = (Mr<=Mrlim[i]) & (Mr>=Mrlim[i+1]) & (lgMh>=12.16) & (lgMh<=14.1) & (icen==0)
  fsatmk[i] = float(len(Mr[ixsat]))/float(len(Mr[ixcen])+len(Mr[ixsat]))

  hods      = hodstat.HODSTAT(Mrlim[i],Mrlim[i+1]) 
  struc     = hods.occupation(np.mean(lgMh[ixcen]))
  cen       = struc['Central']
  sat       = struc['Satellite']
  fsatsm[i] = sat/(cen+sat)
  #print Mrlim[i]-0.06,fsatsm[i],fsatmk[i],np.mean(lgMh[ixcen]),np.mean(lgMh[ixsat])
  #print len(Mr[ixsat]),len(Mr[ixcen])
plt.plot(Mrlim[0:49]-0.06,fsatsm,'k-',lw=3,label='HOD')
plt.plot(Mrlim[0:49]-0.06,fsatmk,'k--',lw=3,label='Mock')
plt.ylim(0.001,0.8)
#plt.yscale('log')
plt.xlim(-19.0,-22.0)
plt.show()



