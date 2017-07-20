from __future__ import absolute_import, unicode_literals, print_function
import json
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pymultinest
import math
import numpy as np
pi = np.pi
from pandas import read_csv
import os
import corner
from astropy.coordinates import SkyCoord

from scipy.optimize import bisect
mpi = True
# mpi = False
if mpi:
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	name = MPI.Get_processor_name()
else: rank = 0
#########################################################################################
from functools import partial
from scipy.stats import halfcauchy,lognorm,norm
from scipy.special import erf

from ModelsKent.kent_distribution import kent4
from ModelsKent.kent import Gamma

dir_  = os.path.expanduser('~') +"/PyAspidistra/"
real  = False
nlive = 40
model = str(sys.argv[1])
rcut  = float(sys.argv[2])
########################
Dist  = 136.0
D2R   = np.pi/180.
R2D   = 180./np.pi
cntr  = [56.65,24.13]
################################### MODEL ######################################################
nameparCtr = ["$\\theta$","$\phi$","$\psi$","$\kappa$","$\\beta$"]
minsCtr    = np.array([      -pi/2,        -pi,-pi/2 ,0.0 ,0.0 ])
maxsCtr    = np.array([       pi/2,         pi, pi/2 ,6e3 ,1.0 ])
paramsCtr  = np.array([cntr[1]*D2R,cntr[0]*D2R, 0.0  ,500 ,0.15 ])

if model == "Plummer":
	from ModelsKent.Plummer import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$"]
	mins    = np.array([0])
	maxs    = np.array([10.0])
	paramsR  = [5.7]

if model == "EFF":
	from ModelsKent.EFF import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$\gamma$"]
	mins    = np.array([0,  2.0])
	maxs    = np.array([5.0,5.0])
	paramsR  = [3.0,3.0]

if model == "King":
	from ModelsKent.King import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$", "$r_t$"]
	mins    = np.array(([ 0.0, 10.0]))
	maxs    = np.array(([ 5.0, 50.0]))
	paramsR = [1.0,20.0]

if model == "GDP":
	from ModelsKent.GDP import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = ["$r_c$","$a$","$b$","$\gamma$"]
	mins    = np.array([ 0.01  ,0.01,  0.01 , 0.01])
	maxs    = np.array([ 200.0, 2.0, 100.0, 2.0])
	#--------- arguments of logLike function


if model == "Centre":
	from ModelsKent.Centre import Module,Number,Density
	#--------- Initial parameters --------------
	namepar = nameparCtr
	support = np.vstack([minsCtr,maxsCtr]).T
	params  = paramsCtr
else:
	namepar = nameparCtr + namepar
	support = np.vstack([np.hstack([minsCtr,mins]),np.hstack([maxsCtr,maxs])]).T
	params  = np.hstack([paramsCtr,paramsR])

##################### DATA ##############################################################

if real :
	fdata = dir_+'Data/OnlyTycho.csv'
	data  = np.array(read_csv(fdata,header=0,sep=','))
	cdtsT = np.array(data[:,[1,2,32]],dtype=np.float32)
	fdata = dir_+'Data/Members-0.84.csv'
	data  = np.array(read_csv(fdata,header=0,sep=','))
	cdtsD = np.array(data[:,[10,11,8]],dtype=np.float32)
	cdts  = np.vstack([cdtsT,cdtsD])
	#---- removes duplicateds --------
	sumc  = np.sum(cdts[:,:2],axis=1)
	idx   = np.unique(sumc,return_index=True)[1]
	cdts  = cdts[idx]
	sumc  = np.sum(cdts[:,:2],axis=1)
	if len(sumc) != len(list(set(sumc))):
		sys.exit("Duplicated entries in Coordinates!")

	dir_out  = dir_+'MultiNest/Samples/'+model+'_Kent_'+str(int(rcut))
else :
	# np.random.seed(345)
	# unifsyn  = np.random.uniform(size=Ntot)
	# a        = np.array(map(lambda x:bisect(lambda r:Number(r,params,rcut)-x,0.0,rcut),unifsyn))
	# tsyn     = np.random.uniform(low=0,high=2*np.pi,size=Ntot)
	# xn_syn   = a*np.cos(tsyn)
	# yn_syn   = np.sqrt(1-params[4]**2)*a*np.sin(tsyn)
	# x_syn    =  xn_syn*np.cos(params[2]) + yn_syn*np.sin(params[2])   # in radians
	# y_syn    = -xn_syn*np.sin(params[2]) + yn_syn*np.cos(params[2])   # in radians
	# cdts     = np.empty((Ntot,3))

	# #------ small angle approxmation -----

	# cdts[:,1]= (params[0] + (y_syn/Dist))*R2D
	# cdts[:,0]= (params[1] + (x_syn/Dist))*R2D
	# cdts[:,2]= np.repeat(1,Ntot)

	k = kent4(Gamma(paramsCtr[0],paramsCtr[1],paramsCtr[2]),paramsCtr[3],0.5*paramsCtr[3]*paramsCtr[4])
	samples = k.rvs(10000)	
	# fdata    = dir_+'Data/Kent-2e3-1e3-Cntr-0.15.csv'
	# samples  = np.array(read_csv(fdata,header=0,sep=','))
	Ntot     = len(samples)
	cdts     = np.empty((Ntot,3))
	cdts[:,0]= (np.arctan2(samples[:,1],samples[:,0]))*R2D
	cdts[:,1]= (np.arcsin(samples[:,2]))*R2D   
	cdts[:,2]= np.repeat(1,Ntot)

	dir_out  = dir_+'MultiNest/Samples/Synthetic/'+model+'_Kent_'+str(int(rcut))+'_'+"{0:.0e}".format(Ntot)

#================= Directory =============
if not os.path.exists(dir_out) and rank==0: os.mkdir(dir_out)

#============== Cut on distance to cluster center ======================
radii     = np.arccos(np.sin(cntr[1]*D2R)*np.sin(cdts[:,1]*D2R)+
	            np.cos(cntr[1]*D2R)*np.cos(cdts[:,1]*D2R)*
	            np.cos((cntr[0]-cdts[:,0])*D2R))*Dist
idx       = np.where(radii <  rcut)[0]
cdts      = cdts[idx]

#========================================
#------------ Load Module -------
Module  = Module(cdts,rcut,support,Dist)
#========================================================
#--------- Dimension, walkers and inital positions -------------------------
# number of dimensions our problem has
ndim     = len(namepar)
n_params = len(namepar)

if mpi: comm.Barrier()
pymultinest.run(Module.LogLike,Module.Priors, n_params,resume = False, verbose = True,n_live_points=nlive,
	wrapped_params=[1,1,1,0,0],
	outputfiles_basename=dir_out+'/0-',multimodal=False, max_modes=1,sampling_efficiency = 'model')

# lets analyse the results
if mpi: comm.Barrier()

if rank ==0:
	a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=dir_out+'/0-')
	s   = a.get_stats()
	MAP = np.array(s['modes'][0]['maximum a posterior'])
	theta,phi,psi,kappa,beta = MAP[0],MAP[1],MAP[2],MAP[3],MAP[4]
	if model == "Centre" : 
		rc = 1
	else : 
		rc = MAP[5]
	samples = a.get_data()[:,2:]
	# print(samples.shape)
	nit     = samples.shape[0]
	nts     = int(0.2*nit)
	samples = samples[nts:]
	# print(samples.shape)
	#@################ Calculates Radii and PA ################

	#---------- Synthetic --------
	if not real :
		#============== Obtains radii and pa ===================
		# Equation 2 from Kacharov 2014.
		x     = np.sin(cdts[:,0]*D2R -params[1])*np.cos(cdts[:,1]*D2R)*Dist
		y     = (np.cos(params[0])*np.sin(cdts[:,1]*D2R)-
		         np.sin(params[0])*np.cos(cdts[:,1]*D2R)*np.cos(cdts[:,0]*D2R-params[1]))*Dist
		
		# x     = (cdts[:,0]*D2R - params[1])*Dist
		# y     = (cdts[:,1]*D2R - params[0])*Dist
		xn    = x*np.cos(params[2]) - y*np.sin(params[2])
		yn    = x*np.sin(params[2]) + y*np.cos(params[2])
		t_syn = np.arctan2(yn,xn)
		r_syn = np.sqrt(xn**2 + yn**2)
		r_syn = r_syn*np.abs(np.cos(t_syn))*np.sqrt(1.0+(np.tan(t_syn)/np.sqrt(1.0-params[4]**2))**2)
		
		idx   = np.argsort(r_syn)

		r_syn     = np.array(r_syn[idx])
		Rmax_syn  = max(r_syn)
		bins_syn  = np.linspace(0,Rmax_syn+0.1,101)

		hist_syn  = np.histogram(r_syn,bins=bins_syn)[0]
		bins_syn  = bins_syn[1:]
		dr_syn    = np.hstack([bins_syn[0]/2,np.diff(bins_syn)])
		da_syn    = 2*np.pi*bins_syn*dr_syn
		densi_syn = hist_syn/da_syn
		densi_syn = densi_syn/sum(densi_syn*bins_syn*dr_syn)
		Nr_syn    = np.cumsum(cdts[idx,2])

		xfitpsi_syn = [params[1]*R2D-np.cos(params[2])*(2*rc/Dist)*R2D,
					   params[1]*R2D+np.cos(params[2])*(2*rc/Dist)*R2D]
		yfitpsi_syn = [params[0]*R2D-np.sin(params[2])*(2*rc/Dist)*R2D,
					   params[0]*R2D+np.sin(params[2])*(2*rc/Dist)*R2D]

	# #---------------------------------------------
	#============== Obtains radii and pa ===================
	# Equation 2 from Kacharov 2014.
	x     = np.sin(cdts[:,0]*D2R -phi)*np.cos(cdts[:,1]*D2R)*Dist
	y     = (np.cos(theta)*np.sin(cdts[:,1]*D2R)-
			np.sin(theta)*np.cos(cdts[:,1]*D2R)*np.cos(cdts[:,0]*D2R-phi))*Dist

	# x     = (cdts[:,0]*D2R - phi  )*Dist
	# y     = (cdts[:,1]*D2R - theta)*Dist
	xn    = x*np.cos(psi) - y*np.sin(psi)
	yn    = x*np.sin(psi) + y*np.cos(psi)
	t     = np.arctan2(yn,xn)
	r     = np.sqrt(xn**2 + yn**2)
	r     = r*np.abs(np.cos(t))*np.sqrt(1.0+(np.tan(t)/np.sqrt(1.0-beta**2))**2)
	
	idx   = np.argsort(r)

	r     = np.array(r[idx])
	t     = np.array(t[idx])
	# pro   = np.array(cdts[idx,2])

	Rmax  = max(r)
	bins  = np.linspace(0,Rmax+0.1,101)


	hist  = np.histogram(r,bins=bins)[0]
	bins  = bins[1:]
	dr    = np.hstack([bins[0]/2,np.diff(bins)])
	da    = 2*np.pi*bins*dr
	densi = hist/da
	densi = densi/sum(densi*bins*dr)
	Nr    = np.cumsum(cdts[:,2])

	xfitpsi = [phi*R2D-np.cos(psi)*(2*rc/Dist)*R2D,phi*R2D+np.cos(psi)*(2*rc/Dist)*R2D]
	yfitpsi = [theta*R2D-np.sin(psi)*(2*rc/Dist)*R2D,theta*R2D+np.sin(psi)*(2*rc/Dist)*R2D]

	print()
	print("-" * 30, 'ANALYSIS', "-" * 30)
	print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))

	with PdfPages(dir_out+'/Fit_'+model+'_'+str(int(rcut))+'.pdf') as pdf:
		
		if not real :
			plt.scatter(r_syn,Nr_syn,s=1,color="blue")
			plt.plot(r_syn,np.max(Nr_syn)*Number(r_syn,params,Rmax_syn), linewidth=1,color="blue")
		else:
			plt.scatter(r,Nr,s=1,color="black")
		plt.plot(r,np.max(Nr)*Number(r,MAP,Rmax), linewidth=1,color="red")
		plt.ylim((0,1.1*max(Nr)))
		plt.xlim((0,1.1*rcut))
		plt.xlabel('Radius [$pc$]')
		plt.ylabel('Number of objects')
		pdf.savefig()  # saves the current figure into a pdf page
		plt.close()
		
		if not real:
			plt.scatter(bins_syn,densi_syn,s=1,color="blue")
			plt.plot(r_syn,Density(r_syn,params,Rmax_syn), linewidth=1,color="blue")
		else:
			plt.scatter(bins,densi,s=1,color="black")
		plt.plot(r,Density(r,MAP,Rmax), linewidth=1,color="red")
		plt.xlabel('Radius [$pc$]')
		plt.ylabel('Density [$pc^{-2}$]')
		pdf.savefig()  # saves the current figure into a pdf page
		plt.close()


		plt.figure()
		ax=plt.gca()
		cntrgal = SkyCoord(phi,theta,unit="rad",frame="fk5").galactic

		grid1 = np.array([np.linspace(cntrgal.l.deg-10,cntrgal.l.deg+10,num=10),np.repeat(cntrgal.b.deg,10)]).T
		grid2 = np.array([np.repeat(cntrgal.l.deg,10),np.linspace(cntrgal.b.deg-10,cntrgal.b.deg+10,num=10)]).T

		gcdt1 =SkyCoord(grid1,unit="deg",frame="galactic").fk5
		gcdt2 =SkyCoord(grid2,unit="deg",frame="galactic").fk5
		ax.text(gcdt2[0].ra.deg,gcdt2[0].dec.deg,s="l="+str(cntrgal.l))
		ax.text(gcdt1[0].ra.deg,gcdt1[0].dec.deg,s="b="+str(cntrgal.b))
		plt.plot(gcdt1.ra,gcdt1.dec,color="grey")
		plt.plot(gcdt2.ra,gcdt2.dec,color="grey")
		plt.scatter(cdts[:,0],cdts[:,1],s=1,color="black")
		plt.xlabel('RA [$deg$]')
		plt.ylabel('Dec [$deg$]')
		plt.axes().set_aspect('equal', 'datalim')	
		ax.add_patch(Ellipse(xy=[phi*R2D,theta*R2D], 
								width=(rc/Dist)*R2D, 
								height=np.sqrt(1-beta**2)*(rc/Dist)*R2D,
						 		angle=psi*R2D,
						 		fc='none',ec='red'))
		plt.plot(xfitpsi,yfitpsi,color='red', linestyle='-')
		if not real :
			ax.add_patch(Ellipse(xy=[params[1]*R2D,params[0]*R2D], 
								width=(rc/Dist)*R2D, 
								height=np.sqrt(1-params[4]**2)*(rc/Dist)*R2D, 
								angle=params[2]*R2D,
								fc='none',ec='blue'))
		
			plt.plot(xfitpsi_syn,yfitpsi_syn,color='blue', linestyle='-')
		pdf.savefig()  # saves the current figure into a pdf page
		plt.close()

		# if not real :
		# 	plt.scatter(t_syn,r_syn,s=1,color="blue")
		# plt.scatter(t,r,s=1,color="black")
		# plt.ylabel('Radius [pc]')
		# plt.xlabel('$PA_0$  [rad]')	
		# pdf.savefig()  # saves the current figure into a pdf page
		# plt.close()

		plt.figure()
		if not real:
			n, bins, patches = plt.hist(t_syn,50, normed=1, facecolor='blue', alpha=0.5)
		n, bins, patches = plt.hist(t,50, normed=1, facecolor='green', alpha=0.5)
		plt.xlabel('Position Angle [$rad$]')
		plt.ylabel('Density ')
		pdf.savefig()  # saves the current figure into a pdf page
		plt.close()

		if real :
			corner.corner(samples, labels=namepar,truths=MAP,truth_color="red")
		else:
			corner.corner(samples, labels=namepar,truths=params,truth_color="blue")
		pdf.savefig()
		plt.close()

	plt.clf()

	# p = pymultinest.PlotMarginalModes(a)
	# plt.figure(figsize=(5*n_params, 5*n_params))
	# #plt.subplots_adjust(wspace=0, hspace=0)
	# for i in range(n_params):
	# 	plt.subplot(n_params, n_params, n_params * i + i + 1)
	# 	p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
	# 	plt.ylabel("Probability")
	# 	plt.xlabel(namepar[i])
		
	# 	for j in range(i):
	# 		plt.subplot(n_params, n_params, n_params * j + i + 1)
	# 		#plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
	# 		p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
	# 		plt.xlabel(namepar[i])
	# 		plt.ylabel(namepar[j])

	# plt.savefig(dir_out+"/marginals_multinest.pdf") #, bbox_inches='tight')


	print("Take a look at the pdf files in "+dir_out)
sys.exit() 




 
