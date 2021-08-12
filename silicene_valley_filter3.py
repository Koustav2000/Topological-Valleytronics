import kwant
from math import pi,sqrt
import numpy as np
from matplotlib import pyplot
import silicene_lattice_test as silicene
import scipy.sparse.linalg as sla


def compute_evs(syst):
	# Compute some eigenvalues of the closed system
	sparse_mat = syst.hamiltonian_submatrix(sparse=True)

	evs = sla.eigs(sparse_mat, 2)[0]
	print(evs.real)

def plot_bandstructure(flead, momenta, params):
	bands = kwant.physics.Bands(flead, params = params)
	energies = [bands(k) for k in momenta]
	pyplot.figure()
	pyplot.plot(momenta, energies)
	#pyplot.ylim([-1,1])
	pyplot.show()
	return np.array(energies)

W = 90*sqrt(3)
L = 100
t = 1.3
grid = (1/2)*sqrt(3)*pi*(t/W)
lambda_so = 0.1
E = t/3
U = E
print(E)
print(lambda_so)
delta = -1*(0.1+lambda_so)
print(delta)
print(np.abs(delta)-lambda_so)
U_disorder = 1*(np.abs(delta)-lambda_so)
print(U_disorder)
Px = 0
Py = 0
Pz = 0


params = dict(U = U, U_disorder = U_disorder, Mex = 0, Px = Px, Py = Py, Pz = Pz, salt = 0)


syst, leads, dum_lead = silicene.make_system(W = W, L = L, delta = delta, t = t, lambda_so = lambda_so)


def family_colors(site):
	return 0 if (site.family == silicene.a) else 1
# Plot the closed system without leads.
#kwant.plot(syst, site_color=family_colors, site_lw=0.1, colorbar=False)

ham = syst.finalized().hamiltonian_submatrix(params=params)


# Compute the band structure of dum_lead.
num_momenta = 100
momenta = np.linspace(-1*pi,1*pi,num_momenta)
band_data = np.zeros((num_momenta,1+int(W*8/sqrt(3))))
band_data[:,0] = momenta
#band_data[:,1:] = plot_bandstructure(dum_lead.finalized(), momenta, dict(Mex=0))
#np.savetxt('/home/bmlabserver05/Koustav/topo_paper_7jul/band_far19.csv',band_data,delimiter=',')
#plot_bandstructure(leads[0].finalized(), momenta, dict(Mex=0))
#plot_bandstructure(leads[1].finalized(), momenta, dict(Mex=0))

# Attach the leads to the system.
for lead in leads:
	syst.attach_lead(lead,add_cells = 0)

# Then, plot the system with leads.
#kwant.plot(syst, site_color=family_colors, site_lw=0.1, lead_site_lw=0, colorbar=False)

syst = syst.finalized()

#params = dict(U = 0, U_disorder = U_disorder, Mex = 0, Px = Px, Py = Py, Pz = Pz, salt = 0)
#test_smatrix = kwant.smatrix(syst,energy=1.2,params=params)
#raise Exception

local_dos = kwant.ldos(syst,energy=E,params=params)
#kwant.plotter.map(syst, local_dos[1::2], vmax=0.04, num_lead_cells=0, a=1/sqrt(3))
#kwant.plotter.density(syst, local_dos[1::2], vmax=0.2)
#raise Exception('HENLO')

def compute_Pv(smatrix) :     
	mat = smatrix.submatrix(1,0)
	Trans = [[],[]]
	num = smatrix.num_propagating(1) 
	for i in range(smatrix.num_propagating(1)) :
		if smatrix.lead_info[1].momenta[num+i] <= 0 :
			Trans[0].append(np.sum((abs(mat[i:i+1,:]))**2))
		else :    
			Trans[1].append(np.sum((abs(mat[i:i+1,:]))**2))

	return np.sum(np.sum(Trans)), np.sum(Trans[0]), np.sum(Trans[1])

def compute_T(smatrix) :     
	mat = smatrix.submatrix(1,0)
	Trans = [0,0,0,0] # Tnn , Tnp, Tpn, Tpp
	num1 = smatrix.num_propagating(1) 
	num0 = smatrix.num_propagating(0) 
	for i in range(smatrix.num_propagating(1)) :
		for j in range(smatrix.num_propagating(0)) :
			if (smatrix.lead_info[1].momenta[num1+i] <= 0) and (smatrix.lead_info[0].momenta[num0+j] <= 0) :
				Trans[0] = Trans[0] + np.abs(mat[i,j]**2)  
			elif (smatrix.lead_info[1].momenta[num1+i] <= 0) and (smatrix.lead_info[0].momenta[num0+j] > 0) :
				Trans[1] = Trans[1] + np.abs(mat[i,j]**2)  
			elif (smatrix.lead_info[1].momenta[num1+i] > 0) and (smatrix.lead_info[0].momenta[num0+j] <= 0) :
				Trans[2] = Trans[2] + np.abs(mat[i,j]**2) 
			else :    
				Trans[3] = Trans[3] + np.abs(mat[i,j]**2)
			 
	return Trans[0],Trans[1],Trans[2],Trans[3]


num_L = 13
disarr = np.linspace(0,24,num_L)
#Larr = np.linspace(30,100,num_L)
#Uarr = np.linspace(-1.5*(np.abs(delta)-lambda_so),1.5*(np.abs(delta)-lambda_so),num_U) + E
Tnn, Tnp, Tpn, Tpp = [], [], [], []
for dis in disarr :
	datann, datanp, datapn, datapp = [], [], [], []
	for salt in range(0,50,1) : 
		#syst, leads, dum_lead = silicene.make_system(W = W, L = L, delta = delta, t = t, lambda_so = lambda_so)
		#for lead in leads:
		#	syst.attach_lead(lead)
		#syst = syst.finalized()
		params = dict(U = U, U_disorder = dis*U_disorder, Mex = 0, Px = Px, Py = Py, Pz = Pz,salt = salt)
		smatrix = kwant.smatrix(syst,energy=E,params=params)
		datapoint = compute_T(smatrix)
		print(dis*U_disorder," ",datapoint)
		datann = datann + [datapoint[0]]
		datanp = datanp + [datapoint[1]]
		datapn = datapn + [datapoint[2]]
		datapp = datapp + [datapoint[3]] 
	Tnn = Tnn+[sum(datann)/len(datann)]
	Tnp = Tnp+[sum(datanp)/len(datanp)]
	Tpn = Tpn+[sum(datapn)/len(datapn)]
	Tpp = Tpp+[sum(datapp)/len(datapp)]
	print(Tnn," ",Tnp," ",Tpn," ",Tpp," ", sum(datann)/len(datann)+sum(datanp)/len(datanp)+sum(datapn)/len(datapn)+sum(datapp)/len(datapp)) 

valley_trans_data = np.zeros((num_L,5))
valley_trans_data[:,0] = disarr*U_disorder
valley_trans_data[:,1] = np.array(Tnn)
valley_trans_data[:,2] = np.array(Tnp)
valley_trans_data[:,3] = np.array(Tpn)
valley_trans_data[:,4] = np.array(Tpp)
#np.savetxt('/home/bmlabserver05/Koustav/topo_paper_7jul/W_comp.csv',valley_trans_data,delimiter=',') 

pyplot.figure()
pyplot.plot(disarr*U_disorder, Tnn)
pyplot.plot(disarr*U_disorder, Tnp)
pyplot.plot(disarr*U_disorder, Tpn)
pyplot.plot(disarr*U_disorder, Tpp)
pyplot.legend(['Tnn','Tnp','Tpn','Tpp'])
pyplot.show()

Tnn = np.array(Tnn)
Tnp = np.array(Tnp)
Tpn = np.array(Tpn)
Tpp = np.array(Tpp)
pyplot.figure()
pyplot.plot(disarr*U_disorder, (Tpp+Tpn-Tnp-Tnn)/(Tpp+Tnn+Tpn+Tnp))
pyplot.show()

# syst, leads, dum_lead = silicene.make_system(W = W, L = L, delta = delta, t = t, lambda_so = lambda_so)
# for lead in leads:
# 	syst.attach_lead(lead)
# syst = syst.finalized()