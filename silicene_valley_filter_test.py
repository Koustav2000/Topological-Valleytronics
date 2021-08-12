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
	pyplot.ylim([-1,1])
	pyplot.show()

W = 40*sqrt(3)
L = 40
t = 1.6
E = 0.5
U = E
lambda_so = 0.2
delta = -1*(lambda_so)
U_disorder = -0*(np.abs(delta)-lambda_so)
Px = 0
Py = 0
Pz = 0


params = dict(U = U, U_disorder = U_disorder, Mex = 0, Px = Px, Py = Py, Pz = Pz, salt=0)


syst, leads, dum_lead = silicene.make_system(W = W, L = L, delta = delta, t = t, lambda_so = lambda_so)

def family_colors(site):
	return 0 if (site.family == silicene.a) else 1
# Plot the closed system without leads.
#kwant.plot(syst, site_color=family_colors, site_lw=0.1, colorbar=False)

ham = syst.finalized().hamiltonian_submatrix(params=params)


# Compute the band structure of dum_lead.
momenta = np.linspace(0,2*pi,100)
#plot_bandstructure(dum_lead.finalized(), momenta, dict(Mex = 0))
# plot_bandstructure(leads[1].finalized(), momenta)

# Attach the leads to the system.
for lead in leads:
	syst.attach_lead(lead)

# Then, plot the system with leads.
#kwant.plot(syst, site_color=family_colors, site_lw=0.1, lead_site_lw=0, colorbar=False)

syst = syst.finalized()


local_dos = kwant.ldos(syst,energy=E,params=params)
kwant.plotter.map(syst, local_dos[1::2], num_lead_cells=0, a=1/sqrt(3))
#kwant.plotter.density(syst, local_dos[1::2])

def compute_Pv(smatrix) :     
	mat = smatrix.submatrix(0,1)
	Trans = [[],[]]
	for i in range(smatrix.num_propagating(0)) :
		if smatrix.lead_info[0].momenta[i] <= 0 :
			Trans[0].append(np.sum((abs(mat[i:i+1,:]))**2))
		else :    
			Trans[1].append(np.sum((abs(mat[i:i+1,:]))**2))

	return np.sum(np.sum(Trans)), np.sum(Trans[0]), np.sum(Trans[1])

Uarr = np.linspace(-np.abs(delta*1.5),np.abs(delta*1.5),30) + E
data, datan, datap = [], [], []
for U in Uarr :
	params = dict(U = U, U_disorder = U_disorder, Mex = 0, Px = Px, Py = Py, Pz = Pz)
	smatrix = kwant.smatrix(syst,energy=E,params=params)
	datapoint = compute_Pv(smatrix)
	print(E-U," ",datapoint)
	data = data + [datapoint[0]]
	datan = datan + [datapoint[1]]
	datap = datap + [datapoint[2]]


pyplot.figure()
pyplot.plot(E-Uarr, data)
pyplot.show()

datan = np.array(datan)
datap = np.array(datap)
pyplot.figure()
pyplot.plot(E-Uarr, (datap-datan)/(datap+datan))
pyplot.show()

# syst, leads, dum_lead = silicene.make_system(W = W, L = L, delta = delta, t = t, lambda_so = lambda_so)
# for lead in leads:
# 	syst.attach_lead(lead)
# syst = syst.finalized()