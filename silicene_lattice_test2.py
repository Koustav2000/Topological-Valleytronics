import kwant
from math import pi,sqrt,tanh
import numpy as np
from matplotlib import pyplot
from kwant.digest import uniform
from types import SimpleNamespace

pauli_z = np.array([[1,0],[0,-1]])
sin_30, cos_30 = (1 / 2, sqrt(3) / 2)
graphene = kwant.lattice.general([(1, 0), (sin_30, cos_30)],
								 [(0, 0), (0, 1 / sqrt(3))])

a, b = graphene.sublattices

def lin0(y,W,jw) :
	if y < -jw :
		return -2 
	elif -jw <= y < jw :
		return 2*y/jw
	else :
		return 2

def lin1(y,W,jw) :
	if y < -W/6-jw :
		return -2 
	elif -W/6-jw <= y < -W/6+jw :
		return (y+W/6)/jw - 1
	elif -W/6+jw <= y < W/6-jw :
		return 0
	elif W/6-jw <= y < W/6+jw :
		return (y-W/6)/jw + 1
	else :
		return 2

def lin2(y,W,jw) :
	if y < -jw :
		return 0 
	elif -jw <= y < jw :
		return y/jw+1
	else :
		return 2

def tan_lin0(y,W,jw) :
	return tanh((y)/(jw/2))

def tan_lin1(y,W,jw) :
	return 0.5*(tanh((y-1*W/6)/(jw/2))+tanh((y+1*W/6)/(jw/2)))


def make_system(W = 10*sqrt(3), L = 10, delta = 0, t = 1.6, lambda_so = 0) :
	
	lambda_so = 1j*lambda_so/(3*sqrt(3))
	t_nn1_a = 1 * lambda_so * pauli_z # [ 1,  0]
	t_nn1_b = -1 * lambda_so * pauli_z
	t_nn2_a = -1 * lambda_so * pauli_z # [ 0,  1]
	t_nn2_b = 1 * lambda_so * pauli_z
	t_nn3_a = -1 * lambda_so * pauli_z # [ 1,  -1]
	t_nn3_b = 1 * lambda_so * pauli_z

	def channel(pos):
		x, y = pos
		return (0 <= x <= L) and (-2*W/3 < y <= 2*W/3)    

	syst = kwant.Builder()
	
	del_fn = lambda y,W : tan_lin1(y,W,W/20)  	
	
	def potential(site, U, U_disorder, Mex, salt):
		(x, y) = site.pos
		d = -1
		if (site.family == a) :
			d = 1
		term1 = d*delta*del_fn(y,W)*np.eye(2)
		term2 = U*np.eye(2)
		term3 = Mex*pauli_z
		term4 = U_disorder * (np.random.uniform() - 0.5) * np.eye(2)
		return term1 + term2 + term3 + term4
    


	def dummy(site, Mex):
		(x, y) = site.pos
		d = -1
		if (site.family == a) :
			d = 1
		term1 = d*delta*del_fn(y,W)*np.eye(2)
		term2 = Mex*pauli_z
		return term1 + term2


	syst[graphene.shape(channel, (0, 0))] = potential
	hoppings = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
	syst[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -t*np.eye(2)
	syst[kwant.builder.HoppingKind((1, 0), a, a)] = t_nn1_a
	syst[kwant.builder.HoppingKind((1, 0), b, b)] = t_nn1_b
	syst[kwant.builder.HoppingKind((0, 1), a, a)] = t_nn2_a
	syst[kwant.builder.HoppingKind((0, 1), b, b)] = t_nn2_b
	syst[kwant.builder.HoppingKind((1, -1), a, a)] = t_nn3_a
	syst[kwant.builder.HoppingKind((1, -1), b, b)] = t_nn3_b

	# left lead
	sym0 = kwant.TranslationalSymmetry(graphene.vec((-1, 0)))
	sym0.add_site_family(graphene.sublattices[0], other_vectors=[(-1, 2)])
	sym0.add_site_family(graphene.sublattices[1], other_vectors=[(-1, 2)])

	def lead0_shape(pos):
		x, y = pos
		return (-2*W/3 < y <= 2*W/3)

	lead0 = kwant.Builder(sym0)
	lead0[graphene.shape(lead0_shape, (0, 0))] = np.zeros((2,2))
	lead0[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -t*np.eye(2)
	#lead0[kwant.builder.HoppingKind((1, 0), a, a)] = t_nn1_a
	#lead0[kwant.builder.HoppingKind((1, 0), b, b)] = t_nn1_b
	#lead0[kwant.builder.HoppingKind((0, 1), a, a)] = t_nn2_a
	#lead0[kwant.builder.HoppingKind((0, 1), b, b)] = t_nn2_b
	#lead0[kwant.builder.HoppingKind((1, -1), a, a)] = t_nn3_a
	#lead0[kwant.builder.HoppingKind((1, -1), b, b)] = t_nn3_b

	# right lead
	sym1 = kwant.TranslationalSymmetry(graphene.vec((1, 0)))
	sym1.add_site_family(graphene.sublattices[0], other_vectors=[(-1, 2)])
	sym1.add_site_family(graphene.sublattices[1], other_vectors=[(-1, 2)])

	def lead1_shape(pos):
		x, y = pos
		return (-2*W/3 < y <= 2*W/3)

	lead1 = kwant.Builder(sym1)
	lead1[graphene.shape(lead1_shape, (0, 0))] = np.zeros((2,2))
	lead1[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -t*np.eye(2)
	#lead1[kwant.builder.HoppingKind((1, 0), a, a)] = t_nn1_a
	#lead1[kwant.builder.HoppingKind((1, 0), b, b)] = t_nn1_b
	#lead1[kwant.builder.HoppingKind((0, 1), a, a)] = t_nn2_a
	#lead1[kwant.builder.HoppingKind((0, 1), b, b)] = t_nn2_b
	#lead1[kwant.builder.HoppingKind((1, -1), a, a)] = t_nn3_a
	#lead1[kwant.builder.HoppingKind((1, -1), b, b)] = t_nn3_b

	# dummy lead
	dum_sym = kwant.TranslationalSymmetry(graphene.vec((1, 0)))
	dum_sym.add_site_family(graphene.sublattices[0], other_vectors=[(-1, 2)])
	dum_sym.add_site_family(graphene.sublattices[1], other_vectors=[(-1, 2)])
	
	def dum_lead_shape(pos):
		x, y = pos
		return (-2*W/3 < y <= 2*W/3)

	dum_lead = kwant.Builder(dum_sym)
	dum_lead[graphene.shape(dum_lead_shape, (0, 0))] = dummy
	dum_lead[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -t*np.eye(2)
	dum_lead[kwant.builder.HoppingKind((1, 0), a, a)] = t_nn1_a
	dum_lead[kwant.builder.HoppingKind((1, 0), b, b)] = t_nn1_b
	dum_lead[kwant.builder.HoppingKind((0, 1), a, a)] = t_nn2_a
	dum_lead[kwant.builder.HoppingKind((0, 1), b, b)] = t_nn2_b
	dum_lead[kwant.builder.HoppingKind((1, -1), a, a)] = t_nn3_a
	dum_lead[kwant.builder.HoppingKind((1, -1), b, b)] = t_nn3_b

	return syst, [lead0, lead1], dum_lead
 
W = 90*sqrt(3)
# L = 100
# t = 1.3
# grid = (1/2)*sqrt(3)*pi*(t/W)
# E = t/3
# U = E
# print(E)
# lambda_so = 0.04
# delta = -1*(0.1+lambda_so)
# print(np.abs(delta)-lambda_so)
# U_disorder = 0*(np.abs(delta)-lambda_so)
# print(U_disorder)
# Px = 0
# Py = 0
# Pz = 0
# salt = 0

# params = dict(U = U, U_disorder = U_disorder, Mex = 0, Px = Px, Py = Py, Pz = Pz, salt = salt)

# syst, leads, dum_lead = make_system(W = W, L = L, delta = delta, t = t, lambda_so = lambda_so)

# #def family_colors(site):
# #	return 0 if (site.family == a) else 1
# # Plot the closed system without leads.
# #kwant.plot(syst, site_color=family_colors, site_lw=0.1, colorbar=False)

# for lead in leads:
	# syst.attach_lead(lead)

# syst = syst.finalized()
# #kwant.plot(syst, site_lw=0.1, colorbar=False)

# print('hi')
# local_dos = kwant.ldos(syst,energy=E,params=params)
# kwant.plotter.map(syst, local_dos[1::2],vmax = 0.01, num_lead_cells=0, a=1/sqrt(3))

 
 
 
 