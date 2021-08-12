import kwant
from math import pi,sqrt,tanh
import numpy as np
from matplotlib import pyplot
from kwant.digest import uniform
from types import SimpleNamespace

pauli_z = np.array([[1,0],[0,-1]])
pauli_x = np.array([[0,1],[1,0]])
pauli_y = np.array([[0,-1j],[1j,0]])

sin_30, cos_30 = (1 / 2, sqrt(3) / 2)
graphene = kwant.lattice.general([(1, 0), (sin_30, cos_30)],
								 [(0, 0), (0, 1 / sqrt(3))])

a, b = graphene.sublattices

# def lin0(y,W,jw) :
# 	if y < -jw :
# 		return -2 
# 	elif -jw <= y < jw :
# 		return 2*y/jw
# 	else :
# 		return 2

def lin0(y,W,jw) :
	return tanh((y)/(jw/2))
 
def dual_lin0(x,y,W,jw,L) :
	return (0 <= x < L/2-1)*tanh((y)/(jw/2))-(L/2+1 < x <= L)*tanh((y)/(jw/2))

def lin1(y,W,jw) :
	if y < -W/4-jw :
		return -1 
	elif -W/4-jw <= y < -W/4+jw :
		return ((y+W/4)/jw - 1)/2
	elif -W/4+jw <= y < W/4-jw :
		return 0
	elif W/4-jw <= y < W/4+jw :
		return ((y-W/4)/jw + 1)/2
	else :
		return 1

def lin2(y,W,jw) :
	if y < -jw :
		return -1 
	elif -jw <= y < jw :
		return y/jw
	else :
		return 1

def tan_lin0(y,W,jw) :
	return tanh((y)/(jw/2))

def tan_lin1(y,W,jw) :
	return 0.5*(tanh((y-1*W/20)/(jw/2))+tanh((y+1*W/20)/(jw/2)))


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
		return (0 <= x <= L) and (-1*(W/2) < y <= 1*(W/2))    

	syst = kwant.Builder()
	
	del_fn = lambda y,W : tan_lin1(y,W,W/20)
	dual_del_fn = lambda x,y,W : dual_lin0(x,y,W,W/20,L) 	  	
	
	def potential(site, U, U_disorder, Mex, Px, Py, Pz, salt):
		(x, y) = site.pos
		d = -1
		if (site.family == a) :
			d = 1
		term1 = d*delta*del_fn(y,W)*np.eye(2)
		term2 = U*np.eye(2)
		term3 = Mex*pauli_z
		term4 = U_disorder * (uniform(repr(site), repr(salt)) - 0.5) * (np.eye(2)+Pz*pauli_z+Px*pauli_x+Py*pauli_y)
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

	def lead0_shape(pos):
		x, y = pos
		return (-W/2 < y <= W/2)

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

	def lead1_shape(pos):
		x, y = pos
		return (-W/2 < y <= W/2)

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

	def dum_lead_shape(pos):
		x, y = pos
		return (-W/2 < y <= W/2)

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