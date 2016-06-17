import numpy as np
from scipy import integrate

def model_names():
	"""Provides a list containing the 8 bad-measurement model names, sorted in order of increasing dimensionality"""
	return ['all-good','P_bad-flat','free-n_all-bad','free-Q_all-bad','free-F_all-bad', 'free-n', 'free-Q', 'free-F']

def model_grids(model, nsteps_fgood=101,
				min_mu0=1.0, max_mu0=10.0, nsteps_mu0=901,
				min_n=1.0, max_n=4.0, nsteps_n=21,
				min_Q=0.0, max_Q=1.0, nsteps_Q=21,
				min_F=0.0, max_F=1.0, nsteps_F=21):

	if model == 'all-good':
		mu0 = np.linspace(min_mu0, max_mu0, nsteps_mu0)
		fgood = 1.0
		n = 1.0
		Q = 0.0
		F = 0.0
		ptype=0.0
		flat_prior = 1/(max_mu0 - min_mu0)
		deltas = (mu0[1]-mu0[0])
	elif model == 'P_bad-flat':
		mu0, fgood = np.indices((nsteps_mu0,nsteps_fgood))
		mu0 = mu0/(nsteps_mu0-1.)*(max_mu0-min_mu0)+min_mu0
		fgood = fgood/(nsteps_fgood-1.)
		n = 1.0
		Q = 0.0
		F = 0.0
		ptype=1.0
		flat_prior = 1./(max_mu0-min_mu0)
		deltas = (mu0[1,0]-mu0[0,0])*(fgood[0,1]-fgood[0,0])
	elif model == 'free-n_all-bad':
		mu0, n = np.indices((nsteps_mu0,nsteps_n))
		mu0 = mu0/(nsteps_mu0-1.)*(max_mu0-min_mu0)+min_mu0
		fgood = 0.
		n = n/(nsteps_n-1.)*(max_n-min_n)+min_n
		Q = 0.0
		F = 0.0
		ptype=0.0
		flat_prior = 1/(max_mu0 - min_mu0)/(max_n - min_n)
		deltas = (mu0[1,0]-mu0[0,0])*(n[0,1]-n[0,0])
	elif model == 'free-n':
		mu0, fgood, n = np.indices((nsteps_mu0,nsteps_fgood,nsteps_n))
		mu0 = mu0/(nsteps_mu0-1.)*(max_mu0-min_mu0)+min_mu0
		fgood = fgood/(nsteps_fgood-1.)
		n = n/(nsteps_n-1.)*(max_n-min_n)+min_n
		Q = 0.0
		F = 0.0
		ptype=0.0
		flat_prior = 1/(max_mu0 - min_mu0)/(max_n - min_n)
		deltas = (mu0[1,0,0]-mu0[0,0,0])*(fgood[0,1,0]-fgood[0,0,0])*(n[0,0,1]-n[0,0,0])
	elif model == 'free-Q_all-bad':
		mu0, Q = np.indices((nsteps_mu0,nsteps_Q))
		mu0 = mu0/(nsteps_mu0-1.)*(max_mu0-min_mu0)+min_mu0
		fgood = 0.
		Q = Q/(nsteps_Q-1.)*(max_Q-min_Q)+min_Q
		n = 1.0
		F = 0.0
		ptype=0.0
		flat_prior = 1/(max_mu0 - min_mu0)/(max_Q - min_Q)
		deltas = (mu0[1,0]-mu0[0,0])*(Q[0,1]-Q[0,0])
	elif model == 'free-Q':
		mu0, fgood, Q = np.indices((nsteps_mu0,nsteps_fgood,nsteps_Q))
		mu0 = mu0/(nsteps_mu0-1.)*(max_mu0-min_mu0)+min_mu0
		fgood = fgood/(nsteps_fgood-1.)
		Q = Q/(nsteps_Q-1.)*(max_Q-min_Q)+min_Q
		n = 1.0
		F = 0.0
		ptype=0.0
		flat_prior = 1/(max_mu0 - min_mu0)/(max_Q - min_Q)
		deltas = (mu0[1,0,0]-mu0[0,0,0])*(fgood[0,1,0]-fgood[0,0,0])*(Q[0,0,1]-Q[0,0,0])
	elif model == 'free-F_all-bad':
		mu0, F = np.indices((nsteps_mu0,nsteps_F))
		mu0 = mu0/(nsteps_mu0-1.)*(max_mu0-min_mu0)+min_mu0
		fgood = 0.
		F = F/(nsteps_F-1.)*(max_F-min_F)+min_F
		n = 1.0
		Q = 0.0
		ptype=0.0
		flat_prior = 1/(max_mu0 - min_mu0)/(max_F - min_F)
		deltas = (mu0[1,0]-mu0[0,0])*(F[0,1]-F[0,0])
	elif model == 'free-F':
		mu0, fgood, F = np.indices((nsteps_mu0,nsteps_fgood,nsteps_F))
		mu0 = mu0/(nsteps_mu0-1.)*(max_mu0-min_mu0)+min_mu0
		fgood = fgood/(nsteps_fgood-1.)
		F = F/(nsteps_F-1.)*(max_F-min_F)+min_F
		n = 1.0
		Q = 0.0
		ptype=0.0
		flat_prior = 1/(max_mu0 - min_mu0)/(max_F - min_F)
		deltas = (mu0[1,0,0]-mu0[0,0,0])*(fgood[0,1,0]-fgood[0,0,0])*(F[0,0,1]-F[0,0,0])
	else:
		print("The model name was not specified correctly")
		return
	return (mu0, fgood, n, Q, F, ptype, flat_prior, deltas)


def likelihood(means, sigmas, median_mean, mu0, fgood, n, Q, F, ptype):
	#all estimates are assumed statistically independent
	#all mean/sigma pairs are assumed to correspond to Gaussian distributions
	ln_likelihood = np.zeros_like(mu0)
	for mean,sigma in zip(means,sigmas):
		if ptype==0.0:
			sigma_bad = np.maximum( np.sqrt( (n*sigma)**2 + (Q*median_mean)**2 ), F*median_mean )
			ln_likelihood += np.log(fgood/np.sqrt(2.0*np.pi*sigma**2)*np.exp(-(mu0-mean)**2/2./sigma**2) + (1.-fgood)/np.sqrt(2.0*np.pi*sigma_bad**2)*np.exp(-(mu0-mean)**2/2./sigma_bad**2) )
		elif ptype==1.0:
			ln_likelihood += np.log(fgood/np.sqrt(2.0*np.pi*sigma**2)*np.exp(-(mu0-mean)**2/2./sigma**2) + (1.-fgood)/(np.max(mu0)-np.min(mu0)) )
		else:
			print("Must specify: ptype=1 for P_bad-flat model, else ptype=0 for all other models")
			return			
	return np.exp(ln_likelihood)


def posterior_evidence(likelihood, prior, deltas):
	
	#integrate over N-dimensional evenly-spaced grid of posterior values
	evidence = likelihood*prior*deltas
	while evidence.size != 1:
		evidence = integrate.simps(evidence)
	posterior = likelihood*prior/evidence
	
	return (posterior, evidence)


def info_criteria(likelihood, n_data):

	n_params = likelihood.ndim

	AIC = -2.0*np.max(np.log(likelihood)) + 2.0*n_params
	AICc = AIC + (2.0*n_params*(n_params+1.0)/(n_data - n_params - 1))
	BIC = -2.0*np.max(np.log(likelihood)) + n_params*np.log(n_data)

	return (AIC, AICc, BIC)

def bayes_factor(model1_evidence, model2_evidence):
	return model1_evidence/model2_evidence

def print_model_results(model_name, means, sigmas, header=True):

	med_mean = np.median(means)
	n_data = float(np.size(means))
	n_params = likelihood.ndim

	model_grids = model_grids(model_name)
	likelihood = likelihood(means, sigmas, med_mean, *model_grids[0:6])
	posterior, evidence = posterior_evidence(likelihood, *model_grids[6:8])
	AIC, AICc, BIC = info_criteria(likelihood, n_data)

	marg_post = 1.0*posterior
	while marg_post.ndim != 1:
		marg_post = integrate.simps(marg_post)

	cdf = np.zeros_like(marg_post)
	for j in range(1,cdf.size):
		cdf[j] = integrate.simps(p_ld[0:j+1],ld[0:j+1])
	interp_from_cdf = interp1d(cdf,ld)
	cdf_prctls = (int_cdf(0.5), int_cdf(0.84)-int_cdf(0.5), int_cdf(0.5)-int_cdf(0.16))

	if header==True:
		print 'model (M_k) \t N_free \t P(mu0 | D, M_k) \t log evidence \t AIC \t AICc \t BIC'
	print model_name+'\t %1i \t %.2f^{+%.2f}_{-%.2f} \t %.2f \t %.2f \t %.2f \t %.2f' % \
			(n_params, interp_from_cdf(0.5), interp_from_cdf(0.84)-interp_from_cdf(0.5), 
				interp_from_cdf(0.5)-interp_from_cdf(0.16), np.log10(evidence), AIC, AICc, BIC)

	return




