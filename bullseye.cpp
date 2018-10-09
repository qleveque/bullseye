#include "pch.h"
#include "bullseye.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

MatrixXd A(MatrixXd const& X, size_t i, size_t p, size_t k)
{
	//return kroneckerProduct(MatrixXd::Identity(k, k), X.row(i)); TODO does not work, no idea why
	MatrixXd ans = MatrixXd::Zero(p, k);
	VectorXd x = X.row(i);
	size_t d = x.rows();
	for (size_t j(0); j < p; ++j)
		ans(j, j / d) = x(j%d, 0);
	return ans;
}

local_params_struct local_params(VectorXd const& y, MatrixXd const& mu, MatrixXd const& sigma, MatrixXd A,
	double(*phi)(VectorXd const&, Eigen::VectorXd const&),
	VectorXd(*grad_phi)(VectorXd const&, Eigen::VectorXd const&),
	MatrixXd(*hess_phi)(VectorXd const&, Eigen::VectorXd const&)
	)
{
	//for a given observation, compute the local parameters

	size_t local_d = A.cols();
	size_t s = 50;

	VectorXd local_mu = A.transpose() * mu; //[k,1]
	MatrixXd local_cov = A.transpose() * sigma * A; //[k,k] TODO

	Eigen::BDCSVD<MatrixXd> dec = Eigen::BDCSVD<MatrixXd>::BDCSVD(local_cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
	MatrixXd local_std = dec.matrixU() * dec.singularValues().array().sqrt().matrix().asDiagonal() * dec.matrixV().transpose();

	//TODO
	sample_struct samp = generate_multivariate_normal_sample(MatrixXd::Zero(local_d,1), MatrixXd::Identity(local_d,local_d), s);
	
	double e(0);
	VectorXd r(MatrixXd::Zero(local_d,1));
	MatrixXd B(MatrixXd::Zero(local_d,local_d));

	for (size_t i(0); i < s; ++i)
	{
		VectorXd a = local_mu + local_std * samp.points.col(i);
		double w = samp.weights[i];
		e+=w*(*phi)(a, y); //[]
		r+=w*(*grad_phi)(a, y); //[d_loc,1]
		B+=w*(*hess_phi)(a, y); //[d_loc,d_loc]
	}

	return { e,A*r,A*B*A.transpose()};
}

global_params_struct global_params(MatrixXd const& x_array, MatrixXd const& y_array,
	MatrixXd const& mu, MatrixXd const& sigma, vector<MatrixXd> & As, VectorXd const& prior_std,
	double(*phi)(VectorXd const&, VectorXd const&),
	VectorXd(*grad_phi)(VectorXd const&, VectorXd const&),
	MatrixXd(*hess_phi)(VectorXd const&, VectorXd const&),
	double(*phi_p)(VectorXd const&, VectorXd const&),
	VectorXd(*grad_phi_p)(VectorXd const&, VectorXd const&),
	MatrixXd(*hess_phi_p)(VectorXd const&, VectorXd const&))
{
	//compute e,rho,beta for prior and likelihood
	double e(0);
	size_t p(x_array.cols() * y_array.cols());
	size_t n(y_array.rows());

	VectorXd rho(MatrixXd::Zero(p, 1));
	MatrixXd beta(MatrixXd::Zero(p,p));
	MatrixXd prior_sigma = prior_std.asDiagonal();
	MatrixXd prior_As = prior_std.array().inverse().matrix().asDiagonal();

	//likelihood
	for (size_t i(0); i < n; ++i)
	{
		local_params_struct p_likelihood = local_params(y_array.row(i), mu, sigma, As[i],
			phi, grad_phi, hess_phi);
		e += p_likelihood.e, rho += p_likelihood.rho, beta += p_likelihood.beta;
	}
	//prior
	for (size_t j(0); j < p; ++j)
	{
		local_params_struct p_prior = local_params(MatrixXd::Zero(1, 1), mu, sigma, prior_As.col(j),
			phi_p, grad_phi_p, hess_phi_p);
		e += p_prior.e, rho += p_prior.rho, beta += p_prior.beta;
	}

	double new_ELBO = -e + 0.5 * log(sigma.determinant()) + 0.5 * log(2 * M_PI * M_E);

	return { e, rho, beta, new_ELBO };
}
void bullseye_iteration(MatrixXd const& x_array, MatrixXd const& y_array, MatrixXd & mu, MatrixXd & sigma,
	vector<MatrixXd> & As, VectorXd const& prior_std, double& ELBO)
{

	//compute e, ρ, β and ELBO
	global_params_struct p = global_params(x_array, y_array, mu, sigma, As, prior_std,
		&phi_multilogit, &grad_phi_multilogit, &hess_phi_multilogit,
		&phi_normal, &grad_phi_normal, &hess_phi_normal
	);

	//init
	double step_size(0.5);
	int nb_tries(0);

	MatrixXd sigma_c = -p.beta.inverse();
	MatrixXd mu_c = mu - p.beta.inverse() * p.rho;

	MatrixXd mu_new = mu_c;
	MatrixXd sigma_new = sigma_c;

	double ELBO_new = p.ELBO;

	while(ELBO_new < ELBO)
	{
		cout << "not accepted : " << ELBO_new << " " << ELBO << endl;
		++nb_tries;
		double gamma(pow(step_size, nb_tries));
		mu_new = mu + gamma *(mu_c - mu);
		sigma_new = sigma + gamma * (sigma_c - sigma);

		global_params_struct pnew = global_params(x_array, y_array, mu_new, sigma_new, As, prior_std,
			&phi_multilogit, &grad_phi_multilogit, &hess_phi_multilogit,
			&phi_normal, &grad_phi_normal, &hess_phi_normal
		);

		ELBO_new = pnew.ELBO;
	}
	cout << "accepted : " << ELBO_new << " " << ELBO << endl;
	mu = mu_new;
	sigma = sigma_new;
	ELBO = ELBO_new;
}