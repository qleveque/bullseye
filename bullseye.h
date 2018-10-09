#ifndef BULLSEYE_H
#define BULLSEYE_H

#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <math.h>

#include "sampling.h"
#include "functions.h"
#include <Eigen/Dense>
#include <Eigen/SVD>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

struct local_params_struct
{
	double e;
	Eigen::VectorXd rho;
	Eigen::MatrixXd beta;
};

struct global_params_struct
{
	double e;
	Eigen::VectorXd rho;
	Eigen::MatrixXd beta;
	double ELBO;
};

Eigen::MatrixXd A(Eigen::MatrixXd const& X, size_t i, size_t p, size_t k);

global_params_struct global_params(Eigen::MatrixXd const& x, Eigen::MatrixXd const& y,
	Eigen::MatrixXd const& mu, Eigen::MatrixXd const& sigma, std::vector<Eigen::MatrixXd> & As, Eigen::VectorXd const& prior_std,
	double(*phi)(Eigen::VectorXd const&, Eigen::VectorXd const&),
	Eigen::VectorXd(*grad_phi)(Eigen::VectorXd const&, Eigen::VectorXd const&),
	Eigen::MatrixXd(*hess_phi)(Eigen::VectorXd const&, Eigen::VectorXd const&),
	double(*phi_p)(Eigen::VectorXd const&, Eigen::VectorXd const&),
	Eigen::VectorXd(*grad_phi_p)(Eigen::VectorXd const&, Eigen::VectorXd const&),
	Eigen::MatrixXd(*hess_phi_p)(Eigen::VectorXd const&, Eigen::VectorXd const&)
);
local_params_struct local_params(Eigen::VectorXd const& y, Eigen::MatrixXd const& mu,
	Eigen::MatrixXd const& sigma, Eigen::MatrixXd A,
	double(*phi)(Eigen::VectorXd const&, Eigen::VectorXd const&),
	Eigen::VectorXd(*grad_phi)(Eigen::VectorXd const&, Eigen::VectorXd const&),
	Eigen::MatrixXd(*hess_phi)(Eigen::VectorXd const&, Eigen::VectorXd const&)
	);
void bullseye_iteration(Eigen::MatrixXd const& x_array, Eigen::MatrixXd const& y_array, Eigen::MatrixXd & mu,
	Eigen::MatrixXd & sigma, std::vector<Eigen::MatrixXd> & As, Eigen::VectorXd const& prior_std, double& ELBO);

#endif