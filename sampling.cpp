#include "pch.h"
#include "sampling.h"

sample_struct generate_multivariate_normal_sample(Eigen::VectorXd const& mu, Eigen::MatrixXd const& sigma, unsigned int s)
{
	srand(time(NULL));
	Eigen::internal::scalar_normal_dist_op<double> randN;
	Eigen::internal::scalar_normal_dist_op<double>::rng.seed(rand()%10000);

	unsigned int size(mu.rows());
	
	Eigen::MatrixXd normTransform(size, size);
	Eigen::LLT<Eigen::MatrixXd> cholSolver(sigma);

	if (cholSolver.info() == Eigen::Success)
	{
		normTransform = cholSolver.matrixL();
	}
	else
	{
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(sigma);
		normTransform = eigenSolver.eigenvectors()
			* eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
	}

	Eigen::MatrixXd samples = (normTransform
		* Eigen::MatrixXd::NullaryExpr(size, s, randN)).colwise()
		+ mu;

	sample_struct ans;
	ans.points = samples;
	ans.weights = Eigen::ArrayXd::Ones(s, 1) * 1. / s;

	return ans;
}