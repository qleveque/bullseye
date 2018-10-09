#include "pch.h"
#include "functions.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

//multilogit
double phi_multilogit(VectorXd const& z, VectorXd const& y)
{
	VectorXd z_ = (z - z.maxCoeff()*MatrixXd::Ones(z.rows(), 1)).array().exp();
	double ans = -1 * log((y.transpose()*z_)(0,0)) / (z_.sum());
	return ans;
}
VectorXd grad_phi_multilogit(VectorXd const& z, VectorXd const& y)
{
	VectorXd z_ = (z - z.maxCoeff()*VectorXd::Ones(z.rows())).array().exp();
	VectorXd ans = -y + z_ / z_.sum();
	return ans;
}
MatrixXd hess_phi_multilogit(VectorXd const& z, VectorXd const& y)
{
	VectorXd z_ = (z - z.maxCoeff()*VectorXd::Ones(z.rows())).array().exp();
	VectorXd probs = z_ / z_.sum();
	MatrixXd ans = MatrixXd(probs.asDiagonal()) - probs * probs.transpose();
	return ans;
}

//normal
double phi_normal(VectorXd const& z, VectorXd const& y)
{
	return pow(z(0, 0),2)/2.0;
}
VectorXd grad_phi_normal(VectorXd const& z, VectorXd const& y)
{
	return z;
}
MatrixXd hess_phi_normal(VectorXd const& z, VectorXd const& y)
{
	return MatrixXd::Ones(1,1);
}
