#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <Eigen/Dense>
#include <math.h>
#include <cmath>

double phi_multilogit(Eigen::VectorXd const& z, Eigen::VectorXd const& y = Eigen::VectorXd::Zero(1, 1));
Eigen::VectorXd grad_phi_multilogit(Eigen::VectorXd const& z, Eigen::VectorXd const& y = Eigen::VectorXd::Zero(1, 1));
Eigen::MatrixXd hess_phi_multilogit(Eigen::VectorXd const& z, Eigen::VectorXd const& y = Eigen::VectorXd::Zero(1, 1));

double phi_normal(Eigen::VectorXd const& z, Eigen::VectorXd const& y = Eigen::VectorXd::Zero(1, 1));
Eigen::VectorXd grad_phi_normal(Eigen::VectorXd const& z, Eigen::VectorXd const& y = Eigen::VectorXd::Zero(1, 1));
Eigen::MatrixXd hess_phi_normal(Eigen::VectorXd const& z, Eigen::VectorXd const& y = Eigen::VectorXd::Zero(1, 1));


#endif