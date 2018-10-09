#ifndef READER_H
#define READER_H

#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/Dense>

Eigen::MatrixXd readMatrix(const char *filename);

#endif