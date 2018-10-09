#include "pch.h"
#include "reader.h"

constexpr auto MAXBUFSIZE = ((int) 6e4);

Eigen::MatrixXd readMatrix(const char *filename)
{
	int cols = 0, rows = 0;
	double buff[MAXBUFSIZE];

	// Read numbers from file into buffer.
	std::ifstream infile;
	infile.open(filename);
	while (!infile.eof())
	{
		std::string line;
		std::getline(infile, line);
		int temp_cols = 0;
		std::stringstream stream(line);
		while (!stream.eof())
			stream >> buff[cols*rows + temp_cols++];
		if (temp_cols == 0)
			continue;
		if (cols == 0)
			cols = temp_cols;
		rows++;
	}
	infile.close();

	rows--;

	// Populate matrix with numbers.
	Eigen::MatrixXd result(rows, cols);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			result(i, j) = buff[cols*i + j];

	return result;
};