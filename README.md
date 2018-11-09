# Bullseye!

"Bullseye!" is a new algorithm for computing the Gaussian Variational Approximation of a target distribution. Its strong point lies in the fact that it can easily be parallelized and distributed.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Installing

Bullseye! is now available as a [PyPI package](https://pypi.python.org/pypi/bullseye_method/):

```
pip install bullseye_method
```

or clone the repository (no installation required, dependencies will be installed automatically):

```
git clone https://github.com/Whenti/bullseye
```

or [download and extract the zip](https://github.com/Whenti/bullseye/archive/master.zip) into your project folder.

## Running the tests

To see if everything is working properly, you can already run the algorithm on a multilogit model with artificially generated data.

```py
from Bullseye.Tests import simple_test
simple_test()
```

## Example

```py
import Bullseye
from Bullseye import generate_multilogit

theta_0, x_array, y_array = generate_multilogit(d = 10, n = 10**3, k = 5)

bull = Bullseye.Graph()
bull.feed_with(x_array,y_array)
bull.set_model("multilogit")
bull.init_with(mu_0 = 0, cov_0 = 1)
bull.set_options(local_std_trick = True,
                 keep_1d_prior = True)
bull.build()

bull.run()
```

## Authors

* **Quentin Lévêque** [Whenti](https://github.com/Whenti)
* **Guillaume Dehaene**

See also the list of [contributors](https://github.com/Whenti/bullseye/contributors) who participated in this project. Hopefully, there will be more.

## License

This project is proudly licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.