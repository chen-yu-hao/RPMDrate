files=rpmdrate/_main.pyf rpmdrate/_math.f90 rpmdrate/_surface.f90 rpmdrate/_main.f90 rpmdrate/blas_lapack.f90
compiler=--fcompiler=gnu95 --f90flags='-O3 -ffree-line-length-none'
python=python3

.PHONY: all build install test clean drpmd-native drpmd-test

all: build

build:

	$(python) -m numpy.f2py -c rpmdrate/_surface.f90 -m rpmdrate._surface 
	$(python) -m numpy.f2py -c $(files) $(compiler)
	mv _main*.so rpmdrate/

test:

	$(python) rpmdrate/test.py

# dRPMD is usable without building the legacy NumPy/f2py extensions.
drpmd-native:

	$(python) -m rpmdrate.drpmd_integrator --build

drpmd-test: drpmd-native

	$(python) -m unittest discover -s tests -p 'drpmd*test.py' -v

clean:
	rm -rf build/
	rm -rf rpmdrate/*.pyc
	rm -rf rpmdrate/*.pyo
	rm -rf rpmdrate/*.pyd
	rm -rf rpmdrate/*.so 
	rm -rf rpmdrate/__pycache__/*pyc
