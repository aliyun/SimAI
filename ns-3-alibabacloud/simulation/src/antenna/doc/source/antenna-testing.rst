+++++++++++++++++++++++++++++++++++++
 Testing Documentation
+++++++++++++++++++++++++++++++++++++

In this section we describe the test suites included with the antenna
module that verify its correct functionality. 


Angles
------

The unit test suite ``angles`` verifies that the Angles class is
constructed properly by correct conversion from 3D Cartesian
coordinates according to the available methods (construction from a
single vector and from a pair of vectors). For each method, several
test cases are provided that compare the values :math:`(\phi, \theta)`
determined by the constructor to known reference values. The test
passes if for each case the values are equal to the reference up to a
tolerance of :math:`10^{-10}` which accounts for numerical errors.


DegreesToRadians
----------------

The unit test suite ``degrees-radians`` verifies that the methods
``DegreesToRadians`` and ``RadiansToDegrees`` work properly by
comparing with known reference values in a number of test
cases. Each test case passes if the comparison is equal up to a
tolerance of :math:`10^{-10}` which accounts for numerical errors.



IsotropicAntennaModel
---------------------

The unit test suite ``isotropic-antenna-model`` checks that the
``IsotropicAntennaModel`` class works properly, i.e., returns always a
0dB gain regardless of the direction.



CosineAntennaModel
------------------

The unit test suite ``cosine-antenna-model`` checks that the
``CosineAntennaModel`` class works properly. Several test cases are
provided that check for the antenna gain value calculated at different
directions and for different values of the orientation, the reference
gain and the beamwidth. The reference gain is calculated by hand. Each
test case passes if the reference gain in dB is equal to the value returned
by ``CosineAntennaModel`` within a tolerance of 0.001, which accounts
for the approximation done for the calculation of the reference
values.



ParabolicAntennaModel
---------------------

The unit test suite ``parabolic-antenna-model`` checks that the
``ParabolicAntennaModel`` class works properly. Several test cases are
provided that check for the antenna gain value calculated at different
directions and for different values of the orientation, the maximum attenuation
and the beamwidth. The reference gain is calculated by hand. Each
test case passes if the reference gain in dB is equal to the value returned
by ``ParabolicAntennaModel`` within a tolerance of 0.001, which accounts
for the approximation done for the calculation of the reference
values.






 
 
