.. include:: replace.txt
.. highlight:: cpp

.. _Propagation:

Propagation
-----------

The |ns3| ``propagation`` module defines two generic interfaces, namely :cpp:class:`PropagationLossModel`
and :cpp:class:`PropagationDelayModel`, to model respectively the propagation loss and the propagation delay.

PropagationLossModel
********************

Propagation loss models calculate the Rx signal power considering the Tx signal power and the
mutual Rx and Tx antennas positions.

A propagation loss model can be "chained" to another one, making a list. The final Rx power
takes into account all the chained models. In this way one can use a slow fading and a fast
fading model (for example), or model separately different fading effects.

The following propagation loss models are implemented:

   * Cost231PropagationLossModel
   * FixedRssLossModel
   * FriisPropagationLossModel
   * ItuR1411LosPropagationLossModel
   * ItuR1411NlosOverRooftopPropagationLossModel
   * JakesPropagationLossModel
   * Kun2600MhzPropagationLossModel
   * LogDistancePropagationLossModel
   * MatrixPropagationLossModel
   * NakagamiPropagationLossModel
   * OkumuraHataPropagationLossModel
   * RandomPropagationLossModel
   * RangePropagationLossModel
   * ThreeLogDistancePropagationLossModel
   * TwoRayGroundPropagationLossModel
   * ThreeGppPropagationLossModel

      * ThreeGppRMaPropagationLossModel
      * ThreeGppUMaPropagationLossModel
      * ThreeGppUmiStreetCanyonPropagationLossModel
      * ThreeGppIndoorOfficePropagationLossModel

Other models could be available thanks to other modules, e.g., the ``building`` module.

Each of the available propagation loss models of ns-3 is explained in
one of the following subsections.

FriisPropagationLossModel
=========================

This model implements the Friis propagation loss model. This model was first described in [friis]_.
The original equation was described as:

.. math::

  \frac{P_r}{P_t} = \frac{A_r A_t}{d^2\lambda^2}

with the following equation for the case of an isotropic antenna with no heat loss:

.. math::

  A_{isotr.} = \frac{\lambda^2}{4\pi}

The final equation becomes:

.. math::

  \frac{P_r}{P_t} = \frac{\lambda^2}{(4 \pi d)^2}

Modern extensions to this original equation are:

.. math::

  P_r = \frac{P_t G_t G_r \lambda^2}{(4 \pi d)^2 L}

With:

  :math:`P_t` : transmission power (W)

  :math:`P_r` : reception power (W)

  :math:`G_t` : transmission gain (unit-less)

  :math:`G_r` : reception gain (unit-less)

  :math:`\lambda` : wavelength (m)

  :math:`d` : distance (m)

  :math:`L` : system loss (unit-less)

In the implementation, :math:`\lambda` is calculated as
:math:`\frac{C}{f}`, where :math:`C = 299792458` m/s is the speed of light in
vacuum, and :math:`f` is the frequency in Hz which can be configured by
the user via the Frequency attribute.

The Friis model is valid only for propagation in free space within
the so-called far field region, which can be considered
approximately as the region for :math:`d > 3 \lambda`.
The model will still return a value for :math:`d < 3 \lambda`, as
doing so (rather than triggering a fatal error) is practical for
many simulation scenarios. However, we stress that the values
obtained in such conditions shall not be considered realistic.

Related with this issue, we note that the Friis formula is
undefined for :math:`d = 0`, and results in
:math:`P_r > P_t` for :math:`d < \lambda / 2 \sqrt{\pi}`.

Both these conditions occur outside of the far field region, so in
principle the Friis model shall not be used in these conditions.
In practice, however, Friis is often used in scenarios where accurate
propagation modeling is not deemed important, and values of
:math:`d = 0` can occur.

To allow practical use of the model in such
scenarios, we have to 1) return some value for :math:`d = 0`, and
2) avoid large discontinuities in propagation loss values (which
could lead to artifacts such as bogus capture effects which are
much worse than inaccurate propagation loss values). The two issues
are conflicting, as, according to the Friis formula,
:math:`\lim_{d \to 0}  P_r = +\infty`;
so if, for :math:`d = 0`, we use a fixed loss value, we end up with an infinitely large
discontinuity, which as we discussed can cause undesirable
simulation artifacts.

To avoid these artifact, this implementation of the Friis model
provides an attribute called MinLoss which allows to specify the
minimum total loss (in dB) returned by the model. This is used in
such a way that
:math:`P_r` continuously increases for :math:`d \to 0`, until
MinLoss is reached, and then stay constant; this allow to
return a value for :math:`d = 0` and at the same time avoid
discontinuities. The model won't be much realistic, but at least
the simulation artifacts discussed before are avoided. The default value of
MinLoss is 0 dB, which means that by default the model will return
:math:`P_r = P_t` for :math:`d <= \lambda / 2 \sqrt{\pi}`.
We note that this value of :math:`d` is outside of the far field
region, hence the validity of the model in the far field region is
not affected.


TwoRayGroundPropagationLossModel
================================

This model implements a Two-Ray Ground propagation loss model ported from NS2

The Two-ray ground reflection model uses the formula

.. math::

  P_r = \frac{P_t * G_t * G_r * (H_t^2 * H_r^2)}{d^4 * L}

The original equation in Rappaport's book assumes :math:`L = 1`.
To be consistent with the free space equation, :math:`L` is added here.

:math:`H_t` and :math:`H_r` are set at the respective nodes :math:`z` coordinate plus a model parameter
set via SetHeightAboveZ.

The two-ray model does not give a good result for short distances, due to the
oscillation caused by constructive and destructive combination of the two
rays. Instead the Friis free-space model is used for small distances.

The crossover distance, below which Friis is used, is calculated as follows:

.. math::

  dCross = \frac{(4 * \pi * H_t * H_r)}{\lambda}

In the implementation,  :math:`\lambda` is calculated as
:math:`\frac{C}{f}`, where :math:`C = 299792458` m/s is the speed of light in
vacuum, and :math:`f` is the frequency in Hz which can be configured by
the user via the Frequency attribute.


LogDistancePropagationLossModel
===============================

This model implements a log distance propagation model.

The reception power is calculated with a so-called
log-distance propagation model:

.. math::

  L = L_0 + 10 n \log(\frac{d}{d_0})

where:

  :math:`n` : the path loss distance exponent

  :math:`d_0` : reference distance (m)

  :math:`L_0` : path loss at reference distance (dB)

  :math:`d` :  - distance (m)

  :math:`L` : path loss (dB)

When the path loss is requested at a distance smaller than
the reference distance, the tx power is returned.

ThreeLogDistancePropagationLossModel
====================================

This model implements a log distance path loss propagation model with three distance
fields. This model is the same as ns3::LogDistancePropagationLossModel
except that it has three distance fields: near, middle and far with
different exponents.

Within each field the reception power is calculated using the log-distance
propagation equation:

.. math::

  L = L_0 + 10 \cdot n_0 \log_{10}(\frac{d}{d_0})

Each field begins where the previous ends and all together form a continuous function.

There are three valid distance fields: near, middle, far. Actually four: the
first from 0 to the reference distance is invalid and returns txPowerDbm.

.. math::

  \underbrace{0 \cdots\cdots}_{=0} \underbrace{d_0 \cdots\cdots}_{n_0} \underbrace{d_1 \cdots\cdots}_{n_1} \underbrace{d_2 \cdots\cdots}_{n_2} \infty

Complete formula for the path loss in dB:


.. math::

  \displaystyle L =
  \begin{cases}
  0 & d < d_0 \\
  L_0 + 10 \cdot n_0 \log_{10}(\frac{d}{d_0}) & d_0 \leq d < d_1 \\
  L_0 + 10 \cdot n_0 \log_{10}(\frac{d_1}{d_0}) + 10 \cdot n_1 \log_{10}(\frac{d}{d_1}) & d_1 \leq d < d_2 \\
  L_0 + 10 \cdot n_0 \log_{10}(\frac{d_1}{d_0}) + 10 \cdot n_1 \log_{10}(\frac{d_2}{d_1}) + 10 \cdot n_2 \log_{10}(\frac{d}{d_2})& d_2 \leq d
  \end{cases}

where:


  :math:`d_0, d_1, d_2` : three distance fields (m)

  :math:`n_0, n_1, n_2` : path loss distance exponent for each field (unitless)

  :math:`L_0` : path loss at reference distance (dB)

  :math:`d` :  - distance (m)

  :math:`L` : path loss (dB)

When the path loss is requested at a distance smaller than the reference
distance :math:`d_0`, the tx power (with no path loss) is returned. The
reference distance defaults to 1m and reference loss defaults to
:cpp:class:`FriisPropagationLossModel` with 5.15 GHz and is thus :math:`L_0` = 46.67 dB.

JakesPropagationLossModel
=========================

ToDo
````

RandomPropagationLossModel
==========================

The propagation loss is totally random, and it changes each time the model is called.
As a consequence, all the packets (even those between two fixed nodes) experience a random
propagation loss.

NakagamiPropagationLossModel
============================

This propagation loss model implements the Nakagami-m fast fading
model, which accounts for the variations in signal strength due to multipath
fading. The model does not account for the path loss due to the
distance traveled by the signal, hence for typical simulation usage it
is recommended to consider using it in combination with other models
that take into account this aspect.

The Nakagami-m distribution is applied to the power level. The probability density function is defined as

.. math::

  p(x; m, \omega) = \frac{2 m^m}{\Gamma(m) \omega^m} x^{2m - 1} e^{-\frac{m}{\omega} x^2} )

with :math:`m` the fading depth parameter and :math:`\omega` the average received power.

It is implemented by either a :cpp:class:`GammaRandomVariable` or a :cpp:class:`ErlangRandomVariable`
random variable.

The implementation of the model allows to specify different values of
the :math:`m` parameter (and hence different fast fading profiles)
for three different distance ranges:

.. math::

  \underbrace{0 \cdots\cdots}_{m_0} \underbrace{d_1 \cdots\cdots}_{m_1} \underbrace{d_2 \cdots\cdots}_{m_2} \infty

For :math:`m = 1` the Nakagami-m distribution equals the Rayleigh distribution. Thus
this model also implements Rayleigh distribution based fast fading.

FixedRssLossModel
=================

This model sets a constant received power level independent of the transmit power.

The received power is constant independent of the transmit power; the user
must set received power level.  Note that if this loss model is chained to other loss
models, it should be the first loss model in the chain.
Else it will disregard the losses computed by loss models that precede it in the chain.

MatrixPropagationLossModel
==========================

The propagation loss is fixed for each pair of nodes and doesn't depend on their actual positions.
This model should be useful for synthetic tests. Note that by default the propagation loss is
assumed to be symmetric.

RangePropagationLossModel
=========================

This propagation loss depends only on the distance (range) between transmitter and receiver.

The single MaxRange attribute (units of meters) determines path loss.
Receivers at or within MaxRange meters receive the transmission at the
transmit power level. Receivers beyond MaxRange receive at power
-1000 dBm (effectively zero).

OkumuraHataPropagationLossModel
===============================

This model is used to model open area pathloss for long distance (i.e., > 1 Km).
In order to include all the possible frequencies usable by LTE we need to consider
several variants of the well known Okumura Hata model. In fact, the original Okumura
Hata model [hata]_ is designed for frequencies ranging from 150 MHz to 1500 MHz,
the COST231 [cost231]_ extends it for the frequency range from 1500 MHz to 2000 MHz.
Another important aspect is the scenarios considered by the models, in fact the all
models are originally designed for urban scenario and then only the standard one and
the COST231 are extended to suburban, while only the standard one has been extended
to open areas. Therefore, the model cannot cover all scenarios at all frequencies.
In the following we detail the models adopted.

The pathloss expression of the COST231 OH is:

.. math::

  L = 46.3 + 33.9\log{f} - 13.82 \log{h_\mathrm{b}} + (44.9 - 6.55\log{h_\mathrm{b}})\log{d} - F(h_\mathrm{M}) + C

where

.. math::

  F(h_\mathrm{M}) = \left\{\begin{array}{ll} (1.1\log(f))-0.7 \times h_\mathrm{M} - (1.56\times \log(f)-0.8) & \mbox{for medium and small size cities} \\ 3.2\times (\log{(11.75\times h_\mathrm{M}}))^2 & \mbox{for large cities}\end{array} \right.

.. math::

  C = \left\{\begin{array}{ll} 0dB & \mbox{for medium-size cities and suburban areas} \\ 3dB & \mbox{for large cities}\end{array} \right.

and

  :math:`f` : frequency [MHz]

  :math:`h_\mathrm{b}` : eNB height above the ground [m]

  :math:`h_\mathrm{M}` : UE height above the ground [m]

  :math:`d` : distance [km]

  :math:`log` : is a logarithm in base 10 (this for the whole document)


This model is only for urban scenarios.

The pathloss expression of the standard OH in urban area is:

.. math::

  L = 69.55 + 26.16\log{f} - 13.82 \log{h_\mathrm{b}} + (44.9 - 6.55\log{h_\mathrm{b}})\log{d} - C_\mathrm{H}

where for small or medium sized city

.. math::

  C_\mathrm{H} = 0.8 + (1.1\log{f} - 0.7)h_\mathrm{M} -1.56\log{f}

and for large cities

.. math::

  C_\mathrm{H} = \left\{\begin{array}{ll} 8.29 (\log{(1.54h_M)})^2 -1.1 & \mbox{if } 150\leq f\leq 200 \\ 3.2(\log{(11.75h_M)})^2 -4.97 & \mbox{if } 200<f\leq 1500\end{array} \right.

There extension for the standard OH in suburban is

.. math::

  L_\mathrm{SU} = L_\mathrm{U} - 2 \left(\log{\frac{f}{28}}\right)^2 - 5.4

where

  :math:`L_\mathrm{U}` : pathloss in urban areas

The extension for the standard OH in open area is

.. math::

  L_\mathrm{O} = L_\mathrm{U} - 4.70 (\log{f})^2 + 18.33\log{f} - 40.94


The literature lacks of extensions of the COST231 to open area (for suburban it seems that
we can just impose C = 0); therefore we consider it a special case fo the suburban one.


Cost231PropagationLossModel
===========================

ToDo
````

ItuR1411LosPropagationLossModel
===============================

This model is designed for Line-of-Sight (LoS) short range outdoor communication in the
frequency range 300 MHz to 100 GHz.  This model provides an upper and lower bound
respectively according to the following formulas

.. math::

  L_\mathrm{LoS,l} = L_\mathrm{bp} + \left\{\begin{array}{ll} 20\log{\frac{d}{R_\mathrm{bp}}} & \mbox{for $d \le R_\mathrm{bp}$} \\ 40\log{\frac{d}{R_\mathrm{bp}}} & \mbox{for $d > R_\mathrm{bp}$}\end{array} \right.

.. math::

  L_\mathrm{LoS,u} = L_\mathrm{bp} + 20 + \left\{\begin{array}{ll} 25\log{\frac{d}{R_\mathrm{bp}}} & \mbox{for $d \le R_\mathrm{bp}$} \\ 40\log{\frac{d}{R_\mathrm{bp}}} & \mbox{for $d > R_\mathrm{bp}$}\end{array} \right.

where the breakpoint distance is given by

.. math::

  R_\mathrm{bp} \approx \frac{4h_\mathrm{b}h_\mathrm{m}}{\lambda}

and the above parameters are

  :math:`\lambda` : wavelength [m]

  :math:`h_\mathrm{b}` : eNB height above the ground [m]

  :math:`h_\mathrm{m}` : UE height above the ground [m]

  :math:`d` : distance [m]

and :math:`L_{bp}` is the value for the basic transmission loss at the break point, defined as:

.. math::

  L_{bp} = \left|20\log \left(\frac{\lambda^2}{8\pi h_\mathrm{b}h\mathrm{m}}\right)\right|

The value used by the simulator is the average one for modeling the median pathloss.


ItuR1411NlosOverRooftopPropagationLossModel
===========================================

This model is designed for Non-Line-of-Sight (LoS) short range outdoor communication over
rooftops in the frequency range 300 MHz to 100 GHz. This model includes several scenario-dependent
parameters, such as average street width, orientation, etc. It is advised to set the values of
these parameters manually (using the ns-3 attribute system) according to the desired scenario.

In detail, the model is based on [walfisch]_ and [ikegami]_, where the loss is expressed
as the sum of free-space loss (:math:`L_{bf}`), the diffraction loss from rooftop to
street (:math:`L_{rts}`) and the reduction due to multiple screen diffraction past
rows of building (:math:`L_{msd}`). The formula is:

.. math::

  L_{NLOS1} = \left\{ \begin{array}{ll} L_{bf} + L_{rts} + L_{msd} & \mbox{for } L_{rts} + L_{msd} > 0 \\ L_{bf} & \mbox{for } L_{rts} + L_{msd} \le 0\end{array}\right.

The free-space loss is given by:

.. math::

  L_{bf} = 32.4 + 20 \log {(d/1000)} + 20\log{(f)}

where:

  :math:`f` : frequency [MHz]

  :math:`d` : distance (where :math:`d > 1`) [m]

The term :math:`L_{rts}` takes into account the width of the street and its orientation, according to the formulas

.. math::

  L_{rts} = -8.2 - 10\log {(w)} + 10\log{(f)} + 20\log{(\Delta h_m)} + L_{ori}

  L_{ori} = \left\{ \begin{array}{lll} -10 + 0.354\varphi & \mbox{for } 0^{\circ} \le \varphi < 35^{\circ} \\ 2.5 + 0.075(\varphi-35) & \mbox{for } 35^{\circ} \le \varphi < 55^{\circ} \\ 4.0 -0.114(\varphi-55) & \mbox{for } 55^{\circ} \varphi \le 90^{\circ}\end{array}\right.

  \Delta h_m = h_r - h_m

where:

  :math:`h_r` : is the height of the rooftop [m]

  :math:`h_m` : is the height of the mobile [m]

  :math:`\varphi` : is the street orientation with respect to the direct path (degrees)


The multiple screen diffraction loss depends on the BS antenna height relative to the building
height and on the incidence angle. The former is selected as the higher antenna in the communication
link. Regarding the latter, the "settled field distance" is used for select the proper model;
its value is given by

.. math::

  d_{s} = \frac{\lambda d^2}{\Delta h_{b}^2}

with

  :math:`\Delta h_b = h_b - h_m`

Therefore, in case of :math:`l > d_s` (where `l` is the distance over which the building extend),
it can be evaluated according to

.. math::

  L_{msd} = L_{bsh} + k_{a} + k_{d}\log{(d/1000)} + k_{f}\log{(f)} - 9\log{(b)}

  L_{bsh} = \left\{ \begin{array}{ll} -18\log{(1+\Delta h_{b})} & \mbox{for } h_{b} > h_{r} \\ 0 & \mbox{for } h_{b} \le h_{hr} \end{array}\right.

  k_a = \left\{ \begin{array}{lll}
      71.4 & \mbox{for } h_{b} > h_{r} \mbox{ and } f>2000 \mbox{ MHz} \\
      54 & \mbox{for } h_{b} > h_{r} \mbox{ and } f\le2000 \mbox{ MHz} \\
      54-0.8\Delta h_b & \mbox{for } h_{b} \le h_{r} \mbox{ and } d \ge 500 \mbox{ m} \\
      54-1.6\Delta h_b & \mbox{for } h_{b} \le h_{r} \mbox{ and } d < 500 \mbox{ m} \\
      \end{array} \right.

  k_d = \left\{ \begin{array}{ll}
        18 & \mbox{for } h_{b} > h_{r} \\
        18 -15\frac{\Delta h_b}{h_r} & \mbox{for } h_{b} \le h_{r}
        \end{array} \right.

  k_f = \left\{ \begin{array}{ll}
        -8 & \mbox{for } f>2000 \mbox{ MHz} \\
        -4 + 0.7(f/925 -1) & \mbox{for medium city and suburban centres and} f\le2000 \mbox{ MHz} \\
        -4 + 1.5(f/925 -1) & \mbox{for metropolitan centres and } f\le2000 \mbox{ MHz}
        \end{array}\right.


Alternatively, in case of :math:`l < d_s`, the formula is:

.. math::

  L_{msd} = -10\log{\left(Q_M^2\right)}

where

.. math::

  Q_M = \left\{ \begin{array}{lll}
        2.35\left(\frac{\Delta h_b}{d}\sqrt{\frac{b}{\lambda}}\right)^{0.9} & \mbox{for } h_{b} > h_{r} \\
        \frac{b}{d} &  \mbox{for } h_{b} \approx h_{r} \\
        \frac{b}{2\pi d}\sqrt{\frac{\lambda}{\rho}}\left(\frac{1}{\theta}-\frac{1}{2\pi + \theta}\right) & \mbox{for }  h_{b} < h_{r}
        \end{array}\right.


where:

.. math::

  \theta = arc tan \left(\frac{\Delta h_b}{b}\right)

  \rho = \sqrt{\Delta h_b^2 + b^2}


Kun2600MhzPropagationLossModel
==============================

This is the empirical model for the pathloss at 2600 MHz for urban areas which is described in [kun2600mhz]_.
The model is as follows. Let :math:`d` be the distance between the transmitter and the receiver
in meters; the pathloss :math:`L` in dB is calculated as:

.. math::

  L = 36 + 26\log{d}

ThreeGppPropagationLossModel
============================

The base class :cpp:class:`ThreeGppPropagationLossModel` and its derived classes implement
the path loss and shadow fading models described in 3GPP TR 38.901 [38901]_.
3GPP TR 38.901 includes multiple scenarios modeling different propagation
environments, i.e., indoor, outdoor urban and rural, for frequencies between
0.5 and 100 GHz.

*Implemented features:*

  * Path loss and shadowing models (3GPP TR 38.901, Sec. 7.4.1)
  * Autocorrelation of shadow fading (3GPP TR 38.901, Sec. 7.4.4)
  * `Channel condition models <propagation.html#threegppchannelconditionmodel>`_ (3GPP TR 38.901, Sec. 7.4.2)

*To be implemented:*

  * O2I penetration loss (3GPP TR 38.901, Sec. 7.4.3)
  * Spatial consistent update of the channel states (3GPP TR 38.901 Sec. 7.6.3.3)

**Configuration**

The :cpp:class:`ThreeGppPropagationLossModel` instance is paired with a :cpp:class:`ChannelConditionModel`
instance used to retrieve the LOS/NLOS channel condition. By default, a 3GPP
channel condition model related to the same scenario is set (e.g., by default,
:cpp:class:`ThreeGppRmaPropagationLossModel` is paired with
:cpp:class:`ThreeGppRmaChannelConditionModel`), but it can be configured using
the method SetChannelConditionModel. The channel condition models are stored inside the
``propagation`` module, for a limitation of the current spectrum API and to avoid
a circular dependency between the spectrum and the ``propagation`` modules. Please
note that it is necessary to install at least one :cpp:class:`ChannelConditionModel` when
using any :cpp:class:`ThreeGppPropagationLossModel` subclass. Please look below for more
information about the Channel Condition models.

The operating frequency has to be set using the attribute "Frequency",
otherwise an assert is raised. The addition of the shadow fading component can
be enabled/disabled through the attribute "ShadowingEnabled".
Other scenario-related parameters can be configured through attributes of the
derived classes.

**Implementation details**

The method DoCalcRxPower computes the propagation loss considering the path loss
and the shadow fading (if enabled). The path loss is computed by the method
GetLossLos or GetLossNlos depending on the LOS/NLOS channel condition, and their
implementation is left to the derived classes. The shadow fading is computed by
the method GetShadowing, which generates an additional random loss component
characterized by Gaussian distribution with zero mean and scenario-specific
standard deviation. Subsequent shadowing components of each BS-UT link are
correlated as described in 3GPP TR 38.901, Sec. 7.4.4 [38901]_.

*Note 1*: The TR defines height ranges for UTs and BSs, depending on the chosen
propagation model (for the exact values, please see below in the specific model
documentation). If the user does not set correct values, the model will emit
a warning but perform the calculation anyway.

*Note 2*: The 3GPP model is originally intended to be used to represent BS-UT
links. However, in ns-3, we may need to compute the pathloss between two BSs
or UTs to evaluate the interference. We have decided to support this case by
considering the tallest node as a BS and the smallest as a UT. As a consequence,
the height values may be outside the validity range of the chosen class:
therefore, an inaccuracy warning may be printed, but it can be ignored.

There are four derived class, each one implementing the propagation model for a different scenario:

ThreeGppRMaPropagationLossModel
```````````````````````````````

This class implements the LOS/NLOS path loss and shadow fading models described in 3GPP TR 38.901 [38901]_, Table 7.4.1-1 for the RMa scenario.
It supports frequencies between 0.5 and 30 GHz.
It is possible to configure some scenario-related parameters through the attributes AvgBuildingHeight and AvgStreetWidth.

As specified in the TR, the 2D distance between the transmitter and the receiver
should be between 10 m and 10 km for the LOS case, or between 10 m and 5 km for
the NLOS case, otherwise the model may not be accurate (a warning message is
printed if the user has enabled logging on the model). Also, the height of the
base station (hBS) should be between 10 m and 150 m, while the height of the
user terminal (hUT) should be between 1 m and 10 m.

ThreeGppUMaPropagationLossModel
```````````````````````````````

This implements the LOS/NLOS path loss and shadow fading models described in 3GPP
TR 38.901 [38901]_, Table 7.4.1-1 for the UMa scenario. It supports frequencies
between 0.5 and 100 GHz.

As specified in the TR, the 2D distance between the transmitter and the receiver
should be between 10 m and 5 km both for the LOS and NLOS cases, otherwise the model may not be
accurate (a warning message is printed if the user has enabled logging on the model).
Also, the height of the base station (hBS) should be 25 m and the height of the
user terminal (hUT) should be between 1.5 m and 22.5 m.

ThreeGppUmiStreetCanyonPropagationLossModel
```````````````````````````````````````````

This implements the LOS/NLOS path loss and shadow fading models described in 3GPP
TR 38.901 [38901]_, Table 7.4.1-1 for the UMi-Street Canyon scenario. It
supports frequencies between 0.5 and 100 GHz.

As specified in the TR, the 2D distance between the transmitter and the receiver
should be between 10 m and 5 km both for the LOS and NLOS cases, otherwise the model may not be
accurate (a warning message is printed if the user has enabled logging on
the model). Also, the height of the base station (hBS) should be 10 m and the
height of the user terminal (hUT) should be between 1.5 m and 10 m (the validity
range is reduced because we assume that the height of the UT nodes is always
lower that the height of the BS nodes).

ThreeGppIndoorOfficePropagationLossModel
````````````````````````````````````````

This implements the LOS/NLOS path loss and shadow fading models described in 3GPP
TR 38.901 [38901]_, Table 7.4.1-1 for the Indoor-Office scenario. It supports
frequencies between 0.5 and 100 GHz.

As specified in the TR, the 3D distance between the transmitter and the receiver
should be between 1 m and 150 m both for the LOS and NLOS cases, otherwise the
model may not be accurate (a warning log message is printed if the user has
enabled logging on the model).

Testing
```````

The test suite :cpp:class:`ThreeGppPropagationLossModelsTestSuite` provides test cases for the classes
implementing the 3GPP propagation loss models.
The test cases :cpp:class:`ThreeGppRmaPropagationLossModelTestCase`,
:cpp:class:`ThreeGppUmaPropagationLossModelTestCase`,
:cpp:class:`ThreeGppUmiPropagationLossModelTestCase` and
:cpp:class:`ThreeGppIndoorOfficePropagationLossModelTestCase` compute the path loss between two nodes and compares it with the value obtained using the formulas in 3GPP TR 38.901 [38901]_, Table 7.4.1-1.
The test case :cpp:class:`ThreeGppShadowingTestCase` checks if the shadowing is correctly computed by testing the deviation of the overall propagation loss from the path loss. The test is carried out for all the scenarios, both in LOS and NLOS condition.

ChannelConditionModel
*********************

The loss models require to know if two nodes are in Line-of-Sight (LoS) or if
they are not. The interface for that is represented by this class. The main
method is GetChannelCondition (a, b), which returns a :cpp:class:`ChannelCondition` object
containing the information about the channel state.

We modeled the LoS condition in two ways: (i) by using a probabilistic model
specified by the 3GPP (), and (ii) by using an ns-3 specific
building-aware model, which checks the space position of the BSs and the UTs.
For what regards the first option, the probability is independent of the node
location: in other words, following the 3GPP model, two UT spatially separated
by an epsilon may have different LoS conditions. To take into account mobility,
we have inserted a parameter called "UpdatePeriod," which indicates how often a
3GPP-based channel condition has to be updated. By default, this attribute is
set to 0, meaning that after the channel condition is generated, it is never
updated. With this default value, we encourage the users to run multiple
simulations with different seeds to get statistical significance from the data.
For the users interested in using mobile nodes, we suggest changing this
parameter to a value that takes into account the node speed and the desired
accuracy. For example, lower-speed node conditions may be updated in terms of
seconds, while high-speed UT or BS may be updated more often.

The two approach are coded, respectively, in the classes:

* :cpp:class:`ThreeGppChannelConditionModel`
* :cpp:class:`BuildingsChannelConditionModel` (see the ``building`` module documentation for further details)

ThreeGppChannelConditionModel
=============================
This is the base class for the 3GPP channel condition models.
It provides the possibility to updated the condition of each channel periodically,
after a given time period which can be configured through the attribute "UpdatePeriod".
If "UpdatePeriod" is set to 0, the channel condition is never updated.
It has five derived classes implementing the channel condition models described in 3GPP TR 38.901 [38901]_ for different propagation scenarios.

ThreeGppRmaChannelConditionModel
````````````````````````````````
This implements the statistical channel condition model described in 3GPP TR 38.901 [38901]_, Table 7.4.2-1, for the RMa scenario.

ThreeGppUmaChannelConditionModel
````````````````````````````````
This implements the statistical channel condition model described in 3GPP TR 38.901 [38901]_, Table 7.4.2-1, for the UMa scenario.

ThreeGppUmiStreetCanyonChannelConditionModel
````````````````````````````````````````````
This implements the statistical channel condition model described in 3GPP TR 38.901 [38901]_, Table 7.4.2-1, for the UMi-Street Canyon scenario.

ThreeGppIndoorMixedOfficeChannelConditionModel
``````````````````````````````````````````````
This implements the statistical channel condition model described in 3GPP TR 38.901 [38901]_, Table 7.4.2-1, for the Indoor-Mixed office scenario.

ThreeGppIndoorOpenOfficeChannelConditionModel
`````````````````````````````````````````````
This implements the statistical channel condition model described in 3GPP TR 38.901 [38901]_, Table 7.4.2-1, for the Indoor-Open office scenario.

Testing
=======
The test suite :cpp:class:`ChannelConditionModelsTestSuite` contains a single test case:

* :cpp:class:`ThreeGppChannelConditionModelTestCase`, which tests all the 3GPP channel condition models. It determines the channel condition between two nodes multiple times, estimates the LOS probability, and compares it with the value given by the formulas in 3GPP TR 38.901 [38901]_, Table 7.4.2-1


PropagationDelayModel
*********************

The following propagation delay models are implemented:

* ConstantSpeedPropagationDelayModel
* RandomPropagationDelayModel

ConstantSpeedPropagationDelayModel
==================================

In this model, the signal travels with constant speed.
The delay is calculated according with the transmitter and receiver positions.
The Euclidean distance between the Tx and Rx antennas is used.
Beware that, according to this model, the Earth is flat.

RandomPropagationDelayModel
===========================

The propagation delay is totally random, and it changes each time the model is called.
All the packets (even those between two fixed nodes) experience a random delay.
As a consequence, the packets order is not preserved.

Models for vehicular environments
*********************************

The 3GPP TR 37.885 [37885]_ specifications extends the channel modeling framework 
described in TR 38.901 [38901]_ to simulate wireless channels in vehicular environments. 
The extended framework supports frequencies between 0.5 to 100 GHz and provides 
the possibility to simulate urban and highway propagation environments. 
To do so, new propagation loss and channel condition models, as well as new 
parameters for the fast fading model, are provided. 

.. _sec-3gpp-v2v-ch-cond:

Vehicular channel condition models
==================================

To properly capture channel dynamics in vehicular environments, three different 
channel conditions have been identified: 

  * LOS (Line Of Sight): represents the case in which the direct path between 
    the transmitter and the receiver is not blocked
  * NLOSv (Non Line Of Sight vehicle): when the direct path 
    between the transmitter and the receiver is blocked by a vehicle
  * NLOS (Non Line Of Sight): when the direct path is blocked by a building

TR 37.885 includes two models that can be used to determine the condition of 
the wireless channel between a pair of nodes, the first for urban and the second 
for highway environments. 
Each model includes both a deterministic and a stochastic part, and works as 
follows:

  1. The model determines the presence of buildings obstructing the direct path 
     between the communicating nodes. This is done in a deterministic way, looking at 
     the possible interceptions between the direct path and the buildings. 
     If the path is obstructed, the channel condition is set to NLOS. 
  2. If not, the model determines the presence of vehicles obstructing the 
     direct path. This is done using a probabilistic model, which is specific 
     for the scenario of interest. If the path is obstructed, the channel 
     condition is set to NLOSv, otherwise is set to LOS.

These models have been implemented by extending the interface 
:cpp:class:`ChannelConditionModel` with the following classes. They have been included in 
the ``building`` module, because they make use of :cpp:class:`Buildings` objects to determine 
the presence of obstructions caused by buildings.

  * :cpp:class:`ThreeGppV2vUrbanChannelConditionModel`: implements the model 
    described in Table 6.2-1 of TR 37.885 for the urban scenario. 
  * :cpp:class:`ThreeGppV2vHighwayChannelConditionModel`: implements the model 
    described in Table 6.2-1 of TR 37.885 for the highway scenario.   

These models rely on :cpp:class:`Buildings` objects to determine the presence 
of obstructing buildings. When considering large scenarios with a large number of 
buildings, this process may become computationally demanding and dramatically 
increase the simulation time. 
To solve this problem, we implemented two fully-probabilistic models 
that can be used as an alternative to the ones included in TR 37.885. 
These models are based on the work carried out by M. Boban et al. [Boban2016Modeling]_, 
which derived a statistical representation of the three channel conditions, 
With the fully-probabilistic models there is no need to determine the presence of blocking 
buildings in a deterministic way, and therefore the computational effort is 
reduced. 
To determine the channel condition, these models account for the propagation 
environment, i.e., urban or highway, as well as for the density of vehicles in the 
scenario, which can be high, medium, or low.

The classes implementing the fully-probabilistic models are:

  * :cpp:class:`ProbabilisticV2vUrbanChannelConditionModel`: implements the model 
    described in [Boban2016Modeling]_ for the urban scenario. 
  * :cpp:class:`ProbabilisticV2vHighwayChannelConditionModel`: implements the model 
    described in [Boban2016Modeling]_ for the highway scenario. 

Both the classes own the attribute "Density", which can be used to select the 
proper value depending on the scenario that have to be simulated. 
Differently from the hybrid models described above, these classes have been 
included in the ``propagation`` module, since they do not have any dependency on the 
``building`` module.

**NOTE:** Both the hybrid and the fully-probabilistic models supports the 
modeling of outdoor scenarios, no support is provided for the modeling of 
indoor scenarios.

Vehicular propagation loss models
=================================

The propagation models described in TR 37.885 determines the attenuation caused 
by path loss and shadowing by considering the propagation environment and the 
channel condition. 

These models have been implemented by extending the interface 
:cpp:class:`ThreeGppPropagationLossModel` with the following classes, which 
are part of the ``propagation`` module:

  * :cpp:class:`ThreeGppV2vUrbanPropagationLossModel`: implements the models 
    defined in Table 6.2.1-1 of TR 37.885 for the urban scenario.
  * :cpp:class:`ThreeGppV2vHighwayPropagationLossModel`: implements the models 
    defined in Table 6.2.1-1 of TR 37.885 for the highway scenario.       

As for all the classes extending the interface :cpp:class:`ThreeGppPropagationLossModel`, 
they have to be paired with an instance of the class :cpp:class:`ChannelConditionModel` 
which is used to determine the channel condition. 
This is done by setting the attribute :cpp:class:`ChannelConditionModel`.
To build the channel modeling framework described in TR 37.885, 
:cpp:class:`ThreeGppV2vUrbanChannelConditionModel` or 
:cpp:class:`ThreeGppV2vHighwayChannelConditionModel` should be used, but users 
are allowed to test any other combination. 

.. _sec-3gpp-v2v-ff:

Vehicular fast fading model
===========================

The fast fading model described in Sec. 6.2.3 of TR 37.885 is based on the one 
specified in TR 38.901, whose implementation is provided in the ``spectrum`` module 
(see the :ref:`spectrum module documentation <sec-3gpp-fast-fading-model>`). 
This model is general and includes different parameters which can 
be tuned to simulate multiple propagation environments.
To better model the channel dynamics in vehicular environments, TR 37.885 
provides new sets of values for these parameters, specific for 
vehicle-to-vehicle transmissions in urban and highway scenarios. 
To select the parameters for vehicular scenarios, it is necessary to set 
the attribute "Scenario" of the class :cpp:class:`ThreeGppChannelModel` using the value 
"V2V-Urban" or "V2V-Highway".

Additionally, TR 37.885 specifies a new equation to compute the Doppler component, 
which accounts for the mobility of both nodes, as well as scattering 
from the environment. 
In particular, the scattering effect is considered by deviating the Doppler 
frequency by a random value, whose distribution depends on the parameter :math:`v_{scatt}`.
TR 37.885 specifies that :math:`v_{scatt}` should be set to the maximum speed of the 
vehicles in the layout and, if :math:`v_{scatt} = 0`, the scattering effect is not considered. 
The Doppler equation is implemented in the class :cpp:class:`ThreeGppSpectrumPropagationLossModel`. 
By means of the attribute "vScatt", it is possible to adjust the value of 
:math:`v_{scatt} = 0` (by default, the value is set to 0).

Example
=======

We implemented the example ``three-gpp-v2v-channel-example.cc`` which shows how to 
configure the different classes to simulate wireless propagation in vehicular 
scenarios, it can be found in the folder ``examples/channel-models``.

We considered two communicating vehicles moving within the scenario, and 
computed the SNR experienced during the entire simulation, with a time 
resolution of 10 ms. 
The vehicles are equipped with 2x2 antenna arrays modeled using the 
:ref:`3GPP antenna model <sec-3gpp-antenna-model>`. 
The bearing and the downtilt angles are properly configured and the 
optimal beamforming vectors are computed at the beginning of the simulation. 
 
The simulation script accepts the following command line parameters: 

  * *frequency*: the operating frequency in Hz
  * *txPow*: the transmission power in dBm
  * *noiseFigure*: the noise figure in dB
  * *scenario*: the simulation scenario, "V2V-Urban" or "V2V-Highway"
  
The "V2V-Urban" scenario simulates urban environment with a rectangular grid of 
buildings. The vehicles moves with a waypoint mobility model. They start from 
the same position and travel in the same direction, along the main street. 
The first vehicle moves at 60 km/h and the second at 30 km/h. 
At a certain point, the first vehicle turns left while the second continues on 
the main street. 

The "V2V-Highway" scenario simulates an highway environment in which the 
two vehicles travel on the same lane, in the same direction, and keep a safety 
distance of 20 m. They maintain a constant speed of 140 km/h.

The example generates the output file ``example-output.txt``. Each row of the
file is organized as follows:

``Time[s] TxPosX[m] TxPosY[m] RxPosX[m] RxPosY[m] ChannelState SNR[dB] Pathloss[dB]``

We also provide the bash script ``three-gpp-v2v-channel-example.sh`` which reads the 
output file and generates two figures:

  1. map.gif, a GIF representing the simulation scenario and vehicle mobility;
  2. snr.png, which represents the behavior of the SNR.


References
**********

.. [friis] Friis, H.T., "A Note on a Simple Transmission Formula," Proceedings of the IRE , vol.34, no.5, pp.254,256, May 1946

.. [hata] M.Hata, "Empirical formula for propagation loss in land mobile radio
   services", IEEE Trans. on Vehicular Technology, vol. 29, pp. 317-325, 1980

.. [cost231] “Digital Mobile Radio: COST 231 View on the Evolution Towards 3rd Generation Systems”, Commission of the European Communities, L-2920, Luxembourg, 1989

.. [walfisch]  J.Walfisch and H.L. Bertoni, "A Theoretical model of UHF propagation in urban environments," in IEEE Trans. Antennas Propagat., vol.36, 1988, pp.1788- 1796

.. [ikegami] F.Ikegami, T.Takeuchi, and S.Yoshida, "Theoretical prediction of mean field strength for Urban Mobile Radio", in IEEE Trans. Antennas Propagat., Vol.39, No.3, 1991

.. [kun2600mhz] Sun Kun, Wang Ping, Li Yingze, "Path loss models for suburban scenario at 2.3GHz, 2.6GHz and 3.5GHz",
   in Proc. of the 8th International Symposium on Antennas, Propagation and EM Theory (ISAPE),  Kunming,  China, Nov 2008.

.. [38901] 3GPP. 2018. TR 38.901, Study on channel model for frequencies from 0.5 to 100 GHz, V15.0.0. (2018-06).

.. [37885] 3GPP. 2019. TR 37.885, Study on evaluation methodology of new Vehicle-to-Everything (V2X) use cases for LTE and NR, V15.3.0. (2019-06).

.. [Boban2016Modeling]  M. Boban,  X. Gong, and  W. Xu, “Modeling the evolution 
   of line-of-sight blockage for V2V channels,” in IEEE 84th Vehicular Technology 
   Conference (VTC-Fall), 2016.
