Energy Framework
----------------

Energy consumption is a key issue for wireless devices, and wireless
network researchers often need to investigate the energy consumption
at a node or in the overall network while running network simulations
in ns-3. This requires ns-3 to support energy consumption
modeling. Further, as concepts such as fuel cells and energy
scavenging are becoming viable for low power wireless devices,
incorporating the effect of these emerging technologies into
simulations requires support for modeling diverse energy sources in
ns-3. The ns-3 Energy Framework provides the basis for energy
consumption, energy source and energy harvesting modeling.


Model Description
=================

The source code for the Energy Framework is currently at: ``src/energy``.

Design
******

The ns-3 Energy Framework is composed of 3 parts: Energy Source,
Device Energy Model and Energy Harvester.  The framework is
implemented into the ``src/energy/models`` folder.

Energy Source
#############

The Energy Source represents the power supply on each node. A node can
have one or more energy sources, and each energy source can be
connected to multiple device energy models. Connecting an energy
source to a device energy model implies that the corresponding device
draws power from the source. The basic functionality of the Energy
Source is to provide energy for devices on the node. When energy is
completely drained from the Energy Source, it notifies the devices on
node such that each device can react to this event. Further, each node
can access the Energy Source Objects for information such as remaining
energy or energy fraction (battery level). This enables the
implementation of energy aware protocols in ns-3.

In order to model a wide range of power supplies such as batteries,
the Energy Source must be able to capture characteristics of these
supplies. There are 2 important characteristics or effects related to
practical batteries:

Rate Capacity Effect
  Decrease of battery lifetime when the current draw is higher
  than the rated value of the battery.
  
Recovery Effect
  Increase of battery lifetime when the battery is alternating
  between discharge and idle states.

In order to incorporate the Rate Capacity Effect, the Energy Source
uses current draw from all the devices on the same node to calculate
energy consumption. Moreover, multiple Energy Harvesters can be
connected to the Energy Source in order to replenish its energy. The
Energy Source periodically polls all the devices and energy harvesters
on the same node to calculate the total current drain and hence the
energy consumption. When a device changes state, its corresponding
Device Energy Model will notify the Energy Source of this change and
new total current draw will be calculated. Similarly, every Energy
Harvester update triggers an update to the connected Energy Source.

The Energy Source base class keeps a list of devices (Device Energy
Model objects) and energy harvesters (Energy Harvester objects) that
are using the particular Energy Source as power supply. When energy is
completely drained, the Energy Source will notify all devices on this
list. Each device can then handle this event independently, based on
the desired behavior that should be followed in case of power outage.


Device Energy Model
###################

The Device Energy Model is the energy consumption model of a device
installed on the node. It is designed to be a state based model where
each device is assumed to have a number of states, and each state is
associated with a power consumption value. Whenever the state of the
device changes, the corresponding Device Energy Model will notify the
Energy Source of the new current draw of the device. The Energy Source
will then calculate the new total current draw and update the
remaining energy.

The Device Energy Model can also be used for devices that do not have
finite number of states. For example, in an electric vehicle, the
current draw of the motor is determined by its speed. Since the
vehicle's speed can take continuous values within a certain range, it
is infeasible to define a set of discrete states of
operation. However, by converting the speed value into current
directly, the same set of Device Energy Model APIs can still be used.

Energy Harvester
################

The energy harvester represents the elements that harvest energy from
the environment and recharge the Energy Source to which it is
connected. The energy harvester includes the complete implementation
of the actual energy harvesting device (e.g., a solar panel) and the
environment (e.g., the solar radiation). This means that in
implementing an energy harvester, the energy contribution of the
environment and the additional energy requirements of the energy
harvesting device such as the conversion efficiency and the internal
power consumption of the device needs to be jointly modeled.


WiFi Radio Energy Model
#######################

The WiFi Radio Energy Model is the energy consumption model of a Wifi
net device. It provides a state for each of the available states of
the PHY layer: Idle, CcaBusy, Tx, Rx, ChannelSwitch, Sleep, Off. Each of
such states is associated with a value (in Ampere) of the current draw
(see below for the corresponding attribute names). A Wifi Radio Energy
Model PHY Listener is registered to the Wifi PHY in order to be
notified of every Wifi PHY state transition. At every transition, the
energy consumed in the previous state is computed and the energy
source is notified in order to update its remaining energy.

The Wifi Tx Current Model gives the possibility to compute the current
draw in the transmit state as a function of the nominal tx power (in
dBm), as observed in several experimental measurements. To this
purpose, the Wifi Radio Energy Model PHY Listener is notified of the
nominal tx power used to transmit the current frame and passes such a
value to the Wifi Tx Current Model which takes care of updating the
current draw in the Tx state. Hence, the energy consumption is
correctly computed even if the Wifi Remote Station Manager performs
per-frame power control. Currently, a Linear Wifi Tx Current Model is
implemented which computes the tx current as a linear function
(according to parameters that can be specified by the user) of the
nominal tx power in dBm.

The Wifi Radio Energy Model offers the possibility to specify a
callback that is invoked when the energy source is depleted. If such a
callback is not specified when the Wifi Radio Energy Model Helper is
used to install the model on a device, a callback is implicitly made
so that the Wifi PHY is put in the OFF mode (hence no frame is
transmitted nor received afterwards) when the energy source is
depleted. Likewise, it is possible to specify a callback that is
invoked when the energy source is recharged (which might occur in case
an energy harvester is connected to the energy source). If such a
callback is not specified when the Wifi Radio Energy Model Helper is
used to install the model on a device, a callback is implicitly made
so that the Wifi PHY is resumed from the OFF mode when the energy
source is recharged.

Future Work
***********

For Device Energy Models, we are planning to include support for other
PHY layer models provided in ns-3 such as WiMAX, and to model the
energy consumptions of other non communicating devices, like a generic
sensor and a CPU. For Energy Sources, we are planning to included new
types of Energy Sources such as Supercapacitor and Nickel-Metal
Hydride (Ni-MH) battery. For the Energy Harvesters, we are planning to
implement an energy harvester that recharges the energy sources
according to the power levels defined in a user customizable dataset
of real measurements.

References
**********

.. [1] ns-2 Energy model:
   http://www.cubinlab.ee.unimelb.edu.au/~jrid/Docs/Manuel-NS2/node204.html

.. [2] H. Wu, S. Nabar and R. Poovendran. An Energy Framework for the
   Network Simulator 3 (ns-3).

.. [3] M. Handy and D. Timmermann. Simulation of mobile wireless
   networks with accurate modelling of non-linear battery effects. In
   Proc. of Applied simulation and Modeling (ASM), 2003.
       
.. [4] D. N. Rakhmatov and S. B. Vrudhula. An analytical high-level
   battery model for use in energy management of portable electronic
   systems. In Proc. of IEEE/ACM International Conference on Computer
   Aided Design (ICCAD'01), pages 488-493, November 2001.
       
.. [5] D. N. Rakhmatov, S. B. Vrudhula, and D. A. Wallach. Battery
   lifetime prediction for energy-aware computing. In Proc. of the 2002
   International Symposium on Low Power Electronics and Design
   (ISLPED'02), pages 154-159, 2002.
       
.. [6] C. Tapparello, H. Ayatollahi and W. Heinzelman. Extending the
   Energy Framework for Network Simulator 3 (ns-3). Workshop on ns-3
   (WNS3), Poster Session, Atlanta, GA, USA. May, 2014.

.. [7] C. Tapparello, H. Ayatollahi and W. Heinzelman. Energy Harvesting 
   Framework for Network Simulator 3 (ns-3). 2nd International Workshop on
   Energy Neutral Sensing Systems (ENSsys), Memphis, TN, USA. November 6,
   2014.

Usage
=====

The main way that ns-3 users will typically interact with the Energy
Framework is through the helper API and through the publicly visible
attributes of the framework. The helper API is defined in
``src/energy/helper/*.h``.

In order to use the energy framework, the user must install an Energy
Source for the node of interest, the corresponding Device Energy Model
for the network devices and, if necessary, the one or more Energy
Harvester. Energy Source (objects) are aggregated onto each node by
the Energy Source Helper. In order to allow multiple energy sources
per node, we aggregate an Energy Source Container rather than directly
aggregating a source object.

The Energy Source object keeps a list of Device Energy Model and
Energy Harvester objects using the source as power supply. Device
Energy Model objects are installed onto the Energy Source by the
Device Energy Model Helper, while Energy Harvester object are
installed by the Energy Harvester Helper. User can access the Device
Energy Model objects through the Energy Source object to obtain energy
consumption information of individual devices. Moreover, the user can
access to the Energy Harvester objects in order to gather information
regarding the current harvestable power and the total energy harvested
by the harvester.


Examples
********

The example directories, ``src/examples/energy`` and
``examples/energy``, contain some basic code that shows how to set up
the framework.

Helpers
*******

Energy Source Helper
####################

Base helper class for Energy Source objects, this helper Aggregates
Energy Source object onto a node. Child implementation of this class
creates the actual Energy Source object.

Device Energy Model Helper
##########################

Base helper class for Device Energy Model objects, this helper
attaches Device Energy Model objects onto Energy Source objects. Child
implementation of this class creates the actual Device Energy Model
object.

Energy Harvesting Helper
##########################

Base helper class for Energy Harvester objects, this helper attaches
Energy Harvester objects onto Energy Source objects. Child
implementation of this class creates the actual Energy Harvester
object.


Attributes
**********

Attributes differ between Energy Sources, Devices Energy Models and
Energy Harvesters implementations, please look at the specific child
class for details.

Basic Energy Source
###################

* ``BasicEnergySourceInitialEnergyJ``: Initial energy stored in
  basic energy source.
* ``BasicEnergySupplyVoltageV``: Initial supply voltage for basic energy source.
* ``PeriodicEnergyUpdateInterval``: Time between two consecutive periodic
  energy updates.

RV Battery Model
################

* ``RvBatteryModelPeriodicEnergyUpdateInterval``: RV battery model sampling
  interval.
* ``RvBatteryModelOpenCircuitVoltage``: RV battery model open circuit voltage.
* ``RvBatteryModelCutoffVoltage``: RV battery model cutoff voltage.
* ``RvBatteryModelAlphaValue``: RV battery model alpha value.
* ``RvBatteryModelBetaValue``: RV battery model beta value.
* ``RvBatteryModelNumOfTerms``: The number of terms of the infinite sum for estimating battery level.

WiFi Radio Energy Model
#######################

* ``IdleCurrentA``: The default radio Idle current in Ampere.
* ``CcaBusyCurrentA``: The default radio CCA Busy State current in Ampere.
* ``TxCurrentA``: The radio Tx current in Ampere.
* ``RxCurrentA``: The radio Rx current in Ampere.
* ``SwitchingCurrentA``: The default radio Channel Switch current in Ampere.
* ``SleepCurrentA``: The radio Sleep current in Ampere.
* ``TxCurrentModel``: A pointer to the attached tx current model.

Basic Energy Harvester
#######################

* ``PeriodicHarvestedPowerUpdateInterval``: Time between two consecutive
  periodic updates of the harvested power.
* ``HarvestablePower``: Random variables that represents the amount of power
  that is provided by the energy harvester.

Tracing
*******

Traced values differ between Energy Sources, Devices Energy Models and
Energy Harvesters implementations, please look at the specific child
class for details.

Basic Energy Source
###################

* ``RemainingEnergy``: Remaining energy at BasicEnergySource.

RV Battery Model
################

* ``RvBatteryModelBatteryLevel``: RV battery model battery level.
* ``RvBatteryModelBatteryLifetime``: RV battery model battery lifetime.

WiFi Radio Energy Model
#######################

* ``TotalEnergyConsumption``: Total energy consumption of the radio device.

Basic Energy Harvester
#######################

* ``HarvestedPower``: Current power provided by the BasicEnergyHarvester.
* ``TotalEnergyHarvested``: Total energy harvested by the BasicEnergyHarvester.


Validation
**********

Comparison of the Energy Framework against actual devices have not
been performed. Current implementation of the Energy Framework is
checked numerically for computation errors. The RV battery model is
validated by comparing results with what was presented in the original
RV battery model paper.
