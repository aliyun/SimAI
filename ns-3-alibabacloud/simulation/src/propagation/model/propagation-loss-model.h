/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006,2007 INRIA
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as 
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 * Contributions: Timo Bingmann <timo.bingmann@student.kit.edu>
 * Contributions: Gary Pei <guangyu.pei@boeing.com> for fixed RSS
 * Contributions: Tom Hewer <tomhewer@mac.com> for two ray ground model
 *                Pavel Boyko <boyko@iitp.ru> for matrix
 */

#ifndef PROPAGATION_LOSS_MODEL_H
#define PROPAGATION_LOSS_MODEL_H

#include "ns3/object.h"
#include "ns3/random-variable-stream.h"
#include <map>

namespace ns3 {

/**
 * \defgroup propagation Propagation Models
 */

/**
 * \ingroup propagation
 * \ingroup tests
 * \defgroup propagation-tests Propagation module tests
 */

class MobilityModel;

/**
 * \ingroup propagation
 *
 * \brief Models the propagation loss through a transmission medium
 *
 * Calculate the receive power (dbm) from a transmit power (dbm)
 * and a mobility model for the source and destination positions.
 */
class PropagationLossModel : public Object
{
public:
  /**
   * Get the type ID.
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  PropagationLossModel ();
  virtual ~PropagationLossModel ();

  // Delete copy constructor and assignment operator to avoid misuse
  PropagationLossModel (const PropagationLossModel &) = delete;
  PropagationLossModel & operator = (const PropagationLossModel &) = delete;

  /**
   * \brief Enables a chain of loss models to act on the signal
   * \param next The next PropagationLossModel to add to the chain
   *
   * This method of chaining propagation loss models only works commutatively
   * if the propagation loss of all models in the chain are independent
   * of transmit power.
   */
  void SetNext (Ptr<PropagationLossModel> next);

  /**
   * \brief Gets the next PropagationLossModel in the chain of loss models
   * that act on the signal.
   * \returns The next PropagationLossModel in the chain
   *
   * This method of chaining propagation loss models only works commutatively
   * if the propagation loss of all models in the chain are independent
   * of transmit power.
   */
  Ptr<PropagationLossModel> GetNext ();

  /**
   * Returns the Rx Power taking into account all the PropagationLossModel(s)
   * chained to the current one.
   *
   * \param txPowerDbm current transmission power (in dBm)
   * \param a the mobility model of the source
   * \param b the mobility model of the destination
   * \returns the reception power after adding/multiplying propagation loss (in dBm)
   */
  double CalcRxPower (double txPowerDbm,
                      Ptr<MobilityModel> a,
                      Ptr<MobilityModel> b) const;

  /**
   * If this loss model uses objects of type RandomVariableStream,
   * set the stream numbers to the integers starting with the offset
   * 'stream'. Return the number of streams (possibly zero) that
   * have been assigned.  If there are PropagationLossModels chained
   * together, this method will also assign streams to the
   * downstream models.
   *
   * \param stream the stream index offset start
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);

protected:
  /**
   * Assign a fixed random variable stream number to the random variables used by this model.
   * 
   * Subclasses must implement this; those not using random variables
   * can return zero.
   * 
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  virtual int64_t DoAssignStreams (int64_t stream) = 0;

private:
  /**
   * PropagationLossModel.
   *
   * \param txPowerDbm current transmission power (in dBm)
   * \param a the mobility model of the source
   * \param b the mobility model of the destination
   * \returns the reception power after adding/multiplying propagation loss (in dBm)
   */
  virtual double DoCalcRxPower (double txPowerDbm,
                                Ptr<MobilityModel> a,
                                Ptr<MobilityModel> b) const = 0;

  Ptr<PropagationLossModel> m_next; //!< Next propagation loss model in the list
};

/**
 * \ingroup propagation
 *
 * \brief The propagation loss follows a random distribution.
 */ 
class RandomPropagationLossModel : public PropagationLossModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  RandomPropagationLossModel ();
  virtual ~RandomPropagationLossModel ();

  // Delete copy constructor and assignment operator to avoid misuse
  RandomPropagationLossModel (const RandomPropagationLossModel &) = delete;
  RandomPropagationLossModel & operator = (const RandomPropagationLossModel &) = delete;

private:
  double DoCalcRxPower (double txPowerDbm,
                        Ptr<MobilityModel> a,
                        Ptr<MobilityModel> b) const override;
  int64_t DoAssignStreams (int64_t stream) override;

  Ptr<RandomVariableStream> m_variable; //!< random generator
};

/**
 * \ingroup propagation
 *
 * \brief a Friis propagation loss model
 *
 * The Friis propagation loss model was first described in
 * "A Note on a Simple Transmission Formula", by 
 * "Harald T. Friis".
 * 
 * The original equation was described as:
 *  \f$ \frac{P_r}{P_t} = \frac{A_r A_t}{d^2\lambda^2} \f$
 *  with the following equation for the case of an
 *  isotropic antenna with no heat loss:
 *  \f$ A_{isotr.} = \frac{\lambda^2}{4\pi} \f$
 *
 * The final equation becomes:
 * \f$ \frac{P_r}{P_t} = \frac{\lambda^2}{(4 \pi d)^2} \f$
 *
 * Modern extensions to this original equation are:
 * \f$ P_r = \frac{P_t G_t G_r \lambda^2}{(4 \pi d)^2 L}\f$
 *
 * With:
 *  - \f$ P_r \f$ : reception power (W)
 *  - \f$ P_t \f$ : transmission power (W)
 *  - \f$ G_t \f$ : transmission gain (unit-less)
 *  - \f$ G_r \f$ : reception gain (unit-less)
 *  - \f$ \lambda \f$ : wavelength (m)
 *  - \f$ d \f$ : distance (m)
 *  - \f$ L \f$ : system loss (unit-less)
 * 
 * In the implementation,  \f$ \lambda \f$ is calculated as 
 * \f$ \frac{C}{f} \f$, where  \f$ C = 299792458\f$ m/s is the speed of light in
 * vacuum, and \f$ f \f$ is the frequency in Hz which can be configured by
 * the user via the Frequency attribute.
 *
 * The Friis model is valid only for propagation in free space within
 * the so-called far field region, which can be considered
 * approximately as the region for \f$ d > 3 \lambda \f$.
 * The model will still return a value for \f$ d < 3 \lambda \f$, as
 * doing so (rather than triggering a fatal error) is practical for
 * many simulation scenarios. However, we stress that the values
 * obtained in such conditions shall not be considered realistic. 
 * 
 * Related with this issue, we note that the Friis formula is
 * undefined for \f$ d = 0 \f$, and results in 
 * \f$ P_r > P_t \f$ for \f$ d < \lambda / 2 \sqrt{\pi} \f$.
 * Both these conditions occur outside of the far field region, so in
 * principle the Friis model shall not be used in these conditions. 
 * In practice, however, Friis is often used in scenarios where accurate
 * propagation modeling is not deemed important, and values of \f$ d =
 * 0 \f$ can occur. To allow practical use of the model in such
 * scenarios, we have to 1) return some value for \f$ d = 0 \f$, and
 * 2) avoid large discontinuities in propagation loss values (which
 * could lead to artifacts such as bogus capture effects which are
 * much worse than inaccurate propagation loss values). The two issues
 * are conflicting, as, according to the Friis formula, 
 * \f$\lim_{d \to 0 }  P_r = +\infty \f$;
 * so if, for \f$ d = 0 \f$, we use a fixed loss value, we end up with an infinitely large
 * discontinuity, which as we discussed can cause undesirable
 * simulation artifacts.
 *
 * To avoid these artifact, this implementation of the Friis model
 * provides an attribute called MinLoss which allows to specify the
 * minimum total loss (in dB) returned by the model. This is used in
 * such a way that 
 * \f$ P_r \f$ continuously increases for \f$ d \to 0 \f$, until
 * MinLoss is reached, and then stay constant; this allow to
 * return a value for \f$ d  = 0 \f$ and at the same time avoid
 * discontinuities. The model won't be much realistic, but at least
 * the simulation artifacts discussed before are avoided. The default value of
 * MinLoss is 0 dB, which means that by default the model will return 
 * \f$ P_r = P_t \f$ for \f$ d <= \lambda / 2 \sqrt{\pi} \f$. We note
 * that this value of \f$ d \f$ is outside of the far field
 * region, hence the validity of the model in the far field region is
 * not affected.
 * 
 */
class FriisPropagationLossModel : public PropagationLossModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  FriisPropagationLossModel ();

  // Delete copy constructor and assignment operator to avoid misuse
  FriisPropagationLossModel (const FriisPropagationLossModel &) = delete;
  FriisPropagationLossModel & operator = (const FriisPropagationLossModel &) = delete;

  /**
   * \param frequency (Hz)
   *
   * Set the carrier frequency used in the Friis model 
   * calculation.
   */
  void SetFrequency (double frequency);
  /**
   * \param systemLoss (dimension-less)
   *
   * Set the system loss used by the Friis propagation model.
   */
  void SetSystemLoss (double systemLoss);

  /**
   * \param minLoss the minimum loss (dB)
   *
   * no matter how short the distance, the total propagation loss (in
   * dB) will always be greater or equal than this value 
   */
  void SetMinLoss (double minLoss);

  /**
   * \return the minimum loss.
   */
  double GetMinLoss (void) const;

  /**
   * \returns the current frequency (Hz)
   */
  double GetFrequency (void) const;
  /**
   * \returns the current system loss (dimension-less)
   */
  double GetSystemLoss (void) const;

private:
  double DoCalcRxPower (double txPowerDbm,
                        Ptr<MobilityModel> a,
                        Ptr<MobilityModel> b) const override;
  int64_t DoAssignStreams (int64_t stream) override;

  /**
   * Transforms a Dbm value to Watt
   * \param dbm the Dbm value
   * \return the Watts
   */
  double DbmToW (double dbm) const;

  /**
   * Transforms a Watt value to Dbm
   * \param w the Watt value
   * \return the Dbm
   */
  double DbmFromW (double w) const;

  double m_lambda;        //!< the carrier wavelength
  double m_frequency;     //!< the carrier frequency
  double m_systemLoss;    //!< the system loss
  double m_minLoss;       //!< the minimum loss
};

/**
 * \ingroup propagation
 *
 * \brief a Two-Ray Ground propagation loss model ported from NS2
 *
 * Two-ray ground reflection model.
 *
 * \f$ Pr = \frac{P_t * G_t * G_r * (H_t^2 * H_r^2)}{d^4 * L} \f$
 *
 * The original equation in Rappaport's book assumes L = 1.
 * To be consistent with the free space equation, L is added here.
 *
 * Ht and Hr are set at the respective nodes z coordinate plus a model parameter
 * set via SetHeightAboveZ.
 *
 * The two-ray model does not give a good result for short distances, due to the
 * oscillation caused by constructive and destructive combination of the two
 * rays. Instead the Friis free-space model is used for small distances. 
 *
 * The crossover distance, below which Friis is used, is calculated as follows:
 *
 * \f$ dCross = \frac{(4 * \pi * H_t * H_r)}{\lambda} \f$
 *
 * In the implementation,  \f$ \lambda \f$ is calculated as 
 * \f$ \frac{C}{f} \f$, where  \f$ C = 299792458\f$ m/s is the speed of light in
 * vacuum, and \f$ f \f$ is the frequency in Hz which can be configured by
 * the user via the Frequency attribute.
 */
class TwoRayGroundPropagationLossModel : public PropagationLossModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  TwoRayGroundPropagationLossModel ();

  // Delete copy constructor and assignment operator to avoid misuse
  TwoRayGroundPropagationLossModel (const TwoRayGroundPropagationLossModel &) = delete;
  TwoRayGroundPropagationLossModel & operator = (const TwoRayGroundPropagationLossModel &) = delete;

  /**
   * \param frequency (Hz)
   *
   * Set the carrier frequency used in the TwoRayGround model 
   * calculation.
   */
  void SetFrequency (double frequency);

  /**
   * \param systemLoss (dimension-less)
   *
   * Set the system loss used by the TwoRayGround propagation model.
   */
  void SetSystemLoss (double systemLoss);
  /**
   * \param minDistance the minimum distance
   *
   * Below this distance, the txpower is returned
   * unmodified as the rxpower.
   */
  void SetMinDistance (double minDistance);
  /**
   * \returns the minimum distance.
   */
  double GetMinDistance (void) const;

  /**
   * \returns the current frequency (Hz)
   */
  double GetFrequency (void) const;

  /**
   * \returns the current system loss (dimension-less)
   */
  double GetSystemLoss (void) const;
  /**
   * \param heightAboveZ the model antenna height above the node's Z coordinate
   *
   * Set the model antenna height above the node's Z coordinate
   */
  void SetHeightAboveZ (double heightAboveZ);

private:
  double DoCalcRxPower (double txPowerDbm,
                        Ptr<MobilityModel> a,
                        Ptr<MobilityModel> b) const override;
  int64_t DoAssignStreams (int64_t stream) override;

  /**
   * Transforms a Dbm value to Watt
   * \param dbm the Dbm value
   * \return the Watts
   */
  double DbmToW (double dbm) const;

  /**
   * Transforms a Watt value to Dbm
   * \param w the Watt value
   * \return the Dbm
   */
  double DbmFromW (double w) const;

  double m_lambda;        //!< the carrier wavelength
  double m_frequency;     //!< the carrier frequency
  double m_systemLoss;    //!< the system loss
  double m_minDistance;   //!< minimum distance for the model
  double m_heightAboveZ;  //!< antenna height above the node's Z coordinate
};

/**
 * \ingroup propagation
 *
 * \brief a log distance propagation model.
 *
 * This model calculates the reception power with a so-called
 * log-distance propagation model:
 * \f$ L = L_0 + 10 n log_{10}(\frac{d}{d_0})\f$
 *
 * where:
 *  - \f$ n \f$ : the path loss distance exponent
 *  - \f$ d_0 \f$ : reference distance (m)
 *  - \f$ L_0 \f$ : path loss at reference distance (dB)
 *  - \f$ d \f$ : distance (m)
 *  - \f$ L \f$ : path loss (dB)
 *
 * When the path loss is requested at a distance smaller than
 * the reference distance, the tx power is returned.
 *
 */
class LogDistancePropagationLossModel : public PropagationLossModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  LogDistancePropagationLossModel ();

  // Delete copy constructor and assignment operator to avoid misuse
  LogDistancePropagationLossModel (const LogDistancePropagationLossModel &) = delete;
  LogDistancePropagationLossModel & operator = (const LogDistancePropagationLossModel &) = delete;

  /**
   * \param n the path loss exponent.
   * Set the path loss exponent.
   */
  void SetPathLossExponent (double n);
  /** 
   * \returns the current path loss exponent.
   */
  double GetPathLossExponent (void) const;

  /**
   * Set the reference path loss at a given distance
   * \param referenceDistance reference distance
   * \param referenceLoss reference path loss
   */
  void SetReference (double referenceDistance, double referenceLoss);

private:
  double DoCalcRxPower (double txPowerDbm,
                        Ptr<MobilityModel> a,
                        Ptr<MobilityModel> b) const override;

  int64_t DoAssignStreams (int64_t stream) override;

  /**
   *  Creates a default reference loss model
   * \return a default reference loss model
   */
  static Ptr<PropagationLossModel> CreateDefaultReference (void);

  double m_exponent; //!< model exponent
  double m_referenceDistance; //!< reference distance
  double m_referenceLoss; //!< reference loss
};

/**
 * \ingroup propagation
 *
 * \brief A log distance path loss propagation model with three distance
 * fields. This model is the same as ns3::LogDistancePropagationLossModel
 * except that it has three distance fields: near, middle and far with
 * different exponents.
 *
 * Within each field the reception power is calculated using the log-distance
 * propagation equation:
 * \f[ L = L_0 + 10 \cdot n_0 log_{10}(\frac{d}{d_0})\f]
 * Each field begins where the previous ends and all together form a continuous function.
 *
 * There are three valid distance fields: near, middle, far. Actually four: the
 * first from 0 to the reference distance is invalid and returns txPowerDbm.
 *
 * \f[ \underbrace{0 \cdots\cdots}_{=0} \underbrace{d_0 \cdots\cdots}_{n_0} \underbrace{d_1 \cdots\cdots}_{n_1} \underbrace{d_2 \cdots\cdots}_{n_2} \infty \f]
 *
 * Complete formula for the path loss in dB:
 *
 * \f[\displaystyle L =
\begin{cases}
0 & d < d_0 \\
L_0 + 10 \cdot n_0 \log_{10}(\frac{d}{d_0}) & d_0 \leq d < d_1 \\
L_0 + 10 \cdot n_0 \log_{10}(\frac{d_1}{d_0}) + 10 \cdot n_1 \log_{10}(\frac{d}{d_1}) & d_1 \leq d < d_2 \\
L_0 + 10 \cdot n_0 \log_{10}(\frac{d_1}{d_0}) + 10 \cdot n_1 \log_{10}(\frac{d_2}{d_1}) + 10 \cdot n_2 \log_{10}(\frac{d}{d_2})& d_2 \leq d
\end{cases}\f]
 *
 * where:
 *  - \f$ L \f$ : resulting path loss (dB)
 *  - \f$ d \f$ : distance (m)
 *  - \f$ d_0, d_1, d_2 \f$ : three distance fields (m)
 *  - \f$ n_0, n_1, n_2 \f$ : path loss distance exponent for each field (unitless)
 *  - \f$ L_0 \f$ : path loss at reference distance (dB)
 *
 * When the path loss is requested at a distance smaller than the reference
 * distance \f$ d_0 \f$, the tx power (with no path loss) is returned. The
 * reference distance defaults to 1m and reference loss defaults to
 * ns3::FriisPropagationLossModel with 5.15 GHz and is thus \f$ L_0 \f$ = 46.67 dB.
 */
class ThreeLogDistancePropagationLossModel : public PropagationLossModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  ThreeLogDistancePropagationLossModel ();

  // Delete copy constructor and assignment operator to avoid misuse
  ThreeLogDistancePropagationLossModel (const ThreeLogDistancePropagationLossModel &) = delete;
  ThreeLogDistancePropagationLossModel & operator = (const ThreeLogDistancePropagationLossModel &) = delete;

  // Parameters are all accessible via attributes.

private:
  double DoCalcRxPower (double txPowerDbm,
                        Ptr<MobilityModel> a,
                        Ptr<MobilityModel> b) const override;

  int64_t DoAssignStreams (int64_t stream) override;

  double m_distance0; //!< Beginning of the first (near) distance field
  double m_distance1; //!< Beginning of the second (middle) distance field.
  double m_distance2; //!< Beginning of the third (far) distance field.

  double m_exponent0; //!< The exponent for the first field.
  double m_exponent1; //!< The exponent for the second field.
  double m_exponent2; //!< The exponent for the third field.

  double m_referenceLoss; //!< The reference loss at distance d0 (dB).
};

/**
 * \ingroup propagation
 *
 * \brief Nakagami-m fast fading propagation loss model.
 * 
 * This propagation loss model implements the Nakagami-m fast fading
 * model, which accounts for the variations in signal strength due to multipath
 * fading. The model does not account for the path loss due to the
 * distance traveled by the signal, hence for typical simulation usage it
 * is recommended to consider using it in combination with other models
 * that take into account this aspect. 
 *
 * The Nakagami-m distribution is applied to the power level. The probability
 * density function is defined as
 * \f[ p(x; m, \omega) = \frac{2 m^m}{\Gamma(m) \omega^m} x^{2m - 1} e^{-\frac{m}{\omega} x^2}  \f]
 * with \f$ m \f$ the fading depth parameter and \f$ \omega \f$ the average received power.
 *
 * It is implemented by either a ns3::GammaRandomVariable or a 
 * ns3::ErlangRandomVariable random variable.
 *
 * The implementation of the model allows to specify different values of the m parameter (and hence different fading profiles)
 * for three different distance ranges:
 * \f[ \underbrace{0 \cdots\cdots}_{m_0} \underbrace{d_1 \cdots\cdots}_{m_1} \underbrace{d_2 \cdots\cdots}_{m_2} \infty \f]
 *
 * For m = 1 the Nakagami-m distribution equals the Rayleigh distribution. Thus
 * this model also implements Rayleigh distribution based fast fading.
 */
class NakagamiPropagationLossModel : public PropagationLossModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  NakagamiPropagationLossModel ();

  // Delete copy constructor and assignment operator to avoid misuse
  NakagamiPropagationLossModel (const NakagamiPropagationLossModel &) = delete;
  NakagamiPropagationLossModel & operator = (const NakagamiPropagationLossModel &) = delete;

  // Parameters are all accessible via attributes.

private:
  double DoCalcRxPower (double txPowerDbm,
                        Ptr<MobilityModel> a,
                        Ptr<MobilityModel> b) const override;

  int64_t DoAssignStreams (int64_t stream) override;

  double m_distance1; //!< Distance1
  double m_distance2; //!< Distance2

  double m_m0;        //!< m for distances smaller than Distance1
  double m_m1;        //!< m for distances smaller than Distance2
  double m_m2;        //!< m for distances greater than Distance2

  Ptr<ErlangRandomVariable>  m_erlangRandomVariable; //!< Erlang random variable
  Ptr<GammaRandomVariable> m_gammaRandomVariable;    //!< Gamma random variable
};

/**
 * \ingroup propagation
 *
 * \brief Return a constant received power level independent of the transmit 
 *  power
 *
 * The received power is constant independent of the transmit power.  The user
 * must set received power level through the Rss attribute or public 
 * SetRss() method.  Note that if this loss model is chained to other loss
 * models via SetNext() method, it can only be the first loss model in such
 * a chain, or else it will disregard the losses computed by loss models
 * that precede it in the chain. 
 */ 
class FixedRssLossModel : public PropagationLossModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  FixedRssLossModel ();
  virtual ~FixedRssLossModel ();

  // Delete copy constructor and assignment operator to avoid misuse
  FixedRssLossModel (const FixedRssLossModel &) = delete;
  FixedRssLossModel & operator = (const FixedRssLossModel &) = delete;

  /**
   * \param rss (dBm) the received signal strength
   *
   * Set the received signal strength (RSS) in dBm.
   */
  void SetRss (double rss);

private:
  double DoCalcRxPower (double txPowerDbm,
                        Ptr<MobilityModel> a,
                        Ptr<MobilityModel> b) const override;

  int64_t DoAssignStreams (int64_t stream) override;

  double m_rss; //!< the received signal strength
};

/**
 * \ingroup propagation
 *
 * \brief The propagation loss is fixed for each pair of nodes and doesn't depend on their actual positions.
 * 
 * This is supposed to be used by synthetic tests. Note that by default propagation loss is assumed to be symmetric.
 */
class MatrixPropagationLossModel : public PropagationLossModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  MatrixPropagationLossModel ();
  virtual ~MatrixPropagationLossModel ();

  // Delete copy constructor and assignment operator to avoid misuse
  MatrixPropagationLossModel (const MatrixPropagationLossModel &) = delete;
  MatrixPropagationLossModel & operator = (const MatrixPropagationLossModel &) = delete;

  /**
   * \brief Set loss (in dB, positive) between pair of ns-3 objects
   * (typically, nodes).
   * 
   * \param a ma          Source mobility model
   * \param b mb          Destination mobility model
   * \param loss        a -> b path loss, positive in dB
   * \param symmetric   If true (default), both a->b and b->a paths will be affected
   */ 
  void SetLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b, double loss, bool symmetric = true);

  /**
   * Set the default propagation loss (in dB, positive) to be used, infinity if not set
   * \param defaultLoss the default proagation loss
   */
  void SetDefaultLoss (double defaultLoss);

private:
  double DoCalcRxPower (double txPowerDbm,
                        Ptr<MobilityModel> a,
                        Ptr<MobilityModel> b) const override;

  int64_t DoAssignStreams (int64_t stream) override;

  double m_default; //!< default loss

  /// Typedef: Mobility models pair
  typedef std::pair< Ptr<MobilityModel>, Ptr<MobilityModel> > MobilityPair; 

  std::map<MobilityPair, double> m_loss; //!< Propagation loss between pair of nodes
};

/**
 * \ingroup propagation
 *
 * \brief The propagation loss depends only on the distance (range) between transmitter and receiver.
 *
 * The single MaxRange attribute (units of meters) determines path loss.
 * Receivers at or within MaxRange meters receive the transmission at the
 * transmit power level. Receivers beyond MaxRange receive at power
 * -1000 dBm (effectively zero).
*/
class RangePropagationLossModel : public PropagationLossModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  RangePropagationLossModel ();

  // Delete copy constructor and assignment operator to avoid misuse
  RangePropagationLossModel (const RangePropagationLossModel &) = delete;
  RangePropagationLossModel & operator = (const RangePropagationLossModel &) = delete;

private:
  double DoCalcRxPower (double txPowerDbm,
                        Ptr<MobilityModel> a,
                        Ptr<MobilityModel> b) const override;

  int64_t DoAssignStreams (int64_t stream) override;

  double m_range; //!< Maximum Transmission Range (meters)
};

} // namespace ns3

#endif /* PROPAGATION_LOSS_MODEL_H */
