/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#ifndef UAN_PROP_MODEL_H
#define UAN_PROP_MODEL_H

#include "ns3/object.h"
#include "ns3/mobility-model.h"
#include "ns3/nstime.h"


#include <vector>
#include <complex>
#include <utility>

namespace ns3 {

class UanTxMode;

/**
 * \ingroup uan
 *
 * Holds PDP Tap information (amplitude and delay)
 */
class Tap
{
public:
  /**
   * Default constructor.  Creates Tap with delay=0, amp=0
   */
  Tap ();
  /**
   * Constructor
   *
   * \param delay Time delay (usually from first arrival) of signal
   * \param amp Complex amplitude of arrival
   */
  Tap (Time delay, std::complex<double> amp);
  /**
   * Get the complex amplitude of arrival.
   *
   * \return The amplitude.
   */
  std::complex<double> GetAmp (void) const;
  /**
   * Get the delay time, usually from first arrival of signal.
   * \return The time delay.
   */
  Time GetDelay (void) const;

private:
  std::complex<double> m_amplitude;  //!< The amplitude.
  Time m_delay;                      //!< The time delay.

};  // class Tap

  
/**
 * \ingroup uan
 *
 * The power delay profile returned by propagation models.
 *
 * Container class to describe power delay profile returned
 * from UAN propagation models using tapped delay line model.
 * This should model a channel impulse response as a set of
 * equally spaced signal arrivals.
 *
 * Generally, the profile should be normalized, such that
 * the sum of all taps should equal 1.  The received signal
 * power on any interval (t1, t2) can then be found from
 * summing the taps on the interval and multiplying by
 * the total received power at the receiver.
 */
class UanPdp
{
public:
  /**
   * Convenience iterator typedef.
   */
  typedef std::vector<Tap>::const_iterator Iterator;
  /**
   * Create empty PDP object.
   */
  UanPdp ();
  /**
   * Create PDP object from a vector of Tap objects.
   *
   * \param taps Taps to include in this PDP.
   * \param resolution Resolution of PDP object.
   */
  UanPdp (std::vector<Tap> taps, Time resolution);
  /**
   * Create PDP object from vector of arrival amplitudes.
   *
   * \param arrivals  Vector of complex amplitude arrivals.
   * \param resolution Time duration between arrivals in vector.
   */
  UanPdp (std::vector<std::complex<double > > arrivals, Time resolution);
  /**
   * Create PDP object from real valued arrival amplitudes.
   *
   * \param arrivals Vector of real valued arrivals.
   * \param resolution Time duration between arrivals in vector.
   */
  UanPdp (std::vector<double> arrivals, Time resolution);
  /** Dummy destructor, see DoDispose. */
  ~UanPdp ();

  /**
   * Set the arrival value for a tap.
   *
   * The delay time is the index multiplied by the resolution.
   * The tap vector will be expanded to accommodate the requested
   * index.
   *
   * \param arrival Complex arrival value.
   * \param index Index of arrival.
   */
  void SetTap (std::complex<double> arrival, uint32_t index);
  /**
   * Resize the tap vector.
   *
   * \param nTaps Number of taps in this PDP
   */
  void SetNTaps (uint32_t nTaps);
  /**
   * Set the time duration (resolution) between arrivals.
   *
   * \param resolution The resolution.
   */
  void SetResolution (Time resolution);
  /**
   * Get the beginning of the tap vector.
   *
   * \return Iterator positioned at first arrival.
   */
  Iterator GetBegin (void) const;
  /**
   * Get the end of the tap list (one beyond the last entry).
   *
   * \return Iterator positioned after last arrival
   */
  Iterator GetEnd (void) const;
  /**
   * Get the number of taps.
   *
   * \return Number of taps in PDP.
   */
  uint32_t GetNTaps (void) const;
  /**
   * Get the Tap at the specified delay index.
   *
   * \param i Index number of tap to return (0 based).
   * \return Tap object at index i.
   */
  const Tap &GetTap (uint32_t i) const;
  /**
   * Get the delay time resolution (time duration between arrivals).
   * 
   * \return Resolution of PDP.
   */
  Time GetResolution (void) const;
  /**
   * Compute the non-coherent sum of tap amplitudes
   * between a start and end time.
   *
   * Assuming that Tap at index 0 arrives at time 0,
   * this function sums non-coherently (sums amplitude of arrivals
   * ignoring phase difference) all arrivals between a start
   * and end time.
   *
   * \param begin Time value to begin summing arrivals.
   * \param end Time value to end summing arrivals.
   * \return Non-coherent sum of arrivals between two time values.
   */
  double SumTapsNc (Time begin, Time end) const;
  /**
   * Compute the coherent sum of tap amplitudes
   * between a start and end time.
   *
   * Assuming that Tap at index 0 arrives at time 0,
   * this function sums coherently (sums amplitude of arrivals
   * considering phase difference) all arrivals between a start
   * and end time.
   *
   * \param begin Time value to begin summing arrivals.
   * \param end Time value to end summing arrivals.
   * \return Coherent sum of arrivals between two time values.
   */
  std::complex<double> SumTapsC (Time begin, Time end) const;
  /**
   * Compute the non-coherent sum of tap amplitudes
   * starting after a delay from the maximum amplitude
   * for a total time duration.
   *
   * This function sums non-coherently (sums amplitude of arrivals
   * ignoring phase difference) all arrivals in a given duration
   * starting the given time after the maximum amplitude arrival received.
   *
   * \param delay Time duratation after max to begin summing arrivals.
   * \param duration Time duration to sum arrivals for.
   * \return Non-coherent sum of arrivals after max in given window.
   */
  double SumTapsFromMaxNc (Time delay, Time duration) const;
  /**
   * Compute the coherent sum of tap amplitudes
   * starting after a delay from the maximum amplitude
   * for a total duration.
   *
   * this function sums coherently (sums amplitude of arrivals
   * considering phase difference) all arrivals in a given duration
   * starting the given time after the maximum amplitude arrival received
   *
   * \param delay Time duratation after max to begin summing arrivals.
   * \param duration Time duration to sum arrivals for.
   * \return Coherent sum of arrivals after max in given window.
   */
  std::complex<double> SumTapsFromMaxC (Time delay, Time duration) const;
   /**
   * Creates a new UanPdp normalized to its non coherent sum.
   * \see SumTapsNc
   * \returns the new PDP
   */
  UanPdp NormalizeToSumNc (void) const;
  /**
   * Get a unit impulse PDP at time 0.
   *
   * \return The unit impulse.
   */ 
  static UanPdp CreateImpulsePdp (void);
  
private:
  friend std::ostream &operator<< (std::ostream &os, const UanPdp &pdp);
  friend std::istream &operator>> (std::istream &is, UanPdp &pdp);
  
  std::vector<Tap> m_taps;  //!< The vector of Taps.
  Time m_resolution;        //!< The time resolution.

};  // class UanPdp


/**
 * \ingroup uan
 *
 * Writes PDP to stream as list of arrivals
 *
 * \param os The output stream.
 * \param pdp The PDP.
 * \return The output stream.
 */
std::ostream &operator<< (std::ostream &os, const UanPdp &pdp);
/**
 * \ingroup uan
 *
 * Reads in list of arrivals from stream is
 *
 * \param is The input stream.
 * \param pdp The PDP variable to set.
 * \return The input stream.
 */
std::istream &operator>> (std::istream &is, UanPdp &pdp);  

  
/**
 * \ingroup uan
 *
 * Base class for implemented underwater propagation models
 */
class UanPropModel : public Object
{
public:
  /**
   * Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * Computes pathloss between nodes a and b.
   *
   * \param a Ptr to mobility model of node a
   * \param b Ptr to mobility model of node b
   * \param txMode TX mode of transmission between a and b
   * \return Pathloss in dB re 1 uPa
   */
  virtual double GetPathLossDb (Ptr<MobilityModel> a, Ptr<MobilityModel> b, UanTxMode txMode) = 0;

  /**
   * Get the PDP for the path between two nodes.
   *
   * \param a Ptr to mobility model of node a.
   * \param b Ptr to mobility model of node b.
   * \param mode TX mode of transmission from a to b.
   * \return PDP for link between nodes a and b.
   */
  virtual UanPdp GetPdp (Ptr<MobilityModel> a, Ptr<MobilityModel> b, UanTxMode mode) = 0;
  /**
   * Finds propagation delay between nodes a and b.
   *
   * \param a Ptr to mobility model of node a.
   * \param b Ptr to mobility model of node b.
   * \param mode TX mode of transmission.
   * \return Propagation delay.
   */
  virtual Time GetDelay (Ptr<MobilityModel> a, Ptr<MobilityModel> b, UanTxMode mode) = 0;

  /** Clear all pointer references. */
  virtual void Clear (void);

  virtual void DoDispose (void);

};  // class UanPropModel

} // namespace ns3

#endif /* UAN_PROP_MODEL_H */
