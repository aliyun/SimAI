/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Authors: Joe Kopena <tjkopena@cs.drexel.edu>
 *
 * These applications are used in the WiFi Distance Test experiment,
 * described and implemented in test02.cc.  That file should be in the
 * same place as this file.  The applications have two very simple
 * jobs, they just generate and receive packets.  We could use the
 * standard Application classes included in the NS-3 distribution.
 * These have been written just to change the behavior a little, and
 * provide more examples.
 *
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/application.h"

#include "ns3/stats-module.h"

using namespace ns3;

/**
 * Sender application.
 */
class Sender : public Application {
public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  Sender();
  virtual ~Sender();

protected:
  virtual void DoDispose (void);

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  /**
   * Send a packet.
   */
  void SendPacket ();

  uint32_t        m_pktSize;    //!< The packet size.
  Ipv4Address     m_destAddr;   //!< Destination address.
  uint32_t        m_destPort;   //!< Destination port.
  Ptr<ConstantRandomVariable> m_interval; //!< Rng for sending packets.
  uint32_t        m_numPkts;    //!< Number of packets to send.

  Ptr<Socket>     m_socket;     //!< Sending socket.
  EventId         m_sendEvent;  //!< Send packet event.

  /// Tx TracedCallback.
  TracedCallback<Ptr<const Packet> > m_txTrace;

  uint32_t        m_count;      //!< Number of packets sent.

  // end class Sender
};


/**
 * Receiver application.
 */
class Receiver : public Application {
public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  Receiver();
  virtual ~Receiver();

  /**
   * Set the counter calculator for received packets.
   * \param calc The CounterCalculator.
   */
  void SetCounter (Ptr<CounterCalculator<> > calc);

  /**
   * Set the delay tracker for received packets.
   * \param delay The Delay calculator.
   */
  void SetDelayTracker (Ptr<TimeMinMaxAvgTotalCalculator> delay);

protected:
  virtual void DoDispose (void);

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  /**
   * Receive a packet.
   * \param socket The receiving socket.
   */
  void Receive (Ptr<Socket> socket);

  Ptr<Socket>     m_socket; //!< Receiving socket.
  uint32_t        m_port;   //!< Listening port.

  Ptr<CounterCalculator<> > m_calc; //!< Counter of the number of received packets.
  Ptr<TimeMinMaxAvgTotalCalculator> m_delay;  //!< Delay calculator.

  // end class Receiver
};


/**
 * Timestamp tag - it carries when the packet has been sent.
 * 
 * It would have been more realistic to include this info in 
 * a header. Here we show how to avoid the extra overhead in
 * a simulation.
 */
class TimestampTag : public Tag {
public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer i) const;
  virtual void Deserialize (TagBuffer i);

  /**
   * Set the timestamp.
   * \param time The timestamp.
   */
  void SetTimestamp (Time time);
  /**
   * Get the timestamp.
   * \return the timestamp.
   */
  Time GetTimestamp (void) const;

  void Print (std::ostream &os) const;

private:
  Time m_timestamp; //!< Timestamp.

  // end class TimestampTag
};
