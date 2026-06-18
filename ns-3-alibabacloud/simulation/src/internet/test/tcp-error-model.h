/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Natale Patriciello <natale.patriciello@gmail.com>
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
 */
#ifndef TCPERRORCHANNEL_H
#define TCPERRORCHANNEL_H

#include "ns3/error-model.h"
#include "ns3/tcp-header.h"
#include "ns3/ipv4-header.h"

namespace ns3 {

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief A general (TCP-aware) error model.
 *
 * The class is responsible to take away the IP and TCP header from the packet,
 * and then to interrogate the method ShouldDrop, dropping the packet accordingly
 * to the returned value.
 */
class TcpGeneralErrorModel : public ErrorModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  TcpGeneralErrorModel ();

  /**
   * \brief Set the drop callback.
   * \param cb The callback to be set.
   */
  void SetDropCallback (Callback<void, const Ipv4Header&, const TcpHeader&, Ptr<const Packet> > cb)
  {
    m_dropCallback = cb;
  }

protected:
  /**
   * \brief Check if the packet should be dropped.
   * \param ipHeader The packet IPv4 header.
   * \param tcpHeader The packet TCP header.
   * \param packetSize The packet size.
   * \returns True if the packet should be dropped.
   */
  virtual bool ShouldDrop (const Ipv4Header &ipHeader, const TcpHeader &tcpHeader,
                           uint32_t packetSize) = 0;


private:
  virtual bool DoCorrupt (Ptr<Packet> p);
  Callback<void, const Ipv4Header&, const TcpHeader&, Ptr<const Packet> > m_dropCallback; //!< Drop callback.
};

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief An error model TCP aware: it drops the sequence number declared.
 *
 * \see AddSeqToKill
 */
class TcpSeqErrorModel : public TcpGeneralErrorModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  TcpSeqErrorModel () : TcpGeneralErrorModel () { }

  /**
   * \brief Add the sequence number to the list of segments to be killed
   *
   * Calling x times this function indicates that you want to kill
   * the segment x times.
   *
   * \param seq sequence number to be killed
   */
  void AddSeqToKill (const SequenceNumber32 &seq)
  {
    m_seqToKill.insert(m_seqToKill.end(), seq);
  }

protected:
  virtual bool ShouldDrop (const Ipv4Header &ipHeader, const TcpHeader &tcpHeader,
                           uint32_t packetSize);

protected:
  std::list<SequenceNumber32> m_seqToKill; //!< List of the sequence numbers to be dropped.

private:
  virtual void DoReset (void);
};

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Error model which drop packets with specified TCP flags.
 *
 * Set the flags with SetFlagToKill and the number of the packets with such flags
 * which should be killed.
 *
 * \see SetFlagToKill
 * \see SetKillRepeat
 *
 */
class TcpFlagErrorModel : public TcpGeneralErrorModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  TcpFlagErrorModel ();

  /**
   * \brief Set the flags of the segment that should be killed
   *
   * \param flags Flags
   */
  void SetFlagToKill (TcpHeader::Flags_t flags)
  {
    m_flagsToKill = flags;
  }

  /**
   * \brief Set how many packets should be killed
   *
   * If the flags are the same, this specified the numbers of drops:
   *
   * # -1 for infinite drops
   * # 0  for no drops
   * # >1 the number of drops
   *
   * \param killNumber Specifies the number of times the packet should be killed
   */
  void SetKillRepeat (int16_t killNumber)
  {
    m_killNumber = killNumber;
  }

protected:
  virtual bool ShouldDrop (const Ipv4Header &ipHeader, const TcpHeader &tcpHeader,
                           uint32_t packetSize);

protected:
  TcpHeader::Flags_t m_flagsToKill; //!< Flags a packet should have to be dropped.
  int16_t m_killNumber;  //!< The number of times the packet should be killed.

private:
  virtual void DoReset (void);
};

} // namespace ns3

#endif // TCPERRORCHANNEL_H
