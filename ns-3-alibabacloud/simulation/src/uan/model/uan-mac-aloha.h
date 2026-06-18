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

#ifndef UAN_MAC_ALOHA_H
#define UAN_MAC_ALOHA_H

#include "uan-mac.h"
#include "ns3/mac8-address.h"

namespace ns3
{


class UanPhy;
class UanTxMode;

/**
 * \ingroup uan
 *
 * ALOHA MAC Protocol, the simplest MAC protocol for wireless networks.
 *
 * Packets enqueued are immediately transmitted.  This MAC attaches
 * a UanHeaderCommon to outgoing packets for address information.
 * (The type field is not used)
 */
class UanMacAloha : public UanMac
{
public:
  /** Default constructor */
  UanMacAloha ();
  /** Dummy destructor, see DoDispose. */
  virtual ~UanMacAloha ();
  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);


  // Inherited methods
  virtual bool Enqueue (Ptr<Packet> pkt, uint16_t protocolNumber, const Address &dest);
  virtual void SetForwardUpCb (Callback<void, Ptr<Packet>, uint16_t, const Mac8Address&> cb);
  virtual void AttachPhy (Ptr<UanPhy> phy);
  virtual void Clear (void);
  int64_t AssignStreams (int64_t stream);

private:
  /** PHY layer attached to this MAC. */
  Ptr<UanPhy> m_phy;
  /** Forwarding up callback. */
  Callback<void, Ptr<Packet>, uint16_t, const Mac8Address& > m_forUpCb;
  /** Flag when we've been cleared. */
  bool m_cleared;

  /**
   * Receive packet from lower layer (passed to PHY as callback).
   *
   * \param pkt Packet being received.
   * \param sinr SINR of received packet.
   * \param txMode Mode of received packet.
   */
  void RxPacketGood (Ptr<Packet> pkt, double sinr, UanTxMode txMode);

  /**
   * Packet received at lower layer in error.
   *
   * \param pkt Packet received in error.
   * \param sinr SINR of received packet.
   */
  void RxPacketError (Ptr<Packet> pkt, double sinr);
protected:
  virtual void DoDispose ();

};  // class UanMacAloha

} // namespace ns3

#endif /* UAN_MAC_ALOHA_H */
