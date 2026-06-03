/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */


#ifndef SPECTRUM_ERROR_MODEL_H
#define SPECTRUM_ERROR_MODEL_H

#include <ns3/spectrum-value.h>
#include <ns3/ptr.h>
#include <ns3/packet.h>
#include <ns3/nstime.h>
#include <ns3/object.h>

namespace ns3 {



/**
 * \ingroup spectrum
 *
 */
class SpectrumErrorModel :  public Object
{
public:

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId ();
  virtual ~SpectrumErrorModel ();

  /**
   * Start a packet reception
   * \param p the packet
   */
  virtual void StartRx (Ptr<const Packet> p) = 0;

  /**
   * Evaluates a chunk
   * \param sinr the SpectrumValue experienced by the Chunk
   * \param duration the Chunk length
   */
  virtual void EvaluateChunk (const SpectrumValue& sinr, Time duration) = 0;

  /**
   * Checks if the packet being received is correct
   * \returns true if the packet is correct.
   */
  virtual bool IsRxCorrect () = 0;
};


/**
 * \ingroup spectrum
 *
 * This class implements the error model described in this paper:
 * N. Baldo and M. Miozzo, "Spectrum-aware Channel and PHY layer modeling
 * for ns3", in Proceedings of International Workshop on Network
 * Simulation Tools (NSTOOLS 2009), 19 October 2009, Pisa (Italy).
 */
class ShannonSpectrumErrorModel : public SpectrumErrorModel
{
protected:
  virtual void DoDispose ();

public:
  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  // inherited from SpectrumErrorModel
  void StartRx (Ptr<const Packet> p);
  void EvaluateChunk (const SpectrumValue& sinr, Time duration);
  bool IsRxCorrect ();

private:
  uint32_t m_bytes;             //!< Length of the packet being received
  uint32_t m_deliverableBytes;  //!< Bytes that can be received according to the Shnanon's formula

};


} // namespace ns3



#endif /* SPECTRUM_ERROR_MODEL_H */
