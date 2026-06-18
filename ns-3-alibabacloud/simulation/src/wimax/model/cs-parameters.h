/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright (c) 2009 INRIA, UDcast
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
 *         Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                              <amine.ismail@udcast.com>
 *
 */

#ifndef WIMAX_CS_PARAMETERS_H
#define WIMAX_CS_PARAMETERS_H

#include "ipcs-classifier-record.h"
#include "wimax-tlv.h"

namespace ns3 {

/**
 * \ingroup wimax
 * \brief CsParameters class
 */
class CsParameters
{
public:
  /// Action enumeration
  enum Action
  {
    ADD = 0,
    REPLACE = 1,
    DELETE = 2
  };
  CsParameters ();
  ~CsParameters ();
  /**
   * \brief creates a convergence sub-layer parameters from a tlv
   * \param tlv the TLV
   */
  CsParameters (Tlv tlv);
  /**
   * \brief creates a convergence sub-layer parameters from an ipcs classifier record
   * \param classifierDscAction the DCS action type
   * \param classifier the IPCS classifier record
   */
  CsParameters (enum Action classifierDscAction, IpcsClassifierRecord classifier);
  /**
   * \brief sets the dynamic service classifier action to ADD, Change or delete. Only ADD is supported
   * \param action the action enumeration
   */
  void SetClassifierDscAction (enum Action action);
  /**
   * \brief sets the packet classifier rules
   * \param packetClassifierRule the IPCS classifier record
   */
  void SetPacketClassifierRule (IpcsClassifierRecord packetClassifierRule);
  /**
   * \return the  dynamic service classifier action
   */
  enum Action GetClassifierDscAction (void) const;
  /**
   * \return the  the packet classifier rules
   */
  IpcsClassifierRecord GetPacketClassifierRule (void) const;
  /**
   * \brief creates a tlv from the classifier record
   * \return the created tlv
   */
  Tlv ToTlv (void) const;

private:
  enum Action m_classifierDscAction; ///< classifier DSC action
  IpcsClassifierRecord m_packetClassifierRule; ///< packet classifier rule
};

}
#endif /* WIMAX_CS_PARAMETERS_H */
