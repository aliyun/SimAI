/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Nicola Baldo  <nbaldo@cttc.es>
 */


#ifndef EPC_TFT_CLASSIFIER_H
#define EPC_TFT_CLASSIFIER_H


#include "ns3/ptr.h"
#include "ns3/simple-ref-count.h"
#include "ns3/epc-tft.h"

#include <map>


namespace ns3 {

class EpcTft;
class Packet;

/**
 * \brief classifies IP packets according to Traffic Flow Templates (TFTs)
 *
 * \note this implementation works with IPv4 and IPv6.
 * When there is fragmentation of IP packets, UDP/TCP ports maybe missing.
 *
 * The following actions are performed to use the port info present in the first segment with
 * the next fragments:
 *  - Port info is stored if it is available, i.e. it is the first fragment with UDP/TCP protocol
 *    and there is enough data in the payload of the IP packet for the port numbers.
 *  - Port info is used for the next fragments.
 *  - Port info is deleted, when the last fragment is processed.
 *
 * When we cannot cache the port info, the TFT of the default bearer is used. This may happen
 * if there is reordering or losses of IP packets.
 */
class EpcTftClassifier : public SimpleRefCount<EpcTftClassifier>
{
public:
  
  EpcTftClassifier ();
  
  /** 
   * add a TFT to the Classifier
   * 
   * \param tft the TFT to be added
   * \param id the ID of the bearer which will be classified by specified TFT classifier
   * 
   */
  void Add (Ptr<EpcTft> tft, uint32_t id);

  /** 
   * delete an existing TFT from the classifier
   * 
   * \param id the identifier of the TFT to be deleted
   */
  void Delete (uint32_t id);
  

  /** 
   * classify an IP packet
   * 
   * \param p the IP packet. The outmost header can only be an IPv4 or an IPv6 header.
   * \param direction the EPC TFT direction (can be downlink, uplink or bi-directional)
   * \param protocolNumber the protocol of the packet. Only IPv4 and IPv6 are supported.
   *
   * \return the identifier (>0) of the first TFT that matches with the IP packet; 0 if no TFT matched.
   */
  uint32_t Classify (Ptr<Packet> p, EpcTft::Direction direction, uint16_t protocolNumber);
  
protected:
  
  std::map <uint32_t, Ptr<EpcTft> > m_tftMap; ///< TFT map

  std::map < std::tuple<uint32_t, uint32_t, uint8_t, uint16_t>,
             std::pair<uint32_t, uint32_t> >
      m_classifiedIpv4Fragments; ///< Map with already classified IPv4 Fragments
                                 ///< An entry is added when the port info is available, i.e.
                                 ///<   first fragment, UDP/TCP protocols and enough payload data
                                 ///< An entry is used if port info is not available, i.e.
                                 ///<   not first fragment or not enough payload data for TCP/UDP
                                 ///< An entry is removed when the last fragment is classified
                                 ///<   Note: If last fragment is lost, entry is not removed
};



} // namespace ns3
  
#endif /* TFT_CLASSIFIER_H */ 
