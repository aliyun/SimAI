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
 * Author: Manuel Requena <manuel.requena@cttc.es> : Based on lte-ue-phy code
 */

#ifndef LTE_TEST_UE_PHY_H
#define LTE_TEST_UE_PHY_H

#include "ns3/lte-phy.h"

#include "ns3/lte-control-messages.h"

namespace ns3 {

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Defines a simplified LtePhy class that is used for testing purposes
 * of downlink and uplink SINR generation. Used in LteDownlinkDataSinrTestCase
 * and LteUplinkDataSinrTestCase as simplified LTE PHY.
 */
class LteTestUePhy : public LtePhy
{
public:
  /**
   * @warning the default constructor should not be used
   */
  LteTestUePhy ();

  /**
   * \param dlPhy the downlink LteSpectrumPhy instance
   * \param ulPhy the uplink LteSpectrumPhy instance
   */
  LteTestUePhy (Ptr<LteSpectrumPhy> dlPhy, Ptr<LteSpectrumPhy> ulPhy);

  virtual ~LteTestUePhy ();

  virtual void DoDispose ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Queue the MAC PDU to be sent
   * \param p the MAC PDU to sent
   */
  virtual void DoSendMacPdu (Ptr<Packet> p);

  /**
   * \brief Create the PSD for the TX
   * \return the pointer to the PSD
   */
  virtual Ptr<SpectrumValue> CreateTxPowerSpectralDensity ();

  virtual void GenerateCtrlCqiReport (const SpectrumValue& sinr);
  
  virtual void GenerateDataCqiReport (const SpectrumValue& sinr);

  virtual void ReportInterference (const SpectrumValue& interf);

  virtual void ReportRsReceivedPower (const SpectrumValue& power);

  /**
   * \brief Reeive LTE Control Message
   * \param msg the control message
   */
  virtual void ReceiveLteControlMessage (Ptr<LteControlMessage> msg);

  /**
   * \brief Get the SINR
   * \return the SINR
   */
  SpectrumValue GetSinr ();

private:
  SpectrumValue m_sinr; ///< the SINR
};


} // namespace ns3

#endif /* LTE_TEST_UE_PHY_H */
