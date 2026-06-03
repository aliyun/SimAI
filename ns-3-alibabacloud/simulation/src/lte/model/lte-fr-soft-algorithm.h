/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Piotr Gawlowicz
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
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 *
 */

#ifndef LTE_FR_SOFT_ALGORITHM_H
#define LTE_FR_SOFT_ALGORITHM_H

#include <ns3/lte-ffr-algorithm.h>
#include <ns3/lte-ffr-sap.h>
#include <ns3/lte-ffr-rrc-sap.h>
#include <ns3/lte-rrc-sap.h>
#include <map>

namespace ns3 {


/**
 * \brief Soft Frequency Reuse algorithm implementation
 */
class LteFrSoftAlgorithm : public LteFfrAlgorithm
{
public:
  /**
   * \brief Creates a trivial ffr algorithm instance.
   */
  LteFrSoftAlgorithm ();

  virtual ~LteFrSoftAlgorithm ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  // inherited from LteFfrAlgorithm
  virtual void SetLteFfrSapUser (LteFfrSapUser* s);
  virtual LteFfrSapProvider* GetLteFfrSapProvider ();

  virtual void SetLteFfrRrcSapUser (LteFfrRrcSapUser* s);
  virtual LteFfrRrcSapProvider* GetLteFfrRrcSapProvider ();

  /// let the forwarder class access the protected and private members
  friend class MemberLteFfrSapProvider<LteFrSoftAlgorithm>;
  /// let the forwarder class access the protected and private members
  friend class MemberLteFfrRrcSapProvider<LteFrSoftAlgorithm>;

protected:
  // inherited from Object
  virtual void DoInitialize ();
  virtual void DoDispose ();

  virtual void Reconfigure ();

  // FFR SAP PROVIDER IMPLEMENTATION
  virtual std::vector <bool> DoGetAvailableDlRbg ();
  virtual bool DoIsDlRbgAvailableForUe (int i, uint16_t rnti);
  virtual std::vector <bool> DoGetAvailableUlRbg ();
  virtual bool DoIsUlRbgAvailableForUe (int i, uint16_t rnti);
  virtual void DoReportDlCqiInfo (const struct FfMacSchedSapProvider::SchedDlCqiInfoReqParameters& params);
  virtual void DoReportUlCqiInfo (const struct FfMacSchedSapProvider::SchedUlCqiInfoReqParameters& params);
  virtual void DoReportUlCqiInfo ( std::map <uint16_t, std::vector <double> > ulCqiMap );
  virtual uint8_t DoGetTpc (uint16_t rnti);
  virtual uint16_t DoGetMinContinuousUlBandwidth ();

  // FFR SAP RRC PROVIDER IMPLEMENTATION
  virtual void DoReportUeMeas (uint16_t rnti, LteRrcSap::MeasResults measResults);
  virtual void DoRecvLoadInformation (EpcX2Sap::LoadInformationParams params);

private:
  /**
   * Set downlink configuration
   *
   * \param cellId the cell ID
   * \param bandwidth the bandwidth
   */
  void SetDownlinkConfiguration (uint16_t cellId, uint8_t bandwidth);
  /**
   * Set uplink configuration
   *
   * \param cellId the cell ID
   * \param bandwidth the bandwidth
   */
  void SetUplinkConfiguration (uint16_t cellId, uint8_t bandwidth);
  /**
   * Initialize downlink RBG maps
   */
  void InitializeDownlinkRbgMaps ();
  /**
   * Initialize uplink RBG maps
   */
  void InitializeUplinkRbgMaps ();

  // FFR SAP
  LteFfrSapUser* m_ffrSapUser; ///< FFR SAP user
  LteFfrSapProvider* m_ffrSapProvider; ///< FFR SAP provider

  // FFR RRF SAP
  LteFfrRrcSapUser* m_ffrRrcSapUser; ///< FFR RRC SAP user
  LteFfrRrcSapProvider* m_ffrRrcSapProvider; ///< FFR RRC SAP provider

  bool m_isEdgeSubBandForCenterUe; ///< is edge subband for center UE?

  uint8_t m_dlEdgeSubBandOffset; ///< DL edge subband offset
  uint8_t m_dlEdgeSubBandwidth; ///< DL edge subbandwidth

  uint8_t m_ulEdgeSubBandOffset; ///< UL edge subband offset
  uint8_t m_ulEdgeSubBandwidth; ///< UL edge subbandwidth

  std::vector <bool> m_dlRbgMap; ///< DL RBG map
  std::vector <bool> m_ulRbgMap; ///< UL RBG map

  std::vector <bool> m_dlEdgeRbgMap; ///< DL edge RBG map
  std::vector <bool> m_ulEdgeRbgMap; ///< UL edge RBG map

  /// SubBand enumeration
  enum SubBand
  {
    AreaUnset,
    CellCenter,
    CellEdge
  };

  std::map< uint16_t, uint8_t > m_ues; ///< UEs
  std::vector<uint16_t> m_edgeUes; ///< Edge UEs

  uint8_t m_edgeSubBandThreshold; ///< Edge subband threshold 

  uint8_t m_centerPowerOffset; ///< center power offset
  uint8_t m_edgePowerOffset; ///< edge power offset

  uint8_t m_centerAreaTpc; ///< center area tpc
  uint8_t m_edgeAreaTpc; ///< edge area tpc

  /// The expected measurement identity
  uint8_t m_measId;

}; // end of class LteFrSoftAlgorithm


} // end of namespace ns3


#endif /* LTE_FR_SOFT_ALGORITHM_H */
