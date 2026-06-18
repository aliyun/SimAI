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

#ifndef LTE_FFR_SOFT_ALGORITHM_H
#define LTE_FFR_SOFT_ALGORITHM_H

#include <ns3/lte-ffr-algorithm.h>
#include <ns3/lte-ffr-sap.h>
#include <ns3/lte-ffr-rrc-sap.h>
#include <ns3/lte-rrc-sap.h>
#include <map>

namespace ns3 {


/**
 * \brief Soft Fractional Frequency Reuse algorithm implementation
 */
class LteFfrSoftAlgorithm : public LteFfrAlgorithm
{
public:
  /**
   * \brief Creates a trivial ffr algorithm instance.
   */
  LteFfrSoftAlgorithm ();

  virtual ~LteFfrSoftAlgorithm ();

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
  friend class MemberLteFfrSapProvider<LteFfrSoftAlgorithm>;
  /// let the forwarder class access the protected and private members
  friend class MemberLteFfrRrcSapProvider<LteFfrSoftAlgorithm>;

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
   * Set downlink configuration function
   *
   * \param cellId the cell ID
   * \param bandwidth the bandwidth
   */
  void SetDownlinkConfiguration (uint16_t cellId, uint8_t bandwidth);
  /**
   * Set uplink configuration function
   *
   * \param cellId the cell ID
   * \param bandwidth the bandwidth
   */
  void SetUplinkConfiguration (uint16_t cellId, uint8_t bandwidth);
  /**
   * Initialize downlink RBG maps function
   */
  void InitializeDownlinkRbgMaps ();
  /**
   * Initialize uplink RBG maps function
   */
  void InitializeUplinkRbgMaps ();


  // FFR SAP
  LteFfrSapUser* m_ffrSapUser; ///< FFR SAP user
  LteFfrSapProvider* m_ffrSapProvider; ///< FFR SAP provider

  // FFR RRF SAP
  LteFfrRrcSapUser* m_ffrRrcSapUser; ///< FFR RRC SAP user
  LteFfrRrcSapProvider* m_ffrRrcSapProvider; ///< FFR RRC SAP provider

  uint8_t m_dlCommonSubBandwidth; ///< DL common subbandwidth
  uint8_t m_dlEdgeSubBandOffset; ///< DL edge subband offset
  uint8_t m_dlEdgeSubBandwidth; ///< DL edge subbandwidth
 
  uint8_t m_ulCommonSubBandwidth; ///< UL common subbandwidth
  uint8_t m_ulEdgeSubBandOffset; ///< UL edge subband offset
  uint8_t m_ulEdgeSubBandwidth; ///< UL edge subbandwidth

  std::vector <bool> m_dlRbgMap; ///< DL RBG Map
  std::vector <bool> m_ulRbgMap; ///< UL RBG map

  std::vector <bool> m_dlCenterRbgMap; ///< DL center RBG map
  std::vector <bool> m_ulCenterRbgMap; ///< UL center RBG map

  std::vector <bool> m_dlMediumRbgMap; ///< DL medium RBG map
  std::vector <bool> m_ulMediumRbgMap; ///< UL medium RBG map

  std::vector <bool> m_dlEdgeRbgMap; ///< DL edge RBG map
  std::vector <bool> m_ulEdgeRbgMap; ///< UL edge RBG map

  /// UE position enumeration
  enum UePosition
  {
    AreaUnset,
    CenterArea,
    MediumArea,
    EdgeArea
  };

  std::map< uint16_t, uint8_t > m_ues; ///< UEs

  uint8_t m_centerSubBandThreshold; ///< center subband threshold
  uint8_t m_edgeSubBandThreshold; ///< edge subband threshold

  uint8_t m_centerAreaPowerOffset; ///< center area power offset
  uint8_t m_mediumAreaPowerOffset; ///< medium area power offset 
  uint8_t m_edgeAreaPowerOffset; ///< edge area power offset

  uint8_t m_centerAreaTpc; ///< center area tpc
  uint8_t m_mediumAreaTpc; ///< medium area tpc
  uint8_t m_edgeAreaTpc; ///< edge area tpc

  /// The expected measurement identity
  uint8_t m_measId;

}; // end of class LteFfrSoftAlgorithm


} // end of namespace ns3


#endif /* LTE_FFR_SOFT_ALGORITHM_H */
