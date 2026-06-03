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

#ifndef LTE_FFR_ENHANCED_ALGORITHM_H
#define LTE_FFR_ENHANCED_ALGORITHM_H

#include <ns3/lte-ffr-algorithm.h>
#include <ns3/lte-ffr-sap.h>
#include <ns3/lte-ffr-rrc-sap.h>
#include <ns3/lte-rrc-sap.h>
#include <map>

// value for SINR outside the range defined by FF-API, used to indicate that there
// is no CQI for this element
#define NO_SINR -5000

namespace ns3 {

/**
 * \brief Enhanced Fractional Frequency Reuse algorithm implementation
 */
class LteFfrEnhancedAlgorithm : public LteFfrAlgorithm
{
public:
  /**
   * \brief Creates a trivial ffr algorithm instance.
   */
  LteFfrEnhancedAlgorithm ();
  virtual ~LteFfrEnhancedAlgorithm ();

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
  friend class MemberLteFfrSapProvider<LteFfrEnhancedAlgorithm>;
  /// let the forwarder class access the protected and private members
  friend class MemberLteFfrRrcSapProvider<LteFfrEnhancedAlgorithm>;

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

  /**
   * Initialize uplink RBG maps
   *
   * \param rnti the RNTI
   * \param rb RB
   * \param ulCqiMap UL CQI map
   * \returns UL SINR
   */
  double EstimateUlSinr (uint16_t rnti, uint16_t rb, std::map <uint16_t, std::vector <double> > ulCqiMap);
  /**
   * Get CQI from spectral efficiency
   *
   * \param s spectral efficiency
   * \returns CQI
   */
  int GetCqiFromSpectralEfficiency (double s);

  // FFR SAP
  LteFfrSapUser* m_ffrSapUser; ///< FFR SAP user
  LteFfrSapProvider* m_ffrSapProvider; ///< FFR SAP provider

  // FFR RRF SAP 
  LteFfrRrcSapUser* m_ffrRrcSapUser; ///< FFR RRC SAP user
  LteFfrRrcSapProvider* m_ffrRrcSapProvider; ///< FFR RRC SAP provider

  uint8_t m_dlSubBandOffset; ///< DL subband offset
  uint8_t m_dlReuse3SubBandwidth; ///< DL reuse 3 subband bandwidth
  uint8_t m_dlReuse1SubBandwidth; ///< DL reuse 1 subband bandwidth

  uint8_t m_ulSubBandOffset; ///< UL subband offset
  uint8_t m_ulReuse3SubBandwidth; ///< UL reuse 3 subbandwidth
  uint8_t m_ulReuse1SubBandwidth; ///< UL reuse 1 subbandwidth 

  std::vector <bool> m_dlRbgMap; ///< DL RBG map
  std::vector <bool> m_ulRbgMap; ///< UL RBG Map

  std::vector <bool> m_dlReuse3RbgMap; ///< DL reuse 3 RBG map
  std::vector <bool> m_dlReuse1RbgMap; ///< DL reuse 1 RBG map
  std::vector <bool> m_dlPrimarySegmentRbgMap; ///< DL primary segment RBG map
  std::vector <bool> m_dlSecondarySegmentRbgMap; ///< DL secondary segment RBG map

  std::vector <bool> m_ulReuse3RbgMap; ///< UL reuse 3 RBG map
  std::vector <bool> m_ulReuse1RbgMap; ///< UL reuse 1 RBG map
  std::vector <bool> m_ulPrimarySegmentRbgMap; ///< UL primary segment RBG map
  std::vector <bool> m_ulSecondarySegmentRbgMap; ///< UL secondary segment RBG map

  /// UE Position enumeration
  enum UePosition
  {
    AreaUnset,
    CenterArea,
    EdgeArea
  };

  std::map< uint16_t, uint8_t > m_ues; ///< UEs

  uint8_t m_rsrqThreshold; ///< RSRQ threshold

  uint8_t m_centerAreaPowerOffset; ///< Center area power offset
  uint8_t m_edgeAreaPowerOffset; ///< Edge area power offset

  uint8_t m_centerAreaTpc; ///< Center area TPC
  uint8_t m_edgeAreaTpc; ///< Edge are TPC


  uint8_t m_dlCqiThreshold; ///< DL CQI threshold
  /**
   * Map of UE's DL CQI A30 received
   */
  std::map <uint16_t,SbMeasResult_s> m_dlCqi;
  std::map <uint16_t, std::vector<bool> > m_dlRbgAvailableforUe; ///< DL RBG available for UE

  uint8_t m_ulCqiThreshold; ///< UL CQI threshold
  std::map <uint16_t, std::vector<int> > m_ulCqi; ///< UL CQI
  std::map <uint16_t, std::vector<bool> > m_ulRbAvailableforUe; ///< UL RB available for UE

  /// The expected measurement identity
  uint8_t m_measId;

}; // end of class LteFfrEnhancedAlgorithm

} // end of namespace ns3

#endif /* LTE_FFR_ENHANCED_ALGORITHM_H */
