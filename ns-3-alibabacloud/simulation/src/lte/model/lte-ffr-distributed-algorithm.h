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

#ifndef LTE_FFR_DISTRIBUTED_ALGORITHM_H
#define LTE_FFR_DISTRIBUTED_ALGORITHM_H

#include <ns3/lte-ffr-algorithm.h>
#include <ns3/lte-ffr-sap.h>
#include <ns3/lte-ffr-rrc-sap.h>
#include <ns3/lte-rrc-sap.h>

namespace ns3 {

/**
 * \brief Distributed Fractional Frequency Reuse algorithm implementation.
 */
class LteFfrDistributedAlgorithm : public LteFfrAlgorithm
{
public:
  LteFfrDistributedAlgorithm ();
  virtual ~LteFfrDistributedAlgorithm ();

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
  friend class MemberLteFfrSapProvider<LteFfrDistributedAlgorithm>;
  /// let the forwarder class access the protected and private members
  friend class MemberLteFfrRrcSapProvider<LteFfrDistributedAlgorithm>;

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
   * Set down link configuration function
   *
   * \param cellId cell ID
   * \param bandwidth the bandwidth
   */
  void SetDownlinkConfiguration (uint16_t cellId, uint8_t bandwidth);
  /**
   * Set up link configuration function
   *
   * \param cellId cell ID
   * \param bandwidth the bandwidth
   */
  void SetUplinkConfiguration (uint16_t cellId, uint8_t bandwidth);
  /**
   * Initialize down link RGB maps function
   */
  void InitializeDownlinkRbgMaps ();
  /**
   * Initialize up link RGB maps function
   */
  void InitializeUplinkRbgMaps ();

  /**
   * Initialize up link RGB maps function
   *
   * \param rnti the RNTI
   * \param cellId the cell ID
   * \param rsrp the RSRP
   * \param rsrq the RSRQ
   */
  void UpdateNeighbourMeasurements (uint16_t rnti, uint16_t cellId, uint8_t rsrp, uint8_t rsrq);

  /// Calculate function
  void Calculate ();
  /**
   * Send load information function
   *
   * \param targetCellId the cell ID
   */
  void SendLoadInformation (uint16_t targetCellId);

  // FFR SAP
  LteFfrSapUser* m_ffrSapUser; ///< FFR SAP User
  LteFfrSapProvider* m_ffrSapProvider; ///< FFR SAP Provider

  // FFR RRF SAP
  LteFfrRrcSapUser* m_ffrRrcSapUser; ///< FFR RRC SAP User
  LteFfrRrcSapProvider* m_ffrRrcSapProvider; ///< FFR RRC SAP Provider

  std::vector <bool> m_dlRbgMap; ///< DL RBG map
  std::vector <bool> m_ulRbgMap; ///< UL RBG map

  uint8_t m_edgeRbNum; ///< edge RB number
  std::vector <bool> m_dlEdgeRbgMap; ///< DL edge RBG map
  std::vector <bool> m_ulEdgeRbgMap; ///< UL edge RBG map

  /// UePosition enumeration
  enum UePosition
  {
    AreaUnset,
    CenterArea,
    EdgeArea
  };

  std::map< uint16_t, uint8_t > m_ues; ///< UEs map

  uint8_t m_edgeSubBandRsrqThreshold; ///< edge sub band RSRQ threshold

  uint8_t m_centerPowerOffset; ///< center power offset
  uint8_t m_edgePowerOffset; ///< edge power offset

  uint8_t m_centerAreaTpc; ///< center area TPC
  uint8_t m_edgeAreaTpc; ///< edge area TCP

  Time m_calculationInterval; ///< calculation interval
  EventId m_calculationEvent; ///< calculation event

  // The expected measurement identity
  uint8_t m_rsrqMeasId; ///< RSRQ measurement ID
  uint8_t m_rsrpMeasId; ///< RSRP measurement ID

  /**
   * \brief Measurements reported by a UE for a cell ID.
   *
   * The values are quantized according 3GPP TS 36.133 section 9.1.4 and 9.1.7.
   */
  class UeMeasure : public SimpleRefCount<UeMeasure>
  {
  public:
    uint16_t m_cellId; ///< Cell ID
    uint8_t m_rsrp; ///< RSRP
    uint8_t m_rsrq; ///< RSRQ
  };

  /// Cell Id is used as the key for the following map
  typedef std::map<uint16_t, Ptr<UeMeasure> > MeasurementRow_t;
  /// RNTI is used as the key for the following map
  typedef std::map<uint16_t, MeasurementRow_t> MeasurementTable_t;
  MeasurementTable_t m_ueMeasures; ///< UE measures

  std::vector<uint16_t> m_neigborCell; ///< neighbor cell

  uint8_t m_rsrpDifferenceThreshold; ///< RSRP difference threshold

  std::map<uint16_t, uint32_t> m_cellWeightMap; ///< cell weight map

  std::map<uint16_t, std::vector <bool> > m_rntp; ///< RNTP

}; // end of class LteFfrDistributedAlgorithm

} // end of namespace ns3

#endif /* LTE_FR_DISTRIBUTED_ALGORITHM_H */
