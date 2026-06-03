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

#ifndef LTE_FFR_SIMPLE_H
#define LTE_FFR_SIMPLE_H

#include <ns3/lte-ffr-algorithm.h>
#include <ns3/lte-ffr-sap.h>
#include <ns3/lte-ffr-rrc-sap.h>
#include <ns3/lte-rrc-sap.h>

#include <ns3/traced-callback.h>
#include <map>

namespace ns3 {


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Simple Frequency Reuse algorithm implementation which uses only 1 sub-band.
 *                Used to test Downlink Power Allocation. When Simple FR receives UE measurements
 *                it immediately call functions to change PdschConfigDedicated (i.e. P_A) value for
 *                this UE.
 */
class LteFfrSimple : public LteFfrAlgorithm
{
public:
  /**
   * \brief Creates a trivial ffr algorithm instance.
   */
  LteFfrSimple ();

  virtual ~LteFfrSimple ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  /**
   * \brief Callback function that is used to be connected to trace ChangePdschConfigDedicated
   * \param change trace fired upon change of PdschConfigDedicated if true
   */
  void ChangePdschConfigDedicated (bool change);
  /**
   * \brief Set PDSCH config dedicated function
   * \param pdschConfigDedicated LteRrcSap::PdschConfigDedicated object
   */
  void SetPdschConfigDedicated (LteRrcSap::PdschConfigDedicated pdschConfigDedicated);

  /**
   * \brief Set transmission power control
   * \param tpc TPC
   * \param num number of TPC configurations in the test case
   * \param acculumatedMode whether TPC accumulated mode is used
   */
  void SetTpc (uint32_t tpc, uint32_t num, bool acculumatedMode);

  // inherited from LteFfrAlgorithm
  virtual void SetLteFfrSapUser (LteFfrSapUser* s);
  virtual LteFfrSapProvider* GetLteFfrSapProvider ();

  virtual void SetLteFfrRrcSapUser (LteFfrRrcSapUser* s);
  virtual LteFfrRrcSapProvider* GetLteFfrRrcSapProvider ();

  /// let the forwarder class access the protected and private members
  friend class MemberLteFfrSapProvider<LteFfrSimple>;
  /// let the forwarder class access the protected and private members
  friend class MemberLteFfrRrcSapProvider<LteFfrSimple>;

  /**
   * TracedCallback signature for change of PdschConfigDedicated.
   *
   * \param [in] rnti
   * \param [in] pdschPa PdschConfiDedicated.pa
   */
  typedef void (* PdschTracedCallback)(uint16_t rnti, uint8_t pdschPa);

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

  /// Update PDSCH config dedicated function
  void UpdatePdschConfigDedicated ();

  // FFR SAP
  LteFfrSapUser* m_ffrSapUser; ///< FFR SAP user
  LteFfrSapProvider* m_ffrSapProvider; ///< FFR SAP provider

  // FFR RRF SAP
  LteFfrRrcSapUser* m_ffrRrcSapUser; ///< FFR RRC SAP user
  LteFfrRrcSapProvider* m_ffrRrcSapProvider; ///< FFR RRC SAP provider

  uint8_t m_dlOffset; ///< DL offset
  uint8_t m_dlSubBand; ///< DL subband

  uint8_t m_ulOffset; ///< UL offset
  uint8_t m_ulSubBand; ///< UL subband

  std::vector <bool> m_dlRbgMap; ///< DL RBG map
  std::vector <bool> m_ulRbgMap; ///< UL RBG map

  std::map <uint16_t, LteRrcSap::PdschConfigDedicated> m_ues; ///< UEs

  // The expected measurement identity
  uint8_t m_measId; ///< measure ID

  bool m_changePdschConfigDedicated; ///< PDSCH config dedicate changed?

  LteRrcSap::PdschConfigDedicated m_pdschConfigDedicated; ///< PDSCH config dedicated

  TracedCallback<uint16_t, uint8_t> m_changePdschConfigDedicatedTrace; ///< PDSCH config dedicated change trace callback


  // Uplink Power Control
  uint32_t m_tpc; ///< transmission power control to be used
  uint32_t m_tpcNum; ///< number of TPC configurations
  bool m_accumulatedMode; ///< whether to use the TPC accumulated mode

}; // end of class LteFfrSimple


} // end of namespace ns3


#endif /* LTE_FFR_SIMPLE_H */
