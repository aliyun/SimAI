/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011, 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 * Copyright (c) 2013 Budiarto Herman
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
 * Original work authors (from lte-enb-rrc.cc):
 * - Nicola Baldo <nbaldo@cttc.es>
 * - Marco Miozzo <mmiozzo@cttc.es>
 * - Manuel Requena <manuel.requena@cttc.es>
 *
 * Converted to ANR interface by:
 * - Budiarto Herman <budiarto.herman@magister.fi>
 */

#ifndef LTE_ANR_H
#define LTE_ANR_H

#include <ns3/object.h>
#include <ns3/lte-rrc-sap.h>
#include <ns3/lte-anr-sap.h>
#include <map>

namespace ns3 {


class LteAnrSapProvider;
class LteAnrSapUser;
class LteNeighbourRelation;

/**
 * \brief Automatic Neighbour Relation function.
 *
 * ANR is a conceptually a list of neighbouring cells called the Neighbour
 * Relation Table (NRT). ANR has the capability of automatically inserting new
 * entries into NRT based on measurement reports obtained from the eNodeB RRC
 * instance. Besides this, ANR also supports manual insertion and accepts
 * queries for the NRT content.
 *
 * The LteHelper class automatically installs one ANR instance for each eNodeB
 * RRC instance. When installed, ANR will assist the eNodeB RRC's handover
 * function, e.g., by preventing an X2-based handover execution if there is no
 * X2 interface to the target neighbour cell. If this is not desired, it can be
 * disabled by the following code sample:
 *
 *     Config::SetDefault ("ns3::LteHelper::AnrEnabled", BooleanValue (false));
 *     Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
 *
 * The communication between an ANR instance and the eNodeB RRC instance is done
 * through the *ANR SAP* interface. The ANR instance corresponds to the
 * "provider" part of this interface, while the eNodeB RRC instance takes the
 * role of the "user" part. The following code skeleton establishes the
 * connection between both instances:
 *
 *     Ptr<LteEnbRrc> u = ...;
 *     Ptr<LteAnr> p = ...;
 *     u->SetLteAnrSapProvider (p->GetLteAnrSapProvider ());
 *     p->SetLteAnrSapUser (u->GetLteAnrSapUser ());
 *
 * However, user rarely needs to use the above code, since it has already been
 * taken care by LteHelper::InstallEnbDevice.
 *
 * The current ANR model is inspired from Section 22.3.2a and 22.3.3 of 3GPP
 * TS 36.300.
 *
 * \sa SetLteAnrSapProvider, SetLteAnrSapUser
 */
class LteAnr : public Object
{
public:
  /**
   * \brief Creates an ANR instance.
   * \param servingCellId the cell ID of the eNodeB instance whom this ANR
   *                      instance is to be associated with
   */
  LteAnr (uint16_t servingCellId);
  virtual ~LteAnr ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  /**
   * \brief Provide an advance information about a related neighbouring cell
   *        and add it as a new Neighbour Relation entry.
   * \param cellId the cell ID of the new neighbour
   *
   * This function simulates the Neighbour Relation addition operation by
   * network operations and maintenance, as depicted in Section 22.3.2a of
   * 3GPP TS 36.300.
   *
   * An entry added by this function will have the NoRemove flag set to TRUE and
   * the NoHo flag set to TRUE. Hence, the cell may not act as the target cell
   * of a handover, unless a measurement report of the cell is received, which
   * will update the NoHo flag to FALSE.
   */
  void AddNeighbourRelation (uint16_t cellId);

  /**
   * \brief Remove an existing Neighbour Relation entry.
   * \param cellId the cell ID to be removed from the NRT
   *
   * This function simulates the Neighbour Relation removal operation by
   * network operations and maintenance, as depicted in Section 22.3.2a of
   * 3GPP TS 36.300.
   */
  void RemoveNeighbourRelation (uint16_t cellId);

  /**
   * \brief Set the "user" part of the ANR SAP interface that this ANR instance
   *        will interact with.
   * \param s a reference to the "user" part of the interface, typically a
   *          member of an LteEnbRrc instance
   */
  virtual void SetLteAnrSapUser (LteAnrSapUser* s);

  /**
   * \brief Export the "provider" part of the ANR SAP interface.
   * \return the reference to the "provider" part of the interface, typically to
   *         be kept by an LteEnbRrc instance
   */
  virtual LteAnrSapProvider* GetLteAnrSapProvider ();

  /// let the forwarder class access the protected and private members
  friend class MemberLteAnrSapProvider<LteAnr>;

protected:
  // inherited from Object
  virtual void DoInitialize ();
  virtual void DoDispose ();

private:

  // ANR SAP PROVIDER IMPLEMENTATION

  /**
   * \brief Implementation of LteAnrSapProvider::ReportUeMeas.
   * \param measResults a single report of one measurement identity
   */
  void DoReportUeMeas (LteRrcSap::MeasResults measResults);

  /**
   * \brief Implementation of LteAnrSapProvider::AddNeighbourRelation.
   * \param cellId the Physical Cell ID of the new neighbouring cell
   */
  void DoAddNeighbourRelation (uint16_t cellId);

  /**
   * \brief Implementation of LteAnrSapProvider::GetNoRemove.
   * \param cellId the Physical Cell ID of the neighbouring cell of interest
   * \return if true, the Neighbour Relation shall *not* be removed from the NRT
   */
  bool DoGetNoRemove (uint16_t cellId) const;

  /**
   * \brief Implementation of LteAnrSapProvider::GetNoHo.
   * \param cellId the Physical Cell ID of the neighbouring cell of interest
   * \return if true, the Neighbour Relation shall *not* be used by the eNodeB
   *         for handover reasons
   */
  bool DoGetNoHo (uint16_t cellId) const;

  /**
   * \brief Implementation of LteAnrSapProvider::GetNoX2.
   * \param cellId the Physical Cell ID of the neighbouring cell of interest
   * \return if true, the Neighbour Relation shall *not* use an X2 interface in
   *         order to initiate procedures towards the eNodeB parenting the
   *         target cell
   */
  bool DoGetNoX2 (uint16_t cellId) const;

  // ANR SAP

  /**
   * \brief Reference to the "provider" part of the ANR SAP interface, which is
   *        automatically created when this class instantiates.
   */
  LteAnrSapProvider* m_anrSapProvider;

  /**
   * \brief Reference to the "user" part of the ANR SAP interface, which is
   *        provided by the eNodeB RRC instance.
   */
  LteAnrSapUser* m_anrSapUser;

  // ATTRIBUTE

  /// The attribute Threshold.
  uint8_t m_threshold;

  /**
   * \brief Neighbour Relation between two eNodeBs (serving eNodeB and neighbour
   *        eNodeB).
   */
  struct NeighbourRelation_t
  {
    bool noRemove; ///< no remove
    bool noHo; ///< no HO
    bool noX2; ///< no X2
    bool detectedAsNeighbour; ///< detected as neighbor
  };

  /// cellId
  typedef std::map<uint16_t, NeighbourRelation_t> NeighbourRelationTable_t;

  /// neighbor relation table
  NeighbourRelationTable_t m_neighbourRelationTable; 

  /**
   * \internal methods
   * \param cellId
   * \returns the neighbor relation
   */
  const NeighbourRelation_t* Find (uint16_t cellId) const;

  /// The expected measurement identity
  uint8_t m_measId;

  /// Serving cell ID
  uint16_t m_servingCellId;

}; // end of class LteAnr


} // end of namespace ns3


#endif /* LTE_ANR_H */
