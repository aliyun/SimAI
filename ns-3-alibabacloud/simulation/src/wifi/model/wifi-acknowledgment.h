/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#ifndef WIFI_ACKNOWLEDGMENT_H
#define WIFI_ACKNOWLEDGMENT_H

#include "ns3/nstime.h"
#include "wifi-tx-vector.h"
#include "block-ack-type.h"
#include "wifi-mac-header.h"
#include "ctrl-headers.h"
#include <map>
#include <memory>


namespace ns3 {

class Mac48Address;

/**
 * \ingroup wifi
 *
 * WifiAcknowledgment is an abstract base struct. Each derived struct defines an acknowledgment
 * method and stores the information needed to perform acknowledgment according to
 * that method.
 */
struct WifiAcknowledgment
{
  /**
   * \enum Method
   * \brief Available acknowledgment methods
   */
  enum Method
    {
      NONE = 0,
      NORMAL_ACK,
      BLOCK_ACK,
      BAR_BLOCK_ACK,
      DL_MU_BAR_BA_SEQUENCE,
      DL_MU_TF_MU_BAR,
      DL_MU_AGGREGATE_TF,
      UL_MU_MULTI_STA_BA,
      ACK_AFTER_TB_PPDU
    };

  /**
   * Constructor.
   * \param m the acknowledgment method for this object
   */
  WifiAcknowledgment (Method m);
  virtual ~WifiAcknowledgment ();

  /**
   * Clone this object.
   * \return a pointer to the cloned object
   */
  virtual std::unique_ptr<WifiAcknowledgment> Copy (void) const = 0;

  /**
   * Get the QoS Ack policy to use for the MPDUs addressed to the given receiver
   * and belonging to the given TID.
   *
   * \param receiver the MAC address of the receiver
   * \param tid the TID
   * \return the QoS Ack policy to use
   */
  WifiMacHeader::QosAckPolicy GetQosAckPolicy (Mac48Address receiver, uint8_t tid) const;

  /**
   * Set the QoS Ack policy to use for the MPDUs addressed to the given receiver
   * and belonging to the given TID. If the pair (receiver, TID) already exists,
   * it is overwritten with the given QoS Ack policy.
   *
   * \param receiver the MAC address of the receiver
   * \param tid the TID
   * \param ackPolicy the QoS Ack policy to use
   */
  void SetQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy);

  /**
   * \brief Print the object contents.
   * \param os output stream in which the data should be printed.
   */
  virtual void Print (std::ostream &os) const = 0;

  const Method method;       //!< acknowledgment method
  Time acknowledgmentTime;   //!< time required by the acknowledgment method

private:
  /**
   * Check whether the given QoS Ack policy can be used for the MPDUs addressed
   * to the given receiver and belonging to the given TID.
   *
   * \param receiver the MAC address of the receiver
   * \param tid the TID
   * \param ackPolicy the QoS Ack policy to use
   * \return true if the given QoS Ack policy can be used, false otherwise
   */
  virtual bool CheckQosAckPolicy (Mac48Address receiver, uint8_t tid,
                                  WifiMacHeader::QosAckPolicy ackPolicy) const = 0;

  /// Qos Ack Policy to set for MPDUs addressed to a given receiver and having a given TID
  std::map<std::pair<Mac48Address, uint8_t>, WifiMacHeader::QosAckPolicy> m_ackPolicy;
};


/**
 * \ingroup wifi
 *
 * WifiNoAck specifies that no acknowledgment is required.
 */
struct WifiNoAck : public WifiAcknowledgment
{
  WifiNoAck ();

  std::unique_ptr<WifiAcknowledgment> Copy (void) const override;
  bool CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const override;
  void Print (std::ostream &os) const override;
};


/**
 * \ingroup wifi
 *
 * WifiNormalAck specifies that acknowledgment via Normal Ack is required.
 */
struct WifiNormalAck : public WifiAcknowledgment
{
  WifiNormalAck ();

  std::unique_ptr<WifiAcknowledgment> Copy (void) const override;
  bool CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const override;
  void Print (std::ostream &os) const override;

  WifiTxVector ackTxVector;       //!< Ack TXVECTOR
};


/**
 * \ingroup wifi
 *
 * WifiBlockAck specifies that acknowledgment via Block Ack is required.
 */
struct WifiBlockAck : public WifiAcknowledgment
{
  WifiBlockAck ();

  std::unique_ptr<WifiAcknowledgment> Copy (void) const override;
  bool CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const override;
  void Print (std::ostream &os) const override;

  WifiTxVector blockAckTxVector;       //!< BlockAck TXVECTOR
  BlockAckType baType;                 //!< BlockAck type
};


/**
 * \ingroup wifi
 *
 * WifiBarBlockAck specifies that a BlockAckReq is sent to solicit a Block Ack response.
 */
struct WifiBarBlockAck : public WifiAcknowledgment
{
  WifiBarBlockAck ();

  std::unique_ptr<WifiAcknowledgment> Copy (void) const override;
  bool CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const override;
  void Print (std::ostream &os) const override;

  WifiTxVector blockAckReqTxVector;    //!< BlockAckReq TXVECTOR
  WifiTxVector blockAckTxVector;       //!< BlockAck TXVECTOR
  BlockAckReqType barType;             //!< BlockAckReq type
  BlockAckType baType;                 //!< BlockAck type
};


/**
 * \ingroup wifi
 *
 * WifiDlMuBarBaSequence specifies that a DL MU PPDU is acknowledged through a
 * sequence of BlockAckReq and BlockAck frames. Only one station may be allowed
 * to reply a SIFS after the DL MU PPDU by sending either a Normal Ack or a BlockAck.
 */
struct WifiDlMuBarBaSequence : public WifiAcknowledgment
{
  WifiDlMuBarBaSequence ();

  std::unique_ptr<WifiAcknowledgment> Copy (void) const override;
  bool CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const override;
  void Print (std::ostream &os) const override;

  /// information related to an Ack frame sent by a station
  struct AckInfo
  {
    WifiTxVector ackTxVector;          //!< TXVECTOR for the Ack frame
  };
  /// information related to a BlockAck frame sent by a station
  struct BlockAckInfo
  {
    WifiTxVector blockAckTxVector;     //!< TXVECTOR for the BlockAck frame
    BlockAckType baType;               //!< BlockAck type
  };
  /// information related to a BlockAckReq frame sent to a station
  struct BlockAckReqInfo
  {
    WifiTxVector blockAckReqTxVector;  //!< TXVECTOR for the BlockAckReq frame
    BlockAckReqType barType;           //!< BlockAckReq type
    WifiTxVector blockAckTxVector;     //!< TXVECTOR for the BlockAck frame
    BlockAckType baType;               //!< BlockAck type
  };

  /// Set of stations replying with an Ack frame (no more than one)
  std::map<Mac48Address, AckInfo> stationsReplyingWithNormalAck;
  /// Set of stations replying with a BlockAck frame (no more than one)
  std::map<Mac48Address, BlockAckInfo> stationsReplyingWithBlockAck;
  /// Set of stations receiving a BlockAckReq frame and replying with a BlockAck frame
  std::map<Mac48Address, BlockAckReqInfo> stationsSendBlockAckReqTo;
};


/**
 * \ingroup wifi
 *
 * WifiDlMuTfMuBar specifies that a DL MU PPDU is followed after a SIFS duration
 * by a MU-BAR Trigger Frame (sent as single user frame) soliciting BlockAck
 * frames sent as HE TB PPDUs.
 */
struct WifiDlMuTfMuBar : public WifiAcknowledgment
{
  WifiDlMuTfMuBar ();

  std::unique_ptr<WifiAcknowledgment> Copy (void) const override;
  bool CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const override;
  void Print (std::ostream &os) const override;

  /// information related to a BlockAck frame sent by a station
  struct BlockAckInfo
  {
    CtrlBAckRequestHeader barHeader;   //!< BlockAckReq header
    WifiTxVector blockAckTxVector;     //!< TXVECTOR for the BlockAck frame
    BlockAckType baType;               //!< BlockAck type
  };

  /// Set of stations replying with a BlockAck frame
  std::map<Mac48Address, BlockAckInfo> stationsReplyingWithBlockAck;
  std::list<BlockAckReqType> barTypes; //!< BAR types
  uint16_t ulLength;                   //!< the UL Length field of the MU-BAR Trigger Frame
  WifiTxVector muBarTxVector;          //!< TXVECTOR used to transmit the MU-BAR Trigger Frame
};


/**
 * \ingroup wifi
 *
 * WifiDlMuAggregateTf specifies that a DL MU PPDU made of PSDUs including each
 * a MU-BAR Trigger Frame is acknowledged through BlockAck frames sent as
 * HE TB PPDUs.
 */
struct WifiDlMuAggregateTf : public WifiAcknowledgment
{
  WifiDlMuAggregateTf ();

  std::unique_ptr<WifiAcknowledgment> Copy (void) const override;
  bool CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const override;
  void Print (std::ostream &os) const override;

  /// information related to a BlockAck frame sent by a station
  struct BlockAckInfo
  {
    uint32_t muBarSize;                //!< size in bytes of a MU-BAR Trigger Frame
    CtrlBAckRequestHeader barHeader;   //!< BlockAckReq header
    WifiTxVector blockAckTxVector;     //!< TXVECTOR for the BlockAck frame
    BlockAckType baType;               //!< BlockAck type
  };

  /// Set of stations replying with a BlockAck frame
  std::map<Mac48Address, BlockAckInfo> stationsReplyingWithBlockAck;
  uint16_t ulLength;                   //!< the UL Length field of the MU-BAR Trigger Frames
};


/**
 * \ingroup wifi
 *
 * WifiUlMuMultiStaBa specifies that a Basic Trigger Frame is being sent to
 * solicit TB PPDUs that will be acknowledged through a multi-STA BlockAck frame.
 */
struct WifiUlMuMultiStaBa : public WifiAcknowledgment
{
  WifiUlMuMultiStaBa ();

  std::unique_ptr<WifiAcknowledgment> Copy (void) const override;
  bool CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const override;
  void Print (std::ostream &os) const override;

  /// Map (originator, tid) pairs to the their index in baType
  std::map<std::pair<Mac48Address, uint8_t>, std::size_t> stationsReceivingMultiStaBa;
  BlockAckType baType;                 //!< BlockAck type
  WifiTxVector tbPpduTxVector;         //!< TXVECTOR for a TB PPDU
  WifiTxVector multiStaBaTxVector;     //!< TXVECTOR for the Multi-STA BlockAck
};


/**
 * \ingroup wifi
 *
 * WifiAckAfterTbPpdu is used when a station prepares a TB PPDU to send in
 * response to a Basic Trigger Frame. The acknowledgment time must be
 * zero because the time taken by the actual acknowledgment is not included
 * in the duration indicated by the Trigger Frame. The QoS ack policy instead
 * must be Normal Ack/Implicit Block Ack Request.
 */
struct WifiAckAfterTbPpdu : public WifiAcknowledgment
{
  WifiAckAfterTbPpdu ();

  std::unique_ptr<WifiAcknowledgment> Copy (void) const override;
  bool CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const override;
  void Print (std::ostream &os) const override;
};


/**
 * \brief Stream insertion operator.
 *
 * \param os the output stream
 * \param acknowledgment the acknowledgment method
 * \returns a reference to the stream
 */
std::ostream& operator<< (std::ostream& os, const WifiAcknowledgment* acknowledgment);

} //namespace ns3

#endif /* WIFI_ACKNOWLEDGMENT_H */
