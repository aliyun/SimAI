/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Dean Armstrong
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
 * Author: Dean Armstrong <deanarm@gmail.com>
 */

#ifndef WIFI_INFORMATION_ELEMENT_H
#define WIFI_INFORMATION_ELEMENT_H

#include "ns3/header.h"

namespace ns3 {

/**
 * This type is used to represent an Information Element ID. An
 * enumeration would be tidier, but doesn't provide for the
 * inheritance that is currently preferable to cleanly support
 * pre-standard modules such as mesh. Maybe there is a nice way of
 * doing this with a class.
 *
 * Until such time as a better way of implementing this is dreamt up
 * and applied, developers will need to be careful to avoid
 * duplication of IE IDs in the defines below (and in files which
 * declare "subclasses" of WifiInformationElement). Sorry.
 */
typedef uint8_t WifiInformationElementId;


/**
 * Here we have definition of all Information Element IDs in IEEE
 * 802.11-2007. See the comments for WifiInformationElementId - this could
 * probably be done in a considerably tidier manner.
 */
#define IE_SSID                                 ((WifiInformationElementId)0)
#define IE_SUPPORTED_RATES                      ((WifiInformationElementId)1)
#define IE_FH_PARAMETER_SET                     ((WifiInformationElementId)2)
#define IE_DSSS_PARAMETER_SET                   ((WifiInformationElementId)3)
#define IE_CF_PARAMETER_SET                     ((WifiInformationElementId)4)
#define IE_TIM                                  ((WifiInformationElementId)5)
#define IE_IBSS_PARAMETER_SET                   ((WifiInformationElementId)6)
#define IE_COUNTRY                              ((WifiInformationElementId)7)
#define IE_HOPPING_PATTERN_PARAMETERS           ((WifiInformationElementId)8)
#define IE_HOPPING_PATTERN_TABLE                ((WifiInformationElementId)9)
#define IE_REQUEST                              ((WifiInformationElementId)10)
#define IE_BSS_LOAD                             ((WifiInformationElementId)11)
#define IE_EDCA_PARAMETER_SET                   ((WifiInformationElementId)12)
#define IE_TSPEC                                ((WifiInformationElementId)13)
#define IE_TCLAS                                ((WifiInformationElementId)14)
#define IE_SCHEDULE                             ((WifiInformationElementId)15)
#define IE_CHALLENGE_TEXT                       ((WifiInformationElementId)16)
// 17 to 31 are reserved
#define IE_POWER_CONSTRAINT                     ((WifiInformationElementId)32)
#define IE_POWER_CAPABILITY                     ((WifiInformationElementId)33)
#define IE_TPC_REQUEST                          ((WifiInformationElementId)34)
#define IE_TPC_REPORT                           ((WifiInformationElementId)35)
#define IE_SUPPORTED_CHANNELS                   ((WifiInformationElementId)36)
#define IE_CHANNEL_SWITCH_ANNOUNCEMENT          ((WifiInformationElementId)37)
#define IE_MEASUREMENT_REQUEST                  ((WifiInformationElementId)38)
#define IE_MEASUREMENT_REPORT                   ((WifiInformationElementId)39)
#define IE_QUIET                                ((WifiInformationElementId)40)
#define IE_IBSS_DFS                             ((WifiInformationElementId)41)
#define IE_ERP_INFORMATION                      ((WifiInformationElementId)42)
#define IE_TS_DELAY                             ((WifiInformationElementId)43)
#define IE_TCLAS_PROCESSING                     ((WifiInformationElementId)44)
#define IE_HT_CAPABILITIES                      ((WifiInformationElementId)45)
#define IE_QOS_CAPABILITY                       ((WifiInformationElementId)46)
// 47 is reserved
#define IE_RSN                                  ((WifiInformationElementId)48)
// 49 is reserved
#define IE_EXTENDED_SUPPORTED_RATES             ((WifiInformationElementId)50)
#define IE_AP_CHANNEL_REPORT                    ((WifiInformationElementId)51)
#define IE_NEIGHBOR_REPORT                      ((WifiInformationElementId)52)
#define IE_RCPI                                 ((WifiInformationElementId)53)
#define IE_MOBILITY_DOMAIN                      ((WifiInformationElementId)54)
#define IE_FAST_BSS_TRANSITION                  ((WifiInformationElementId)55)
#define IE_TIMEOUT_INTERVAL                     ((WifiInformationElementId)56)
#define IE_RIC_DATA                             ((WifiInformationElementId)57)
#define IE_DSE_REGISTERED_LOCATION              ((WifiInformationElementId)58)
#define IE_SUPPORTED_OPERATING_CLASSES          ((WifiInformationElementId)59)
#define IE_EXTENDED_CHANNEL_SWITCH_ANNOUNCEMENT ((WifiInformationElementId)60)
#define IE_HT_OPERATION                         ((WifiInformationElementId)61)
#define IE_SECONDARY_CHANNEL_OFFSET             ((WifiInformationElementId)62)
#define IE_BSS_AVERAGE_ACCESS_DELAY             ((WifiInformationElementId)63)
#define IE_ANTENNA                              ((WifiInformationElementId)64)
#define IE_RSNI                                 ((WifiInformationElementId)65)
#define IE_MEASUREMENT_PILOT_TRANSMISSION       ((WifiInformationElementId)66)
#define IE_BSS_AVAILABLE_ADMISSION_CAPACITY     ((WifiInformationElementId)67)
#define IE_BSS_AC_ACCESS_DELAY                  ((WifiInformationElementId)68)
#define IE_TIME_ADVERTISEMENT                   ((WifiInformationElementId)69)
#define IE_RM_ENABLED_CAPACITIES                ((WifiInformationElementId)70)
#define IE_MULTIPLE_BSSID                       ((WifiInformationElementId)71)
#define IE_20_40_BSS_COEXISTENCE                ((WifiInformationElementId)72)
#define IE_20_40_BSS_INTOLERANT_CHANNEL_REPORT  ((WifiInformationElementId)73)
#define IE_OVERLAPPING_BSS_SCAN_PARAMETERS      ((WifiInformationElementId)74)
#define IE_RIC_DESCRIPTOR                       ((WifiInformationElementId)75)
#define IE_MANAGEMENT_MIC                       ((WifiInformationElementId)76)
// 77 is reserved
#define IE_EVENT_REQUEST                        ((WifiInformationElementId)78)
#define IE_EVENT_REPORT                         ((WifiInformationElementId)79)
#define IE_DIAGNOSTIC_REQUEST                   ((WifiInformationElementId)80)
#define IE_DIAGNOSTIC_REPORT                    ((WifiInformationElementId)81)
#define IE_LOCATION_PARAMETERS                  ((WifiInformationElementId)82)
#define IE_NONTRANSMITTED_BSSID_CAPABILITY      ((WifiInformationElementId)83)
#define IE_SSID_LIST                            ((WifiInformationElementId)84)
#define IE_MULTIPLE_BSSID_INDEX                 ((WifiInformationElementId)85)
#define IE_FMS_DESCRIPTOR                       ((WifiInformationElementId)86)
#define IE_FMS_REQUEST                          ((WifiInformationElementId)87)
#define IE_FMS_RESPONSE                         ((WifiInformationElementId)88)
#define IE_QOS_TRAFFIC_CAPABILITY               ((WifiInformationElementId)89)
#define IE_BSS_MAX_IDLE_PERIOD                  ((WifiInformationElementId)90)
#define IE_TFS_REQUEST                          ((WifiInformationElementId)91)
#define IE_TFS_RESPONSE                         ((WifiInformationElementId)92)
#define IE_WNM_SLEEP_MODE                       ((WifiInformationElementId)93)
#define IE_TIM_BROADCAST_REQUEST                ((WifiInformationElementId)94)
#define IE_TIM_BROADCAST_RESPONSE               ((WifiInformationElementId)95)
#define IE_COLLOCATED_INTERFERENCE_REPORT       ((WifiInformationElementId)96)
#define IE_CHANNEL_USAGE                        ((WifiInformationElementId)97)
#define IE_TIME_ZONE                            ((WifiInformationElementId)98)
#define IE_DMS_REQUEST                          ((WifiInformationElementId)99)
#define IE_DMS_RESPONSE                         ((WifiInformationElementId)100)
#define IE_LINK_IDENTIFIER                      ((WifiInformationElementId)101)
#define IE_WAKEUP_SCHEDULE                      ((WifiInformationElementId)102)
// 103 is reserved
#define IE_CHANNEL_SWITCH_TIMING                ((WifiInformationElementId)104)
#define IE_PTI_CONTROL                          ((WifiInformationElementId)105)
#define IE_TPU_BUFFER_STATUS                    ((WifiInformationElementId)106)
#define IE_INTERWORKING                         ((WifiInformationElementId)107)
#define IE_ADVERTISEMENT_PROTOCOL               ((WifiInformationElementId)108)
#define IE_EXPEDITED_BANDWIDTH_REQUEST          ((WifiInformationElementId)109)
#define IE_QOS_MAP_SET                          ((WifiInformationElementId)110)
#define IE_ROAMING_CONSORTIUM                   ((WifiInformationElementId)111)
#define IE_EMERGENCY_ALART_IDENTIFIER           ((WifiInformationElementId)112)
#define IE_MESH_CONFIGURATION                   ((WifiInformationElementId)113)
#define IE_MESH_ID                              ((WifiInformationElementId)114)
#define IE_MESH_LINK_METRIC_REPORT              ((WifiInformationElementId)115)
#define IE_CONGESTION_NOTIFICATION              ((WifiInformationElementId)116)
#define IE_MESH_PEERING_MANAGEMENT              ((WifiInformationElementId)117)
#define IE_MESH_CHANNEL_SWITCH_PARAMETERS       ((WifiInformationElementId)118)
#define IE_MESH_AWAKE_WINDOW                    ((WifiInformationElementId)119)
#define IE_BEACON_TIMING                        ((WifiInformationElementId)120)
#define IE_MCCAOP_SETUP_REQUEST                 ((WifiInformationElementId)121)
#define IE_MCCAOP_SETUP_REPLY                   ((WifiInformationElementId)122)
#define IE_MCCAOP_ADVERTISEMENT                 ((WifiInformationElementId)123)
#define IE_MCCAOP_TEARDOWN                      ((WifiInformationElementId)124)
#define IE_GANN                                 ((WifiInformationElementId)125)
#define IE_RANN                                 ((WifiInformationElementId)126)
// 67 to 126 are reserved
#define IE_EXTENDED_CAPABILITIES                ((WifiInformationElementId)127)
// 128 to 129 are reserved
#define IE_PREQ                                 ((WifiInformationElementId)130)
#define IE_PREP                                 ((WifiInformationElementId)131)
#define IE_PERR                                 ((WifiInformationElementId)132)
// 133 to 136 are reserved
#define IE_PROXY_UPDATE                         ((WifiInformationElementId)137)
#define IE_PROXY_UPDATE_CONFIRMATION            ((WifiInformationElementId)138)
#define IE_AUTHENTICATED_MESH_PEERING_EXCHANGE  ((WifiInformationElementId)139)
#define IE_MIC                                  ((WifiInformationElementId)140)
#define IE_DESTINATION_URI                      ((WifiInformationElementId)141)
#define IE_UAPSD_COEXISTENCE                    ((WifiInformationElementId)142)
// 143 to 173 are reserved
#define IE_MCCAOP_ADVERTISEMENT_OVERVIEW        ((WifiInformationElementId)174)
// 175 to 190 are reserved
#define IE_VHT_CAPABILITIES                     ((WifiInformationElementId)191)
#define IE_VHT_OPERATION                        ((WifiInformationElementId)192)
#define IE_EXTENDED_BSS_LOAD                    ((WifiInformationElementId)193)
#define IE_WIDE_BANDWIDTH_CHANNEL_SWITCH        ((WifiInformationElementId)194)
#define IE_VHT_TRANSMIT_POWER_ENVELOPE          ((WifiInformationElementId)195)
#define IE_CHANNEL_SWITCH_WRAPPER               ((WifiInformationElementId)196)
#define IE_AID                                  ((WifiInformationElementId)197)
#define IE_QUIET_CHANNEL                        ((WifiInformationElementId)198)
#define IE_OPERATING_MODE_NOTIFICATION          ((WifiInformationElementId)199)
// 200 to 220 are reserved
#define IE_VENDOR_SPECIFIC                      ((WifiInformationElementId)221)
// 222 to 254 are reserved
#define IE_EXTENSION                            ((WifiInformationElementId)255)

#define IE_EXT_HE_CAPABILITIES                  ((WifiInformationElementId)35)
#define IE_EXT_HE_OPERATION                     ((WifiInformationElementId)36)
#define IE_EXT_UORA_PARAMETER_SET               ((WifiInformationElementId)37)
#define IE_EXT_MU_EDCA_PARAMETER_SET            ((WifiInformationElementId)38)

/**
 * \brief Information element, as defined in 802.11-2007 standard
 * \ingroup wifi
 *
 * The IEEE 802.11 standard includes the notion of Information
 * Elements, which are encodings of management information to be
 * communicated between STAs in the payload of various frames of type
 * Management. Information Elements (IEs) have a common format, each
 * starting with a single octet - the Element ID, which indicates the
 * specific type of IE (a type to represent the options here is
 * defined as WifiInformationElementId). The next octet is a length field and
 * encodes the number of octets in the third and final field, which is
 * the IE Information field.
 *
 * The class ns3::WifiInformationElement provides a base for classes
 * which represent specific Information Elements. This class defines
 * pure virtual methods for serialisation
 * (ns3::WifiInformationElement::SerializeInformationField) and
 * deserialisation
 * (ns3::WifiInformationElement::DeserializeInformationField) of IEs, from
 * or to data members or other objects that simulation objects use to
 * maintain the relevant state.
 *
 * This class also provides an implementation of the equality
 * operator, which operates by comparing the serialised versions of
 * the two WifiInformationElement objects concerned.
 *
 * Elements are defined to have a common general format consisting of
 * a 1 octet Element ID field, a 1 octet length field, and a
 * variable-length element-specific information field. Each element is
 * assigned a unique Element ID as defined in this standard. The
 * Length field specifies the number of octets in the Information
 * field.
 *
 * This class is pure virtual and acts as base for classes which know
 * how to serialize specific IEs.
 */
class WifiInformationElement : public SimpleRefCount<WifiInformationElement>
{
public:
  virtual ~WifiInformationElement ();
  /**
   * Serialize entire IE including Element ID and length fields
   *
   * \param i an iterator which points to where the IE should be written.
   *
   * \return an iterator
   */
  virtual Buffer::Iterator Serialize (Buffer::Iterator i) const;
  /**
   * Deserialize entire IE, which must be present. The iterator
   * passed in must be pointing at the Element ID (i.e., the very
   * first octet) of the correct type of information element,
   * otherwise this method will generate a fatal error.
   *
   * \param i an iterator which points to where the IE should be read.
   *
   * \return an iterator
   */
  Buffer::Iterator Deserialize (Buffer::Iterator i);
  /**
   * Deserialize entire IE if it is present. The iterator passed in
   * must be pointing at the Element ID of an information element. If
   * the Element ID is not the one that the given class is interested
   * in then it will return the same iterator.
   *
   * \param i an iterator which points to where the IE should be read.
   *
   * \return an iterator
   */
  Buffer::Iterator DeserializeIfPresent (Buffer::Iterator i);
  /**
   * Get the size of the serialized IE including Element ID and
   * length fields.
   *
   * \return the size of the serialized IE in bytes
   */
  virtual uint16_t GetSerializedSize () const;

  // Each subclass must implement these pure virtual functions:
  /**
   * \returns Own unique Element ID
   */
  virtual WifiInformationElementId ElementId () const = 0;
  /**
   * Length of serialized information (i.e., the length of the body
   * of the IE, not including the Element ID and length octets. This
   * is the value that will appear in the second octet of the entire
   * IE - the length field)
   *
   * \return the length of serialized information
   */
  virtual uint8_t GetInformationFieldSize () const = 0;
  /**
   * Serialize information (i.e., the body of the IE, not including
   * the Element ID and length octets)
   *
   * \param start an iterator which points to where the information should
   *        be written.
   */
  virtual void SerializeInformationField (Buffer::Iterator start) const = 0;
  /**
   * Deserialize information (i.e., the body of the IE, not including
   * the Element ID and length octets)
   *
   * \param start an iterator which points to where the information should be written.
   * \param length
   *
   * \return the number of bytes read
   */
  virtual uint8_t DeserializeInformationField (Buffer::Iterator start,
                                               uint8_t length) = 0;

  /**
   * \returns Own unique Element ID Extension
   */
  virtual WifiInformationElementId ElementIdExt () const;

  // In addition, a subclass may optionally override the following...
  /**
   * Generate human-readable form of IE
   *
   * \param os output stream
   */
  virtual void Print (std::ostream &os) const;
  /**
   * Compare two IEs for equality by ID & Length, and then through
   * memcmp of serialised version
   *
   * \param a another information element to compare with
   *
   * \return true if the two IEs are equal,
   *         false otherwise
   */
  virtual bool operator== (WifiInformationElement const & a) const;

};

} //namespace ns3

#endif /* WIFI_INFORMATION_ELEMENT_H */
