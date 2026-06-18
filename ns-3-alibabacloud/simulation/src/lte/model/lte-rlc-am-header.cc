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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#include "ns3/log.h"

#include "ns3/lte-rlc-am-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteRlcAmHeader");

NS_OBJECT_ENSURE_REGISTERED (LteRlcAmHeader);

LteRlcAmHeader::LteRlcAmHeader ()
  : m_headerLength (0),
    m_dataControlBit (0xff),
    m_resegmentationFlag (0xff),
    m_pollingBit (0xff),
    m_framingInfo (0xff),
    m_sequenceNumber (0xfffa),
    m_segmentOffset (0xffff),
    m_lastOffset (0xffff),
    m_controlPduType (0xff),
    m_ackSn (0xffff)
{
}

LteRlcAmHeader::~LteRlcAmHeader ()
{
    m_headerLength = 0;
    m_dataControlBit = 0xff;
    m_resegmentationFlag = 0xff;
    m_pollingBit = 0xff;
    m_framingInfo = 0xff;
    m_sequenceNumber = 0xfffb;
    m_segmentOffset = 0xffff;
    m_lastOffset = 0xffff;
    m_controlPduType = 0xff;
    m_ackSn = 0xffff;
}

void
LteRlcAmHeader::SetDataPdu (void)
{
  m_headerLength = 4;
  m_dataControlBit = DATA_PDU;
}
void
LteRlcAmHeader::SetControlPdu (uint8_t controlPduType)
{
  m_headerLength = 2;
  m_dataControlBit = CONTROL_PDU;
  m_controlPduType = controlPduType;
}
bool
LteRlcAmHeader::IsDataPdu (void) const
{
  return m_dataControlBit == DATA_PDU;
}
bool
LteRlcAmHeader::IsControlPdu (void) const
{
  return m_dataControlBit == CONTROL_PDU;
}

void
LteRlcAmHeader::SetFramingInfo (uint8_t framingInfo)
{
  m_framingInfo = framingInfo & 0x03;
}

void
LteRlcAmHeader::SetSequenceNumber (SequenceNumber10 sequenceNumber)
{
  m_sequenceNumber = sequenceNumber;
}

uint8_t
LteRlcAmHeader::GetFramingInfo () const
{
  return m_framingInfo;
}

SequenceNumber10
LteRlcAmHeader::GetSequenceNumber () const
{
  return m_sequenceNumber;
}


void
LteRlcAmHeader::PushExtensionBit (uint8_t extensionBit)
{
  m_extensionBits.push_back (extensionBit);
  if (m_extensionBits.size() > 1)
    {
      if (m_extensionBits.size() % 2)
        {
          m_headerLength += 1;
        }
      else
        {
          m_headerLength += 2;
        }
    }
}

void
LteRlcAmHeader::PushLengthIndicator (uint16_t lengthIndicator)
{
  m_lengthIndicators.push_back (lengthIndicator);
}


uint8_t
LteRlcAmHeader::PopExtensionBit (void)
{
  uint8_t extensionBit = m_extensionBits.front ();
  m_extensionBits.pop_front ();

  return extensionBit;
}

uint16_t
LteRlcAmHeader::PopLengthIndicator (void)
{
  uint16_t lengthIndicator = m_lengthIndicators.front ();
  m_lengthIndicators.pop_front ();

  return lengthIndicator;
}


void
LteRlcAmHeader::SetResegmentationFlag (uint8_t resegFlag)
{
  m_resegmentationFlag = resegFlag & 0x01;
}

uint8_t
LteRlcAmHeader::GetResegmentationFlag () const
{
  return m_resegmentationFlag;
}


void
LteRlcAmHeader::SetPollingBit (uint8_t pollingBit)
{
  m_pollingBit = pollingBit & 0x01;
}

uint8_t
LteRlcAmHeader::GetPollingBit (void) const
{
  return m_pollingBit;
}


void
LteRlcAmHeader::SetLastSegmentFlag (uint8_t lsf)
{
  m_lastSegmentFlag = lsf & 0x01;
}

uint8_t
LteRlcAmHeader::GetLastSegmentFlag (void) const
{
  return m_lastSegmentFlag;
}


void
LteRlcAmHeader::SetSegmentOffset (uint16_t segmentOffset)
{
  m_segmentOffset = segmentOffset & 0x7FFF;
}

uint16_t
LteRlcAmHeader::GetSegmentOffset (void) const
{
  return m_segmentOffset;
}

uint16_t
LteRlcAmHeader::GetLastOffset (void) const
{
  return m_lastOffset;
}


void
LteRlcAmHeader::SetAckSn (SequenceNumber10 ackSn)
{
  m_ackSn = ackSn;
}

bool
LteRlcAmHeader::OneMoreNackWouldFitIn (uint16_t bytes)
{
  NS_LOG_FUNCTION (this << bytes);
  NS_ASSERT_MSG (m_dataControlBit == CONTROL_PDU && m_controlPduType == LteRlcAmHeader::STATUS_PDU,
                 "method allowed only for STATUS PDUs");
  if (m_nackSnList.size () % 2 == 0)
    {
      return (m_headerLength < bytes);
    }
  else
    {
      return (m_headerLength < (bytes - 1));
    }
}

void
LteRlcAmHeader::PushNack (int nack)
{
  NS_LOG_FUNCTION (this << nack);
  NS_ASSERT_MSG (m_dataControlBit == CONTROL_PDU && m_controlPduType == LteRlcAmHeader::STATUS_PDU,
                 "method allowed only for STATUS PDUs");
  m_nackSnList.push_back (nack);

  if (m_nackSnList.size () % 2 == 0)
    {
      m_headerLength++;
    }
  else
    {
      m_headerLength+=2;
    }
}

bool
LteRlcAmHeader::IsNackPresent (SequenceNumber10 nack)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (m_dataControlBit == CONTROL_PDU && m_controlPduType == LteRlcAmHeader::STATUS_PDU,
                 "method allowed only for STATUS PDUs");
  for (std::list<int>::iterator nackIt = m_nackSnList.begin ();
       nackIt != m_nackSnList.end ();
       ++nackIt)
    {
      if ((*nackIt) == nack.GetValue ())
        {
          return true;
        }
    }
  return false;
}

int
LteRlcAmHeader::PopNack (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (m_dataControlBit == CONTROL_PDU && m_controlPduType == LteRlcAmHeader::STATUS_PDU,
                 "method allowed only for STATUS PDUs");
  if ( m_nackSnList.empty () )
    {
      return -1;
    }

  int nack = m_nackSnList.front ();
  m_nackSnList.pop_front ();

  return nack;
}


SequenceNumber10
LteRlcAmHeader::GetAckSn (void) const
{
  return m_ackSn;
}


TypeId
LteRlcAmHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LteRlcAmHeader")
    .SetParent<Header> ()
    .SetGroupName("Lte")
    .AddConstructor<LteRlcAmHeader> ()
  ;
  return tid;
}

TypeId
LteRlcAmHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
LteRlcAmHeader::Print (std::ostream &os)  const
{
  std::list <uint8_t>::const_iterator it1 = m_extensionBits.begin ();
  std::list <uint16_t>::const_iterator it2 = m_lengthIndicators.begin ();
  std::list <int>::const_iterator it3 = m_nackSnList.begin ();

  os << "Len=" << m_headerLength;
  os << " D/C=" << (uint16_t)m_dataControlBit;

  if ( m_dataControlBit == DATA_PDU )
    {
      os << " RF=" << (uint16_t)m_resegmentationFlag;
      os << " P=" << (uint16_t)m_pollingBit;
      os << " FI=" << (uint16_t)m_framingInfo;
      os << " E=" << (uint16_t)(*it1);
      os << " SN=" << m_sequenceNumber;
      os << " LSF=" << (uint16_t)(m_lastSegmentFlag);
      os << " SO=" << m_segmentOffset;

      it1++;
      if (it1 != m_extensionBits.end ())
        {
          os << " E=";
        }
      while ( it1 != m_extensionBits.end () )
        {
          os << (uint16_t)(*it1);
          it1++;
        }

      if (it2 != m_lengthIndicators.end ())
        {
          os << " LI=";
        }
      while ( it2 != m_lengthIndicators.end () )
        {
          os << (uint16_t)(*it2) << " ";
          it2++;
        }
    }
  else // if ( m_dataControlBit == CONTROL_PDU )
    {
      os << " ACK_SN=" << m_ackSn;

       while ( it3 != m_nackSnList.end () )
        {
          os << " NACK_SN=" << (int)(*it3);
          it3++;
        }


    }
}

uint32_t LteRlcAmHeader::GetSerializedSize (void) const
{
  return m_headerLength;
}

void LteRlcAmHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  std::list <uint8_t>::const_iterator it1 = m_extensionBits.begin ();
  std::list <uint16_t>::const_iterator it2 = m_lengthIndicators.begin ();
  std::list <int>::const_iterator it3 = m_nackSnList.begin ();

  if ( m_dataControlBit == DATA_PDU )
    {
      i.WriteU8 ( ((DATA_PDU << 7) & 0x80) |
                  ((m_resegmentationFlag << 6) & 0x40) |
                  ((m_pollingBit << 5) & 0x20) |
                  ((m_framingInfo << 3) & 0x18) |
                  (((*it1) << 2) & 0x04) |
                  ((m_sequenceNumber.GetValue () >> 8) & 0x0003) );
      i.WriteU8 ( m_sequenceNumber.GetValue () & 0x00FF );
      i.WriteU8 ( ((m_lastSegmentFlag << 7) & 0x80) |
                  ((m_segmentOffset >> 8) & 0x007F) );
      i.WriteU8 ( m_segmentOffset & 0x00FF );
      it1++;

      while ( it1 != m_extensionBits.end () &&
              it2 != m_lengthIndicators.end () )
        {
          uint16_t oddLi, evenLi;
          uint8_t oddE, evenE;

          oddE = *it1;
          oddLi = *it2;

          it1++;
          it2++;

          if ( it1 != m_extensionBits.end () &&
              it2 != m_lengthIndicators.end () )
            {
              evenE = *it1;
              evenLi = *it2;

              i.WriteU8 ( ((oddE << 7) & 0x80) | ((oddLi >> 4) & 0x007F) );
              i.WriteU8 ( ((oddLi << 4) & 0x00F0) | ((evenE << 3) & 0x08) | ((evenLi >> 8) & 0x0007) );
              i.WriteU8 ( evenLi & 0x00FF );

              it1++;
              it2++;
            }
          else
            {
              i.WriteU8 ( ((oddE << 7) & 0x80) | ((oddLi >> 4) & 0x007F) );
              i.WriteU8 ( ((oddLi << 4) & 0x00F0) ); // Padding is implicit
            }
        }
    }
  else // if ( m_dataControlBit == CONTROL_PDU )
    {
      i.WriteU8 ( ((CONTROL_PDU << 7) & 0x80) |
                  ((m_controlPduType << 4) & 0x70) |
                  ((m_ackSn.GetValue () >> 6) & 0x0F) );
      // note: second part of ackSn will be written later

      // serialize the NACKs
      if ( it3 == m_nackSnList.end () )
        {
          NS_LOG_LOGIC (this << " no NACKs");
           // If there are no NACKs then this line adds the rest of the ACK
           // along with 0x00, indicating an E1 value of 0 or no NACKs follow.
           i.WriteU8 ( ((m_ackSn.GetValue () << 2) & 0xFC) );
        }
      else
        {
          int oddNack = *it3;
          int evenNack = -1;
          // Else write out a series of E1 = 1 and NACK values. Note since we
          // are not supporting SO start/end the value of E2 will always be 0.


          // First write out the ACK along with the very first NACK
          // And the remaining NACK with 0x02 or 10 in binary to set
          // E1 to 1, then Or in the first bit of the NACK
          i.WriteU8 ( ((m_ackSn.GetValue () << 2) & 0xFC)
            | (0x02)
            | ((*it3 >> 9) & 0x01));

          while ( it3 != m_nackSnList.end () )
            {
              // The variable oddNack has the current NACK value to write, also
              // either the setup to enter this loop or the previous loop would
              // have written the highest order bit to the previous octet.
              // Write the next set of bits (2 - 9) into the next octet
              i.WriteU8( ((oddNack >> 1) & 0xFF) );

              // Next check to see if there is going to be another NACK after
              // this
              it3++;
              if ( it3 != m_nackSnList.end () )
                {
                  // Yes there will be another NACK after this, so E1 will be 1
                  evenNack = *it3;
                  i.WriteU8( ((oddNack << 7) & 0x80)
                   | (0x40)  // E1 = 1 E2 = 0, more NACKs
                   | ( (evenNack >> 5) & 0x1F) );

                  // The final octet of this loop will have the rest of the
                  // NACK and another E1, E2. Check to see if there will be
                  // one more NACK after this.
                  it3++;
                  if ( it3 != m_nackSnList.end () )
                    {
                      // Yes there is at least one more NACK. Finish writing
                      // this octet and the next iteration will do the rest.
                      oddNack = *it3;
                      i.WriteU8 ( ((evenNack << 3) & 0xF8)
                        | (0x04)
                        | ((oddNack >> 9) & 0x01));
                    }
                  else
                    {
                      // No, there are no more NACKs
                      i.WriteU8 ( ((evenNack << 3) & 0xF8) );
                    }
                }
              else
                {
                  // No, this is the last NACK so E1 will be 0
                  i.WriteU8 ( ((oddNack << 7) & 0x80) );
                }
            }
        }

    }
}

uint32_t LteRlcAmHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint8_t byte_1;
  uint8_t byte_2;
  uint8_t byte_3;
  uint8_t byte_4;
  uint8_t extensionBit;

  byte_1 = i.ReadU8 ();
  m_headerLength = 1;
  m_dataControlBit = (byte_1 & 0x80) >> 7;

  if ( m_dataControlBit == DATA_PDU )
    {
      byte_2 = i.ReadU8 ();
      byte_3 = i.ReadU8 ();
      byte_4 = i.ReadU8 ();
      m_headerLength += 3;

      m_resegmentationFlag = (byte_1 & 0x40) >> 6;
      m_pollingBit         = (byte_1 & 0x20) >> 5;
      m_framingInfo        = (byte_1 & 0x18) >> 3;
      m_sequenceNumber     = ((byte_1 & 0x03) << 8) | byte_2;

      m_lastSegmentFlag    = (byte_3 & 0x80) >> 7;
      m_segmentOffset      = (byte_3 & 0x7F) | byte_4;

      extensionBit = (byte_1 & 0x04) >> 2;
      m_extensionBits.push_back (extensionBit);

      if (extensionBit == DATA_FIELD_FOLLOWS)
        {
          return GetSerializedSize ();
        }

      uint16_t oddLi, evenLi;
      uint8_t oddE, evenE;
      bool moreLiFields = (extensionBit == E_LI_FIELDS_FOLLOWS);

      while (moreLiFields)
        {
          byte_1 = i.ReadU8 ();
          byte_2 = i.ReadU8 ();

          oddE = (byte_1 & 0x80) >> 7;
          oddLi = ((byte_1 & 0x7F) << 4) | ((byte_2 & 0xF0) >> 4);
          moreLiFields = (oddE == E_LI_FIELDS_FOLLOWS);

          m_extensionBits.push_back (oddE);
          m_lengthIndicators.push_back (oddLi);
          m_headerLength += 2;

          if (moreLiFields)
            {
              byte_3 = i.ReadU8 ();

              evenE = (byte_2 & 0x08) >> 3;
              evenLi = ((byte_2 & 0x07) << 8) | (byte_3 & 0xFF);
              moreLiFields = (evenE == E_LI_FIELDS_FOLLOWS);

              m_extensionBits.push_back (evenE);
              m_lengthIndicators.push_back (evenLi);

              m_headerLength += 1;
            }
        }

      if (m_resegmentationFlag == SEGMENT)
        {
          m_lastOffset = m_segmentOffset + start.GetSize () - m_headerLength;
        }
    }
  else // if ( m_dataControlBit == CONTROL_PDU )
    {
      byte_2 = i.ReadU8 ();

      m_controlPduType = (byte_1 & 0x70) >> 4;
      m_ackSn = ((byte_1 & 0x0F) << 6 ) | ((byte_2 & 0xFC) >> 2);

      int moreNacks = (byte_2 & 0x02) >> 1;
      // Get the first NACK outside the loop as it is not preceded by an E2
      // field but all following NACKs will.
      if ( moreNacks == 1 )
        {
          byte_3 = i.ReadU8 ();
          byte_4 = i.ReadU8 ();
          m_headerLength = 4;

          m_nackSnList.push_back (
            ((byte_2 & 0x01) << 9)
            | (byte_3 << 1)
            | ((byte_4 & 0x80) >> 7)
          );

         // Loop until all NACKs are found
         moreNacks = ((byte_4 & 0x40) >> 6);
         uint8_t byte = byte_4;
         uint8_t nextByte;
         uint8_t finalByte;
         while (moreNacks == 1)
           {
             // Ignore E2, read next NACK
             nextByte = i.ReadU8 ();
             m_nackSnList.push_back (
              ((byte & 0x1F) << 5)
              | ((nextByte & 0xF8) >> 3)
            );

            // Check for another NACK, after this any following NACKs will
            // be aligned properly for the next iteration of this loop.
            moreNacks = (nextByte & 0x04) >> 2;
            byte = nextByte;
            if (moreNacks == 1)
              {
                nextByte = i.ReadU8 ();
                finalByte = i.ReadU8 ();

                 m_nackSnList.push_back (
                   ((byte & 0x01) << 9)
                   | (nextByte << 1)
                   | ((finalByte & 0x80) >> 7)
                 );

                moreNacks = ((finalByte & 0x40) >> 6);
                byte = finalByte;
                m_headerLength+=3;
              }
            else
              {
                m_headerLength++;
              }

           }
        }
      else
        {
          m_headerLength++;
        }
    }

  return GetSerializedSize ();
}

}; // namespace ns3
