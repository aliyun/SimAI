/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018
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

#include "he-ru.h"
#include "ns3/abort.h"
#include "ns3/assert.h"

namespace ns3 {

const HeRu::SubcarrierGroups HeRu::m_heRuSubcarrierGroups =
{
  // RUs in a 20 MHz HE PPDU (Table 28-6)
  { {20, HeRu::RU_26_TONE}, { /* 1 */ {{-121, -96}},
      /* 2 */ {{-95, -70}},
      /* 3 */ {{-68, -43}},
      /* 4 */ {{-42, -17}},
      /* 5 */ {{-16, -4}, {4, 16}},
      /* 6 */ {{17, 42}},
      /* 7 */ {{43, 68}},
      /* 8 */ {{70, 95}},
      /* 9 */ {{96, 121}} } },
  { {20, HeRu::RU_52_TONE}, { /* 1 */ {{-121, -70}},
      /* 2 */ {{-68, -17}},
      /* 3 */ {{17, 68}},
      /* 4 */ {{70, 121}} } },
  { {20, HeRu::RU_106_TONE}, { /* 1 */ {{-122, -17}},
      /* 2 */ {{17, 122}} } },
  { {20, HeRu::RU_242_TONE}, { /* 1 */ {{-122, -2}, {2, 122}} } },
  // RUs in a 40 MHz HE PPDU (Table 28-7)
  { {40, HeRu::RU_26_TONE}, { /* 1 */ {{-243, -218}},
      /* 2 */ {{-217, -192}},
      /* 3 */ {{-189, -164}},
      /* 4 */ {{-163, -138}},
      /* 5 */ {{-136, -111}},
      /* 6 */ {{-109, -84}},
      /* 7 */ {{-83, -58}},
      /* 8 */ {{-55, -30}},
      /* 9 */ {{-29, -4}},
      /* 10 */ {{4, 29}},
      /* 11 */ {{30, 55}},
      /* 12 */ {{58, 83}},
      /* 13 */ {{84, 109}},
      /* 14 */ {{111, 136}},
      /* 15 */ {{138, 163}},
      /* 16 */ {{164, 189}},
      /* 17 */ {{192, 217}},
      /* 18 */ {{218, 243}} } },
  { {40, HeRu::RU_52_TONE}, { /* 1 */ {{-243, -192}},
      /* 2 */ {{-189, -138}},
      /* 3 */ {{-109, -58}},
      /* 4 */ {{-55, -4}},
      /* 5 */ {{4, 55}},
      /* 6 */ {{58, 109}},
      /* 7 */ {{138, 189}},
      /* 8 */ {{192, 243}} } },
  { {40, HeRu::RU_106_TONE}, { /* 1 */ {{-243, -138}},
      /* 2 */ {{-109, -4}},
      /* 3 */ {{4, 109}},
      /* 4 */ {{138, 243}} } },
  { {40, HeRu::RU_242_TONE}, { /* 1 */ {{-244, -3}},
      /* 2 */ {{3, 244}} } },
  { {40, HeRu::RU_484_TONE}, { /* 1 */ {{-244, -3}, {3, 244}} } },
  // RUs in an 80 MHz HE PPDU (Table 28-8)
  { {80, HeRu::RU_26_TONE}, { /* 1 */ {{-499, -474}},
      /* 2 */ {{-473, -448}},
      /* 3 */ {{-445, -420}},
      /* 4 */ {{-419, -394}},
      /* 5 */ {{-392, -367}},
      /* 6 */ {{-365, -340}},
      /* 7 */ {{-339, -314}},
      /* 8 */ {{-311, -286}},
      /* 9 */ {{-285, -260}},
      /* 10 */ {{-257, -232}},
      /* 11 */ {{-231, -206}},
      /* 12 */ {{-203, -178}},
      /* 13 */ {{-177, -152}},
      /* 14 */ {{-150, -125}},
      /* 15 */ {{-123, -98}},
      /* 16 */ {{-97, -72}},
      /* 17 */ {{-69, -44}},
      /* 18 */ {{-43, -18}},
      /* 19 */ {{-16, -4}, {4, 16}},
      /* 20 */ {{18, 43}},
      /* 21 */ {{44, 69}},
      /* 22 */ {{72, 97}},
      /* 23 */ {{98, 123}},
      /* 24 */ {{125, 150}},
      /* 25 */ {{152, 177}},
      /* 26 */ {{178, 203}},
      /* 27 */ {{206, 231}},
      /* 28 */ {{232, 257}},
      /* 29 */ {{260, 285}},
      /* 30 */ {{286, 311}},
      /* 31 */ {{314, 339}},
      /* 32 */ {{340, 365}},
      /* 33 */ {{367, 392}},
      /* 34 */ {{394, 419}},
      /* 35 */ {{420, 445}},
      /* 36 */ {{448, 473}},
      /* 37 */ {{474, 499}} } },
  { {80, HeRu::RU_52_TONE}, { /* 1 */ {{-499, -448}},
      /* 2 */ {{-445, -394}},
      /* 3 */ {{-365, -314}},
      /* 4 */ {{-311, -260}},
      /* 5 */ {{-257, -206}},
      /* 6 */ {{-203, -152}},
      /* 7 */ {{-123, -72}},
      /* 8 */ {{-69, -18}},
      /* 9 */ {{18, 69}},
      /* 10 */ {{72, 123}},
      /* 11 */ {{152, 203}},
      /* 12 */ {{206, 257}},
      /* 13 */ {{260, 311}},
      /* 14 */ {{314, 365}},
      /* 15 */ {{394, 445}},
      /* 16 */ {{448, 499}} } },
  { {80, HeRu::RU_106_TONE}, { /* 1 */ {{-499, -394}},
      /* 2 */ {{-365, -260}},
      /* 3 */ {{-257, -152}},
      /* 4 */ {{-123, -18}},
      /* 5 */ {{18, 123}},
      /* 6 */ {{152, 257}},
      /* 7 */ {{260, 365}},
      /* 8 */ {{394, 499}} } },
  { {80, HeRu::RU_242_TONE}, { /* 1 */ {{-500, -259}},
      /* 2 */ {{-258, -17}},
      /* 3 */ {{17, 258}},
      /* 4 */ {{259, 500}} } },
  { {80, HeRu::RU_484_TONE}, { /* 1 */ {{-500, -17}},
      /* 2 */ {{17, 500}} } },
  { {80, HeRu::RU_996_TONE}, { /* 1 */ {{-500, -3}, {3, 500}} } }
};


HeRu::RuSpec::RuSpec ()
  : m_index (0)    // indicates undefined RU
{
}

HeRu::RuSpec::RuSpec (RuType ruType, std::size_t index, bool primary80MHz)
  : m_ruType (ruType),
    m_index (index),
    m_primary80MHz (primary80MHz),
    m_phyIndex (0)
{
  NS_ABORT_MSG_IF (index == 0, "Index cannot be zero");
}

HeRu::RuType
HeRu::RuSpec::GetRuType (void) const
{
  NS_ABORT_MSG_IF (m_index == 0, "Undefined RU");
  return m_ruType;
}

std::size_t
HeRu::RuSpec::GetIndex (void) const
{
  NS_ABORT_MSG_IF (m_index == 0, "Undefined RU");
  return m_index;
}

bool
HeRu::RuSpec::GetPrimary80MHz (void) const
{
  NS_ABORT_MSG_IF (m_index == 0, "Undefined RU");
  return m_primary80MHz;
}

void
HeRu::RuSpec::SetPhyIndex (uint16_t bw, uint8_t p20Index)
{
  bool primary80IsLower80 = (p20Index < bw / 40);

  if (bw < 160
      || m_ruType == HeRu::RU_2x996_TONE
      || (primary80IsLower80 && m_primary80MHz)
      || (!primary80IsLower80 && !m_primary80MHz))
    {
      m_phyIndex = m_index;
    }
  else
    {
      m_phyIndex = m_index + GetNRus (bw, m_ruType) / 2;
    }
}

bool
HeRu::RuSpec::IsPhyIndexSet (void) const
{
  return (m_phyIndex != 0);
}

std::size_t
HeRu::RuSpec::GetPhyIndex (void) const
{
  NS_ABORT_MSG_IF (m_phyIndex == 0, "RU PHY index not set");
  return m_phyIndex;
}

std::size_t
HeRu::GetNRus (uint16_t bw, RuType ruType)
{
  if (bw == 160 && ruType == RU_2x996_TONE)
    {
      return 1;
    }

  // if the bandwidth is 160MHz, search for the number of RUs available
  // in 80MHz and double the result.
  auto it = m_heRuSubcarrierGroups.find ({(bw == 160 ? 80 : bw), ruType});

  if (it == m_heRuSubcarrierGroups.end ())
    {
      return 0;
    }

  return (bw == 160 ? 2 : 1) * it->second.size ();
}

std::vector<HeRu::RuSpec>
HeRu::GetRusOfType (uint16_t bw, HeRu::RuType ruType)
{
  if (ruType == HeRu::RU_2x996_TONE)
    {
      NS_ASSERT (bw >= 160);
      return {{ruType, 1, true}};
    }

  std::vector<HeRu::RuSpec> ret;
  std::vector<bool> primary80MHzSet {true};

  if (bw == 160)
    {
      primary80MHzSet.push_back (false);
      bw = 80;
    }

  for (auto primary80MHz : primary80MHzSet)
    {
      for (std::size_t ruIndex = 1; ruIndex <= HeRu::m_heRuSubcarrierGroups.at ({bw, ruType}).size (); ruIndex++)
        {
          ret.push_back ({ruType, ruIndex, primary80MHz});
        }
    }
  return ret;
}

std::vector<HeRu::RuSpec>
HeRu::GetCentral26TonesRus (uint16_t bw, HeRu::RuType ruType)
{
  std::vector<std::size_t> indices;

  if (ruType == HeRu::RU_52_TONE || ruType == HeRu::RU_106_TONE)
    {
      if (bw == 20)
        {
          indices.push_back (5);
        }
      else if (bw == 40)
        {
          indices.insert (indices.end (), {5, 14});
        }
      else if (bw >= 80)
        {
          indices.insert (indices.end (), {5, 14, 19, 24, 33});
        }
    }
  else if (ruType == HeRu::RU_242_TONE || ruType == HeRu::RU_484_TONE)
    {
      if (bw >= 80)
        {
          indices.push_back (19);
        }
    }

  std::vector<HeRu::RuSpec> ret;
  std::vector<bool> primary80MHzSet {true};

  if (bw == 160)
    {
      primary80MHzSet.push_back (false);
    }

  for (auto primary80MHz : primary80MHzSet)
    {
      for (const auto& index : indices)
        {
          ret.push_back ({HeRu::RU_26_TONE, index, primary80MHz});
        }
    }
  return ret;
}

HeRu::SubcarrierGroup
HeRu::GetSubcarrierGroup (uint16_t bw, RuType ruType, std::size_t phyIndex)
{
  if (ruType == HeRu::RU_2x996_TONE) //handle special case of RU covering 160 MHz channel
    {
      NS_ABORT_MSG_IF (bw != 160, "2x996 tone RU can only be used on 160 MHz band");
      return {{-1012, -3}, {3, 1012}};
    }

  // Determine the shift to apply to tone indices for 160 MHz channel (i.e. -1012 to 1012), since
  // m_heRuSubcarrierGroups contains indices for lower 80 MHz subchannel (i.e. from -500 to 500).
  // The phyIndex is used to that aim.
  std::size_t indexInLower80MHz = phyIndex;
  std::size_t numRus = GetNRus (bw, ruType);
  int16_t shift = (bw == 160) ? -512 : 0;
  if (bw == 160 && phyIndex > (numRus / 2))
    {
      // The provided index is that of the upper 80 MHz subchannel
      indexInLower80MHz = phyIndex - (numRus / 2);
      shift = 512;
    }

  auto it = m_heRuSubcarrierGroups.find ({(bw == 160 ? 80 : bw), ruType});

  NS_ABORT_MSG_IF (it == m_heRuSubcarrierGroups.end (), "RU not found");
  NS_ABORT_MSG_IF (indexInLower80MHz > it->second.size (), "RU index not available");

  SubcarrierGroup group = it->second.at (indexInLower80MHz - 1);
  if (bw == 160)
    {
      for (auto & range : group)
        {
          range.first += shift;
          range.second += shift;
        }
    }
  return group;
}

bool
HeRu::DoesOverlap (uint16_t bw, RuSpec ru, const std::vector<RuSpec> &v)
{
  // A 2x996-tone RU spans 160 MHz, hence it overlaps with any other RU
  if (bw == 160 && ru.GetRuType () == RU_2x996_TONE && !v.empty ())
    {
      return true;
    }

  // This function may be called by the MAC layer, hence the PHY index may have
  // not been set yet. Hence, we pass the "MAC" index to GetSubcarrierGroup instead
  // of the PHY index. This is fine because we compare the primary 80 MHz bands of
  // the two RUs below.
  SubcarrierGroup rangesRu = GetSubcarrierGroup (bw, ru.GetRuType (), ru.GetIndex ());
  for (auto& p : v)
    {
      if (ru.GetPrimary80MHz () != p.GetPrimary80MHz ())
        {
          // the two RUs are located in distinct 80MHz bands
          continue;
        }
      for (const auto& rangeRu : rangesRu)
        {
          SubcarrierGroup rangesP = GetSubcarrierGroup (bw, p.GetRuType (), p.GetIndex ());
          for (auto& rangeP : rangesP)
            {
              if (rangeP.second >= rangeRu.first && rangeRu.second >= rangeP.first)
                {
                  return true;
                }
            }
        }
    }
  return false;
}

bool
HeRu::DoesOverlap (uint16_t bw, RuSpec ru, const SubcarrierGroup &toneRanges)
{
  for (const auto & range : toneRanges)
    {
      if (bw == 160 && ru.GetRuType () == RU_2x996_TONE)
        {
          return true;
        }

      SubcarrierGroup rangesRu = GetSubcarrierGroup (bw, ru.GetRuType (), ru.GetPhyIndex ());
      for (auto& r : rangesRu)
        {
          if (range.second >= r.first && r.second >= range.first)
            {
              return true;
            }
        }
    }
  return false;
}

HeRu::RuSpec
HeRu::FindOverlappingRu (uint16_t bw, RuSpec referenceRu, RuType searchedRuType)
{
  std::size_t numRus = HeRu::GetNRus (bw, searchedRuType);

  std::size_t numRusPer80Mhz;
  std::vector<bool> primary80MhzFlags;
  if (bw == 160)
    {
      primary80MhzFlags.push_back (true);
      primary80MhzFlags.push_back (false);
      numRusPer80Mhz = (searchedRuType == HeRu::RU_2x996_TONE ? 1 : numRus / 2);
    }
  else
    {
      primary80MhzFlags.push_back (referenceRu.GetPrimary80MHz ());
      numRusPer80Mhz = numRus;
    }

  for (const auto primary80MHz : primary80MhzFlags)
    {
      std::size_t index = 1;
      for (std::size_t indexPer80Mhz = 1; indexPer80Mhz <= numRusPer80Mhz; ++indexPer80Mhz, ++index)
        {
          RuSpec searchedRu (searchedRuType, index, primary80MHz);
          if (DoesOverlap (bw, referenceRu, {searchedRu}))
            {
              return searchedRu;
            }
        }
    }
  NS_ABORT_MSG ("The searched RU type " << searchedRuType << " was not found for bw=" << bw << " and referenceRu=" << referenceRu);
  return HeRu::RuSpec ();
}

std::ostream& operator<< (std::ostream& os, const HeRu::RuType &ruType)
{
  switch (ruType)
    {
      case HeRu::RU_26_TONE:
        os << "26-tones";
        break;
      case HeRu::RU_52_TONE:
        os << "52-tones";
        break;
      case HeRu::RU_106_TONE:
        os << "106-tones";
        break;
      case HeRu::RU_242_TONE:
        os << "242-tones";
        break;
      case HeRu::RU_484_TONE:
        os << "484-tones";
        break;
      case HeRu::RU_996_TONE:
        os << "996-tones";
        break;
      case HeRu::RU_2x996_TONE:
        os << "2x996-tones";
        break;
      default:
        NS_FATAL_ERROR ("Unknown RU type");
    }
  return os;
}

std::ostream& operator<< (std::ostream& os, const HeRu::RuSpec &ru)
{
  os << "RU{" << ru.GetRuType () << "/" << ru.GetIndex () << "/" << (ru.GetPrimary80MHz () ? "primary80MHz" : "secondary80MHz");
  if (ru.IsPhyIndexSet ())
    {
      os << "[" << ru.GetPhyIndex () << "]";
    }
  os << "}";
  return os;
}

uint16_t
HeRu::GetBandwidth (RuType ruType)
{
  switch (ruType)
    {
      case RU_26_TONE:
        return 2;
      case RU_52_TONE:
        return 4;
      case RU_106_TONE:
        return 8;
      case RU_242_TONE:
        return 20;
      case RU_484_TONE:
        return 40;
      case RU_996_TONE:
        return 80;
      case RU_2x996_TONE:
        return 160;
      default:
        NS_ABORT_MSG ("RU type " << ruType << " not found");
        return 0;
    }
}

HeRu::RuType
HeRu::GetRuType (uint16_t bandwidth)
{
  switch (bandwidth)
    {
      case 2:
        return RU_26_TONE;
      case 4:
        return RU_52_TONE;
      case 8:
        return RU_106_TONE;
      case 20:
        return RU_242_TONE;
      case 40:
        return RU_484_TONE;
      case 80:
        return RU_996_TONE;
      case 160:
        return RU_2x996_TONE;
      default:
        NS_ABORT_MSG (bandwidth << " MHz bandwidth not found");
        return RU_242_TONE;
    }
}

HeRu::RuType
HeRu::GetEqualSizedRusForStations (uint16_t bandwidth, std::size_t& nStations,
                                   std::size_t& nCentral26TonesRus)
{
  RuType ruType;
  uint8_t nRusAssigned = 0;

  // iterate over all the available RU types
  for (auto& ru : m_heRuSubcarrierGroups)
    {
      if (ru.first.first == bandwidth && ru.second.size () <= nStations)
        {
          ruType = ru.first.second;
          nRusAssigned = ru.second.size ();
          break;
        }
      else if (bandwidth == 160 && ru.first.first == 80 && (2 * ru.second.size () <= nStations))
        {
          ruType = ru.first.second;
          nRusAssigned = 2 * ru.second.size ();
          break;
        }
    }
  if (nRusAssigned == 0)
    {
      NS_ABORT_IF (bandwidth != 160 || nStations != 1);
      nRusAssigned = 1;
      ruType = RU_2x996_TONE;
    }

  nStations = nRusAssigned;

  switch (ruType)
    {
      case RU_52_TONE:
      case RU_106_TONE:
        if (bandwidth == 20)
          {
            nCentral26TonesRus = 1;
          }
        else if (bandwidth == 40)
          {
            nCentral26TonesRus = 2;
          }
        else
          {
            nCentral26TonesRus = 5;
          }
        break;
      case RU_242_TONE:
      case RU_484_TONE:
        nCentral26TonesRus = (bandwidth >= 80 ? 1 : 0);
        break;
      default:
        nCentral26TonesRus = 0;
    }

  if (bandwidth == 160)
    {
      nCentral26TonesRus *= 2;
    }

  return ruType;
}

bool
HeRu::RuSpec::operator== (const RuSpec& other) const
{
  // we do not compare the RU PHY indices because they may be uninitialized for
  // one of the compared RUs. This event should not cause the comparison to evaluate
  // to false
  return m_ruType == other.m_ruType
         && m_index == other.m_index
         && m_primary80MHz == other.m_primary80MHz;
}

bool
HeRu::RuSpec::operator!= (const RuSpec& other) const
{
  return !(*this == other);
}

} //namespace ns3
