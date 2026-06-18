/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#ifndef UAN_TX_MODE_H
#define UAN_TX_MODE_H

#include "ns3/object.h"
#include <map>

namespace ns3 {

class UanTxModeFactory;
class UanTxMode;

/**
 * \ingroup uan
 *
 * Abstraction of packet modulation information.
 *
 * This contains a lightweight globally unique id for the mode.
 * Mode details are held in the UanTxModeFactory.  Attributes
 * are set in by the UanTxModeFactory constructor.
 */
class UanTxMode
{
public:
  UanTxMode ();   //!< Constructor.
  ~UanTxMode ();  //!< Destructor.

  /**
   * Modulation type.
   */
  typedef enum {
    PSK,   //!< Phase shift keying.
    QAM,   //!< Quadrature amplitude modulation.
    FSK,   //!< Frequency shift keying.
    OTHER  //!< Unspecified/undefined.
  } ModulationType;
  /**
   * Get the modulation type of the mode.
   *
   * \return The modulation type.
   */
  ModulationType GetModType (void) const;
  /**
   * Get the data rate of the transmit mode.
   *
   * \return Data rate of the TX mode, in bits per second.
   */
  uint32_t GetDataRateBps (void) const;
  /**
   * Get the physical signaling rate.
   *
   * \return PHY rate in symbols per second.
   */
  uint32_t GetPhyRateSps (void) const;
  /**
   * Get the transmission center frequency.
   *
   * \return Center frequency, in Hz.
   */
  uint32_t GetCenterFreqHz (void) const;
  /**
   * Get the transmission signal bandwidth.
   *
   * \return Bandwidth in Hz.
   */
  uint32_t GetBandwidthHz (void) const;
  /**
   * Get the number of constellation points in the modulation scheme.
   *
   * \return Number of constellation points.
   */
  uint32_t GetConstellationSize (void) const;
  /**
   * Get the mode name.
   *
   * \return Name
   */
  std::string GetName (void) const;
  /**
   * Get a unique id for the mode.
   *
   * \return Unique ID.
   */
  uint32_t GetUid (void) const;

private:
  friend class UanTxModeFactory;
  friend std::ostream &operator<< (std::ostream & os, const UanTxMode &mode);
  friend std::istream &operator>> (std::istream & is, UanTxMode &mode);


  uint32_t m_uid;  //!< Mode id

};  // class UanTxMode

  
/**
 * Writes tx mode entry to stream os.
 *
 * \param os The output stream.
 * \param mode The mode.
 * \return The stream.
 */
std::ostream & operator << (std::ostream & os, const UanTxMode &mode);
/**
 * Reads tx mode entry from stream is
 *
 * \param is The input stream.
 * \param mode The mode.
 * \return The stream.
 */
std::istream & operator >> (std::istream & is, UanTxMode &mode);

/**
 * \ingroup uan
 *
 * Global database of UanTxMode objects, retrievable by id or name.
 */
class UanTxModeFactory
{
public:
  UanTxModeFactory ();   //!< Constructor.
  ~UanTxModeFactory ();  //!< Destructor.

  /**
   *
   * \param type Modulation type.
   * \param dataRateBps Data rate in BPS.
   * \param phyRateSps  Symbol rate in symbols per second.
   * \param cfHz Center frequency in Hz.
   * \param bwHz Bandwidth in Hz.
   * \param constSize Modulation constellation size (2 for BPSK, 4 for QPSK).
   * \param name Unique string name for this transmission mode.
   *
   * \return the transmit mode object
   */
  static UanTxMode CreateMode (UanTxMode::ModulationType type,
                               uint32_t dataRateBps,
                               uint32_t phyRateSps,
                               uint32_t cfHz,
                               uint32_t bwHz,
                               uint32_t constSize,
                               std::string name);

  /**
   * Get a mode by name.
   *
   * \param name String name of mode.
   * \return Mode with given name.
   */
  static UanTxMode GetMode (std::string name);
  /**
   * Get a mode by id.
   *
   * \param uid Unique ID of mode.
   * \return The mode with given uid.
   */
  static UanTxMode GetMode (uint32_t uid);

private:
  friend class UanTxMode;
  uint32_t m_nextUid;                  //!< next id number

  /**
   * \ingroup uan
   * Container for the UanTxMode properties.
   */
  struct UanTxModeItem
  {
    UanTxMode::ModulationType m_type;  //!< Modulation type.
    uint32_t m_cfHz;                   //!< Center frequency in Hz.
    uint32_t m_bwHz;                   //!< Bandwidth in Hz.
    uint32_t m_dataRateBps;            //!< Data rate in BPS.
    uint32_t m_phyRateSps;             //!< Symbol rate in symbols per second.
    uint32_t m_constSize;              //!< Modulation constellation size (2 for BPSK, 4 for QPSK).
    uint32_t m_uid;                    //!< Unique id.
    std::string m_name;                //!< Unique string name for this transmission mode.
  };

  /**
   * Container for modes
   *
   * \internal
   *   Accessed internally by uid and name, so a multimap might be more
   *   appropriate.  If name accesses are predominant, perhaps a map
   *   indexed by name, with a find for uid.  If accesses by uid dominate
   *   then vector (since uid's are sequential), and find by name.
   */
  std::map<uint32_t, UanTxModeItem> m_modes;

  /**
   * Check if the mode \pname{name} already exists.
   *
   * \param name The mode name to test.
   * \return True if \pname{name} exists.
   */
  bool NameUsed (std::string name);

  /**
   * Construct and get the static global factory instance.
   *
   * \return The global instance.
   */
  static UanTxModeFactory &GetFactory (void);

  /**
   * Get a mode by id.
   *
   * \param uid The unique id to find.
   * \return The corresponding mode.
   */
  UanTxModeItem &GetModeItem (uint32_t uid);

  /**
   * Get a mode by name.
   * \param name The mode name to find.
   * \return The corresponding mode.
   */
  UanTxModeItem &GetModeItem (std::string name);

  /**
   * Create a public UanTxMode from an internal UanTxModeItem.
   *
   * \param item The UanTxModeItem to reference.
   * \return A public UanTxMode.
   */
  UanTxMode MakeModeFromItem (const UanTxModeItem &item);

};  // class UanTxModeFactory

/**
 * \ingroup uan
 *
 * Container for UanTxModes.
 *
 * \see attribute_UanModesList
 */
class UanModesList
{
public:
  UanModesList ();           //!< Constructor
  virtual ~UanModesList ();  //!< Destructor

  /**
   * Add mode to this list.
   * \param mode The mode to add.
   */
  void AppendMode (UanTxMode mode);
  /**
   * Delete the mode at given index.
   * \param num Index of mode to delete.
   */
  void DeleteMode (uint32_t num);
  /**
   * Retrieve a mode by index.
   *
   * \param index Mode index.
   * \return Mode at given index.
   */
  UanTxMode operator[] (uint32_t index) const;
  /**
   * Get the number of modes in this list.
   *
   * \return Number of modes.
   */
  uint32_t GetNModes (void) const;


private:
  /** The vector of modes in this list. */
  std::vector<UanTxMode> m_modes;
  
  friend std::ostream &operator << (std::ostream &os, const UanModesList &ml);
  friend std::istream &operator >> (std::istream &is, UanModesList &ml);

};  // class UanModesList

/**
 * Write UanModesList to stream os
 *
 * \param os The output stream.
 * \param ml The mode list.
 * \return The stream.
 */
std::ostream &operator << (std::ostream &os, const UanModesList &ml);
/**
 * Read UanModesList from stream is.
 *
 * \param is The input stream.
 * \param ml The mode list to fill.
 * \return The stream.
 */
std::istream &operator >> (std::istream &is, UanModesList &ml);

ATTRIBUTE_HELPER_HEADER (UanModesList);

} // namespace ns3

#endif /* UAN_TX_MODE_H */
