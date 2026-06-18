/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Fraunhofer FKIE
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
 * Author:
 *  Sascha Alexander Jopen <jopen@cs.uni-bonn.de>
 */
#ifndef LR_WPAN_INTERFERENCE_HELPER_H
#define LR_WPAN_INTERFERENCE_HELPER_H

#include <ns3/simple-ref-count.h>
#include <ns3/ptr.h>
#include <set>

namespace ns3 {

class SpectrumValue;
class SpectrumModel;

/**
 * \ingroup lr-wpan
 *
 * \brief This class provides helper functions for LrWpan interference handling.
 */
class LrWpanInterferenceHelper : public SimpleRefCount<LrWpanInterferenceHelper>
{
public:
  /**
   * Create a new interference helper for the given SpectrumModel.
   *
   * \param spectrumModel the SpectrumModel to be used
   */
  LrWpanInterferenceHelper (Ptr<const SpectrumModel> spectrumModel);

  ~LrWpanInterferenceHelper (void);

  /**
   * Add the given signal to the set of accumulated signals. Never add the same
   * signal more than once. The SpectrumModels of the signal and the one used
   * for instantiation of the helper have to be the same.
   *
   * \param signal the signal to be added
   * \return false, if the signal was not added because the SpectrumModel of the
   * signal does not match the one of the helper, true otherwise.
   */
  bool AddSignal (Ptr<const SpectrumValue> signal);

  /**
   * Remove the given signal to the set of accumulated signals.
   *
   * \param signal the signal to be removed
   * \return false, if the signal was not removed (because it was not added
   * before), true otherwise.
   */
  bool RemoveSignal (Ptr<const SpectrumValue> signal);

  /**
   * Remove all currently accumulated signals.
   */
  void ClearSignals (void);

  /**
   * Get the sum of all accumulated signals.
   *
   * \return the sum of the signals
   */
  Ptr<SpectrumValue> GetSignalPsd (void) const;

  /**
   * Get the SpectrumModel used by the helper.
   *
   * \return the helpers SpectrumModel
   */
  Ptr<const SpectrumModel> GetSpectrumModel (void) const;
private:
  // Disable implicit copy constructors
  /**
   * \brief Copy constructor - defined and not implemented.
   */
  LrWpanInterferenceHelper (LrWpanInterferenceHelper const &);
  /**
   * \brief Copy constructor - defined and not implemented.
   * \returns
   */
  LrWpanInterferenceHelper& operator= (LrWpanInterferenceHelper const &);
  /**
   * The helpers SpectrumModel.
   */
  Ptr<const SpectrumModel> m_spectrumModel;

  /**
   * The set of accumulated signals.
   */
  std::set<Ptr<const SpectrumValue> > m_signals;

  /**
   * The precomputed sum of all accumulated signals.
   */
  mutable Ptr<SpectrumValue> m_signal;

  /**
   * Mark m_signal as dirty, whenever a signal is added or removed. m_signal has
   * to be recomputed before next use.
   */
  mutable bool m_dirty;
};

}

#endif /* LR_WPAN_INTERFERENCE_HELPER_H */
