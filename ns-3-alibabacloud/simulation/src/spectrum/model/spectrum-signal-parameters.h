/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */


#ifndef SPECTRUM_SIGNAL_PARAMETERS_H
#define SPECTRUM_SIGNAL_PARAMETERS_H


#include <ns3/simple-ref-count.h>
#include <ns3/ptr.h>
#include <ns3/nstime.h>


namespace ns3 {

class SpectrumPhy;
class SpectrumValue;
class AntennaModel;

/**
 * \ingroup spectrum
 *
 * This struct provides the generic signal representation to be used by
 * all wireless technologies. Any specific wireless technology is
 * allowed to define additional signal parameters by inheriting from this
 * struct and providing additional member variables. This makes sure
 * that a minimum set of parameters (in particular, the ones needed
 * for interference calculation) is
 * common across all wireless technologies, while at the same time
 * allowing each technology to have its own specific signal parameters.
 *
 * Furthermore, since the signal parameters specific of every technology inherit
 * directly from this struct, each PHY can test (by using a dynamic
 * cast) if a signal being received belongs to a given technology or not.
 *
 * \note when inheriting from this class, make sure that the assignment operator and the copy constructor work properly, making deep copies if needed.
 */
struct SpectrumSignalParameters : public SimpleRefCount<SpectrumSignalParameters>
{
  /**
   * default constructor
   */
  SpectrumSignalParameters ();

  /**
   * destructor
   */
  virtual ~SpectrumSignalParameters ();

  /**
   * copy constructor
   * \param p object to copy
   */
  SpectrumSignalParameters (const SpectrumSignalParameters& p);

  /**
   * make a "virtual" copy of this class, where "virtual" refers to
   * the fact that if the actual object is a derived class of
   * SpectrumSignalParameters, then the copy is also a derived class
   * of the same type.
   * Each class inheriting from
   * SpectrumSignalParameters should override this method and use it
   * to call the copy constructor of the derived class.
   *
   * \return a copy of the (possibly derived) class
   */
  virtual Ptr<SpectrumSignalParameters> Copy ();

  /**
   * The Power Spectral Density of the
   * waveform, in linear units. The exact unit will depend on the
   * type of transmission medium involved: W for radio communications, Pa for
   * underwater acoustic communications. Other transmission media to
   * be defined.
   *
   * \note when SpectrumSignalParameters is copied, only the pointer to the PSD will be copied. This is because SpectrumChannel objects normally overwrite the psd anyway, so there is no point in making a copy.
   */
  Ptr <SpectrumValue> psd;

  /**
   * The duration of the packet transmission. It is
   * assumed that the Power Spectral Density remains constant for the
   * whole duration of the transmission. In other words, all waveform
   * have a rect shape with respect to time.
   */
  Time duration;

  /**
   * The SpectrumPhy instance that is making the transmission
   */
  Ptr<SpectrumPhy> txPhy;

  /**
   * The AntennaModel instance that was used to transmit this signal.
   */
  Ptr<AntennaModel> txAntenna;
};


}




#endif /* SPECTRUM_SIGNAL_PARAMETERS_H */
