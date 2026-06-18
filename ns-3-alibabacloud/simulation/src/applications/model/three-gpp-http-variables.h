/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Magister Solutions
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
 * Author: Budiarto Herman <budiarto.herman@magister.fi>
 *
 */

#ifndef THREE_GPP_HTTP_VARIABLES_H
#define THREE_GPP_HTTP_VARIABLES_H

#include <ns3/object.h>
#include <ns3/nstime.h>
#include <ns3/random-variable-stream.h>


namespace ns3 {


/**
 * \ingroup http
 * Container of various random variables to assist in generating web browsing
 * traffic pattern.
 *
 * The available random values to be retrieved are:
 *   - MTU size --- 536 bytes or 1460 bytes;
 *   - request size --- constant 350 bytes;
 *   - delay in generating a main object --- 0 second;
 *   - main object size --- truncated LogNormal distribution with a mean of
 *     10710 bytes;
 *     \image html http-main-object-size.png
 *   - delay in generating an embedded object --- 0 second;
 *   - embedded object size (in bytes) --- truncated LogNormal distribution
 *     with a mean of 7758 bytes;
 *     \image html http-embedded-object-size.png
 *   - number of embedded object per web page --- truncated Pareto distribution
 *     with a mean of approximately 3.95 (after truncation);
 *     \image html http-num-of-embedded-objects.png
 *   - length of reading time (in seconds) --- unbounded exponential
 *     distribution with a mean of 30 seconds; and
 *     \image html http-reading-time.png
 *   - length of parsing time (in seconds) --- unbounded exponential
 *     distribution with a mean of 0.13 seconds.
 *     \image html http-parsing-time.png
 *
 * Most parameters of the random distributions are configurable via attributes
 * and methods of this class. The default values are according to the following
 * references:
 *   - 3GPP TR 25.892, "Feasibility Study for Orthogonal Frequency Division
 *     Multiplexing (OFDM) for UTRAN enhancement"
 *   - IEEE 802.16m, "Evaluation Methodology Document (EMD)",
 *     IEEE 802.16m-08/004r5, July 2008.
 *   - NGMN Alliance, "NGMN Radio Access Performance Evaluation Methodology",
 *     v1.0, January 2008.
 *   - 3GPP2-TSGC5, "HTTP, FTP and TCP models for 1xEV-DV simulations", 2001.
 */
class ThreeGppHttpVariables : public Object
{
public:
  /// Create a new instance with default configuration of random distributions.
  ThreeGppHttpVariables ();

  /**
   * Returns the object TypeId.
   * \return The object TypeId.
   */
  static TypeId GetTypeId ();

  /**
   * Draws a random value of maximum transmission unit (MTU) size in bytes.
   *
   * The possible MTU sizes are 1460 bytes and 536 bytes with 76% and 24%
   * chances, respectively. The selected value is typically used by the sockets
   * of HTTP servers to send the response packets (both main objects and
   * embedded objects) to the requesting HTTP clients.
   *
   * \return MTU size in bytes.
   */
  uint32_t GetMtuSize ();

  /**
   * Returns the constant HTTP request size in bytes.
   *
   * By default, HTTP request size is 350 bytes, which can be modified by
   * setting the `RequestSize` attribute or calling the SetRequestSize()
   * method. This value applies to requests by HTTP client for main objects
   * and embedded objects alike.
   *
   * \return Request size in bytes.
   */
  uint32_t GetRequestSize ();

  /**
   * Returns the constant length of time needed by an HTTP server to generate
   * a main object.
   *
   * By default, main objects are generated instantly, i.e., zero delay. This
   * can be modified by setting the `MainObjectGenerationDelay` attribute or by
   * calling the SetMainObjectGenerationDelay() method.
   *
   * \return The delay for generating a main object.
   */
  Time GetMainObjectGenerationDelay ();

  /**
   * Draws a random main object size (in bytes) to be sent by an HTTP server.
   *
   * The size of main objects are determined by a truncated log-normal random
   * distribution. The default distribution settings produces random integers
   * with a mean of 10710 bytes and a standard deviation of 25032 bytes, and
   * then truncated to fit between 100 bytes and 2 MB. These default settings
   * can be modified via attributes or class methods.
   *
   * \return Main object size in bytes.
   */
  uint32_t GetMainObjectSize ();

  /**
   * Returns the constant length of time needed by an HTTP server to generate
   * an embedded object.
   *
   * By default, embedded objects are generated instantly, i.e., zero delay.
   * This can be modified by setting the `EmbeddedObjectGenerationDelay`
   * attribute or by calling the SetEmbeddedObjectGenerationDelay() method.
   *
   * \return The delay for generating an embedded object.
   */
  Time GetEmbeddedObjectGenerationDelay ();

  /**
   * Draws a random embedded object size (in bytes) to be sent by an HTTP
   * server.
   *
   * The size of embedded objects are determined by a truncated log-normal
   * random distribution. The default distribution settings produces random
   * integers with a mean of 7758 bytes and a standard deviation of 126168
   * bytes, and then truncated to fit between 50 bytes and 2 MB. These default
   * settings can be modified via attributes or class methods.
   *
   * \return Embedded object size in bytes.
   */
  uint32_t GetEmbeddedObjectSize ();

  /**
   * Draws a random integer indicating the number of embedded objects in a
   * main object.
   *
   * The number of embedded objects in a main object is typically discovered
   * when the HTTP client is parsing the main object in question. This number
   * is determined by a truncated Pareto distribution. The default distribution
   * settings produces (after truncation) random integers within [0, 53)
   * interval, with an actual mean of approximately 3.95.
   *
   * \return The number of embedded objects.
   */
  uint32_t GetNumOfEmbeddedObjects ();

  /**
   * Draws a random length of time which is spent by a hypothetical human user
   * (HTTP client) to read a web page before transitioning to another web page.
   *
   * Reading time is determined by an exponential distribution. The default
   * distribution settings produces random values with a mean of 30 seconds
   * without any maximum bound. The mean can be modified by setting the
   * `ReadingTimeMean` attribute or by calling the SetReadingTimeMean() method.
   *
   * \return Time interval for reading a web page.
   */
  Time GetReadingTime ();

  /**
   * Draws a random length of time which simulate the small delay caused by
   * HTTP client looking for any embedded objects within the received main
   * object.
   *
   * Parsing time is determined by an exponential distribution. The default
   * distribution settings produces random values with a mean of 130 ms without
   * any maximum bound. The mean can be modified by setting the
   * `ParsingTimeMean` attribute or by calling the SetParsingTimeMean() method.
   *
   * \return time interval for parsing a main object
   */
  Time GetParsingTime ();

  /**
   * Assign a fixed random variable stream number to the random variables used
   * by this model.
   *
   * Different random variable stream number makes random number generators to
   * produce different set of random values, thus possibly resulting to
   * different simulation results. On the other hand, two identical simulations
   * which use the same stream number should produce identical results (the
   * repeatability property of ns-3 simulation).
   *
   * \param stream The first stream index to use.
   * \return The number of stream indices assigned by this model.
   */
  int64_t AssignStreams (int64_t stream);

  // SETTER METHODS

  /**
   * \param constant Request size in bytes.
   */
  void SetRequestSize (uint32_t constant);
  /**
   * \param constant The delay for generating a main object.
   */
  void SetMainObjectGenerationDelay (Time constant);
  /**
   * \param mean The mean of main object sizes in bytes. Must be greater than
   *             zero.
   */
  void SetMainObjectSizeMean (uint32_t mean);
  /**
   * \param stdDev The standard deviation of main object sizes in bytes.
   */
  void SetMainObjectSizeStdDev (uint32_t stdDev);
  /**
   * \param constant The delay for generating an embedded object.
   */
  void SetEmbeddedObjectGenerationDelay (Time constant);
  /**
   * \param mean The mean of embedded object sizes in bytes. Must be greater
   *             than zero.
   */
  void SetEmbeddedObjectSizeMean (uint32_t mean);
  /**
   * \param stdDev The standard deviation of embedded object sizes in bytes.
   */
  void SetEmbeddedObjectSizeStdDev (uint32_t stdDev);
  /**
   * \param max The upper bound parameter of the Pareto distribution for
   *            determining the number of embedded objects per web page.
   */
  void SetNumOfEmbeddedObjectsMax (uint32_t max);
  /**
   * \param shape The shape parameter of the Pareto distribution for
   *              determining the number of embedded objects per web page.
   */
  void SetNumOfEmbeddedObjectsShape (double shape);
  /**
   * \param scale The scale parameter of the Pareto distribution for
   *              determining the number of embedded objects per web page.
   */
  void SetNumOfEmbeddedObjectsScale (uint32_t scale);
  /**
   * \param mean The mean length of time needed for reading a web page.
   */
  void SetReadingTimeMean (Time mean);
  /**
   * \param mean The mean length of time needed for parsing a main object.
   */
  void SetParsingTimeMean (Time mean);

private:
  /**
   * \brief Upon and after object initialization, update random variable
   * Mu and Sigma based on changes to attribute values.
   */
  void UpdateMainObjectMuAndSigma (void);
  /**
   * \brief Upon and after object initialization, update random variable
   * Mu and Sigma based on changes to attribute values.
   */
  void UpdateEmbeddedObjectMuAndSigma (void);
  
  void DoInitialize (void);   // overridden from base class

  /**
   * Random variable for determining MTU size (in bytes).
   */
  Ptr<UniformRandomVariable>      m_mtuSizeRng;
  /**
   * Random variable for determining request size (in bytes).
   */
  Ptr<ConstantRandomVariable>     m_requestSizeRng;
  /**
   * Random variable for determining the delay needed to generate a main object
   * (in seconds).
   */
  Ptr<ConstantRandomVariable>     m_mainObjectGenerationDelayRng;
  /**
   * Random variable for determining main object size (in bytes).
   */
  Ptr<LogNormalRandomVariable>    m_mainObjectSizeRng;
  /// Mean parameter for #m_mainObjectSizeRng;
  uint32_t                        m_mainObjectSizeMean;
  /// Standard deviation parameter for #m_mainObjectSizeRng;
  uint32_t                        m_mainObjectSizeStdDev;
  /// Lower bound parameter for #m_mainObjectSizeRng;
  uint32_t                        m_mainObjectSizeMin;
  /// Upper bound parameter for #m_mainObjectSizeRng;
  uint32_t                        m_mainObjectSizeMax;
  /// Lower MTU size
  uint32_t m_lowMtu;
  /// Higher MTU size
  uint32_t m_highMtu;
  /// High MTU size probability
  double m_highMtuProbability;
  /**
   * Random variable for determining the delay needed to generate an embedded
   * object (in seconds).
   */
  Ptr<ConstantRandomVariable>     m_embeddedObjectGenerationDelayRng;
  /**
   * Random variable for determining embedded object size (in bytes).
   */
  Ptr<LogNormalRandomVariable>    m_embeddedObjectSizeRng;
  /// Mean parameter for #m_embeddedObjectSizeRng.
  uint32_t                        m_embeddedObjectSizeMean;
  /// Standard deviation parameter for #m_embeddedObjectSizeRng.
  uint32_t                        m_embeddedObjectSizeStdDev;
  /// Lower bound parameter for #m_embeddedObjectSizeRng.
  uint32_t                        m_embeddedObjectSizeMin;
  /// Upper bound parameter for #m_embeddedObjectSizeRng.
  uint32_t                        m_embeddedObjectSizeMax;
  /**
   * Random variable for determining the number of embedded objects.
   */
  Ptr<ParetoRandomVariable>       m_numOfEmbeddedObjectsRng;
  /// Scale parameter for #m_numOfEmbeddedObjectsRng.
  uint32_t                        m_numOfEmbeddedObjectsScale;
  /**
   * Random variable for determining the length of reading time (in seconds).
   */
  Ptr<ExponentialRandomVariable>  m_readingTimeRng;
  /**
   * Random variable for determining the length of parsing time (in seconds).
   */
  Ptr<ExponentialRandomVariable>  m_parsingTimeRng;

}; // end of `class ThreeGppHttpVariables`


} // end of `namespace ns3`

#endif /* THREE_GPP_HTTP_VARIABLES_H */

