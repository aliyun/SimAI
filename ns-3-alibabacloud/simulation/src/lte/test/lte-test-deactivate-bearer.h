/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#ifndef LENA_TEST_DEACTIVATE_BEARER_H
#define LENA_TEST_DEACTIVATE_BEARER_H

#include "ns3/simulator.h"
#include "ns3/test.h"


namespace ns3 {

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test case that for testing the deactivation of the bearer
 * functionality. The test consist in deactivating the bearer and checking
 * if some traffic will appear afterwards on the given bearer. If some traffic
 * is detected the test will fail. Firs the de-activation is performed by
 * using Simulator::Schedule() method which will initiate bearer de-activation
 * after deActivateTime that is specified as the test parameter. Additionally,
 * test instantiate de-activation in sequence.
 */

class LenaDeactivateBearerTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param dist distance between the nodes
   * \param estThrPssDl estimated throughput PSS DL
   * \param packetSize packet size in bytes
   * \param interval interval time in ms
   * \param errorModelEnabled whether the error model is enabled
   * \param useIdealRrc whether to use ideal RRC
   */
  LenaDeactivateBearerTestCase (std::vector<uint16_t> dist, std::vector<uint32_t> estThrPssDl, std::vector<uint16_t> packetSize, uint16_t interval, bool  errorModelEnabled, bool useIdealRrc);
  virtual ~LenaDeactivateBearerTestCase ();

private:
  /**
   * Build name string function
   *
   * \param nUser number of UE nodes
   * \param dist distance between nodes
   * \returns name string
   */
  static std::string BuildNameString (uint16_t nUser, std::vector<uint16_t> dist);
  virtual void DoRun (void);
  uint16_t m_nUser; ///< number of UE nodes
  std::vector<uint16_t> m_dist; ///< distance between the nodes
  std::vector<uint16_t> m_packetSize;  ///< packet size in bytes
  uint16_t m_interval;    ///< interval time in ms
  std::vector<uint32_t> m_estThrPssDl; ///< estimated threshold PSS DL
  bool m_errorModelEnabled; ///< error mode enabled?
};


/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief The test suite class for the LenaDeactivateBearerTestCase.
 */

class LenaTestBearerDeactivateSuite : public TestSuite
{
public:
  LenaTestBearerDeactivateSuite ();
};

} // namespace ns3

#endif
