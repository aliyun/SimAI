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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */



#include "ns3/test.h"
#include "ns3/log.h"

#include "ns3/spectrum-test.h"
#include "ns3/lte-spectrum-value-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LteTestSpectrumValueHelper");

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test checks if lte spectrum model is generated properly. Different 
 * test cases are configured by defining different frequency and banwidth.
 */
class LteSpectrumModelTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param str the test reference name
   * \param earfcn the central carrier frequency
   * \param bw bandwidth tha bandwidth
   * \param fcs the expected spectrum model
   */
  LteSpectrumModelTestCase (const char* str, uint16_t earfcn, uint8_t bw, std::vector<double> fcs);
  virtual ~LteSpectrumModelTestCase ();

protected:
  Ptr<SpectrumModel> m_actual; ///< actual spectrum model
  Ptr<SpectrumModel> m_expected; ///< expected spectrum model

private:
  virtual void DoRun (void);
};

LteSpectrumModelTestCase::LteSpectrumModelTestCase (const char* str, uint16_t earfcn, uint8_t bw, std::vector<double> fcs)
  :   TestCase (std::string ("SpectrumModel ") + str)
{
  NS_LOG_FUNCTION (this << str << earfcn << bw);
  m_actual = LteSpectrumValueHelper::GetSpectrumModel (earfcn, bw);
  m_expected = Create<SpectrumModel> (fcs);
}

LteSpectrumModelTestCase::~LteSpectrumModelTestCase ()
{
}

void 
LteSpectrumModelTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  NS_TEST_ASSERT_MSG_SPECTRUM_MODEL_EQ_TOL ((*m_actual), (*m_expected), 0.0000001, "spectrum model mismatch");
}




/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test that the function for creation of LTE noise power spectral 
 * density is working properly.
 */
class LteNoisePsdTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param str the test reference name
   * \param earfcn
   * \param bw bandwidth
   * \param noiseFigureDb noise figure in dB
   * \param expected executed spectrum value
   */
  LteNoisePsdTestCase (const char* str, uint16_t earfcn, uint8_t bw, double noiseFigureDb, SpectrumValue& expected);
  virtual ~LteNoisePsdTestCase ();

protected:
  Ptr<SpectrumValue> m_actual; ///< actual spectrum value
  Ptr<SpectrumValue> m_expected; ///< expected spectrum value

private:
  virtual void DoRun (void);
};

LteNoisePsdTestCase::LteNoisePsdTestCase (const char* str, uint16_t earfcn, uint8_t bw, double noiseFigureDb, SpectrumValue& expected)
  :   TestCase (str),
    m_actual (LteSpectrumValueHelper::CreateNoisePowerSpectralDensity (earfcn, bw, noiseFigureDb)),
    m_expected (Create<SpectrumValue> (expected))
{
  NS_LOG_FUNCTION (this << str << earfcn << bw << noiseFigureDb);
}

LteNoisePsdTestCase::~LteNoisePsdTestCase ()
{
}

void 
LteNoisePsdTestCase::DoRun (void)
{
  NS_TEST_ASSERT_MSG_EQ (m_actual->GetSpectrumModelUid (), m_expected->GetSpectrumModelUid (), "SpectrumModel UID mismatch");
  NS_TEST_ASSERT_MSG_SPECTRUM_VALUE_EQ_TOL ((*m_actual), (*m_expected), 0.0000001, "SpectrumValue not equal");
}





/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test that the funtcion for the creation of the Lte transmission power 
 * spectral density is working as expected.
 */
class LteTxPsdTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param str the reference name
   * \param earfcn
   * \param bw bandwidth
   * \param txPowerDbm tranmit power in dBm
   * \param activeRbs active RBs
   * \param expected executed spectrum value
   */
  LteTxPsdTestCase (const char* str, uint16_t earfcn, uint8_t bw, double txPowerDbm, std::vector<int> activeRbs, SpectrumValue& expected);
  virtual ~LteTxPsdTestCase ();

protected:
  Ptr<SpectrumValue> m_actual; ///< actual spectrum value
  Ptr<SpectrumValue> m_expected; ///< expected spectrum value

private:
  virtual void DoRun (void);
};

LteTxPsdTestCase::LteTxPsdTestCase (const char* str, uint16_t earfcn, uint8_t bw, double txPowerDbm, std::vector<int> activeRbs, SpectrumValue& expected)
  :   TestCase (str),
    m_actual (LteSpectrumValueHelper::CreateTxPowerSpectralDensity (earfcn, bw, txPowerDbm, activeRbs)),
    m_expected (Create<SpectrumValue> (expected))
{
  NS_LOG_FUNCTION (this << str << earfcn << bw << txPowerDbm);
}

LteTxPsdTestCase::~LteTxPsdTestCase ()
{
}

void 
LteTxPsdTestCase::DoRun (void)
{
  NS_TEST_ASSERT_MSG_EQ (m_actual->GetSpectrumModelUid (), m_expected->GetSpectrumModelUid (), "SpectrumModel UID mismatch");
  NS_TEST_ASSERT_MSG_SPECTRUM_VALUE_EQ_TOL ((*m_actual), (*m_expected), 0.0000001, "SpectrumValues not equal");
}




/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief Test suite for LteSpectrumValueHelper. Test suite is 
 * checking different functionalities of LteSpectrumValueHelper.
 */
class LteSpectrumValueHelperTestSuite : public TestSuite
{
public:
  LteSpectrumValueHelperTestSuite ();
};

static LteSpectrumValueHelperTestSuite g_lteSpectrumValueHelperTestSuite;

LteSpectrumValueHelperTestSuite::LteSpectrumValueHelperTestSuite ()
  : TestSuite ("lte-spectrum-value-helper", UNIT)
{
//   LogLevel logLevel = (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_LEVEL_ALL);
//   LogComponentEnable ("LteSpectrumModelTestCase", logLevel);
//   LogComponentEnable ("LteSpectrumValueHelperTestSuite", logLevel);
//   LogComponentEnable ("LteSpectrumValueHelper", logLevel);

 
  NS_LOG_INFO ("Creating LteSpectrumValueHelperTestSuite");



  std::vector<double> fc2160nrb6 (6);
  fc2160nrb6[0] = 2.159550000000e+09;
  fc2160nrb6[1] = 2.159730000000e+09;
  fc2160nrb6[2] = 2.159910000000e+09;
  fc2160nrb6[3] = 2.160090000000e+09;
  fc2160nrb6[4] = 2.160270000000e+09;
  fc2160nrb6[5] = 2.160450000000e+09;
  AddTestCase (new LteSpectrumModelTestCase ("fc2160nrb6", 500, 6, fc2160nrb6), TestCase::QUICK);

  std::vector<double> fc2160nrb15 (15);
  fc2160nrb15[0] = 2.158740000000e+09;
  fc2160nrb15[1] = 2.158920000000e+09;
  fc2160nrb15[2] = 2.159100000000e+09;
  fc2160nrb15[3] = 2.159280000000e+09;
  fc2160nrb15[4] = 2.159460000000e+09;
  fc2160nrb15[5] = 2.159640000000e+09;
  fc2160nrb15[6] = 2.159820000000e+09;
  fc2160nrb15[7] = 2.160000000000e+09;
  fc2160nrb15[8] = 2.160180000000e+09;
  fc2160nrb15[9] = 2.160360000000e+09;
  fc2160nrb15[10] = 2.160540000000e+09;
  fc2160nrb15[11] = 2.160720000000e+09;
  fc2160nrb15[12] = 2.160900000000e+09;
  fc2160nrb15[13] = 2.161080000000e+09;
  fc2160nrb15[14] = 2.161260000000e+09;
  AddTestCase (new LteSpectrumModelTestCase ("fc2160nrb15", 500, 15, fc2160nrb15), TestCase::QUICK);

  std::vector<double> fc2160nrb25 (25);
  fc2160nrb25[0] = 2.157840000000e+09;
  fc2160nrb25[1] = 2.158020000000e+09;
  fc2160nrb25[2] = 2.158200000000e+09;
  fc2160nrb25[3] = 2.158380000000e+09;
  fc2160nrb25[4] = 2.158560000000e+09;
  fc2160nrb25[5] = 2.158740000000e+09;
  fc2160nrb25[6] = 2.158920000000e+09;
  fc2160nrb25[7] = 2.159100000000e+09;
  fc2160nrb25[8] = 2.159280000000e+09;
  fc2160nrb25[9] = 2.159460000000e+09;
  fc2160nrb25[10] = 2.159640000000e+09;
  fc2160nrb25[11] = 2.159820000000e+09;
  fc2160nrb25[12] = 2.160000000000e+09;
  fc2160nrb25[13] = 2.160180000000e+09;
  fc2160nrb25[14] = 2.160360000000e+09;
  fc2160nrb25[15] = 2.160540000000e+09;
  fc2160nrb25[16] = 2.160720000000e+09;
  fc2160nrb25[17] = 2.160900000000e+09;
  fc2160nrb25[18] = 2.161080000000e+09;
  fc2160nrb25[19] = 2.161260000000e+09;
  fc2160nrb25[20] = 2.161440000000e+09;
  fc2160nrb25[21] = 2.161620000000e+09;
  fc2160nrb25[22] = 2.161800000000e+09;
  fc2160nrb25[23] = 2.161980000000e+09;
  fc2160nrb25[24] = 2.162160000000e+09;
  AddTestCase (new LteSpectrumModelTestCase ("fc2160nrb25", 500, 25, fc2160nrb25), TestCase::QUICK);

  std::vector<double> fc2160nrb50 (50);
  fc2160nrb50[0] = 2.155590000000e+09;
  fc2160nrb50[1] = 2.155770000000e+09;
  fc2160nrb50[2] = 2.155950000000e+09;
  fc2160nrb50[3] = 2.156130000000e+09;
  fc2160nrb50[4] = 2.156310000000e+09;
  fc2160nrb50[5] = 2.156490000000e+09;
  fc2160nrb50[6] = 2.156670000000e+09;
  fc2160nrb50[7] = 2.156850000000e+09;
  fc2160nrb50[8] = 2.157030000000e+09;
  fc2160nrb50[9] = 2.157210000000e+09;
  fc2160nrb50[10] = 2.157390000000e+09;
  fc2160nrb50[11] = 2.157570000000e+09;
  fc2160nrb50[12] = 2.157750000000e+09;
  fc2160nrb50[13] = 2.157930000000e+09;
  fc2160nrb50[14] = 2.158110000000e+09;
  fc2160nrb50[15] = 2.158290000000e+09;
  fc2160nrb50[16] = 2.158470000000e+09;
  fc2160nrb50[17] = 2.158650000000e+09;
  fc2160nrb50[18] = 2.158830000000e+09;
  fc2160nrb50[19] = 2.159010000000e+09;
  fc2160nrb50[20] = 2.159190000000e+09;
  fc2160nrb50[21] = 2.159370000000e+09;
  fc2160nrb50[22] = 2.159550000000e+09;
  fc2160nrb50[23] = 2.159730000000e+09;
  fc2160nrb50[24] = 2.159910000000e+09;
  fc2160nrb50[25] = 2.160090000000e+09;
  fc2160nrb50[26] = 2.160270000000e+09;
  fc2160nrb50[27] = 2.160450000000e+09;
  fc2160nrb50[28] = 2.160630000000e+09;
  fc2160nrb50[29] = 2.160810000000e+09;
  fc2160nrb50[30] = 2.160990000000e+09;
  fc2160nrb50[31] = 2.161170000000e+09;
  fc2160nrb50[32] = 2.161350000000e+09;
  fc2160nrb50[33] = 2.161530000000e+09;
  fc2160nrb50[34] = 2.161710000000e+09;
  fc2160nrb50[35] = 2.161890000000e+09;
  fc2160nrb50[36] = 2.162070000000e+09;
  fc2160nrb50[37] = 2.162250000000e+09;
  fc2160nrb50[38] = 2.162430000000e+09;
  fc2160nrb50[39] = 2.162610000000e+09;
  fc2160nrb50[40] = 2.162790000000e+09;
  fc2160nrb50[41] = 2.162970000000e+09;
  fc2160nrb50[42] = 2.163150000000e+09;
  fc2160nrb50[43] = 2.163330000000e+09;
  fc2160nrb50[44] = 2.163510000000e+09;
  fc2160nrb50[45] = 2.163690000000e+09;
  fc2160nrb50[46] = 2.163870000000e+09;
  fc2160nrb50[47] = 2.164050000000e+09;
  fc2160nrb50[48] = 2.164230000000e+09;
  fc2160nrb50[49] = 2.164410000000e+09;
  AddTestCase (new LteSpectrumModelTestCase ("fc2160nrb50", 500, 50, fc2160nrb50), TestCase::QUICK);

  std::vector<double> fc2160nrb75 (75);
  fc2160nrb75[0] = 2.153340000000e+09;
  fc2160nrb75[1] = 2.153520000000e+09;
  fc2160nrb75[2] = 2.153700000000e+09;
  fc2160nrb75[3] = 2.153880000000e+09;
  fc2160nrb75[4] = 2.154060000000e+09;
  fc2160nrb75[5] = 2.154240000000e+09;
  fc2160nrb75[6] = 2.154420000000e+09;
  fc2160nrb75[7] = 2.154600000000e+09;
  fc2160nrb75[8] = 2.154780000000e+09;
  fc2160nrb75[9] = 2.154960000000e+09;
  fc2160nrb75[10] = 2.155140000000e+09;
  fc2160nrb75[11] = 2.155320000000e+09;
  fc2160nrb75[12] = 2.155500000000e+09;
  fc2160nrb75[13] = 2.155680000000e+09;
  fc2160nrb75[14] = 2.155860000000e+09;
  fc2160nrb75[15] = 2.156040000000e+09;
  fc2160nrb75[16] = 2.156220000000e+09;
  fc2160nrb75[17] = 2.156400000000e+09;
  fc2160nrb75[18] = 2.156580000000e+09;
  fc2160nrb75[19] = 2.156760000000e+09;
  fc2160nrb75[20] = 2.156940000000e+09;
  fc2160nrb75[21] = 2.157120000000e+09;
  fc2160nrb75[22] = 2.157300000000e+09;
  fc2160nrb75[23] = 2.157480000000e+09;
  fc2160nrb75[24] = 2.157660000000e+09;
  fc2160nrb75[25] = 2.157840000000e+09;
  fc2160nrb75[26] = 2.158020000000e+09;
  fc2160nrb75[27] = 2.158200000000e+09;
  fc2160nrb75[28] = 2.158380000000e+09;
  fc2160nrb75[29] = 2.158560000000e+09;
  fc2160nrb75[30] = 2.158740000000e+09;
  fc2160nrb75[31] = 2.158920000000e+09;
  fc2160nrb75[32] = 2.159100000000e+09;
  fc2160nrb75[33] = 2.159280000000e+09;
  fc2160nrb75[34] = 2.159460000000e+09;
  fc2160nrb75[35] = 2.159640000000e+09;
  fc2160nrb75[36] = 2.159820000000e+09;
  fc2160nrb75[37] = 2.160000000000e+09;
  fc2160nrb75[38] = 2.160180000000e+09;
  fc2160nrb75[39] = 2.160360000000e+09;
  fc2160nrb75[40] = 2.160540000000e+09;
  fc2160nrb75[41] = 2.160720000000e+09;
  fc2160nrb75[42] = 2.160900000000e+09;
  fc2160nrb75[43] = 2.161080000000e+09;
  fc2160nrb75[44] = 2.161260000000e+09;
  fc2160nrb75[45] = 2.161440000000e+09;
  fc2160nrb75[46] = 2.161620000000e+09;
  fc2160nrb75[47] = 2.161800000000e+09;
  fc2160nrb75[48] = 2.161980000000e+09;
  fc2160nrb75[49] = 2.162160000000e+09;
  fc2160nrb75[50] = 2.162340000000e+09;
  fc2160nrb75[51] = 2.162520000000e+09;
  fc2160nrb75[52] = 2.162700000000e+09;
  fc2160nrb75[53] = 2.162880000000e+09;
  fc2160nrb75[54] = 2.163060000000e+09;
  fc2160nrb75[55] = 2.163240000000e+09;
  fc2160nrb75[56] = 2.163420000000e+09;
  fc2160nrb75[57] = 2.163600000000e+09;
  fc2160nrb75[58] = 2.163780000000e+09;
  fc2160nrb75[59] = 2.163960000000e+09;
  fc2160nrb75[60] = 2.164140000000e+09;
  fc2160nrb75[61] = 2.164320000000e+09;
  fc2160nrb75[62] = 2.164500000000e+09;
  fc2160nrb75[63] = 2.164680000000e+09;
  fc2160nrb75[64] = 2.164860000000e+09;
  fc2160nrb75[65] = 2.165040000000e+09;
  fc2160nrb75[66] = 2.165220000000e+09;
  fc2160nrb75[67] = 2.165400000000e+09;
  fc2160nrb75[68] = 2.165580000000e+09;
  fc2160nrb75[69] = 2.165760000000e+09;
  fc2160nrb75[70] = 2.165940000000e+09;
  fc2160nrb75[71] = 2.166120000000e+09;
  fc2160nrb75[72] = 2.166300000000e+09;
  fc2160nrb75[73] = 2.166480000000e+09;
  fc2160nrb75[74] = 2.166660000000e+09;
  AddTestCase (new LteSpectrumModelTestCase ("fc2160nrb75", 500, 75, fc2160nrb75), TestCase::QUICK);

  std::vector<double> fc2160nrb100 (100);
  fc2160nrb100[0] = 2.151090000000e+09;
  fc2160nrb100[1] = 2.151270000000e+09;
  fc2160nrb100[2] = 2.151450000000e+09;
  fc2160nrb100[3] = 2.151630000000e+09;
  fc2160nrb100[4] = 2.151810000000e+09;
  fc2160nrb100[5] = 2.151990000000e+09;
  fc2160nrb100[6] = 2.152170000000e+09;
  fc2160nrb100[7] = 2.152350000000e+09;
  fc2160nrb100[8] = 2.152530000000e+09;
  fc2160nrb100[9] = 2.152710000000e+09;
  fc2160nrb100[10] = 2.152890000000e+09;
  fc2160nrb100[11] = 2.153070000000e+09;
  fc2160nrb100[12] = 2.153250000000e+09;
  fc2160nrb100[13] = 2.153430000000e+09;
  fc2160nrb100[14] = 2.153610000000e+09;
  fc2160nrb100[15] = 2.153790000000e+09;
  fc2160nrb100[16] = 2.153970000000e+09;
  fc2160nrb100[17] = 2.154150000000e+09;
  fc2160nrb100[18] = 2.154330000000e+09;
  fc2160nrb100[19] = 2.154510000000e+09;
  fc2160nrb100[20] = 2.154690000000e+09;
  fc2160nrb100[21] = 2.154870000000e+09;
  fc2160nrb100[22] = 2.155050000000e+09;
  fc2160nrb100[23] = 2.155230000000e+09;
  fc2160nrb100[24] = 2.155410000000e+09;
  fc2160nrb100[25] = 2.155590000000e+09;
  fc2160nrb100[26] = 2.155770000000e+09;
  fc2160nrb100[27] = 2.155950000000e+09;
  fc2160nrb100[28] = 2.156130000000e+09;
  fc2160nrb100[29] = 2.156310000000e+09;
  fc2160nrb100[30] = 2.156490000000e+09;
  fc2160nrb100[31] = 2.156670000000e+09;
  fc2160nrb100[32] = 2.156850000000e+09;
  fc2160nrb100[33] = 2.157030000000e+09;
  fc2160nrb100[34] = 2.157210000000e+09;
  fc2160nrb100[35] = 2.157390000000e+09;
  fc2160nrb100[36] = 2.157570000000e+09;
  fc2160nrb100[37] = 2.157750000000e+09;
  fc2160nrb100[38] = 2.157930000000e+09;
  fc2160nrb100[39] = 2.158110000000e+09;
  fc2160nrb100[40] = 2.158290000000e+09;
  fc2160nrb100[41] = 2.158470000000e+09;
  fc2160nrb100[42] = 2.158650000000e+09;
  fc2160nrb100[43] = 2.158830000000e+09;
  fc2160nrb100[44] = 2.159010000000e+09;
  fc2160nrb100[45] = 2.159190000000e+09;
  fc2160nrb100[46] = 2.159370000000e+09;
  fc2160nrb100[47] = 2.159550000000e+09;
  fc2160nrb100[48] = 2.159730000000e+09;
  fc2160nrb100[49] = 2.159910000000e+09;
  fc2160nrb100[50] = 2.160090000000e+09;
  fc2160nrb100[51] = 2.160270000000e+09;
  fc2160nrb100[52] = 2.160450000000e+09;
  fc2160nrb100[53] = 2.160630000000e+09;
  fc2160nrb100[54] = 2.160810000000e+09;
  fc2160nrb100[55] = 2.160990000000e+09;
  fc2160nrb100[56] = 2.161170000000e+09;
  fc2160nrb100[57] = 2.161350000000e+09;
  fc2160nrb100[58] = 2.161530000000e+09;
  fc2160nrb100[59] = 2.161710000000e+09;
  fc2160nrb100[60] = 2.161890000000e+09;
  fc2160nrb100[61] = 2.162070000000e+09;
  fc2160nrb100[62] = 2.162250000000e+09;
  fc2160nrb100[63] = 2.162430000000e+09;
  fc2160nrb100[64] = 2.162610000000e+09;
  fc2160nrb100[65] = 2.162790000000e+09;
  fc2160nrb100[66] = 2.162970000000e+09;
  fc2160nrb100[67] = 2.163150000000e+09;
  fc2160nrb100[68] = 2.163330000000e+09;
  fc2160nrb100[69] = 2.163510000000e+09;
  fc2160nrb100[70] = 2.163690000000e+09;
  fc2160nrb100[71] = 2.163870000000e+09;
  fc2160nrb100[72] = 2.164050000000e+09;
  fc2160nrb100[73] = 2.164230000000e+09;
  fc2160nrb100[74] = 2.164410000000e+09;
  fc2160nrb100[75] = 2.164590000000e+09;
  fc2160nrb100[76] = 2.164770000000e+09;
  fc2160nrb100[77] = 2.164950000000e+09;
  fc2160nrb100[78] = 2.165130000000e+09;
  fc2160nrb100[79] = 2.165310000000e+09;
  fc2160nrb100[80] = 2.165490000000e+09;
  fc2160nrb100[81] = 2.165670000000e+09;
  fc2160nrb100[82] = 2.165850000000e+09;
  fc2160nrb100[83] = 2.166030000000e+09;
  fc2160nrb100[84] = 2.166210000000e+09;
  fc2160nrb100[85] = 2.166390000000e+09;
  fc2160nrb100[86] = 2.166570000000e+09;
  fc2160nrb100[87] = 2.166750000000e+09;
  fc2160nrb100[88] = 2.166930000000e+09;
  fc2160nrb100[89] = 2.167110000000e+09;
  fc2160nrb100[90] = 2.167290000000e+09;
  fc2160nrb100[91] = 2.167470000000e+09;
  fc2160nrb100[92] = 2.167650000000e+09;
  fc2160nrb100[93] = 2.167830000000e+09;
  fc2160nrb100[94] = 2.168010000000e+09;
  fc2160nrb100[95] = 2.168190000000e+09;
  fc2160nrb100[96] = 2.168370000000e+09;
  fc2160nrb100[97] = 2.168550000000e+09;
  fc2160nrb100[98] = 2.168730000000e+09;
  fc2160nrb100[99] = 2.168910000000e+09;
  AddTestCase (new LteSpectrumModelTestCase ("fc2160nrb100", 500, 100, fc2160nrb100), TestCase::QUICK);

  std::vector<double> fc1730nrb6 (6);
  fc1730nrb6[0] = 1.729550000000e+09;
  fc1730nrb6[1] = 1.729730000000e+09;
  fc1730nrb6[2] = 1.729910000000e+09;
  fc1730nrb6[3] = 1.730090000000e+09;
  fc1730nrb6[4] = 1.730270000000e+09;
  fc1730nrb6[5] = 1.730450000000e+09;
  AddTestCase (new LteSpectrumModelTestCase ("fc1730nrb6", 19400, 6, fc1730nrb6), TestCase::QUICK);

  std::vector<double> fc1730nrb15 (15);
  fc1730nrb15[0] = 1.728740000000e+09;
  fc1730nrb15[1] = 1.728920000000e+09;
  fc1730nrb15[2] = 1.729100000000e+09;
  fc1730nrb15[3] = 1.729280000000e+09;
  fc1730nrb15[4] = 1.729460000000e+09;
  fc1730nrb15[5] = 1.729640000000e+09;
  fc1730nrb15[6] = 1.729820000000e+09;
  fc1730nrb15[7] = 1.730000000000e+09;
  fc1730nrb15[8] = 1.730180000000e+09;
  fc1730nrb15[9] = 1.730360000000e+09;
  fc1730nrb15[10] = 1.730540000000e+09;
  fc1730nrb15[11] = 1.730720000000e+09;
  fc1730nrb15[12] = 1.730900000000e+09;
  fc1730nrb15[13] = 1.731080000000e+09;
  fc1730nrb15[14] = 1.731260000000e+09;
  AddTestCase (new LteSpectrumModelTestCase ("fc1730nrb15", 19400, 15, fc1730nrb15), TestCase::QUICK);

  std::vector<double> fc1730nrb25 (25);
  fc1730nrb25[0] = 1.727840000000e+09;
  fc1730nrb25[1] = 1.728020000000e+09;
  fc1730nrb25[2] = 1.728200000000e+09;
  fc1730nrb25[3] = 1.728380000000e+09;
  fc1730nrb25[4] = 1.728560000000e+09;
  fc1730nrb25[5] = 1.728740000000e+09;
  fc1730nrb25[6] = 1.728920000000e+09;
  fc1730nrb25[7] = 1.729100000000e+09;
  fc1730nrb25[8] = 1.729280000000e+09;
  fc1730nrb25[9] = 1.729460000000e+09;
  fc1730nrb25[10] = 1.729640000000e+09;
  fc1730nrb25[11] = 1.729820000000e+09;
  fc1730nrb25[12] = 1.730000000000e+09;
  fc1730nrb25[13] = 1.730180000000e+09;
  fc1730nrb25[14] = 1.730360000000e+09;
  fc1730nrb25[15] = 1.730540000000e+09;
  fc1730nrb25[16] = 1.730720000000e+09;
  fc1730nrb25[17] = 1.730900000000e+09;
  fc1730nrb25[18] = 1.731080000000e+09;
  fc1730nrb25[19] = 1.731260000000e+09;
  fc1730nrb25[20] = 1.731440000000e+09;
  fc1730nrb25[21] = 1.731620000000e+09;
  fc1730nrb25[22] = 1.731800000000e+09;
  fc1730nrb25[23] = 1.731980000000e+09;
  fc1730nrb25[24] = 1.732160000000e+09;
  AddTestCase (new LteSpectrumModelTestCase ("fc1730nrb25", 19400, 25, fc1730nrb25), TestCase::QUICK);

  std::vector<double> fc1730nrb50 (50);
  fc1730nrb50[0] = 1.725590000000e+09;
  fc1730nrb50[1] = 1.725770000000e+09;
  fc1730nrb50[2] = 1.725950000000e+09;
  fc1730nrb50[3] = 1.726130000000e+09;
  fc1730nrb50[4] = 1.726310000000e+09;
  fc1730nrb50[5] = 1.726490000000e+09;
  fc1730nrb50[6] = 1.726670000000e+09;
  fc1730nrb50[7] = 1.726850000000e+09;
  fc1730nrb50[8] = 1.727030000000e+09;
  fc1730nrb50[9] = 1.727210000000e+09;
  fc1730nrb50[10] = 1.727390000000e+09;
  fc1730nrb50[11] = 1.727570000000e+09;
  fc1730nrb50[12] = 1.727750000000e+09;
  fc1730nrb50[13] = 1.727930000000e+09;
  fc1730nrb50[14] = 1.728110000000e+09;
  fc1730nrb50[15] = 1.728290000000e+09;
  fc1730nrb50[16] = 1.728470000000e+09;
  fc1730nrb50[17] = 1.728650000000e+09;
  fc1730nrb50[18] = 1.728830000000e+09;
  fc1730nrb50[19] = 1.729010000000e+09;
  fc1730nrb50[20] = 1.729190000000e+09;
  fc1730nrb50[21] = 1.729370000000e+09;
  fc1730nrb50[22] = 1.729550000000e+09;
  fc1730nrb50[23] = 1.729730000000e+09;
  fc1730nrb50[24] = 1.729910000000e+09;
  fc1730nrb50[25] = 1.730090000000e+09;
  fc1730nrb50[26] = 1.730270000000e+09;
  fc1730nrb50[27] = 1.730450000000e+09;
  fc1730nrb50[28] = 1.730630000000e+09;
  fc1730nrb50[29] = 1.730810000000e+09;
  fc1730nrb50[30] = 1.730990000000e+09;
  fc1730nrb50[31] = 1.731170000000e+09;
  fc1730nrb50[32] = 1.731350000000e+09;
  fc1730nrb50[33] = 1.731530000000e+09;
  fc1730nrb50[34] = 1.731710000000e+09;
  fc1730nrb50[35] = 1.731890000000e+09;
  fc1730nrb50[36] = 1.732070000000e+09;
  fc1730nrb50[37] = 1.732250000000e+09;
  fc1730nrb50[38] = 1.732430000000e+09;
  fc1730nrb50[39] = 1.732610000000e+09;
  fc1730nrb50[40] = 1.732790000000e+09;
  fc1730nrb50[41] = 1.732970000000e+09;
  fc1730nrb50[42] = 1.733150000000e+09;
  fc1730nrb50[43] = 1.733330000000e+09;
  fc1730nrb50[44] = 1.733510000000e+09;
  fc1730nrb50[45] = 1.733690000000e+09;
  fc1730nrb50[46] = 1.733870000000e+09;
  fc1730nrb50[47] = 1.734050000000e+09;
  fc1730nrb50[48] = 1.734230000000e+09;
  fc1730nrb50[49] = 1.734410000000e+09;
  AddTestCase (new LteSpectrumModelTestCase ("fc1730nrb50", 19400, 50, fc1730nrb50), TestCase::QUICK);

  std::vector<double> fc1730nrb75 (75);
  fc1730nrb75[0] = 1.723340000000e+09;
  fc1730nrb75[1] = 1.723520000000e+09;
  fc1730nrb75[2] = 1.723700000000e+09;
  fc1730nrb75[3] = 1.723880000000e+09;
  fc1730nrb75[4] = 1.724060000000e+09;
  fc1730nrb75[5] = 1.724240000000e+09;
  fc1730nrb75[6] = 1.724420000000e+09;
  fc1730nrb75[7] = 1.724600000000e+09;
  fc1730nrb75[8] = 1.724780000000e+09;
  fc1730nrb75[9] = 1.724960000000e+09;
  fc1730nrb75[10] = 1.725140000000e+09;
  fc1730nrb75[11] = 1.725320000000e+09;
  fc1730nrb75[12] = 1.725500000000e+09;
  fc1730nrb75[13] = 1.725680000000e+09;
  fc1730nrb75[14] = 1.725860000000e+09;
  fc1730nrb75[15] = 1.726040000000e+09;
  fc1730nrb75[16] = 1.726220000000e+09;
  fc1730nrb75[17] = 1.726400000000e+09;
  fc1730nrb75[18] = 1.726580000000e+09;
  fc1730nrb75[19] = 1.726760000000e+09;
  fc1730nrb75[20] = 1.726940000000e+09;
  fc1730nrb75[21] = 1.727120000000e+09;
  fc1730nrb75[22] = 1.727300000000e+09;
  fc1730nrb75[23] = 1.727480000000e+09;
  fc1730nrb75[24] = 1.727660000000e+09;
  fc1730nrb75[25] = 1.727840000000e+09;
  fc1730nrb75[26] = 1.728020000000e+09;
  fc1730nrb75[27] = 1.728200000000e+09;
  fc1730nrb75[28] = 1.728380000000e+09;
  fc1730nrb75[29] = 1.728560000000e+09;
  fc1730nrb75[30] = 1.728740000000e+09;
  fc1730nrb75[31] = 1.728920000000e+09;
  fc1730nrb75[32] = 1.729100000000e+09;
  fc1730nrb75[33] = 1.729280000000e+09;
  fc1730nrb75[34] = 1.729460000000e+09;
  fc1730nrb75[35] = 1.729640000000e+09;
  fc1730nrb75[36] = 1.729820000000e+09;
  fc1730nrb75[37] = 1.730000000000e+09;
  fc1730nrb75[38] = 1.730180000000e+09;
  fc1730nrb75[39] = 1.730360000000e+09;
  fc1730nrb75[40] = 1.730540000000e+09;
  fc1730nrb75[41] = 1.730720000000e+09;
  fc1730nrb75[42] = 1.730900000000e+09;
  fc1730nrb75[43] = 1.731080000000e+09;
  fc1730nrb75[44] = 1.731260000000e+09;
  fc1730nrb75[45] = 1.731440000000e+09;
  fc1730nrb75[46] = 1.731620000000e+09;
  fc1730nrb75[47] = 1.731800000000e+09;
  fc1730nrb75[48] = 1.731980000000e+09;
  fc1730nrb75[49] = 1.732160000000e+09;
  fc1730nrb75[50] = 1.732340000000e+09;
  fc1730nrb75[51] = 1.732520000000e+09;
  fc1730nrb75[52] = 1.732700000000e+09;
  fc1730nrb75[53] = 1.732880000000e+09;
  fc1730nrb75[54] = 1.733060000000e+09;
  fc1730nrb75[55] = 1.733240000000e+09;
  fc1730nrb75[56] = 1.733420000000e+09;
  fc1730nrb75[57] = 1.733600000000e+09;
  fc1730nrb75[58] = 1.733780000000e+09;
  fc1730nrb75[59] = 1.733960000000e+09;
  fc1730nrb75[60] = 1.734140000000e+09;
  fc1730nrb75[61] = 1.734320000000e+09;
  fc1730nrb75[62] = 1.734500000000e+09;
  fc1730nrb75[63] = 1.734680000000e+09;
  fc1730nrb75[64] = 1.734860000000e+09;
  fc1730nrb75[65] = 1.735040000000e+09;
  fc1730nrb75[66] = 1.735220000000e+09;
  fc1730nrb75[67] = 1.735400000000e+09;
  fc1730nrb75[68] = 1.735580000000e+09;
  fc1730nrb75[69] = 1.735760000000e+09;
  fc1730nrb75[70] = 1.735940000000e+09;
  fc1730nrb75[71] = 1.736120000000e+09;
  fc1730nrb75[72] = 1.736300000000e+09;
  fc1730nrb75[73] = 1.736480000000e+09;
  fc1730nrb75[74] = 1.736660000000e+09;
  AddTestCase (new LteSpectrumModelTestCase ("fc1730nrb75", 19400, 75, fc1730nrb75), TestCase::QUICK);

  std::vector<double> fc1730nrb100 (100);
  fc1730nrb100[0] = 1.721090000000e+09;
  fc1730nrb100[1] = 1.721270000000e+09;
  fc1730nrb100[2] = 1.721450000000e+09;
  fc1730nrb100[3] = 1.721630000000e+09;
  fc1730nrb100[4] = 1.721810000000e+09;
  fc1730nrb100[5] = 1.721990000000e+09;
  fc1730nrb100[6] = 1.722170000000e+09;
  fc1730nrb100[7] = 1.722350000000e+09;
  fc1730nrb100[8] = 1.722530000000e+09;
  fc1730nrb100[9] = 1.722710000000e+09;
  fc1730nrb100[10] = 1.722890000000e+09;
  fc1730nrb100[11] = 1.723070000000e+09;
  fc1730nrb100[12] = 1.723250000000e+09;
  fc1730nrb100[13] = 1.723430000000e+09;
  fc1730nrb100[14] = 1.723610000000e+09;
  fc1730nrb100[15] = 1.723790000000e+09;
  fc1730nrb100[16] = 1.723970000000e+09;
  fc1730nrb100[17] = 1.724150000000e+09;
  fc1730nrb100[18] = 1.724330000000e+09;
  fc1730nrb100[19] = 1.724510000000e+09;
  fc1730nrb100[20] = 1.724690000000e+09;
  fc1730nrb100[21] = 1.724870000000e+09;
  fc1730nrb100[22] = 1.725050000000e+09;
  fc1730nrb100[23] = 1.725230000000e+09;
  fc1730nrb100[24] = 1.725410000000e+09;
  fc1730nrb100[25] = 1.725590000000e+09;
  fc1730nrb100[26] = 1.725770000000e+09;
  fc1730nrb100[27] = 1.725950000000e+09;
  fc1730nrb100[28] = 1.726130000000e+09;
  fc1730nrb100[29] = 1.726310000000e+09;
  fc1730nrb100[30] = 1.726490000000e+09;
  fc1730nrb100[31] = 1.726670000000e+09;
  fc1730nrb100[32] = 1.726850000000e+09;
  fc1730nrb100[33] = 1.727030000000e+09;
  fc1730nrb100[34] = 1.727210000000e+09;
  fc1730nrb100[35] = 1.727390000000e+09;
  fc1730nrb100[36] = 1.727570000000e+09;
  fc1730nrb100[37] = 1.727750000000e+09;
  fc1730nrb100[38] = 1.727930000000e+09;
  fc1730nrb100[39] = 1.728110000000e+09;
  fc1730nrb100[40] = 1.728290000000e+09;
  fc1730nrb100[41] = 1.728470000000e+09;
  fc1730nrb100[42] = 1.728650000000e+09;
  fc1730nrb100[43] = 1.728830000000e+09;
  fc1730nrb100[44] = 1.729010000000e+09;
  fc1730nrb100[45] = 1.729190000000e+09;
  fc1730nrb100[46] = 1.729370000000e+09;
  fc1730nrb100[47] = 1.729550000000e+09;
  fc1730nrb100[48] = 1.729730000000e+09;
  fc1730nrb100[49] = 1.729910000000e+09;
  fc1730nrb100[50] = 1.730090000000e+09;
  fc1730nrb100[51] = 1.730270000000e+09;
  fc1730nrb100[52] = 1.730450000000e+09;
  fc1730nrb100[53] = 1.730630000000e+09;
  fc1730nrb100[54] = 1.730810000000e+09;
  fc1730nrb100[55] = 1.730990000000e+09;
  fc1730nrb100[56] = 1.731170000000e+09;
  fc1730nrb100[57] = 1.731350000000e+09;
  fc1730nrb100[58] = 1.731530000000e+09;
  fc1730nrb100[59] = 1.731710000000e+09;
  fc1730nrb100[60] = 1.731890000000e+09;
  fc1730nrb100[61] = 1.732070000000e+09;
  fc1730nrb100[62] = 1.732250000000e+09;
  fc1730nrb100[63] = 1.732430000000e+09;
  fc1730nrb100[64] = 1.732610000000e+09;
  fc1730nrb100[65] = 1.732790000000e+09;
  fc1730nrb100[66] = 1.732970000000e+09;
  fc1730nrb100[67] = 1.733150000000e+09;
  fc1730nrb100[68] = 1.733330000000e+09;
  fc1730nrb100[69] = 1.733510000000e+09;
  fc1730nrb100[70] = 1.733690000000e+09;
  fc1730nrb100[71] = 1.733870000000e+09;
  fc1730nrb100[72] = 1.734050000000e+09;
  fc1730nrb100[73] = 1.734230000000e+09;
  fc1730nrb100[74] = 1.734410000000e+09;
  fc1730nrb100[75] = 1.734590000000e+09;
  fc1730nrb100[76] = 1.734770000000e+09;
  fc1730nrb100[77] = 1.734950000000e+09;
  fc1730nrb100[78] = 1.735130000000e+09;
  fc1730nrb100[79] = 1.735310000000e+09;
  fc1730nrb100[80] = 1.735490000000e+09;
  fc1730nrb100[81] = 1.735670000000e+09;
  fc1730nrb100[82] = 1.735850000000e+09;
  fc1730nrb100[83] = 1.736030000000e+09;
  fc1730nrb100[84] = 1.736210000000e+09;
  fc1730nrb100[85] = 1.736390000000e+09;
  fc1730nrb100[86] = 1.736570000000e+09;
  fc1730nrb100[87] = 1.736750000000e+09;
  fc1730nrb100[88] = 1.736930000000e+09;
  fc1730nrb100[89] = 1.737110000000e+09;
  fc1730nrb100[90] = 1.737290000000e+09;
  fc1730nrb100[91] = 1.737470000000e+09;
  fc1730nrb100[92] = 1.737650000000e+09;
  fc1730nrb100[93] = 1.737830000000e+09;
  fc1730nrb100[94] = 1.738010000000e+09;
  fc1730nrb100[95] = 1.738190000000e+09;
  fc1730nrb100[96] = 1.738370000000e+09;
  fc1730nrb100[97] = 1.738550000000e+09;
  fc1730nrb100[98] = 1.738730000000e+09;
  fc1730nrb100[99] = 1.738910000000e+09;
  AddTestCase (new LteSpectrumModelTestCase ("fc1730nrb100", 19400, 100, fc1730nrb100), TestCase::QUICK);


  SpectrumValue nfdB0earfcn500nrb6 (LteSpectrumValueHelper::GetSpectrumModel (500, 6));
  nfdB0earfcn500nrb6[0] = 4.002000000000e-21;
  nfdB0earfcn500nrb6[1] = 4.002000000000e-21;
  nfdB0earfcn500nrb6[2] = 4.002000000000e-21;
  nfdB0earfcn500nrb6[3] = 4.002000000000e-21;
  nfdB0earfcn500nrb6[4] = 4.002000000000e-21;
  nfdB0earfcn500nrb6[5] = 4.002000000000e-21;
  AddTestCase (new LteNoisePsdTestCase ("nfdB0earfcn500nrb6", 500, 6, 0.000000, nfdB0earfcn500nrb6), TestCase::QUICK);

  SpectrumValue nfdB5earfcn500nrb6 (LteSpectrumValueHelper::GetSpectrumModel (500, 6));
  nfdB5earfcn500nrb6[0] = 1.265543519599e-20;
  nfdB5earfcn500nrb6[1] = 1.265543519599e-20;
  nfdB5earfcn500nrb6[2] = 1.265543519599e-20;
  nfdB5earfcn500nrb6[3] = 1.265543519599e-20;
  nfdB5earfcn500nrb6[4] = 1.265543519599e-20;
  nfdB5earfcn500nrb6[5] = 1.265543519599e-20;
  AddTestCase (new LteNoisePsdTestCase ("nfdB5earfcn500nrb6", 500, 6, 5.000000, nfdB5earfcn500nrb6), TestCase::QUICK);

  SpectrumValue nfdB10earfcn500nrb6 (LteSpectrumValueHelper::GetSpectrumModel (500, 6));
  nfdB10earfcn500nrb6[0] = 4.002000000000e-20;
  nfdB10earfcn500nrb6[1] = 4.002000000000e-20;
  nfdB10earfcn500nrb6[2] = 4.002000000000e-20;
  nfdB10earfcn500nrb6[3] = 4.002000000000e-20;
  nfdB10earfcn500nrb6[4] = 4.002000000000e-20;
  nfdB10earfcn500nrb6[5] = 4.002000000000e-20;
  AddTestCase (new LteNoisePsdTestCase ("nfdB10earfcn500nrb6", 500, 6, 10.000000, nfdB10earfcn500nrb6), TestCase::QUICK);

  SpectrumValue nfdB0earfcn500nrb15 (LteSpectrumValueHelper::GetSpectrumModel (500, 15));
  nfdB0earfcn500nrb15[0] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[1] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[2] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[3] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[4] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[5] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[6] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[7] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[8] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[9] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[10] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[11] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[12] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[13] = 4.002000000000e-21;
  nfdB0earfcn500nrb15[14] = 4.002000000000e-21;
  AddTestCase (new LteNoisePsdTestCase ("nfdB0earfcn500nrb15", 500, 15, 0.000000, nfdB0earfcn500nrb15), TestCase::QUICK);

  SpectrumValue nfdB5earfcn500nrb15 (LteSpectrumValueHelper::GetSpectrumModel (500, 15));
  nfdB5earfcn500nrb15[0] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[1] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[2] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[3] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[4] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[5] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[6] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[7] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[8] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[9] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[10] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[11] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[12] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[13] = 1.265543519599e-20;
  nfdB5earfcn500nrb15[14] = 1.265543519599e-20;
  AddTestCase (new LteNoisePsdTestCase ("nfdB5earfcn500nrb15", 500, 15, 5.000000, nfdB5earfcn500nrb15), TestCase::QUICK);

  SpectrumValue nfdB10earfcn500nrb15 (LteSpectrumValueHelper::GetSpectrumModel (500, 15));
  nfdB10earfcn500nrb15[0] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[1] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[2] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[3] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[4] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[5] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[6] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[7] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[8] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[9] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[10] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[11] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[12] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[13] = 4.002000000000e-20;
  nfdB10earfcn500nrb15[14] = 4.002000000000e-20;


  std::vector<int> activeRbs_txpowdB10nrb6run1earfcn500 (2);
  activeRbs_txpowdB10nrb6run1earfcn500[0] = 0.000000000000e+00;
  activeRbs_txpowdB10nrb6run1earfcn500[1] = 1.000000000000e+00;
  SpectrumValue spectrumValue_txpowdB10nrb6run1earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 6));
  spectrumValue_txpowdB10nrb6run1earfcn500[0] = 9.259259259259e-09;
  spectrumValue_txpowdB10nrb6run1earfcn500[1] = 9.259259259259e-09;
  spectrumValue_txpowdB10nrb6run1earfcn500[2] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb6run1earfcn500[3] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb6run1earfcn500[4] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb6run1earfcn500[5] = 0.000000000000e+00;
  AddTestCase (new LteTxPsdTestCase ("txpowdB10nrb6run1earfcn500", 500, 6, 10.000000, activeRbs_txpowdB10nrb6run1earfcn500, spectrumValue_txpowdB10nrb6run1earfcn500), TestCase::QUICK);

  std::vector<int> activeRbs_txpowdB30nrb6run1earfcn500 (1);
  activeRbs_txpowdB30nrb6run1earfcn500[0] = 5.000000000000e+00;
  SpectrumValue spectrumValue_txpowdB30nrb6run1earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 6));
  spectrumValue_txpowdB30nrb6run1earfcn500[0] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb6run1earfcn500[1] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb6run1earfcn500[2] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb6run1earfcn500[3] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb6run1earfcn500[4] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb6run1earfcn500[5] = 9.259259259259e-07;
  AddTestCase (new LteTxPsdTestCase ("txpowdB30nrb6run1earfcn500", 500, 6, 30.000000, activeRbs_txpowdB30nrb6run1earfcn500, spectrumValue_txpowdB30nrb6run1earfcn500), TestCase::QUICK);

  std::vector<int> activeRbs_txpowdB10nrb25run1earfcn500 (8);
  activeRbs_txpowdB10nrb25run1earfcn500[0] = 1.000000000000e+00;
  activeRbs_txpowdB10nrb25run1earfcn500[1] = 2.000000000000e+00;
  activeRbs_txpowdB10nrb25run1earfcn500[2] = 5.000000000000e+00;
  activeRbs_txpowdB10nrb25run1earfcn500[3] = 1.000000000000e+01;
  activeRbs_txpowdB10nrb25run1earfcn500[4] = 1.500000000000e+01;
  activeRbs_txpowdB10nrb25run1earfcn500[5] = 1.700000000000e+01;
  activeRbs_txpowdB10nrb25run1earfcn500[6] = 1.800000000000e+01;
  activeRbs_txpowdB10nrb25run1earfcn500[7] = 2.400000000000e+01;
  SpectrumValue spectrumValue_txpowdB10nrb25run1earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 25));
  spectrumValue_txpowdB10nrb25run1earfcn500[0] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[1] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run1earfcn500[2] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run1earfcn500[3] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[4] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[5] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run1earfcn500[6] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[7] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[8] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[9] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[10] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run1earfcn500[11] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[12] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[13] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[14] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[15] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run1earfcn500[16] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[17] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run1earfcn500[18] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run1earfcn500[19] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[20] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[21] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[22] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[23] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run1earfcn500[24] = 2.222222222222e-09;
  AddTestCase (new LteTxPsdTestCase ("txpowdB10nrb25run1earfcn500", 500, 25, 10.000000, activeRbs_txpowdB10nrb25run1earfcn500, spectrumValue_txpowdB10nrb25run1earfcn500), TestCase::QUICK);

  std::vector<int> activeRbs_txpowdB30nrb25run1earfcn500 (13);
  activeRbs_txpowdB30nrb25run1earfcn500[0] = 0.000000000000e+00;
  activeRbs_txpowdB30nrb25run1earfcn500[1] = 1.000000000000e+00;
  activeRbs_txpowdB30nrb25run1earfcn500[2] = 2.000000000000e+00;
  activeRbs_txpowdB30nrb25run1earfcn500[3] = 5.000000000000e+00;
  activeRbs_txpowdB30nrb25run1earfcn500[4] = 6.000000000000e+00;
  activeRbs_txpowdB30nrb25run1earfcn500[5] = 1.000000000000e+01;
  activeRbs_txpowdB30nrb25run1earfcn500[6] = 1.100000000000e+01;
  activeRbs_txpowdB30nrb25run1earfcn500[7] = 1.300000000000e+01;
  activeRbs_txpowdB30nrb25run1earfcn500[8] = 1.400000000000e+01;
  activeRbs_txpowdB30nrb25run1earfcn500[9] = 1.500000000000e+01;
  activeRbs_txpowdB30nrb25run1earfcn500[10] = 1.600000000000e+01;
  activeRbs_txpowdB30nrb25run1earfcn500[11] = 1.800000000000e+01;
  activeRbs_txpowdB30nrb25run1earfcn500[12] = 2.400000000000e+01;
  SpectrumValue spectrumValue_txpowdB30nrb25run1earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 25));
  spectrumValue_txpowdB30nrb25run1earfcn500[0] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run1earfcn500[1] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run1earfcn500[2] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run1earfcn500[3] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run1earfcn500[4] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run1earfcn500[5] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run1earfcn500[6] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run1earfcn500[7] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run1earfcn500[8] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run1earfcn500[9] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run1earfcn500[10] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run1earfcn500[11] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run1earfcn500[12] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run1earfcn500[13] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run1earfcn500[14] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run1earfcn500[15] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run1earfcn500[16] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run1earfcn500[17] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run1earfcn500[18] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run1earfcn500[19] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run1earfcn500[20] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run1earfcn500[21] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run1earfcn500[22] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run1earfcn500[23] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run1earfcn500[24] = 2.222222222222e-07;
  AddTestCase (new LteTxPsdTestCase ("txpowdB30nrb25run1earfcn500", 500, 25, 30.000000, activeRbs_txpowdB30nrb25run1earfcn500, spectrumValue_txpowdB30nrb25run1earfcn500), TestCase::QUICK);

  std::vector<int> activeRbs_txpowdB10nrb100run1earfcn500 (60);
  activeRbs_txpowdB10nrb100run1earfcn500[0] = 0.000000000000e+00;
  activeRbs_txpowdB10nrb100run1earfcn500[1] = 1.000000000000e+00;
  activeRbs_txpowdB10nrb100run1earfcn500[2] = 4.000000000000e+00;
  activeRbs_txpowdB10nrb100run1earfcn500[3] = 5.000000000000e+00;
  activeRbs_txpowdB10nrb100run1earfcn500[4] = 6.000000000000e+00;
  activeRbs_txpowdB10nrb100run1earfcn500[5] = 8.000000000000e+00;
  activeRbs_txpowdB10nrb100run1earfcn500[6] = 1.000000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[7] = 1.200000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[8] = 1.300000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[9] = 1.400000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[10] = 1.500000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[11] = 1.600000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[12] = 1.800000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[13] = 2.000000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[14] = 2.400000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[15] = 2.500000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[16] = 2.600000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[17] = 2.700000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[18] = 2.800000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[19] = 2.900000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[20] = 3.000000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[21] = 3.100000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[22] = 3.300000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[23] = 3.500000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[24] = 3.600000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[25] = 3.700000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[26] = 3.900000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[27] = 4.000000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[28] = 4.500000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[29] = 4.600000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[30] = 4.800000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[31] = 4.900000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[32] = 5.000000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[33] = 5.300000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[34] = 5.700000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[35] = 5.800000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[36] = 6.000000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[37] = 6.300000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[38] = 6.600000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[39] = 6.800000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[40] = 6.900000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[41] = 7.100000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[42] = 7.300000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[43] = 7.700000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[44] = 7.800000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[45] = 7.900000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[46] = 8.200000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[47] = 8.500000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[48] = 8.600000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[49] = 8.700000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[50] = 8.800000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[51] = 9.000000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[52] = 9.100000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[53] = 9.200000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[54] = 9.300000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[55] = 9.400000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[56] = 9.500000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[57] = 9.600000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[58] = 9.700000000000e+01;
  activeRbs_txpowdB10nrb100run1earfcn500[59] = 9.800000000000e+01;
  SpectrumValue spectrumValue_txpowdB10nrb100run1earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 100));
  spectrumValue_txpowdB10nrb100run1earfcn500[0] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[1] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[2] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[3] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[4] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[5] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[6] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[7] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[8] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[9] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[10] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[11] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[12] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[13] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[14] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[15] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[16] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[17] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[18] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[19] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[20] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[21] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[22] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[23] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[24] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[25] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[26] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[27] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[28] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[29] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[30] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[31] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[32] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[33] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[34] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[35] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[36] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[37] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[38] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[39] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[40] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[41] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[42] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[43] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[44] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[45] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[46] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[47] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[48] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[49] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[50] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[51] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[52] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[53] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[54] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[55] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[56] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[57] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[58] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[59] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[60] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[61] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[62] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[63] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[64] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[65] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[66] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[67] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[68] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[69] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[70] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[71] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[72] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[73] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[74] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[75] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[76] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[77] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[78] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[79] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[80] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[81] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[82] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[83] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[84] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[85] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[86] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[87] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[88] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[89] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run1earfcn500[90] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[91] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[92] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[93] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[94] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[95] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[96] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[97] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[98] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run1earfcn500[99] = 0.000000000000e+00;
  AddTestCase (new LteTxPsdTestCase ("txpowdB10nrb100run1earfcn500", 500, 100, 10.000000, activeRbs_txpowdB10nrb100run1earfcn500, spectrumValue_txpowdB10nrb100run1earfcn500), TestCase::QUICK);

  std::vector<int> activeRbs_txpowdB30nrb100run1earfcn500 (39);
  activeRbs_txpowdB30nrb100run1earfcn500[0] = 0.000000000000e+00;
  activeRbs_txpowdB30nrb100run1earfcn500[1] = 5.000000000000e+00;
  activeRbs_txpowdB30nrb100run1earfcn500[2] = 6.000000000000e+00;
  activeRbs_txpowdB30nrb100run1earfcn500[3] = 8.000000000000e+00;
  activeRbs_txpowdB30nrb100run1earfcn500[4] = 1.000000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[5] = 1.100000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[6] = 1.300000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[7] = 1.700000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[8] = 1.800000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[9] = 1.900000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[10] = 2.200000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[11] = 2.400000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[12] = 2.500000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[13] = 2.800000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[14] = 2.900000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[15] = 3.300000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[16] = 3.600000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[17] = 3.800000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[18] = 4.000000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[19] = 4.500000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[20] = 4.600000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[21] = 4.800000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[22] = 5.200000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[23] = 5.300000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[24] = 5.400000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[25] = 5.600000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[26] = 6.600000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[27] = 6.700000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[28] = 7.000000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[29] = 7.300000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[30] = 7.700000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[31] = 8.200000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[32] = 8.800000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[33] = 9.000000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[34] = 9.100000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[35] = 9.200000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[36] = 9.600000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[37] = 9.800000000000e+01;
  activeRbs_txpowdB30nrb100run1earfcn500[38] = 9.900000000000e+01;
  SpectrumValue spectrumValue_txpowdB30nrb100run1earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 100));
  spectrumValue_txpowdB30nrb100run1earfcn500[0] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[1] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[2] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[3] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[4] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[5] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[6] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[7] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[8] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[9] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[10] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[11] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[12] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[13] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[14] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[15] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[16] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[17] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[18] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[19] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[20] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[21] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[22] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[23] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[24] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[25] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[26] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[27] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[28] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[29] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[30] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[31] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[32] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[33] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[34] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[35] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[36] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[37] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[38] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[39] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[40] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[41] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[42] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[43] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[44] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[45] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[46] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[47] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[48] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[49] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[50] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[51] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[52] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[53] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[54] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[55] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[56] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[57] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[58] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[59] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[60] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[61] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[62] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[63] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[64] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[65] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[66] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[67] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[68] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[69] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[70] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[71] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[72] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[73] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[74] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[75] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[76] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[77] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[78] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[79] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[80] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[81] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[82] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[83] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[84] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[85] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[86] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[87] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[88] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[89] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[90] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[91] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[92] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[93] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[94] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[95] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[96] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[97] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run1earfcn500[98] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run1earfcn500[99] = 5.555555555556e-08;
  AddTestCase (new LteTxPsdTestCase ("txpowdB30nrb100run1earfcn500", 500, 100, 30.000000, activeRbs_txpowdB30nrb100run1earfcn500, spectrumValue_txpowdB30nrb100run1earfcn500), TestCase::QUICK);

  std::vector<int> activeRbs_txpowdB10nrb6run2earfcn500 (3);
  activeRbs_txpowdB10nrb6run2earfcn500[0] = 1.000000000000e+00;
  activeRbs_txpowdB10nrb6run2earfcn500[1] = 4.000000000000e+00;
  activeRbs_txpowdB10nrb6run2earfcn500[2] = 5.000000000000e+00;
  SpectrumValue spectrumValue_txpowdB10nrb6run2earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 6));
  spectrumValue_txpowdB10nrb6run2earfcn500[0] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb6run2earfcn500[1] = 9.259259259259e-09;
  spectrumValue_txpowdB10nrb6run2earfcn500[2] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb6run2earfcn500[3] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb6run2earfcn500[4] = 9.259259259259e-09;
  spectrumValue_txpowdB10nrb6run2earfcn500[5] = 9.259259259259e-09;
  AddTestCase (new LteTxPsdTestCase ("txpowdB10nrb6run2earfcn500", 500, 6, 10.000000, activeRbs_txpowdB10nrb6run2earfcn500, spectrumValue_txpowdB10nrb6run2earfcn500), TestCase::QUICK);

  std::vector<int> activeRbs_txpowdB30nrb6run2earfcn500 (3);
  activeRbs_txpowdB30nrb6run2earfcn500[0] = 3.000000000000e+00;
  activeRbs_txpowdB30nrb6run2earfcn500[1] = 4.000000000000e+00;
  activeRbs_txpowdB30nrb6run2earfcn500[2] = 5.000000000000e+00;
  SpectrumValue spectrumValue_txpowdB30nrb6run2earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 6));
  spectrumValue_txpowdB30nrb6run2earfcn500[0] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb6run2earfcn500[1] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb6run2earfcn500[2] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb6run2earfcn500[3] = 9.259259259259e-07;
  spectrumValue_txpowdB30nrb6run2earfcn500[4] = 9.259259259259e-07;
  spectrumValue_txpowdB30nrb6run2earfcn500[5] = 9.259259259259e-07;
  AddTestCase (new LteTxPsdTestCase ("txpowdB30nrb6run2earfcn500", 500, 6, 30.000000, activeRbs_txpowdB30nrb6run2earfcn500, spectrumValue_txpowdB30nrb6run2earfcn500), TestCase::QUICK);

  std::vector<int> activeRbs_txpowdB10nrb25run2earfcn500 (15);
  activeRbs_txpowdB10nrb25run2earfcn500[0] = 0.000000000000e+00;
  activeRbs_txpowdB10nrb25run2earfcn500[1] = 1.000000000000e+00;
  activeRbs_txpowdB10nrb25run2earfcn500[2] = 2.000000000000e+00;
  activeRbs_txpowdB10nrb25run2earfcn500[3] = 4.000000000000e+00;
  activeRbs_txpowdB10nrb25run2earfcn500[4] = 5.000000000000e+00;
  activeRbs_txpowdB10nrb25run2earfcn500[5] = 6.000000000000e+00;
  activeRbs_txpowdB10nrb25run2earfcn500[6] = 8.000000000000e+00;
  activeRbs_txpowdB10nrb25run2earfcn500[7] = 9.000000000000e+00;
  activeRbs_txpowdB10nrb25run2earfcn500[8] = 1.100000000000e+01;
  activeRbs_txpowdB10nrb25run2earfcn500[9] = 1.200000000000e+01;
  activeRbs_txpowdB10nrb25run2earfcn500[10] = 1.400000000000e+01;
  activeRbs_txpowdB10nrb25run2earfcn500[11] = 1.600000000000e+01;
  activeRbs_txpowdB10nrb25run2earfcn500[12] = 1.800000000000e+01;
  activeRbs_txpowdB10nrb25run2earfcn500[13] = 1.900000000000e+01;
  activeRbs_txpowdB10nrb25run2earfcn500[14] = 2.400000000000e+01;
  SpectrumValue spectrumValue_txpowdB10nrb25run2earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 25));
  spectrumValue_txpowdB10nrb25run2earfcn500[0] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[1] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[2] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[3] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run2earfcn500[4] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[5] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[6] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[7] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run2earfcn500[8] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[9] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[10] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run2earfcn500[11] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[12] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[13] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run2earfcn500[14] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[15] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run2earfcn500[16] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[17] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run2earfcn500[18] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[19] = 2.222222222222e-09;
  spectrumValue_txpowdB10nrb25run2earfcn500[20] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run2earfcn500[21] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run2earfcn500[22] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run2earfcn500[23] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb25run2earfcn500[24] = 2.222222222222e-09;
  AddTestCase (new LteTxPsdTestCase ("txpowdB10nrb25run2earfcn500", 500, 25, 10.000000, activeRbs_txpowdB10nrb25run2earfcn500, spectrumValue_txpowdB10nrb25run2earfcn500), TestCase::QUICK);

  std::vector<int> activeRbs_txpowdB30nrb25run2earfcn500 (13);
  activeRbs_txpowdB30nrb25run2earfcn500[0] = 0.000000000000e+00;
  activeRbs_txpowdB30nrb25run2earfcn500[1] = 1.000000000000e+00;
  activeRbs_txpowdB30nrb25run2earfcn500[2] = 2.000000000000e+00;
  activeRbs_txpowdB30nrb25run2earfcn500[3] = 5.000000000000e+00;
  activeRbs_txpowdB30nrb25run2earfcn500[4] = 6.000000000000e+00;
  activeRbs_txpowdB30nrb25run2earfcn500[5] = 8.000000000000e+00;
  activeRbs_txpowdB30nrb25run2earfcn500[6] = 1.000000000000e+01;
  activeRbs_txpowdB30nrb25run2earfcn500[7] = 1.200000000000e+01;
  activeRbs_txpowdB30nrb25run2earfcn500[8] = 1.300000000000e+01;
  activeRbs_txpowdB30nrb25run2earfcn500[9] = 1.600000000000e+01;
  activeRbs_txpowdB30nrb25run2earfcn500[10] = 1.900000000000e+01;
  activeRbs_txpowdB30nrb25run2earfcn500[11] = 2.300000000000e+01;
  activeRbs_txpowdB30nrb25run2earfcn500[12] = 2.400000000000e+01;
  SpectrumValue spectrumValue_txpowdB30nrb25run2earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 25));
  spectrumValue_txpowdB30nrb25run2earfcn500[0] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run2earfcn500[1] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run2earfcn500[2] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run2earfcn500[3] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run2earfcn500[4] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run2earfcn500[5] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run2earfcn500[6] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run2earfcn500[7] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run2earfcn500[8] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run2earfcn500[9] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run2earfcn500[10] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run2earfcn500[11] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run2earfcn500[12] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run2earfcn500[13] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run2earfcn500[14] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run2earfcn500[15] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run2earfcn500[16] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run2earfcn500[17] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run2earfcn500[18] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run2earfcn500[19] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run2earfcn500[20] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run2earfcn500[21] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run2earfcn500[22] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb25run2earfcn500[23] = 2.222222222222e-07;
  spectrumValue_txpowdB30nrb25run2earfcn500[24] = 2.222222222222e-07;
  AddTestCase (new LteTxPsdTestCase ("txpowdB30nrb25run2earfcn500", 500, 25, 30.000000, activeRbs_txpowdB30nrb25run2earfcn500, spectrumValue_txpowdB30nrb25run2earfcn500), TestCase::QUICK);

  std::vector<int> activeRbs_txpowdB10nrb100run2earfcn500 (48);
  activeRbs_txpowdB10nrb100run2earfcn500[0] = 2.000000000000e+00;
  activeRbs_txpowdB10nrb100run2earfcn500[1] = 4.000000000000e+00;
  activeRbs_txpowdB10nrb100run2earfcn500[2] = 8.000000000000e+00;
  activeRbs_txpowdB10nrb100run2earfcn500[3] = 9.000000000000e+00;
  activeRbs_txpowdB10nrb100run2earfcn500[4] = 1.000000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[5] = 1.100000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[6] = 1.300000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[7] = 1.400000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[8] = 1.800000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[9] = 1.900000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[10] = 2.100000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[11] = 2.200000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[12] = 2.500000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[13] = 2.900000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[14] = 3.200000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[15] = 3.700000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[16] = 3.900000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[17] = 4.000000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[18] = 4.100000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[19] = 4.300000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[20] = 4.600000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[21] = 4.800000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[22] = 5.100000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[23] = 5.300000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[24] = 5.400000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[25] = 5.500000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[26] = 5.700000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[27] = 5.900000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[28] = 6.000000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[29] = 6.100000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[30] = 6.300000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[31] = 6.500000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[32] = 6.600000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[33] = 6.800000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[34] = 7.000000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[35] = 7.200000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[36] = 7.300000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[37] = 7.600000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[38] = 7.800000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[39] = 8.000000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[40] = 8.100000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[41] = 8.300000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[42] = 8.400000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[43] = 8.600000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[44] = 8.700000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[45] = 9.100000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[46] = 9.200000000000e+01;
  activeRbs_txpowdB10nrb100run2earfcn500[47] = 9.700000000000e+01;
  SpectrumValue spectrumValue_txpowdB10nrb100run2earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 100));
  spectrumValue_txpowdB10nrb100run2earfcn500[0] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[1] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[2] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[3] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[4] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[5] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[6] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[7] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[8] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[9] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[10] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[11] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[12] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[13] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[14] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[15] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[16] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[17] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[18] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[19] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[20] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[21] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[22] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[23] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[24] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[25] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[26] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[27] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[28] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[29] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[30] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[31] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[32] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[33] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[34] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[35] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[36] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[37] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[38] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[39] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[40] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[41] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[42] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[43] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[44] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[45] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[46] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[47] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[48] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[49] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[50] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[51] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[52] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[53] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[54] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[55] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[56] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[57] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[58] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[59] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[60] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[61] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[62] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[63] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[64] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[65] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[66] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[67] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[68] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[69] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[70] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[71] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[72] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[73] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[74] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[75] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[76] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[77] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[78] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[79] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[80] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[81] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[82] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[83] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[84] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[85] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[86] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[87] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[88] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[89] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[90] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[91] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[92] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[93] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[94] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[95] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[96] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[97] = 5.555555555556e-10;
  spectrumValue_txpowdB10nrb100run2earfcn500[98] = 0.000000000000e+00;
  spectrumValue_txpowdB10nrb100run2earfcn500[99] = 0.000000000000e+00;
  AddTestCase (new LteTxPsdTestCase ("txpowdB10nrb100run2earfcn500", 500, 100, 10.000000, activeRbs_txpowdB10nrb100run2earfcn500, spectrumValue_txpowdB10nrb100run2earfcn500), TestCase::QUICK);

  std::vector<int> activeRbs_txpowdB30nrb100run2earfcn500 (52);
  activeRbs_txpowdB30nrb100run2earfcn500[0] = 0.000000000000e+00;
  activeRbs_txpowdB30nrb100run2earfcn500[1] = 2.000000000000e+00;
  activeRbs_txpowdB30nrb100run2earfcn500[2] = 3.000000000000e+00;
  activeRbs_txpowdB30nrb100run2earfcn500[3] = 4.000000000000e+00;
  activeRbs_txpowdB30nrb100run2earfcn500[4] = 7.000000000000e+00;
  activeRbs_txpowdB30nrb100run2earfcn500[5] = 8.000000000000e+00;
  activeRbs_txpowdB30nrb100run2earfcn500[6] = 9.000000000000e+00;
  activeRbs_txpowdB30nrb100run2earfcn500[7] = 1.300000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[8] = 1.400000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[9] = 1.900000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[10] = 2.000000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[11] = 2.100000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[12] = 2.300000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[13] = 2.500000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[14] = 2.700000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[15] = 2.800000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[16] = 3.100000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[17] = 3.200000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[18] = 3.600000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[19] = 3.800000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[20] = 3.900000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[21] = 4.000000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[22] = 4.300000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[23] = 4.600000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[24] = 5.100000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[25] = 5.200000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[26] = 5.300000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[27] = 5.400000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[28] = 5.500000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[29] = 5.800000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[30] = 5.900000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[31] = 6.200000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[32] = 6.600000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[33] = 6.700000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[34] = 6.800000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[35] = 7.000000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[36] = 7.100000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[37] = 7.200000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[38] = 7.300000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[39] = 7.400000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[40] = 7.600000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[41] = 7.800000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[42] = 8.200000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[43] = 8.300000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[44] = 8.500000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[45] = 8.900000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[46] = 9.000000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[47] = 9.100000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[48] = 9.400000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[49] = 9.500000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[50] = 9.800000000000e+01;
  activeRbs_txpowdB30nrb100run2earfcn500[51] = 9.900000000000e+01;
  SpectrumValue spectrumValue_txpowdB30nrb100run2earfcn500 (LteSpectrumValueHelper::GetSpectrumModel (500, 100));
  spectrumValue_txpowdB30nrb100run2earfcn500[0] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[1] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[2] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[3] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[4] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[5] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[6] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[7] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[8] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[9] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[10] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[11] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[12] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[13] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[14] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[15] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[16] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[17] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[18] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[19] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[20] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[21] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[22] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[23] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[24] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[25] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[26] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[27] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[28] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[29] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[30] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[31] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[32] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[33] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[34] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[35] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[36] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[37] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[38] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[39] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[40] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[41] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[42] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[43] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[44] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[45] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[46] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[47] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[48] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[49] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[50] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[51] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[52] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[53] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[54] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[55] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[56] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[57] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[58] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[59] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[60] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[61] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[62] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[63] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[64] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[65] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[66] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[67] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[68] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[69] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[70] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[71] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[72] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[73] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[74] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[75] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[76] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[77] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[78] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[79] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[80] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[81] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[82] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[83] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[84] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[85] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[86] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[87] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[88] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[89] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[90] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[91] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[92] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[93] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[94] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[95] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[96] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[97] = 0.000000000000e+00;
  spectrumValue_txpowdB30nrb100run2earfcn500[98] = 5.555555555556e-08;
  spectrumValue_txpowdB30nrb100run2earfcn500[99] = 5.555555555556e-08;
  AddTestCase (new LteTxPsdTestCase ("txpowdB30nrb100run2earfcn500", 500, 100, 30.000000, activeRbs_txpowdB30nrb100run2earfcn500, spectrumValue_txpowdB30nrb100run2earfcn500), TestCase::QUICK);



}
