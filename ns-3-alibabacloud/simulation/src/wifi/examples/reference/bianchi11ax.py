#
# Copyright 2020 University of Washington
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Authors:  Hao Yin and Sebastien Deronne
# 
import numpy as np
import math


def bianchi_ax(data_rate, ack_rate, k, difs):
    # Parameters for 11ax
    nA = np.linspace(5, 50, 10)
    CWmin = 15 
    CWmax = 1023 
    L_DATA = 1500 * 8 # data size in bits
    L_ACK = 14 * 8    # ACK size in bits
    #B = 1/(CWmin+1) 
    B=0 
    EP = L_DATA/(1-B)
    T_GI = 800e-9                  # guard interval in seconds
    T_SYMBOL_ACK = 4e-6            # symbol duration in seconds (for ACK)
    T_SYMBOL_DATA = 12.8e-6 + T_GI # symbol duration in seconds (for DATA)
    T_PHY_ACK = 20e-6              # PHY preamble & header duration in seconds (for ACK)
    T_PHY_DATA = 44e-6             # PHY preamble & header duration in seconds (for DATA)
    L_SERVICE = 16                 # service field length in bits
    L_TAIL = 6                     # tail lengthh in bits
    L_MAC = (30) * 8               # MAC header size in bits
    L_APP_HDR = 8 * 8              # bits added by the upper layer(s)
    T_SIFS = 16e-6 
    T_DIFS = 34e-6 
    T_SLOT = 9e-6 
    delta = 1e-7 

    Aggregation_Type = 'A_MPDU'   #A_MPDU or A_MSDU (HYBRID not fully supported)
    K_MSDU = 1 
    K_MPDU = k 
    L_MPDU_HEADER = 4
    L_MSDU_HEADER = 14 * 8 
    if (k <= 1):
        Aggregation_Type = 'NONE' 

    N_DBPS = data_rate * T_SYMBOL_DATA   # number of data bits per OFDM symbol

    if (Aggregation_Type == 'NONE'):
        N_SYMBOLS = math.ceil((L_SERVICE + (L_MAC + L_DATA + L_APP_HDR) + L_TAIL)/N_DBPS)
        T_DATA = T_PHY_DATA + (T_SYMBOL_DATA * N_SYMBOLS)
        K_MPDU = 1
        K_MSDU = 1 

    if (Aggregation_Type == 'A_MSDU'):
        N_SYMBOLS = math.ceil((L_SERVICE + K_MPDU*(L_MAC + L_MPDU_HEADER + K_MSDU*(L_MSDU_HEADER + L_DATA + L_APP_HDR)) + L_TAIL)/N_DBPS)
        T_DATA = T_PHY_DATA + (T_SYMBOL_DATA * N_SYMBOLS)

    if (Aggregation_Type == 'A_MPDU'):
        N_SYMBOLS = math.ceil((L_SERVICE + K_MPDU*(L_MAC + L_MPDU_HEADER + L_DATA + L_APP_HDR) + L_TAIL)/N_DBPS)
        T_DATA = T_PHY_DATA + (T_SYMBOL_DATA * N_SYMBOLS)

    #Calculate ACK Duration
    N_DBPS = ack_rate * T_SYMBOL_ACK   # number of data bits per OFDM symbol
    N_SYMBOLS = math.ceil((L_SERVICE + L_ACK + L_TAIL)/N_DBPS)
    T_ACK = T_PHY_ACK + (T_SYMBOL_ACK * N_SYMBOLS)

    T_s = T_DATA + T_SIFS + T_ACK + T_DIFS 
    if difs == 1: #DIFS
        T_C = T_DATA + T_DIFS 
    else:
        T_s = T_DATA + T_SIFS + T_ACK + T_DIFS + delta 
        T_C = T_DATA + T_DIFS + T_SIFS + T_ACK + delta 

    T_S = T_s/(1-B) + T_SLOT 

    S_bianchi = np.zeros(len(nA)) 
    for j in range(len(nA)):
        n = nA[j]*1 
        W = CWmin + 1 
        m = math.log2((CWmax + 1)/(CWmin + 1)) 
        tau1 = np.linspace(0, 0.1, 100000)  
        p = 1 - np.power((1 - tau1),(n - 1))
        ps = p*0 

        for i in range(int(m)):
            ps = ps + np.power(2*p, i) 

        taup = 2./(1 + W + p*W*ps) 
        b = np.argmin(np.abs(tau1 - taup))
        tau = taup[b] 

        Ptr = 1 - math.pow((1 - tau), int(n))
        Ps = n*tau*math.pow((1 - tau), int(n-1))/Ptr

        S_bianchi[j] = K_MSDU*K_MPDU*Ps*Ptr*EP/((1-Ptr)*T_SLOT+Ptr*Ps*T_S+Ptr*(1-Ps)*T_C)/1e6 

    bianchi_result = S_bianchi
    return bianchi_result

def str_result(bianchi_result, mcs, bw):
    str_bianchi = '    {' + '\"HeMcs{:d}'.format(mcs) + '_{:d}MHz\"'.format(bw) + ', {\n'
    for  i in range (len(bianchi_result)):
        str_tmp = '        {' + '{:d}, {:.4f}'.format(5*(i+1), bianchi_result[i]) +'},\n'
        str_bianchi  = str_bianchi + str_tmp
    str_bianchi = str_bianchi + "    }},\n"
    print(str_bianchi)
    return str_bianchi

# Settings for different MCS and mode
data_rates_20MHz = [8.603e6, 17.206e6, 25.8e6, 34.4e6, 51.5e6, 68.8e6, 77.4e6, 86e6, 103.2e6, 114.7e6, 129e6, 143.4e6]
ack_rates_20MHz = [6e6, 12e6, 12e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6]
data_rates_40MHz = [17.2e6, 34.4e6, 51.5e6, 68.8e6, 103.2e6, 137.6e6, 154.9e6, 172.1e6, 206.5e6, 229.4e6, 258.1e6, 286.8e6]
ack_rates_40MHz = [6e6, 12e6, 12e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6]
data_rates_80MHz = [36e6, 72.1e6, 108.1e6, 144.1e6, 216.2e6, 288.2e6, 324.3e6, 360.3e6, 432.4e6, 480.4e6, 540.4e6, 600.5e6]
ack_rates_80MHz = [6e6, 12e6, 12e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6]
data_rates_160MHz = [72.1e6, 144.1e6, 216.2e6, 288.2e6, 432.4e6, 576.5e6, 648.5e6, 720.6e6, 864.7e6, 960.8e6, 1080.9e6, 1201e6]
ack_rates_160MHz = [6e6, 12e6, 12e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6, 24e6]

# Generate results with frame aggregation disabled
k = 1

difs = 1
fo = open("bianchi_11ax_difs.txt", "w")
for i in range(len(data_rates_20MHz)):
    bianchi_result = bianchi_ax(data_rates_20MHz[i], ack_rates_20MHz[i], k, difs)
    str_s = str_result(bianchi_result, i, 20)
    fo.write(str_s)
for i in range(len(data_rates_40MHz)):
    bianchi_result = bianchi_ax(data_rates_40MHz[i], ack_rates_40MHz[i], k, difs)
    str_s = str_result(bianchi_result, i, 40)
    fo.write(str_s)
for i in range(len(data_rates_80MHz)):
    bianchi_result = bianchi_ax(data_rates_80MHz[i], ack_rates_80MHz[i], k, difs)
    str_s = str_result(bianchi_result, i, 80)
    fo.write(str_s)
for i in range(len(data_rates_160MHz)):
    bianchi_result = bianchi_ax(data_rates_160MHz[i], ack_rates_160MHz[i], k, difs)
    str_s = str_result(bianchi_result, i, 160)
    fo.write(str_s)
fo.close()

difs = 0
fo = open("bianchi_11ax_eifs.txt", "w")
for i in range(len(data_rates_20MHz)):
    bianchi_result = bianchi_ax(data_rates_20MHz[i], ack_rates_20MHz[i], k, difs)
    str_s = str_result(bianchi_result, i, 20)
    fo.write(str_s)
for i in range(len(data_rates_40MHz)):
    bianchi_result = bianchi_ax(data_rates_40MHz[i], ack_rates_40MHz[i], k, difs)
    str_s = str_result(bianchi_result, i, 40)
    fo.write(str_s)
for i in range(len(data_rates_80MHz)):
    bianchi_result = bianchi_ax(data_rates_80MHz[i], ack_rates_80MHz[i], k, difs)
    str_s = str_result(bianchi_result, i, 80)
    fo.write(str_s)
for i in range(len(data_rates_160MHz)):
    bianchi_result = bianchi_ax(data_rates_160MHz[i], ack_rates_160MHz[i], k, difs)
    str_s = str_result(bianchi_result, i, 160)
    fo.write(str_s)
fo.close()
