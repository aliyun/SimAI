"""
This file can generate topology of AlibabaHPN, Spectrum-X, DCN+.
Users can freely customize the topology according to their needs。
"""

import argparse
import warnings

def Rail_Opti_SingleToR(parameters):
    nodes_per_asw = parameters['nics_per_aswitch']
    asw_switch_num_per_segment = parameters['gpu_per_server']
    if(parameters['gpu'] % (nodes_per_asw * asw_switch_num_per_segment) == 0):
        segment_num = (int)(parameters['gpu']/ (nodes_per_asw * asw_switch_num_per_segment))
    else:
        segment_num = (int)(parameters['gpu']/ (nodes_per_asw * asw_switch_num_per_segment))+1
    
    if(segment_num != parameters['asw_switch_num'] / asw_switch_num_per_segment):
        warnings.warn("Error relations between total GPU Nums and total aws_switch_num.\n \
                         The correct asw_switch_num is set to "+str(segment_num * asw_switch_num_per_segment))
        parameters['asw_switch_num'] = segment_num * asw_switch_num_per_segment
    print("asw_switch_num: " + str(parameters['asw_switch_num']))
    if segment_num > int(parameters['asw_per_psw'] /  asw_switch_num_per_segment):
        raise ValueError("Number of GPU exceeds the capacity of Rail_Optimized_SingleToR(One Pod)")
    pod_num = 1
    print("psw_switch_num: " + str(parameters['psw_switch_num']))
    print("Creating Topology of totally " + str(segment_num) + " segment(s), totally "+ str(pod_num) + " pod(s)." )  

    nv_switch_num = (int)(parameters['gpu'] / parameters['gpu_per_server']) * parameters['nv_switch_per_server']
    nodes = (int) (parameters['gpu'] + parameters['asw_switch_num'] + parameters['psw_switch_num']+ nv_switch_num ) # 
    servers = parameters['gpu'] / parameters['gpu_per_server']
    switch_nodes = (int)(parameters['psw_switch_num'] + parameters['asw_switch_num'] + nv_switch_num) # 
    links = (int)(parameters['psw_switch_num']/pod_num * parameters['asw_switch_num'] + servers * asw_switch_num_per_segment
                  + servers * parameters['nv_switch_per_server'] * parameters['gpu_per_server']) # 
    if parameters['topology'] == 'Spectrum-X':
        file_name = "Spectrum-X_"+str(parameters['gpu'])+"g_"+str(parameters['gpu_per_server'])+"gps_"+parameters['bandwidth']+"_"+parameters['gpu_type']
    else:
        file_name = "Rail_Opti_SingleToR_"+str(parameters['gpu'])+"g_"+str(parameters['gpu_per_server'])+"gps_"+parameters['bandwidth']+"_"+parameters['gpu_type']
    with open(file_name, 'w') as f:
        print(file_name)
        first_line = str(nodes)+" "+str(parameters['gpu_per_server'])+" "+str(nv_switch_num)+" "+str(switch_nodes-nv_switch_num)+" "+str(int(links))+" "+str(parameters['gpu_type'])
        f.write(first_line)
        f.write('\n')
        nv_switch = []
        asw_switch = []
        psw_switch = []
        dsw_switch = []
        sec_line = ""
        nnodes = nodes - switch_nodes
        for i in range(nnodes, nodes):
            sec_line = sec_line + str(i) + " "
            if len(nv_switch) < nv_switch_num:
                nv_switch.append(i)
            elif len(asw_switch) < parameters['asw_switch_num']:
                asw_switch.append(i)
            elif len(psw_switch) < parameters['psw_switch_num']:
                psw_switch.append(i)
            else:
                dsw_switch.append(i)
        f.write(sec_line)
        f.write('\n')
        ind_asw = 0
        curr_node = 0
        group_num = 0
        group_account = 0
        ind_nv = 0
        for i in range(parameters['gpu']):
            curr_node = curr_node + 1
            if curr_node > parameters['gpu_per_server']:
                curr_node = 1
                ind_nv = ind_nv + parameters['nv_switch_per_server']
            for j in range(0, parameters['nv_switch_per_server']):
                #cnt += 1
                line = str(i)+" "+str(nv_switch[ind_nv+j])+" "+str(parameters['nvlink_bw'])+" "+str(parameters['nv_latency'])+" "+str(parameters['error_rate'])
                f.write(line)
                f.write('\n')
            line = str(i)+" "+str(asw_switch[group_num*asw_switch_num_per_segment+ind_asw])+" "+str(parameters['bandwidth'])+" "+str(parameters['latency'])+" "+str(parameters['error_rate'])
            f.write(line)
            f.write('\n')
            ind_asw = ind_asw + 1
            group_account = group_account + 1
            
            if ind_asw == asw_switch_num_per_segment:
                ind_asw = 0
            if group_account == (parameters['gpu_per_server'] * parameters['nics_per_aswitch']):
                group_num = group_num + 1
                group_account = 0

        for i in asw_switch: # asw - psw
                for j in psw_switch:
                    line = str(i) + " " + str(j) +" "+ str(parameters['ap_bandwidth'])+" " +str(parameters['latency'])+" "+str(parameters['error_rate'])
                    f.write(line)
                    f.write('\n')

def Rail_Opti_DualToR_SinglePlane(parameters):
    nodes_per_asw = parameters['nics_per_aswitch']
    asw_switch_num_per_segment = parameters['gpu_per_server']*2
    if(parameters['gpu'] % (nodes_per_asw * asw_switch_num_per_segment/2) == 0):
        segment_num = (int)(parameters['gpu']/ (nodes_per_asw * asw_switch_num_per_segment/2))
    else:
        segment_num = (int)(parameters['gpu']/ (nodes_per_asw * asw_switch_num_per_segment/2))+1
    
    if(segment_num != parameters['asw_switch_num'] / asw_switch_num_per_segment):
        warnings.warn("Error relations between total GPU Nums and total aws_switch_num.\n \
                         The correct asw_switch_num is set to "+str(segment_num * asw_switch_num_per_segment))
        parameters['asw_switch_num'] = segment_num * asw_switch_num_per_segment
    print("asw_switch_num: " + str(parameters['asw_switch_num']))
    if segment_num > int(parameters['asw_per_psw'] / (asw_switch_num_per_segment/2)):
        raise ValueError("Number of GPU exceeds the capacity of Rail_Optimized_SingleToR(One Pod)")
    pod_num = 1
    print("psw_switch_num: " + str(parameters['psw_switch_num']))
    print("Creating Topology of totally " + str(segment_num) + " segment(s), totally "+ str(pod_num) + " pod(s)." )  

    nv_switch_num = (int)(parameters['gpu'] / parameters['gpu_per_server']) * parameters['nv_switch_per_server']
    nodes = (int) (parameters['gpu'] + parameters['asw_switch_num'] + parameters['psw_switch_num']+ nv_switch_num ) # 
    servers = parameters['gpu'] / parameters['gpu_per_server']
    switch_nodes = (int)(parameters['psw_switch_num'] + parameters['asw_switch_num'] + nv_switch_num) # 
    links = (int)(parameters['psw_switch_num']/pod_num * parameters['asw_switch_num'] + servers * asw_switch_num_per_segment
                  + servers * parameters['nv_switch_per_server'] * parameters['gpu_per_server']) # 
    if parameters['topology'] == 'AlibabaHPN':
        file_name = "AlibabaHPN_"+str(parameters['gpu'])+"g_"+str(parameters['gpu_per_server'])+"gps_DualToR_SinglePlane_"+parameters['bandwidth']+"_"+parameters['gpu_type']
    else:
        file_name = "Rail_Opti_"+str(parameters['gpu'])+"g_"+str(parameters['gpu_per_server'])+"gps_DualToR_SinglePlane_"+parameters['bandwidth']+"_"+parameters['gpu_type']
    with open(file_name, 'w') as f:
        print(file_name)
        first_line = str(nodes)+" "+str(parameters['gpu_per_server'])+" "+str(nv_switch_num)+" "+str(switch_nodes-nv_switch_num)+" "+str(int(links))+" "+str(parameters['gpu_type'])
        f.write(first_line)
        f.write('\n')
        nv_switch = []
        asw_switch_1 = []
        asw_switch_2 = []
        psw_switch = []
        dsw_switch = []
        sec_line = ""
        nnodes = nodes - switch_nodes
        for i in range(nnodes, nodes):
            sec_line = sec_line + str(i) + " "
            if len(nv_switch) < nv_switch_num:
                nv_switch.append(i)
            elif len(asw_switch_1) < parameters['asw_switch_num']/2:
                asw_switch_1.append(i)
            elif len(asw_switch_2) < parameters['asw_switch_num']/2:
                asw_switch_2.append(i)
            elif len(psw_switch) < parameters['psw_switch_num']:
                psw_switch.append(i)
            else:
                dsw_switch.append(i)
        f.write(sec_line)
        f.write('\n')
        ind_asw = 0
        curr_node = 0
        group_num = 0
        group_account = 0
        ind_nv = 0
        for i in range(parameters['gpu']):
            curr_node = curr_node + 1
            if curr_node > parameters['gpu_per_server']:
                curr_node = 1
                ind_nv = ind_nv + parameters['nv_switch_per_server']
            for j in range(0, parameters['nv_switch_per_server']):
                #cnt += 1
                line = str(i)+" "+str(nv_switch[ind_nv+j])+" "+str(parameters['nvlink_bw'])+" "+str(parameters['nv_latency'])+" "+str(parameters['error_rate'])
                f.write(line)
                f.write('\n')
            line = str(i)+" "+str(asw_switch_1[group_num*int(asw_switch_num_per_segment/2)+ind_asw])+" "+str(parameters['bandwidth'])+" "+str(parameters['latency'])+" "+str(parameters['error_rate'])
            f.write(line)
            f.write('\n')
            
            line = str(i)+" "+str(asw_switch_2[group_num*int(asw_switch_num_per_segment/2)+ind_asw])+" "+str(parameters['bandwidth'])+" "+str(parameters['latency'])+" "+str(parameters['error_rate'])
            f.write(line)
            f.write('\n')

            ind_asw = ind_asw + 1
            group_account = group_account + 1
            
            if ind_asw == int(asw_switch_num_per_segment/2):
                ind_asw = 0
            if group_account == (parameters['gpu_per_server'] * parameters['nics_per_aswitch']):
                group_num = group_num + 1
                group_account = 0

        for i in asw_switch_1: # asw - psw
            for j in psw_switch:
                line = str(i) + " " + str(j) +" "+ str(parameters['ap_bandwidth'])+" " +str(parameters['latency'])+" "+str(parameters['error_rate'])
                f.write(line)
                f.write('\n')
        for i in asw_switch_2: # asw - psw
            for j in psw_switch:
                line = str(i) + " " + str(j) +" "+ str(parameters['ap_bandwidth'])+" " +str(parameters['latency'])+" "+str(parameters['error_rate'])
                f.write(line)
                f.write('\n')

def Rail_Opti_DualToR_DualPlane(parameters):
    nodes_per_asw = parameters['nics_per_aswitch']
    asw_switch_num_per_segment = parameters['gpu_per_server']*2
    if(parameters['gpu'] % (nodes_per_asw * asw_switch_num_per_segment/2) == 0):
        segment_num = (int)(parameters['gpu']/ (nodes_per_asw * asw_switch_num_per_segment/2))
    else:
        segment_num = (int)(parameters['gpu']/ (nodes_per_asw * asw_switch_num_per_segment/2))+1
    
    if(segment_num != parameters['asw_switch_num'] / asw_switch_num_per_segment):
        warnings.warn("Error relations between total GPU Nums and total aws_switch_num.\n \
                         The correct asw_switch_num is set to "+str(segment_num * asw_switch_num_per_segment))
        parameters['asw_switch_num'] = segment_num * asw_switch_num_per_segment
    print("asw_switch_num: " + str(parameters['asw_switch_num']))
    if segment_num > int(parameters['asw_per_psw'] / (asw_switch_num_per_segment/2)):
        raise ValueError("Number of GPU exceeds the capacity of Rail_Optimized_SingleToR(One Pod)")
    pod_num = 1
    print("psw_switch_num: " + str(parameters['psw_switch_num']))
    print("Creating Topology of totally " + str(segment_num) + " segment(s), totally "+ str(pod_num) + " pod(s)." )  

    nv_switch_num = (int)(parameters['gpu'] / parameters['gpu_per_server']) * parameters['nv_switch_per_server']
    nodes = (int) (parameters['gpu'] + parameters['asw_switch_num'] + parameters['psw_switch_num']+ nv_switch_num ) # 
    servers = parameters['gpu'] / parameters['gpu_per_server']
    switch_nodes = (int)(parameters['psw_switch_num'] + parameters['asw_switch_num'] + nv_switch_num) # 
    links = (int)(parameters['psw_switch_num']/pod_num/2 * parameters['asw_switch_num'] + servers * asw_switch_num_per_segment
                  + servers * parameters['nv_switch_per_server'] * parameters['gpu_per_server']) # 
    if parameters['topology'] == 'AlibabaHPN':
        file_name = "AlibabaHPN_"+str(parameters['gpu'])+"g_"+str(parameters['gpu_per_server'])+"gps_DualToR_DualPlane_"+parameters['bandwidth']+"_"+parameters['gpu_type']
    else:
        file_name = "Rail_Opti_"+str(parameters['gpu'])+"g_"+str(parameters['gpu_per_server'])+"gps_DualToR_DualPlane_"+parameters['bandwidth']+"_"+parameters['gpu_type']
    with open(file_name, 'w') as f:
        print(file_name)
        first_line = str(nodes)+" "+str(parameters['gpu_per_server'])+" "+str(nv_switch_num)+" "+str(switch_nodes-nv_switch_num)+" "+str(int(links))+" "+str(parameters['gpu_type'])
        f.write(first_line)
        f.write('\n')
        nv_switch = []
        asw_switch_1 = []
        asw_switch_2 = []
        psw_switch_1 = []
        psw_switch_2 = []
        dsw_switch = []
        sec_line = ""
        nnodes = nodes - switch_nodes
        for i in range(nnodes, nodes):
            sec_line = sec_line + str(i) + " "
            if len(nv_switch) < nv_switch_num:
                nv_switch.append(i)
            elif len(asw_switch_1) < parameters['asw_switch_num']/2:
                asw_switch_1.append(i)
            elif len(asw_switch_2) < parameters['asw_switch_num']/2:
                asw_switch_2.append(i)
            elif len(psw_switch_1) < parameters['psw_switch_num']/2:
                psw_switch_1.append(i)
            elif len(psw_switch_2) < parameters['psw_switch_num']/2:
                psw_switch_2.append(i)
            else:
                dsw_switch.append(i)
        f.write(sec_line)
        f.write('\n')
        ind_asw = 0
        curr_node = 0
        group_num = 0
        group_account = 0
        ind_nv = 0
        for i in range(parameters['gpu']):
            curr_node = curr_node + 1
            if curr_node > parameters['gpu_per_server']:
                curr_node = 1
                ind_nv = ind_nv + parameters['nv_switch_per_server']
            for j in range(0, parameters['nv_switch_per_server']):
                #cnt += 1
                line = str(i)+" "+str(nv_switch[ind_nv+j])+" "+str(parameters['nvlink_bw'])+" "+str(parameters['nv_latency'])+" "+str(parameters['error_rate'])
                f.write(line)
                f.write('\n')
            line = str(i)+" "+str(asw_switch_1[group_num*int(asw_switch_num_per_segment/2)+ind_asw])+" "+str(parameters['bandwidth'])+" "+str(parameters['latency'])+" "+str(parameters['error_rate'])
            f.write(line)
            f.write('\n')
            
            line = str(i)+" "+str(asw_switch_2[group_num*int(asw_switch_num_per_segment/2)+ind_asw])+" "+str(parameters['bandwidth'])+" "+str(parameters['latency'])+" "+str(parameters['error_rate'])
            f.write(line)
            f.write('\n')

            ind_asw = ind_asw + 1
            group_account = group_account + 1
            
            if ind_asw == int(asw_switch_num_per_segment/2):
                ind_asw = 0
            if group_account == (parameters['gpu_per_server'] * parameters['nics_per_aswitch']):
                group_num = group_num + 1
                group_account = 0

        for i in asw_switch_1: # asw - psw
            for j in psw_switch_1:
                line = str(i) + " " + str(j) +" "+ str(parameters['ap_bandwidth'])+" " +str(parameters['latency'])+" "+str(parameters['error_rate'])
                f.write(line)
                f.write('\n')
        for i in asw_switch_2: # asw - psw
            for j in psw_switch_2:
                line = str(i) + " " + str(j) +" "+ str(parameters['ap_bandwidth'])+" " +str(parameters['latency'])+" "+str(parameters['error_rate'])
                f.write(line)
                f.write('\n')

def No_Rail_Opti_SingleToR(parameters):
    nodes_per_asw = parameters['nics_per_aswitch']
    asw_switch_num_per_segment = 1
    if(parameters['gpu'] % (nodes_per_asw * asw_switch_num_per_segment) == 0):
        segment_num = (int)(parameters['gpu']/ (nodes_per_asw * asw_switch_num_per_segment))
    else:
        segment_num = (int)(parameters['gpu']/ (nodes_per_asw * asw_switch_num_per_segment))+1
    
    if(segment_num != parameters['asw_switch_num'] / asw_switch_num_per_segment):
        warnings.warn("Error relations between total GPU Nums and total aws_switch_num.\n \
                         The correct asw_switch_num is set to "+str(segment_num * asw_switch_num_per_segment))
        parameters['asw_switch_num'] = segment_num * asw_switch_num_per_segment
    print("asw_switch_num: " + str(parameters['asw_switch_num']))
    if segment_num > int(parameters['asw_per_psw'] /  asw_switch_num_per_segment):
        raise ValueError("Number of GPU exceeds the capacity of Rail_Optimized_SingleToR(One Pod)")
    pod_num = 1
    print("psw_switch_num: " + str(parameters['psw_switch_num']))
    print("Creating Topology of totally " + str(segment_num) + " segment(s), totally "+ str(pod_num) + " pod(s)." )  

    nv_switch_num = (int)(parameters['gpu'] / parameters['gpu_per_server']) * parameters['nv_switch_per_server']
    nodes = (int) (parameters['gpu'] + parameters['asw_switch_num'] + parameters['psw_switch_num']+ nv_switch_num ) # 
    servers = parameters['gpu'] / parameters['gpu_per_server']
    switch_nodes = (int)(parameters['psw_switch_num'] + parameters['asw_switch_num'] + nv_switch_num) # 
    links = (int)(parameters['psw_switch_num']/pod_num * parameters['asw_switch_num'] + servers * parameters['gpu_per_server']
                  + servers * parameters['nv_switch_per_server'] * parameters['gpu_per_server']) # 
    if parameters['topology'] == 'DCN+':
        file_name = "DCN+SingleToR_"+str(parameters['gpu'])+"g_"+str(parameters['gpu_per_server'])+"gps_"+parameters['bandwidth']+"_"+parameters['gpu_type']
    else:
        file_name = "No_Rail_Opti_"+str(parameters['gpu'])+"g_"+str(parameters['gpu_per_server'])+"gps_SingleToR_"+parameters['bandwidth']+"_"+parameters['gpu_type']
    with open(file_name, 'w') as f:
        print(file_name)
        first_line = str(nodes)+" "+str(parameters['gpu_per_server'])+" "+str(nv_switch_num)+" "+str(switch_nodes-nv_switch_num)+" "+str(int(links))+" "+str(parameters['gpu_type'])
        f.write(first_line)
        f.write('\n')
        nv_switch = []
        asw_switch = []
        psw_switch = []
        dsw_switch = []
        sec_line = ""
        nnodes = nodes - switch_nodes
        for i in range(nnodes, nodes):
            sec_line = sec_line + str(i) + " "
            if len(nv_switch) < nv_switch_num:
                nv_switch.append(i)
            elif len(asw_switch) < parameters['asw_switch_num']:
                asw_switch.append(i)
            elif len(psw_switch) < parameters['psw_switch_num']:
                psw_switch.append(i)
            else:
                dsw_switch.append(i)
        f.write(sec_line)
        f.write('\n')
        ind_asw = 0
        curr_node = 0
        group_num = 0
        group_account = 0
        ind_nv = 0
        for i in range(parameters['gpu']):
            curr_node = curr_node + 1
            if curr_node > parameters['gpu_per_server']:
                curr_node = 1
                ind_nv = ind_nv + parameters['nv_switch_per_server']
            for j in range(0, parameters['nv_switch_per_server']):
                #cnt += 1
                line = str(i)+" "+str(nv_switch[ind_nv+j])+" "+str(parameters['nvlink_bw'])+" "+str(parameters['nv_latency'])+" "+str(parameters['error_rate'])
                f.write(line)
                f.write('\n')
            line = str(i)+" "+str(asw_switch[group_num*asw_switch_num_per_segment+ind_asw])+" "+str(parameters['bandwidth'])+" "+str(parameters['latency'])+" "+str(parameters['error_rate'])
            f.write(line)
            f.write('\n')
            group_account = group_account + 1
            
            if group_account == nodes_per_asw:
                group_num = group_num + 1
                group_account = 0

        for i in asw_switch: # asw - psw
                for j in psw_switch:
                    line = str(i) + " " + str(j) +" "+ str(parameters['ap_bandwidth'])+" " +str(parameters['latency'])+" "+str(parameters['error_rate'])
                    f.write(line)
                    f.write('\n')

def No_Rail_Opti_DualToR(parameters):
    nodes_per_asw = parameters['nics_per_aswitch']
    asw_switch_num_per_segment = 2
    if(parameters['gpu'] % (nodes_per_asw * (asw_switch_num_per_segment/2)) == 0):
        segment_num = (int)(parameters['gpu']/ (nodes_per_asw * asw_switch_num_per_segment/2))
    else:
        segment_num = (int)(parameters['gpu']/ (nodes_per_asw * asw_switch_num_per_segment/2))+1
    if(segment_num != parameters['asw_switch_num'] / asw_switch_num_per_segment):
        warnings.warn("Error relations between total GPU Nums and total aws_switch_num.\n \
                         The correct asw_switch_num is set to "+str(segment_num * asw_switch_num_per_segment))
        parameters['asw_switch_num'] = segment_num * asw_switch_num_per_segment
    print("asw_switch_num: " + str(parameters['asw_switch_num']))
    if segment_num > int(parameters['asw_per_psw'] /  asw_switch_num_per_segment):
        raise ValueError("Number of GPU exceeds the capacity of Rail_Optimized_SingleToR(One Pod)")
    pod_num = 1
    print("psw_switch_num: " + str(parameters['psw_switch_num']))
    print("Creating Topology of totally " + str(segment_num) + " segment(s), totally "+ str(pod_num) + " pod(s)." )  

    nv_switch_num = (int)(parameters['gpu'] / parameters['gpu_per_server']) * parameters['nv_switch_per_server']
    nodes = (int) (parameters['gpu'] + parameters['asw_switch_num'] + parameters['psw_switch_num']+ nv_switch_num ) # 
    servers = parameters['gpu'] / parameters['gpu_per_server']
    switch_nodes = (int)(parameters['psw_switch_num'] + parameters['asw_switch_num'] + nv_switch_num) # 
    links = (int)(parameters['psw_switch_num']/pod_num * parameters['asw_switch_num'] + servers * parameters['gpu_per_server']*2
                  + servers * parameters['nv_switch_per_server'] * parameters['gpu_per_server']) # 
    if parameters['topology'] == 'DCN+':
        file_name = "DCN+DualToR_"+str(parameters['gpu'])+"g_"+str(parameters['gpu_per_server'])+"gps_"+parameters['bandwidth']+"_"+parameters['gpu_type']
    else:
        file_name = "No_Rail_Opti_"+str(parameters['gpu'])+"g_"+str(parameters['gpu_per_server'])+"gps_DualToR_"+parameters['bandwidth']+"_"+parameters['gpu_type']
    with open(file_name, 'w') as f:
        print(file_name)
        first_line = str(nodes)+" "+str(parameters['gpu_per_server'])+" "+str(nv_switch_num)+" "+str(switch_nodes-nv_switch_num)+" "+str(int(links))+" "+str(parameters['gpu_type'])
        f.write(first_line)
        f.write('\n')
        nv_switch = []
        asw_switch_1 = []
        asw_switch_2 = []
        psw_switch = []
        dsw_switch = []
        sec_line = ""
        nnodes = nodes - switch_nodes
        for i in range(nnodes, nodes):
            sec_line = sec_line + str(i) + " "
            if len(nv_switch) < nv_switch_num:
                nv_switch.append(i)
            elif len(asw_switch_1) < parameters['asw_switch_num']/2:
                asw_switch_1.append(i)
            elif len(asw_switch_2) < parameters['asw_switch_num']/2:
                asw_switch_2.append(i)
            elif len(psw_switch) < parameters['psw_switch_num']:
                psw_switch.append(i)
            else:
                dsw_switch.append(i)
        f.write(sec_line)
        f.write('\n')
        ind_asw = 0
        curr_node = 0
        group_num = 0
        group_account = 0
        ind_nv = 0
        for i in range(parameters['gpu']):
            curr_node = curr_node + 1
            if curr_node > parameters['gpu_per_server']:
                curr_node = 1
                ind_nv = ind_nv + parameters['nv_switch_per_server']
            for j in range(0, parameters['nv_switch_per_server']):
                #cnt += 1
                line = str(i)+" "+str(nv_switch[ind_nv+j])+" "+str(parameters['nvlink_bw'])+" "+str(parameters['nv_latency'])+" "+str(parameters['error_rate'])
                f.write(line)
                f.write('\n')
            line = str(i)+" "+str(asw_switch_1[group_num*int(asw_switch_num_per_segment/2)+ind_asw])+" "+str(parameters['bandwidth'])+" "+str(parameters['latency'])+" "+str(parameters['error_rate'])
            f.write(line)
            f.write('\n')

            line = str(i)+" "+str(asw_switch_2[group_num*int(asw_switch_num_per_segment/2)+ind_asw])+" "+str(parameters['bandwidth'])+" "+str(parameters['latency'])+" "+str(parameters['error_rate'])
            f.write(line)
            f.write('\n')
            group_account = group_account + 1

            if group_account == nodes_per_asw:
                group_num = group_num + 1
                group_account = 0

        for i in asw_switch_1: # asw - psw
                for j in psw_switch:
                    line = str(i) + " " + str(j) +" "+ str(parameters['ap_bandwidth'])+" " +str(parameters['latency'])+" "+str(parameters['error_rate'])
                    f.write(line)
                    f.write('\n')
        for i in asw_switch_2: # asw - psw
                for j in psw_switch:
                    line = str(i) + " " + str(j) +" "+ str(parameters['ap_bandwidth'])+" " +str(parameters['latency'])+" "+str(parameters['error_rate'])
                    f.write(line)
                    f.write('\n')


def main():
    parser = argparse.ArgumentParser(description='Python script for generating a topology for SimAI')

    #Whole Structure Parameters:
    parser.add_argument('-topo','--topology', type=str, default=None,help='Template for AlibabaHPN, Spectrum-X, DCN+')
    parser.add_argument('--ro', action='store_true',help='use rail-optimized structure')
    parser.add_argument('--dt',action='store_true', help='enable dual ToR, only for DCN+')
    parser.add_argument('--dp', action='store_true', help='enable dual_plane, only for AlibabaHPN')
    parser.add_argument('-g','--gpu',type=int,default=None,help='gpus num, default 32')
    parser.add_argument('-er','--error_rate',type=str,default=None,help='error_rate, default 0')
    #Intra-Host Parameters:
    parser.add_argument('-gps','--gpu_per_server',type=int,default=None,help='gpu_per_server, default 8')
    parser.add_argument('-gt','--gpu_type',type=str,default=None,help='gpu_type, default H100')
    parser.add_argument('-nsps','--nv_switch_per_server',type=int,default=None,help='nv_switch_per_server, default 1')
    parser.add_argument('-nvbw','--nvlink_bw',type=str,default=None,help='nvlink_bw, default 2880Gbps')
    parser.add_argument('-nl','--nv_latency',type=str,default=None,help='nv switch latency, default 0.000025ms')
    parser.add_argument('-l','--latency',type=str,default=None,help='nic latency, default 0.0005ms')
    #Intra-Segment Parameters:
    parser.add_argument('-bw','--bandwidth',type=str,default=None,help='nic to asw bandwidth, default 400Gbps')
    parser.add_argument('-asn','--asw_switch_num',type=int,default=None,help='asw_switch_num, default 8')
    parser.add_argument('-npa','--nics_per_aswitch',type=int,default=None,help='nnics per asw, default 64')
    #Intra-Pod Parameters:
    parser.add_argument('-psn','--psw_switch_num',type=int,default=None,help='psw_switch_num, default 64')
    parser.add_argument('-apbw','--ap_bandwidth',type=str,default=None,help='asw to psw bandwidth,default 400Gbps')   
    parser.add_argument('-app','--asw_per_psw',type=int,default=None,help='asw for psw')
    args = parser.parse_args()

    default_parameters = []
    parameters = analysis_template(args, default_parameters)
    if not parameters['rail_optimized']:
        if parameters['dual_plane']:
            raise ValueError("Sorry, None Rail-Optimized Structure doesn't support Dual Plane")
        if parameters['dual_ToR']:
            No_Rail_Opti_DualToR(parameters)
        else:
            No_Rail_Opti_SingleToR(parameters)
    else:
        if parameters['dual_ToR']:
            if parameters['dual_plane']:
                Rail_Opti_DualToR_DualPlane(parameters)
            else:
                Rail_Opti_DualToR_SinglePlane(parameters)
        else:
            if parameters['dual_plane']:
                raise ValueError("Sorry, Rail-optimized Single-ToR Structure doesn't support Dual Plane")
            Rail_Opti_SingleToR(parameters)


def analysis_template(args, default_parameters):
    # Basic default parameters
    default_parameters = {'rail_optimized': True, 'dual_ToR': False, 'dual_plane': False, 'gpu': 32, 'error_rate':0,
                          'gpu_per_server': 8, 'gpu_type': 'H100', 'nv_switch_per_server': 1, 
                          'nvlink_bw': '2880Gbps','nv_latency': '0.000025ms', 'latency': '0.0005ms',
                          'bandwidth': '400Gbps', 'asw_switch_num': 8,  'nics_per_aswitch': 64,
                          'psw_switch_num': 64, 'ap_bandwidth': "400Gbps", 'asw_per_psw' : 64}
    parameters = {}
    parameters['topology'] = args.topology
    parameters['rail_optimized'] = bool(args.ro)
    parameters['dual_ToR'] = bool(args.dt)
    parameters['dual_plane'] = bool(args.dp)

    
    if parameters['topology'] == 'Spectrum-X':
        default_parameters.update({
            'gpu': 4096
        })
        parameters.update({
            'rail_optimized': True, 
            'dual_ToR': False, 
            'dual_plane': False,
        })
    elif parameters['topology'] == 'AlibabaHPN':
        default_parameters.update({
            'gpu': 15360, 
            'bandwidth': '200Gbps', 
            'asw_switch_num': 240, 
            'nics_per_aswitch': 128, 
            'psw_switch_num': 120,
            'asw_per_psw':240
        })
        parameters.update({
            'rail_optimized': True, 
            'dual_ToR': True, 
            'dual_plane': False,
            
        })
        if args.dp:
            default_parameters.update({
                'asw_per_psw':120
            })
            parameters.update({
                'rail_optimized': True, 
                'dual_ToR': True, 
                'dual_plane': True, 
            })
    elif parameters['topology'] == 'DCN+':
        default_parameters.update({
            'gpu': 512, 
            'asw_switch_num': 8, 
            'asw_per_psw':8,
            'psw_switch_num': 8
        })
        parameters.update({
            'rail_optimized': False, 
            'dual_ToR': False, 
            'dual_plane': False, 
        })
        if args.dt:
            default_parameters.update({
                'bandwidth': '200Gbps',
                'nics_per_aswitch': 128, 
            })
            parameters.update({
                'rail_optimized': False, 
                'dual_ToR': True, 
                'dual_plane': False,
            })
    
    parameter_keys = [
        'gpu', 'error_rate', 'gpu_per_server', 'gpu_type', 'nv_switch_per_server',
        'nvlink_bw', 'nv_latency', 'latency', 'bandwidth', 'asw_switch_num',
        'nics_per_aswitch', 'psw_switch_num', 'ap_bandwidth','asw_per_psw'
    ]
    for key in parameter_keys:
        parameters[key] = getattr(args, key, None) if getattr(args, key, None) is not None else default_parameters[key]
    # for key, value in parameters.items():
    #     print(f'{key}: {value}')
    # print("==================================")
    return parameters


if __name__ =='__main__':
    main()