"""
This file can generate topology of AlibabaHPN, Spectrum-X, DCN+.
Users can freely customize the topology according to their needs。
"""

import argparse
import warnings

def gen_SingleToR(args):
    print("SingleToR")
    nodes_per_asw = args.nics_per_aswitch
    asw_switch_num_per = args.gpu_per_server
    if(args.gpu % (nodes_per_asw * asw_switch_num_per) == 0):
        segment_num = (int)(args.gpu / (nodes_per_asw * asw_switch_num_per))
    else:
        segment_num = (int)(args.gpu / (nodes_per_asw * asw_switch_num_per))+1
        
    if(segment_num != args.asw_switch_num / asw_switch_num_per):
        warnings.warn("Error relations between total GPU Nums and total aws_switch_num.\n \
                         The correct asw_switch_num is set to "+str(segment_num * asw_switch_num_per))
        args.asw_switch_num = segment_num * asw_switch_num_per
    print("asw_switch_num: " + str(args.asw_switch_num))
    nv_switch_num = (int)(args.gpu / args.gpu_per_server) * args.nv_switch_per_server
    
    if segment_num <= 1:
        # args.psw_switch_num = 0
        dsw_switch_num = 0
        pod_num = 1
        print("psw_switch_num: " + str(args.psw_switch_num))
    elif 1< segment_num <= int(args.asw_per_psw /  asw_switch_num_per):
        pod_num = 1
        print("psw_switch_num: " + str(args.psw_switch_num))
        dsw_switch_num = 0
    else:  #More Than one Pod  
        if segment_num % (int(args.asw_per_psw /  asw_switch_num_per)) == 0 :
            pod_num = int(segment_num / (int(args.asw_per_psw /  asw_switch_num_per)))
        else:
            pod_num = int(segment_num / (int(args.asw_per_psw /  asw_switch_num_per))) + 1
        
        if pod_num != int(args.psw_switch_num / args.asw_per_psw):
            warnings.warn("Error relations between total GPU Nums and total aws_switch_num.\n \
                         The correct psw_switch_num is set to "+str(pod_num * args.asw_per_psw))
            args.psw_switch_num = pod_num * args.asw_per_psw
        print("psw_switch_num: " + str(args.psw_switch_num))
        dsw_switch_num = args.dsw_switch_num
        if pod_num > args.psw_per_dsw:
            raise ValueError("Number of GPU exceeds the capacity of Single_ToR_Rail_Optimzied")
        elif args.dsw_switch_num != pod_num * args.asw_per_psw:
            warnings.warn("Error relations between total GPU Nums and total aws_switch_num.\n \
                         The correct dsw_switch_num is set to "+str(args.asw_per_psw * pod_num))
            dsw_switch_num = args.asw_per_psw * pod_num
        print("dsw_switch_num: " + str(dsw_switch_num))   
    print("Creating Topology of totally " + str(segment_num) + " segment(s), totally "+ str(pod_num) + " pod(s)." )  
        
    nodes = (int) (args.gpu + args.asw_switch_num + args.psw_switch_num + nv_switch_num + dsw_switch_num) # 
    servers = args.gpu / args.gpu_per_server
    switch_nodes = (int)(args.psw_switch_num + args.asw_switch_num + nv_switch_num + dsw_switch_num) # 
    links = (int)(args.psw_switch_num/pod_num * args.asw_switch_num + servers * asw_switch_num_per+ servers * args.nv_switch_per_server * args.gpu_per_server \
                  + dsw_switch_num * pod_num) # 
    if args.topology == 'Spectrum-X':
        file_name = "Spectrum-X_"+str(args.gpu)+"g_"+str(args.gpu_per_server)+"gps_"+args.bandwidth+"_"+args.gpu_type
    else:
        file_name = "SingleToR_"+str(args.gpu)+"g_"+str(args.gpu_per_server)+"gps_"+args.bandwidth+"_"+args.gpu_type
    with open(file_name, 'w') as f:
        print(file_name)
        first_line = str(nodes)+" "+str(args.gpu_per_server)+" "+str(nv_switch_num)+" "+str(switch_nodes-nv_switch_num)+" "+str(int(links))+" "+str(args.gpu_type)
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
            elif len(asw_switch) < args.asw_switch_num:
                asw_switch.append(i)
            elif len(psw_switch) < args.psw_switch_num:
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
        for i in range(args.gpu):
            curr_node = curr_node + 1
            if curr_node > args.gpu_per_server:
                curr_node = 1
                ind_nv = ind_nv + args.nv_switch_per_server
            for j in range(0, args.nv_switch_per_server):
                #cnt += 1
                line = str(i)+" "+str(nv_switch[ind_nv+j])+" "+args.nvlink_bw+" "+args.nv_latency+" "+args.error_rate
                f.write(line)
                f.write('\n')
            line = str(i)+" "+str(asw_switch[group_num*asw_switch_num_per+ind_asw])+" "+args.bandwidth+" "+args.latency+" "+args.error_rate
            f.write(line)
            f.write('\n')
            ind_asw = ind_asw + 1
            group_account = group_account + 1
            
            if ind_asw == asw_switch_num_per:
                ind_asw = 0
            if group_account == (args.gpu_per_server * args.nics_per_aswitch):
                group_num = group_num + 1
                group_account = 0
        ind_psw = 0
        pod_ind = 0
        pod_account = 0
        if dsw_switch_num == 0:
            for i in asw_switch: # asw - psw
                for j in psw_switch:
                    line = str(i) + " " + str(j) +" "+ args.ap_bandwidth+" " + args.latency+" " +args.error_rate
                    f.write(line)
                    f.write('\n')
        else:
            for i in asw_switch: 
                for j in range(args.asw_per_psw):
                    line = str(i)+" "+str(psw_switch[pod_ind*args.asw_per_psw+j])+" "+args.ap_bandwidth+" "+args.latency+" "+args.error_rate
                    f.write(line)
                    f.write('\n')
                pod_account = pod_account +1

                if pod_account == args.asw_per_psw:
                    pod_ind = pod_ind +1
                    pod_account = 0

            for k in range(len(dsw_switch)):
                for q in range(pod_num):
                    line = str(dsw_switch[k])+" "+str(psw_switch[q * args.asw_per_psw + int(k/pod_num)])+" "+args.pd_bandwidth+" "+args.latency+" "+args.error_rate
                    f.write(line)
                    f.write('\n')
                                      
def gen_DualToR_SinglePlane(args):
    print("DualToR_SinglePlane")
    print(args.asw_per_psw)
    nodes_per_asw = args.nics_per_aswitch
    asw_switch_num_per = args.gpu_per_server*2
    if(args.gpu % (nodes_per_asw * (asw_switch_num_per/2)) == 0):
        segment_num = (int)(args.gpu / (nodes_per_asw * (asw_switch_num_per/2)))
    else:
        segment_num = (int)(args.gpu / (nodes_per_asw * (asw_switch_num_per/2)))+1
        
    if(segment_num != args.asw_switch_num / asw_switch_num_per):
        warnings.warn("Error relations between total GPU Nums and total aws_switch_num.\n \
                         The correct asw_switch_num is set to "+str(segment_num * asw_switch_num_per))
        args.asw_switch_num = segment_num * asw_switch_num_per
    print("asw_switch_num: " + str(args.asw_switch_num))
    nv_switch_num = (int)(args.gpu / args.gpu_per_server) * args.nv_switch_per_server
    
    if segment_num <= 1:
        # args.psw_switch_num = 0
        dsw_switch_num = 0
        pod_num = 1
        print("psw_switch_num: " + str(args.psw_switch_num))
    elif 1< segment_num <= int(args.asw_per_psw /  (asw_switch_num_per /2)):
        pod_num = 1
        # if pod_num != int(args.psw_switch_num / int(args.asw_per_psw/2)):
        #     raise ValueError("Error relations between total GPU Nums and total pws_switch_num with asw_per_psw.")
        print("psw_switch_num: " + str(args.psw_switch_num))
        dsw_switch_num = 0
    else:  #More Than one Pod
        raise ValueError("Number of GPUs exceed one Pod") 
    print("Creating Topology of totally " + str(segment_num) + " segment(s), totally "+ str(pod_num) + " pod(s)." )  
        
    nodes = (int) (args.gpu + args.asw_switch_num + args.psw_switch_num + nv_switch_num + dsw_switch_num) # 
    servers = args.gpu / args.gpu_per_server
    switch_nodes = (int)(args.psw_switch_num + args.asw_switch_num + nv_switch_num + dsw_switch_num) # 
    links = (int)(int(args.psw_switch_num/pod_num) * args.asw_switch_num + servers * asw_switch_num_per + servers * args.nv_switch_per_server * args.gpu_per_server \
                  + dsw_switch_num * pod_num) # 
    # print(args.psw_switch_num/pod_num * args.asw_switch_num)
    # print(args.psw_switch_num/pod_num * args.asw_switch_num + servers * asw_switch_num_per+ servers * args.nv_switch_per_server * args.gpu_per_server)
    file_name = "AlibabaHPN_"+str(args.gpu)+"g_"+str(args.gpu_per_server)+"gps_SinglePlane_"+args.bandwidth+"_"+args.gpu_type
    with open(file_name, 'w') as f:
        print(file_name)
        first_line = str(nodes)+" "+str(args.gpu_per_server)+" "+str(nv_switch_num)+" "+str(switch_nodes-nv_switch_num)+" "+str(int(links))+" "+str(args.gpu_type)
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
            elif len(asw_switch_1) < args.asw_switch_num / 2:
                asw_switch_1.append(i)
            elif len(asw_switch_2) < args.asw_switch_num / 2:
                asw_switch_2.append(i)
            elif len(psw_switch) < args.psw_switch_num:
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
        for i in range(args.gpu):
            curr_node = curr_node + 1
            if curr_node > args.gpu_per_server:
                curr_node = 1
                ind_nv = ind_nv + args.nv_switch_per_server
            for j in range(0, args.nv_switch_per_server):
                #cnt += 1
                line = str(i)+" "+str(nv_switch[ind_nv+j])+" "+args.nvlink_bw+" "+args.nv_latency+" "+args.error_rate
                f.write(line)
                f.write('\n')
            line = str(i)+" "+str(asw_switch_1[group_num*int(asw_switch_num_per/2)+ind_asw])+" "+args.bandwidth+" "+args.latency+" "+args.error_rate
            f.write(line)
            f.write('\n')
            
            line = str(i)+" "+str(asw_switch_2[group_num*int(asw_switch_num_per/2)+ind_asw])+" "+args.bandwidth+" "+args.latency+" "+args.error_rate
            f.write(line)
            f.write('\n')
            ind_asw = ind_asw + 1
            group_account = group_account + 1
            
            if ind_asw == int(asw_switch_num_per/2):
                ind_asw = 0
            if group_account == (args.gpu_per_server * args.nics_per_aswitch):
                group_num = group_num + 1
                group_account = 0

        ind_psw = 0
        pod_ind = 0
        pod_account = 0
        if dsw_switch_num == 0:
            for i in asw_switch_1: # asw - psw
                for j in psw_switch:
                    line = str(i) + " " + str(j) +" "+ args.ap_bandwidth+" " + args.latency+" " +args.error_rate
                    f.write(line)
                    f.write('\n') 
            for i in asw_switch_2: # asw - psw
                for j in psw_switch:
                    line = str(i) + " " + str(j) +" "+ args.ap_bandwidth+" " + args.latency+" " +args.error_rate
                    f.write(line)
                    f.write('\n')

        else:
            for i in asw_switch_1: 
                # print("print:"+str(int(args.asw_per_psw/2)))
                for j in range(int(args.asw_per_psw/2)):
                    # print("This is "+ str(j))
                    line = str(i)+" "+str(psw_switch[pod_ind*int(args.asw_per_psw/2)+j])+" "+args.ap_bandwidth+" "+args.latency+" "+args.error_rate
                    f.write(line)
                    f.write('\n')
                pod_account = pod_account +1

                if pod_account == args.asw_per_psw/2:
                    pod_ind = pod_ind +1
                    pod_account = 0

            for i in asw_switch_2: 
                pod_ind = 0
                for j in range(int(args.asw_per_psw/2)):
                    line = str(i)+" "+str(psw_switch[pod_ind*int(args.asw_per_psw/2)+j])+" "+args.ap_bandwidth+" "+args.latency+" "+args.error_rate
                    f.write(line)
                    f.write('\n')
                pod_account = pod_account +1

                if pod_account == args.asw_per_psw/2:
                    pod_ind = pod_ind +1
                    pod_account = 0

def gen_DualToR_DualPlane(args):
    print("DualToR_DualPlane")
    print(args.asw_per_psw)
    nodes_per_asw = args.nics_per_aswitch
    asw_switch_num_per = args.gpu_per_server*2
    if(args.gpu % (nodes_per_asw * (asw_switch_num_per/2)) == 0):
        segment_num = (int)(args.gpu / (nodes_per_asw * (asw_switch_num_per/2)))
    else:
        segment_num = (int)(args.gpu / (nodes_per_asw * (asw_switch_num_per/2)))+1
        
    if(segment_num != args.asw_switch_num / asw_switch_num_per):
        warnings.warn("Error relations between total GPU Nums and total aws_switch_num.\n \
                         The correct asw_switch_num is set to "+str(segment_num * asw_switch_num_per))
        args.asw_switch_num = segment_num * asw_switch_num_per
    print("asw_switch_num: " + str(args.asw_switch_num))
    nv_switch_num = (int)(args.gpu / args.gpu_per_server) * args.nv_switch_per_server
    
    if segment_num <= 1:
        # args.psw_switch_num = 0
        dsw_switch_num = 0
        pod_num = 1
        print("psw_switch_num: " + str(args.psw_switch_num))
    elif 1< segment_num <= int(args.asw_per_psw /  (asw_switch_num_per /2)):
        pod_num = 1
        print("psw_switch_num: " + str(args.psw_switch_num))
        dsw_switch_num = 0
    else:  #More Than one Pod
        raise ValueError("Number of GPUs exceed one Pod") 
    print("Creating Topology of totally " + str(segment_num) + " segment(s), totally "+ str(pod_num) + " pod(s)." )  
        
    nodes = (int) (args.gpu + args.asw_switch_num + args.psw_switch_num + nv_switch_num + dsw_switch_num) # 
    servers = args.gpu / args.gpu_per_server
    switch_nodes = (int)(args.psw_switch_num + args.asw_switch_num + nv_switch_num + dsw_switch_num) # 
    links = (int)(int(args.psw_switch_num/pod_num/2) * args.asw_switch_num + servers * asw_switch_num_per + servers * args.nv_switch_per_server * args.gpu_per_server \
                  + dsw_switch_num * pod_num) # 
    file_name = "AlibabaHPN_"+str(args.gpu)+"g_"+str(args.gpu_per_server)+"gps_DualPlane_"+args.bandwidth+"_"+args.gpu_type
    with open(file_name, 'w') as f:
        print(file_name)
        first_line = str(nodes)+" "+str(args.gpu_per_server)+" "+str(nv_switch_num)+" "+str(switch_nodes-nv_switch_num)+" "+str(int(links))+" "+str(args.gpu_type)
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
            elif len(asw_switch_1) < int(args.asw_switch_num / 2):
                asw_switch_1.append(i)
            elif len(asw_switch_2) < int(args.asw_switch_num / 2):
                asw_switch_2.append(i)
            elif len(psw_switch_1) < int(args.psw_switch_num / 2):
                psw_switch_1.append(i)
            elif len(psw_switch_2) < int(args.psw_switch_num / 2):
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
        for i in range(args.gpu):
            curr_node = curr_node + 1
            if curr_node > args.gpu_per_server:
                curr_node = 1
                ind_nv = ind_nv + args.nv_switch_per_server
            for j in range(0, args.nv_switch_per_server):
                #cnt += 1
                line = str(i)+" "+str(nv_switch[ind_nv+j])+" "+args.nvlink_bw+" "+args.nv_latency+" "+args.error_rate
                f.write(line)
                f.write('\n')
            line = str(i)+" "+str(asw_switch_1[group_num*int(asw_switch_num_per/2)+ind_asw])+" "+args.bandwidth+" "+args.latency+" "+args.error_rate
            f.write(line)
            f.write('\n')
            
            line = str(i)+" "+str(asw_switch_2[group_num*int(asw_switch_num_per/2)+ind_asw])+" "+args.bandwidth+" "+args.latency+" "+args.error_rate
            f.write(line)
            f.write('\n')
            ind_asw = ind_asw + 1
            group_account = group_account + 1
            
            if ind_asw == int(asw_switch_num_per/2):
                ind_asw = 0
            if group_account == (args.gpu_per_server * args.nics_per_aswitch):
                group_num = group_num + 1
                group_account = 0

        ind_psw = 0
        pod_ind = 0
        pod_account = 0
        if dsw_switch_num == 0:
            for i in asw_switch_1: # asw - psw
                for j in psw_switch_1:
                    line = str(i) + " " + str(j) +" "+ args.ap_bandwidth+" " + args.latency+" " +args.error_rate
                    f.write(line)
                    f.write('\n')

            for i in asw_switch_2: # asw - psw
                for j in psw_switch_2:
                    line = str(i) + " " + str(j) +" "+ args.ap_bandwidth+" " + args.latency+" " +args.error_rate
                    f.write(line)
                    f.write('\n')
        else:
            for i in asw_switch_1: 
                # print("print:"+str(int(args.asw_per_psw/2)))
                for j in range(int(args.asw_per_psw/2)):
                    # print("This is "+ str(j))
                    line = str(i)+" "+str(psw_switch_1[pod_ind*int(args.asw_per_psw/2)+j])+" "+args.ap_bandwidth+" "+args.latency+" "+args.error_rate
                    f.write(line)
                    f.write('\n')
                pod_account = pod_account +1
                # print("This is j: " + str(j) +"This is pod_account: " + str(pod_account))
                if pod_account == args.asw_per_psw:
                    pod_ind = pod_ind +1
                    pod_account = 0
            for i in asw_switch_2: 
                pod_ind = 0
                for j in range(int(args.asw_per_psw/2)):
                    line = str(i)+" "+str(psw_switch_2[pod_ind*int(args.asw_per_psw/2)+j])+" "+args.ap_bandwidth+" "+args.latency+" "+args.error_rate
                    f.write(line)
                    f.write('\n')
                pod_account = pod_account +1
                if pod_account == args.asw_per_psw:
                    pod_ind = pod_ind +1
                    pod_account = 0

def DCN_DualToR(args):
    print("DCN+DualToR")
    nodes_per_asw = args.nics_per_aswitch
    asw_switch_num_per = 2
    if(args.gpu % (nodes_per_asw) == 0):
        segment_num = (int)(args.gpu / (nodes_per_asw * (asw_switch_num_per/2)))
    else:
        segment_num = (int)(args.gpu / (nodes_per_asw * (asw_switch_num_per/2)))+1
        
    if(segment_num != args.asw_switch_num / asw_switch_num_per):
        warnings.warn("Error relations between total GPU Nums and total aws_switch_num.\n \
                         The correct asw_switch_num is set to "+str(segment_num * asw_switch_num_per))
        args.asw_switch_num = segment_num * asw_switch_num_per
    print("asw_switch_num: " + str(args.asw_switch_num))
    nv_switch_num = (int)(args.gpu / args.gpu_per_server) * args.nv_switch_per_server
    
    if segment_num <= 1:
        # args.psw_switch_num = 0
        dsw_switch_num = 0
        pod_num = 1
        print("psw_switch_num: " + str(args.psw_switch_num))
    elif 1< segment_num <= int(args.asw_per_psw / asw_switch_num_per):
        pod_num = 1
        # if pod_num != int(args.psw_switch_num / args.asw_per_psw):
        #     raise ValueError("Error relations between total GPU Nums and total pws_switch_num with asw per psw.")
        print("psw_switch_num: " + str(args.psw_switch_num))
        dsw_switch_num = 0
    else:  #More Than one Pod
        raise ValueError("Number of GPUs exceed one Pod") 
    print("Creating Topology of totally " + str(segment_num) + " segment(s), totally "+ str(pod_num) + " pod(s)." )  
        
    nodes = (int) (args.gpu + args.asw_switch_num + args.psw_switch_num + nv_switch_num + dsw_switch_num) # 
    servers = args.gpu / args.gpu_per_server
    switch_nodes = (int)(args.psw_switch_num + args.asw_switch_num + nv_switch_num + dsw_switch_num) # 
    links = (int)(int(args.psw_switch_num/pod_num) * args.asw_switch_num + servers * args.gpu_per_server * 2 + servers * args.nv_switch_per_server * args.gpu_per_server \
                  + dsw_switch_num * pod_num) # 
    print(links)
    file_name = "DCN+_"+str(args.gpu)+"g_"+str(args.gpu_per_server)+"gps_"+args.bandwidth+"_"+args.gpu_type
    with open(file_name, 'w') as f:
        print(file_name)
        first_line = str(nodes)+" "+str(args.gpu_per_server)+" "+str(nv_switch_num)+" "+str(switch_nodes-nv_switch_num)+" "+str(int(links))+" "+str(args.gpu_type)
        f.write(first_line)
        f.write('\n')
        nv_switch = []
        asw_switch_1 = []
        asw_switch_2 = []
        psw_switch = []
        dsw_switch = []
        sec_line = ""
        nnodes = nodes - switch_nodes
        nnodes = nodes - switch_nodes
        for i in range(nnodes, nodes):
            sec_line = sec_line + str(i) + " "
            if len(nv_switch) < nv_switch_num:
                nv_switch.append(i)
            elif len(asw_switch_1) < args.asw_switch_num / 2:
                asw_switch_1.append(i)
            elif len(asw_switch_2) < args.asw_switch_num / 2:
                asw_switch_2.append(i)
            elif len(psw_switch) < args.psw_switch_num:
                psw_switch.append(i)
            else:
                dsw_switch.append(i)
        f.write(sec_line)
        f.write('\n')
        ind_asw = 0
        curr_node = 0
        group_num = 0
        group_account = 0
        asw_node = 0
        ind_nv = 0
        for i in range(args.gpu):
            curr_node = curr_node + 1
            asw_node = asw_node + 1
            if curr_node > args.gpu_per_server:
                curr_node = 1
                ind_nv = ind_nv + args.nv_switch_per_server
            for j in range(0, args.nv_switch_per_server):
                #cnt += 1
                line = str(i)+" "+str(nv_switch[ind_nv+j])+" "+args.nvlink_bw+" "+args.nv_latency+" "+args.error_rate
                f.write(line)
                f.write('\n')
            
            line = str(i)+" "+str(asw_switch_1[group_num*int(asw_switch_num_per/2)+ind_asw])+" "+args.bandwidth+" "+args.latency+" "+args.error_rate
            f.write(line)
            f.write('\n')
            
            line = str(i)+" "+str(asw_switch_2[group_num*int(asw_switch_num_per/2)+ind_asw])+" "+args.bandwidth+" "+args.latency+" "+args.error_rate
            f.write(line)
            f.write('\n')
            group_account = group_account + 1
            
            if asw_node == nodes_per_asw:
                ind_asw = ind_asw+1
                asw_node = 0
            if group_account == (args.gpu_per_server * args.nics_per_aswitch):
                group_num = group_num + 1
                group_account = 0
        ind_psw = 0
        pod_ind = 0
        pod_account = 0
        if dsw_switch_num == 0:
            for i in asw_switch_1: # asw - psw
                for j in psw_switch:
                    line = str(i) + " " + str(j) +" "+ args.ap_bandwidth+" " + args.latency+" " +args.error_rate
                    f.write(line)
                    f.write('\n') 
            for i in asw_switch_2: # asw - psw
                for j in psw_switch:
                    line = str(i) + " " + str(j) +" "+ args.ap_bandwidth+" " + args.latency+" " +args.error_rate
                    f.write(line)
                    f.write('\n')
        else:
            for i in asw_switch_1: 
                # print("print:"+str(int(args.asw_per_psw/2)))
                for j in range(args.asw_per_psw):
                    # print("This is "+ str(j))
                    line = str(i)+" "+str(psw_switch[pod_ind*int(args.asw_per_psw)+j])+" "+args.ap_bandwidth+" "+args.latency+" "+args.error_rate
                    f.write(line)
                    f.write('\n')
                pod_account = pod_account +1
                # print("This is j: " + str(j) +"This is pod_account: " + str(pod_account))

                if pod_account == args.asw_per_psw:
                    pod_ind = pod_ind +1
                    pod_account = 0
                # print("Pod Index:" + str(pod_ind))
            for i in asw_switch_2: 
                pod_ind = 0
                for j in range(int(args.asw_per_psw)):
                    line = str(i)+" "+str(psw_switch[pod_ind*int(args.asw_per_psw)+j])+" "+args.ap_bandwidth+" "+args.latency+" "+args.error_rate
                    f.write(line)
                    f.write('\n')
                pod_account = pod_account +1

                if pod_account == args.asw_per_psw:
                    pod_ind = pod_ind +1
                    pod_account = 0
                    
def DCN_SingleToR(args):
    print("DCN+SingleToR")
    nodes_per_asw = args.nics_per_aswitch
    asw_switch_num_per = 1
    if(args.gpu % (nodes_per_asw) == 0):
        segment_num = (int)(args.gpu / (nodes_per_asw * asw_switch_num_per))
    else:
        segment_num = (int)(args.gpu / (nodes_per_asw * asw_switch_num_per))+1
        
    if(segment_num != args.asw_switch_num / asw_switch_num_per):
        warnings.warn("Error relations between total GPU Nums and total aws_switch_num.\n \
                         The correct asw_switch_num is set to "+str(segment_num * asw_switch_num_per))
        args.asw_switch_num = segment_num * asw_switch_num_per
    print("asw_switch_num: " + str(args.asw_switch_num))
    nv_switch_num = (int)(args.gpu / args.gpu_per_server) * args.nv_switch_per_server
    
    if segment_num <= 1:
        args.psw_switch_num = 0
        dsw_switch_num = 0
        pod_num = 1
    elif 1< segment_num <= int(args.asw_per_psw / asw_switch_num_per):
        pod_num = 1
        # if pod_num != int(args.psw_switch_num / args.asw_per_psw):
        #     raise ValueError("Error relations between total GPU Nums and total pws_switch_num with asw per psw.")
        print("psw_switch_num: " + str(args.psw_switch_num))
        dsw_switch_num = 0
    else:  #More Than one Pod
        raise ValueError("Number of GPUs exceed one Pod") 
    print("Creating Topology of totally " + str(segment_num) + " segment(s), totally "+ str(pod_num) + " pod(s)." )  
        
    nodes = (int) (args.gpu + args.asw_switch_num + args.psw_switch_num + nv_switch_num + dsw_switch_num) # 
    servers = args.gpu / args.gpu_per_server
    switch_nodes = (int)(args.psw_switch_num + args.asw_switch_num + nv_switch_num + dsw_switch_num) # 
    links = (int)(int(args.psw_switch_num/pod_num) * args.asw_switch_num + servers * args.gpu_per_server + servers * args.nv_switch_per_server * args.gpu_per_server \
                  + dsw_switch_num * pod_num) # 
    file_name = "DCN+SingleToR_"+str(args.gpu)+"g_"+str(args.gpu_per_server)+"gps_"+args.bandwidth+"_"+args.gpu_type
    with open(file_name, 'w') as f:
        print(file_name)
        first_line = str(nodes)+" "+str(args.gpu_per_server)+" "+str(nv_switch_num)+" "+str(switch_nodes-nv_switch_num)+" "+str(int(links))+" "+str(args.gpu_type)
        f.write(first_line)
        f.write('\n')
        nv_switch = []
        asw_switch = []
        psw_switch = []
        dsw_switch = []
        sec_line = ""
        nnodes = nodes - switch_nodes
        nnodes = nodes - switch_nodes
        for i in range(nnodes, nodes):
            sec_line = sec_line + str(i) + " "
            if len(nv_switch) < nv_switch_num:
                nv_switch.append(i)
            elif len(asw_switch) < args.asw_switch_num:
                asw_switch.append(i)
            elif len(psw_switch) < args.psw_switch_num:
                psw_switch.append(i)
            else:
                dsw_switch.append(i)
        f.write(sec_line)
        f.write('\n')
        ind_asw = 0
        curr_node = 0
        group_num = 0
        group_account = 0
        asw_node = 0
        ind_nv = 0
        for i in range(args.gpu):
            curr_node = curr_node + 1
            asw_node = asw_node + 1
            if curr_node > args.gpu_per_server:
                curr_node = 1
                ind_nv = ind_nv + args.nv_switch_per_server
            for j in range(0, args.nv_switch_per_server):
                #cnt += 1
                line = str(i)+" "+str(nv_switch[ind_nv+j])+" "+args.nvlink_bw+" "+args.nv_latency+" "+args.error_rate
                f.write(line)
                f.write('\n')
            
            line = str(i)+" "+str(asw_switch[group_num*asw_switch_num_per+ind_asw])+" "+args.bandwidth+" "+args.latency+" "+args.error_rate
            f.write(line)
            f.write('\n')
            
            group_account = group_account + 1
            
            if asw_node == nodes_per_asw:
                ind_asw = ind_asw+1
                asw_node = 0
            if group_account == (args.gpu_per_server * args.nics_per_aswitch):
                group_num = group_num + 1
                group_account = 0
        ind_psw = 0
        pod_ind = 0
        pod_account = 0
        if dsw_switch_num == 0:
            for i in asw_switch: # asw - psw
                for j in psw_switch:
                    line = str(i) + " " + str(j) +" "+ args.ap_bandwidth+" " + args.latency+" " +args.error_rate
                    f.write(line)
                    f.write('\n')
        else:
            for i in asw_switch: 
                # print("print:"+str(int(args.asw_per_psw/2)))
                for j in range(args.asw_per_psw):
                    # print("This is "+ str(j))
                    line = str(i)+" "+str(psw_switch[pod_ind*int(args.asw_per_psw)+j])+" "+args.ap_bandwidth+" "+args.latency+" "+args.error_rate
                    f.write(line)
                    f.write('\n')
                pod_account = pod_account +1
                # print("This is j: " + str(j) +"This is pod_account: " + str(pod_account))

                if pod_account == args.asw_per_psw:
                    pod_ind = pod_ind +1
                    pod_account = 0
                # print("Pod Index:" + str(pod_ind))

def main():
    parser = argparse.ArgumentParser(description='Python script for generate the AlibabaHPN network topo')
     #Whole Structure Parameters:
    parser.add_argument('-er','--error_rate',type=str,default='0',help='error_rate,default 0')
    parser.add_argument('--dp', action='store_true', help='enable dual_plane, default single plane')
    parser.add_argument('--st', action='store_true', help='enable DCN architecture , default AlibabaHPN')
    parser.add_argument('-g','--gpu',type=int,default=0,help='gpus num,default 0')
    parser.add_argument('-topo','--topology', type=str, default='',help='Template for AlibabaHPN, Spectrum-X, DCN+ \
                        Pay Attention that only gpu num and dp for AlibabaHPN are enable, other parameters are invalid.')
    parser.add_argument('-dualToR','--enableDualToR', type=int, default=0, help="Whether enable dual ToR, default with false for Spectrum")

    #Intra-Host Parameters:
    parser.add_argument('-gps','--gpu_per_server',type=int,default=8,help='gpu_per_server,default 8')
    parser.add_argument('-gt','--gpu_type',type=str,default='H100',help='gpu_type,default H100')
    parser.add_argument('-nsps','--nv_switch_per_server',type=int,default=1,help='nv_switch_per_server,default 1')
    parser.add_argument('-nvbw','--nvlink_bw',type=str,default='2880Gbps',help='nvlink_bw,default 2880Gbps')
    parser.add_argument('-nl','--nv_latency',type=str,default='0.000025ms',help='nv switch latency,default 0.000025ms')
    parser.add_argument('-l','--latency',type=str,default='0.0005ms',help='nic latency,default 0.0005ms')
    #Intra-Segment Parameters:
    parser.add_argument('-bw','--bandwidth',type=str,default='100Gbps',help='nic to asw bandwidth,default 100Gbps')
    parser.add_argument('-asn','--asw_switch_num',type=int,default=8,help='asw_switch_num,default 8')
    parser.add_argument('-npa','--nics_per_aswitch',type=int,default=64,help='nnics per asw,default 64') #Downstream port num of ASW
    #Intra-Pod Parameters:
    parser.add_argument('-psn','--psw_switch_num',type=int,default=64,help='psw_switch_num,default for Specrrum-X 64')
    parser.add_argument('-apbw','--ap_bandwidth',type=str,default='400Gbps',help='asw to psw bandwidth,default 400Gbps')
    parser.add_argument('-app','--asw_per_psw',type=int,default=64,help='asw for psw,default for default for Specrrum-X 64')
    #Inter-Pod Parameters:
    parser.add_argument('-dsn','--dsw_switch_num',type=int,default=0,help='dsw_switch_num, default for Specrrum-X 0')
    parser.add_argument('-pdbw','--pd_bandwidth',type=str,default='400Gbps',help='psw to dsw bandwidth,default 400Gbps')
    parser.add_argument('-ppd','--psw_per_dsw',type=int,default=64,help='psw for dsw, default for Specrrum-X 64')
    args = parser.parse_args()

    topo_template(args)

    if args.gpu == 0:
        raise ValueError("Please enter GPU Num or Template Name")
    if args.st: #Not use rail-optimized:
        if args.enableDualToR == 1:
            DCN_DualToR(args)
        else:
            DCN_SingleToR(args)
    elif args.dp:
        if args.enableDualToR == 1:
            gen_DualToR_DualPlane(args)
        else:
            gen_DualToR_SinglePlane(args)
    else:
        gen_SingleToR(args)

def topo_template(args):
    if args.topology == 'AlibabaHPN':
        AlibabaHPNTem(args)
    elif args.topology == 'Spectrum-X':
        SpectrumXTem(args)
    elif args.topology == 'DCN+':
        if args.enableDualToR == 1:
            DCNDualToR(args)
        else:
            DCNSingleToR(args)
    else:
        print('No Template is used.')

def DCNSingleToR(args):
    print("SingleDCN")
    args.st = True
    args.enableDualToR = 0
    args.gpu_per_server = 8
    args.gpu_type = 'H100'
    args.nv_switch_per_server = 1
    args.nvlink_bw = '2880Gbps'
    args.nv_latency = '0.000025ms'
    args.latency = '0.0005ms'
    args.bandwidth = '400Gbps'

    if args.gpu == 0:
        print("Complete DCN+ with 512 GPU in one Pod")
        args.gpu = 512
    args.nics_per_aswitch = 128
    segment_num_gpu = args.nics_per_aswitch
    if(args.gpu % segment_num_gpu == 0):
        segment_num = int(args.gpu / segment_num_gpu)
    else:
        segment_num = int(args.gpu / segment_num_gpu) + 1
       
    args.asw_switch_num  = segment_num

    if(segment_num % 4 == 0):
        pod_num = int(segment_num /4)
    else:
        pod_num = int(segment_num / 4 + 1)

    args.ap_bandwidth = '400Gbps'
        
    if pod_num > 1:
        args.psw_per_dsw = 128
        if pod_num > args.psw_per_dsw:
            raise ValueError("Number of GPU exceeds the capacity of DCN+")
        print("Creating DCN+ with " + pod_num + "pods.")
        args.pd_bandwidth = '400Gbps'
        args.psw_switch_num = 4 * pod_num
        args.asw_per_psw = 4
        args.dsw_switch_num = 4
    else:
        args.dsw_switch_num = 0
        args.psw_switch_num = 4 * pod_num
        args.asw_per_psw = 4
        args.dsw_switch_num = 4


def DCNDualToR(args):
    args.st = True
    args.enableDualToR = 1
    args.gpu_per_server = 8
    args.gpu_type = 'H100'
    args.nv_switch_per_server = 1
    args.nvlink_bw = '2880Gbps'
    args.nv_latency = '0.000025ms'
    args.latency = '0.0005ms'

    args.bandwidth = '200Gbps'
    if args.gpu == 0:
        print("Complete DCN+ with 512 GPU in one Pod")
        args.gpu = 512
    args.nics_per_aswitch = 64
    segment_num_gpu = args.nics_per_aswitch
    if(args.gpu % segment_num_gpu == 0):
        segment_num = int(args.gpu / segment_num_gpu)
    else:
        segment_num = int(args.gpu / segment_num_gpu) + 1
       
    args.asw_switch_num  = segment_num * 2


    if(segment_num % 4 == 0):
        pod_num = int(segment_num /4)
    else:
        pod_num = int(segment_num / 4 + 1)

    args.ap_bandwidth = '400Gbps'
        
    if pod_num > 1:
        args.psw_per_dsw = 128
        if pod_num > args.psw_per_dsw:
            raise ValueError("Number of GPU exceeds the capacity of DCN+")
        print("Creating DCN+ with " + pod_num + "pods.")
        args.pd_bandwidth = '400Gbps'
        args.psw_switch_num = 8 * pod_num
        args.asw_per_psw = 8
        args.dsw_switch_num = 8
    else:
        args.dsw_switch_num = 0
        args.psw_switch_num = 8 * pod_num
        args.asw_per_psw = 8
        args.dsw_switch_num = 8

def AlibabaHPNTem(args):
    #Dual-ToR for both single plane and dual plane
    args.enableDualToR = True
    args.gpu_per_server = 8
    args.gpu_type = 'H100'
    args.nv_switch_per_server = 1
    args.nvlink_bw = '2880Gbps'
    args.nv_latency = '0.000025ms'
    args.latency = '0.0005ms'

    args.bandwidth = '200Gbps'
    if args.gpu == 0:
        print("Complete AlibabaHPN with 15K GPU in one Pod")
        args.gpu = 15360
    args.nics_per_aswitch = 128
    segment_num_gpu = args.nics_per_aswitch * 8
    if(args.gpu % segment_num_gpu == 0):
        segment_num = int(args.gpu / segment_num_gpu)
    else:
        segment_num = int(args.gpu / segment_num_gpu) + 1
       
    args.asw_switch_num  = segment_num * 16

    if(segment_num % 15 == 0):
        pod_num = int(segment_num /15)
    else:
        pod_num = int(segment_num / 15 + 1)

    args.ap_bandwidth = '400Gbps'
        
    if pod_num > 1:
        args.psw_per_dsw = 128
        if pod_num > args.psw_per_dsw:
            raise ValueError("Number of GPU exceeds the capacity of AlibabaHPN")
        print("Creating AlibabaHPN with " + pod_num + "pods.")
        args.pd_bandwidth = '400Gbps'
        
        
        if args.dp:
            args.psw_switch_num = 120 * pod_num
            args.asw_per_psw = 120
            args.dsw_switch_num = 32  
        else:
            args.psw_switch_num = 120 * pod_num
            args.asw_per_psw = 240
            args.dsw_switch_num = 64
    else:
        args.dsw_switch_num = 0
        if args.dp:
            args.psw_switch_num = 120 * pod_num
            args.asw_per_psw = 120
        else:
            args.psw_switch_num = 120 * pod_num
            args.asw_per_psw = 240

def SpectrumXTem(args):
    #Spectrum only support single ToR with single Plane
    args.enableDualToR = 0
    args.dp = False
    args.gpu_type = 'H100'
    args.nv_switch_per_server = 1
    args.nvlink_bw = '2880Gbps'
    args.nv_latency = '0.000025ms'
    args.latency = '0.0005ms'
    args.bandwidth = '400Gbps'
    if args.gpu == 0:
        print("Complete Spectrum-X with 4096 GPU in one Pod")
        args.gpu = 4096
    args.nics_per_aswitch = 64
    segment_num_gpu = args.nics_per_aswitch * 8
    if(args.gpu % segment_num_gpu == 0):
        segment_num = int(args.gpu / segment_num_gpu)
    else:
        segment_num = int(args.gpu / segment_num_gpu) + 1
    args.asw_switch_num  = segment_num * 8
    if(segment_num % 8 == 0):
        pod_num = int(segment_num /8)
    else:
        pod_num = int(segment_num / 8 + 1)

    args.ap_bandwidth = '400Gbps'
    args.psw_switch_num = 64 
    args.asw_per_psw = 64
    args.dsw_switch_num = 0 
    if pod_num > 1:
        args.psw_per_dsw = 64
        if pod_num > args.psw_per_dsw:
            raise ValueError("Number of GPU exceeds the capacity of Spectrum-X")
        print("Creating SpectrumX with " + str(pod_num) + "pods.")
        args.pd_bandwidth = '400Gbps'
        
        args.psw_switch_num = 64 * pod_num
        
        args.asw_per_psw = 64
        args.dsw_switch_num = 64 * pod_num  


if __name__ =='__main__':
    main()
    