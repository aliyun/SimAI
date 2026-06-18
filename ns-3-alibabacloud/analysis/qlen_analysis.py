import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as offline
import plotly.graph_objects as go
import sys

curr_work_path = os.path.dirname(__file__)
file_name = sys.argv[1]
file_path = curr_work_path + "/../simulation/monitor_output/" + file_name + ".txt"
if os.path.exists(curr_work_path+'/../simulation/monitor_output/figs/'+file_name) == False:
    os.mkdir(curr_work_path+'/../simulation/monitor_output/figs/'+file_name)

all_data = pd.read_table(file_path, sep=', ', engine='python')

node_id_scale = all_data['sw_id'].value_counts().sort_index()
node_start_id = node_id_scale.index[0]

def get_data(node_id : int):
    data = all_data[all_data['sw_id'] == node_id]
    return data

def get_qlen_list(tx_bytes_diff, interval):
    bw_list = [x*1e9/interval for x in tx_bytes_diff] # B/s
    bw_list = [x*8 / 1e9 for x in bw_list] # Gbps
    return bw_list

def get_fig_for_node(node_id):
    # get data of node
    data = get_data(int(node_id))
    # group data by port
    groups = data.groupby('port_id')
    
    # plot for every port in node
    fig = px.line()
    lables=[]
    for key, df in groups:
        lables.append('port-'+ str(key))
        df = df.reset_index(drop=True)

        q_cnt = len(df['q_id'].value_counts())

        time = list(df['time'])
        time = time[0:len(time):q_cnt]
        time = [t / 1e6 for t in time] # ms

        qlen_list = list(df['port_len'])
        qlen_list = qlen_list[0:len(qlen_list):q_cnt] # Bytes
        qlen_list = [qlen / 1e3 for qlen in qlen_list] # KB
        fig.add_scatter(x=time, y=qlen_list, name='port-'+str(key))
    fig.update_layout(title='Switch-'+str(node_id)+'-qlen', xaxis_title='time(ms)', yaxis_title='QueueLength(KB)')
    offline.plot(fig, filename=curr_work_path+'/../simulation/monitor_output/figs/'+file_name+'/qlen_switch-'+str(node_id)+'.html')
    # fig.show()   
 
if __name__ == "__main__":
    for i in range(len(node_id_scale)):
        print("get fig for node: "+str(i+node_start_id))
        get_fig_for_node(i+node_start_id)


# if __name__ == "__main__":
#     # get data of node
#     data = get_data(int(node_id))
#     # group data by port
#     groups = data.groupby('port_id')
    
#     # plot for every port in node
#     plt.figure(figsize=(9,6))
#     lables=[]
#     for key, df in groups:
#         lables.append('port-'+ str(key))
#         df = df.reset_index(drop=True)

#         q_cnt = len(df['q_id'].value_counts())

#         time = list(df['time'])
#         time = time[0:len(time):q_cnt]
#         time = [t / 1e6 for t in time] # ms

#         qlen_list = list(df['port_len'])
#         qlen_list = qlen_list[0:len(qlen_list):q_cnt] # Bytes
#         qlen_list = [qlen / 1e3 for qlen in qlen_list] # KB
#         print(time)
#         print(qlen_list)

#         plt.plot(time, qlen_list)

#     plt.xlabel("Time(ms)")
#     plt.ylabel("QueueLength(KB)")
#     plt.legend(lables)
#     plt.savefig("/home/xuemo.lc/AliLMN-Sim/figs/" + fig_name)
#     plt.show()