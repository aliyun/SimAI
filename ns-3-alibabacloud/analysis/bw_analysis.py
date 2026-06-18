import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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

node_id_scale = all_data['node_id'].value_counts().sort_index()
node_start_id = node_id_scale.index[0]

def get_data(node_id : int):
    data = all_data[all_data['node_id'] == node_id]
    return data

def get_bw_list(tx_bytes_diff, interval):
    bw_list = [x*1e9/interval for x in tx_bytes_diff] # B/s
    bw_list = [x*8 / 1e9 for x in bw_list] # Gbps
    # bw_list = map(lambda x : x*1e9/interval, tx_bytes_diff)
    return bw_list

# this function is used for monitor recording tx_bytes. 
def get_fig_for_node(node_id):
    # get data of node
    data = get_data(int(node_id))
    # group data by port
    groups = data.groupby('port_id')
    port_bw = {}
    # plot for every port in node
    fig = px.line()
    lables=[]
    for key, df in groups:
        lables.append('port-'+ str(key))
        df = df.reset_index(drop=True)
        interval = df['time'][1]-df['time'][0] # ns
        df['time'] = df['time'].map(lambda x : x/1e6)

        tx_list = list(df['tx_bytes'])
        nxt_tx = list(df['tx_bytes'])
        nxt_tx.pop(0)
        nxt_tx.append(tx_list[-1])
        diff = [nxt_tx[i] - tx_list[i] for i in range(len(tx_list))]
        bw_list = get_bw_list(diff, interval)
        df['bw'] = pd.DataFrame(bw_list)
        fig.add_scatter(x=df['time'],y=df['bw'], name='port-'+str(key))
    fig.update_layout(title='Node-'+str(node_id)+'-throughput', xaxis_title='time(ms)', yaxis_title='Throughput(Gbps)')
    offline.plot(fig, filename=curr_work_path+'/../simulation/monitor_output/figs/'+file_name+'/bw_node_'+str(node_id)+'.html')
    # fig.show()

# this function is used for monitor recording bandwidth. 
def get_fig_for_node_bw(node_id):
    # get data of node
    data = get_data(int(node_id))
    # group data by port
    groups = data.groupby('port_id')
    port_bw = {}
    # plot for every port in node
    fig = px.line()
    lables=[]
    for key, df in groups:
        lables.append('port-'+ str(key))
        df = df.reset_index(drop=True)
        df['time'] = df['time'].map(lambda x : x/1e6) # ms
        fig.add_scatter(x=df['time'],y=df['bandwidth'], name='port-'+str(key))
    fig.update_layout(title='Node-'+str(node_id)+'-throughput', xaxis_title='time(ms)', yaxis_title='Throughput(Gbps)')
    offline.plot(fig, filename=curr_work_path+'/../simulation/monitor_output/figs/'+file_name+'/bw_node_'+str(node_id)+'.html')

if __name__ == "__main__":
     for i in range(len(node_id_scale)):
        print("get fig for node: "+str(i+node_start_id))
        get_fig_for_node_bw(i+node_start_id)

# if __name__ == "__main__":
#     # get data of node
#     data = get_data(int(node_id))
#     # group data by port
#     groups = data.groupby('port_id')
#     port_bw = {}
#     # plot for every port in node
#     plt.figure(figsize=(9,6))
#     lables=[]
#     for key, df in groups:
#         lables.append('port-'+ str(key))
#         df = df.reset_index(drop=True)

#         time = list(df['time'])
#         interval = time[1]-time[0] # ns
#         time = [t / 1e6 for t in time] # ms

#         tx_list = list(df['tx_bytes'])
#         nxt_tx = list(df['tx_bytes'])
#         nxt_tx.pop(0)
#         nxt_tx.append(tx_list[-1])
#         diff = [nxt_tx[i] - tx_list[i] for i in range(len(tx_list))]
#         bw_list = get_bw_list(diff, interval)
#         # print(nxt_tx)
#         # print(tx_list)
#         # print(diff)
#         # print(bw_list)

#         plt.plot(time, bw_list)

#     plt.xlabel("Time(ms)")
#     plt.ylabel("Througput(Gbps)")
#     plt.legend(lables)
#     plt.savefig("/home/xuemo.lc/AliLMN-Sim/figs/" + fig_name)
#     plt.show()