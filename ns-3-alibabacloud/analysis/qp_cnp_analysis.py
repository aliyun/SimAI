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

def get_data():
    groups = all_data.groupby(['src', 'dst', 'sport', 'dport', 'size'])
    return groups

def get_fig_for_qp():
    # get data group by <src, dst, sport, dport, size>
    groups = get_data()
    # plot for every qp
    fig = px.line()
    lables=[]
    for key, df in groups:
        lables.append('flow-'+ str(key))
        df = df.reset_index(drop=True)
        df['time'] = df['time'].map(lambda x : x/1e6) # ms
        fig.add_scatter(x=df['time'],y=df['cnp_number'], name='flow-'+str(key))
    fig.update_layout(title='Flow CNP Number', xaxis_title='time(ms)', yaxis_title='Flow CNP Number')
    offline.plot(fig, filename=curr_work_path+'/../simulation/monitor_output/figs/'+file_name+'/flow_cnp.html')
    # fig.show()

if __name__ == "__main__":
    get_fig_for_qp()