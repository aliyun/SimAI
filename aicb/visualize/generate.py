
from log_analyzer.log import *
from pyecharts.charts import Pie,Scatter,Line,Graph,Bar
from pyecharts import options as opts
from typing import List, Dict
import random
import math
from collections import defaultdict
from jinja2 import Template
from pyecharts.commons.utils import JsCode
import json
import sys
import ast

colors_group = {
    "tp_group": "#376AB3",
    "dp_group": "#87C0CA",
    "ep_group": "#E8EDB9" ,
    "pp_group": "#8cc540" ,
    "all_reduce": "#376AB3",
    "broadcast": "#87C0CA",
    "all_gather": "#E8EDB9"  ,
    "reduce_scatter": "#8cc540",
    "all_to_all":"#009f5d", 
    "isend":   "#A9A9A9" ,
    "irecv":    "#FFD700"
}
def parse_msg_size(msg_size_str: str) -> float:
    try:
        result = ast.literal_eval(msg_size_str)
        if isinstance(result, tuple):
            return 0.0
    except (ValueError, SyntaxError):
        pass

    try:
        return float(msg_size_str)
    except ValueError:
        return 0.0

def custom_csv_reader(file_path: str, only_workload: bool):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)

        if only_workload:
            next(reader)
        
        for row in reader:
            combined_row = []
            temp = ""
            inside_tuple = False
            for item in row:
                if item.startswith("((") and not item.endswith("))"):
                    inside_tuple = True
                    temp += item
                elif inside_tuple:
                    temp += "," + item
                    if item.endswith("))"):
                        inside_tuple = False
                        combined_row.append(temp)
                        temp = ""
                else:
                    combined_row.append(item)

            yield dict(zip(header, combined_row))

def read_csv_and_structure_data(file_path: str, only_workload: bool) -> List[LogItem]:
    log_items = []
    for row in custom_csv_reader(file_path, only_workload):
        
        # parse msg_size
        msg_size = parse_msg_size(row['msg_size'])

        log_item = LogItem(
            comm_type=CommType[row['comm_type'].split('.')[-1]],
            comm_group=CommGroup[row['comm_group'].split('.')[-1]] if row['comm_group'] != 'None' else None,
            comm_group_size=int(row['comm_group_size']) if row['comm_group_size'] != 'None' else None,
            msg_size=msg_size,
            stage=row['stage'],
            dst=int(row['dst']) if row['dst'] != 'None' else None,
            src=int(row['src']) if row['src'] != 'None' else None,
            additional=row['additional'],
            _elapsed_time=None if only_workload else float(row['_elapsed_time']),
            algbw=None if only_workload else float(row['algbw']),
            busbw=None if only_workload else float(row['busbw']),
            count=float(row['count'])
        )
        log_items.append(log_item)

    return log_items

def split_data_by_epoch(is_comm: int,log_items: List[LogItem]) -> Dict[str, List[LogItem]]:
    data_by_epoch = {'init': []}
    epoch_count = 0
    current_epoch = 'init'

    for log_item in log_items:
        if is_comm and log_item.comm_type == CommType.computation:
            continue

        if log_item.is_epoch_end():
            data_by_epoch[current_epoch].append(log_item)
            current_epoch = f'epoch_{epoch_count}'
            data_by_epoch[current_epoch] = []
            epoch_count += 1
        else:
            data_by_epoch[current_epoch].append(log_item)
    if not data_by_epoch[current_epoch]:
        del data_by_epoch[current_epoch]
    return data_by_epoch
def count_by_epoch(data_by_epoch: Dict[str, List[LogItem]]) -> Dict[str, Dict[str, int]]:
    comm_type_counts = {}

    for epoch, log_items in data_by_epoch.items():
        if epoch == 'init':
            continue

        comm_type_counts[epoch] = {}
        for log_item in log_items:
            if log_item.comm_type == CommType.epoch_end:
                continue

            comm_type_str = log_item.comm_type.name
            if comm_type_str not in comm_type_counts[epoch]:
                comm_type_counts[epoch][comm_type_str] = 0
            comm_type_counts[epoch][comm_type_str] += 1

    return comm_type_counts
def extract_data_from_log_items(log_items: List[LogItem]) -> List:
    data = []
    count_dict = defaultdict(int)

    # Count the occurrences of each (comm_type, msg_size, group) combination
    for item in log_items:
        if item.stage != 'init' and item.comm_type != CommType.epoch_end:
            group = item.comm_group.name if item.comm_group else "unknown"
            if item.msg_size > 0:
                item.msg_size = item.msg_size  #B
                key = (item.comm_type.name, item.msg_size, group)
                count_dict[key] += 1

    # Construct the return data, including the busbw value
    for item in log_items:
        if item.stage != 'init' and item.comm_type != CommType.epoch_end:
            group = item.comm_group.name if item.comm_group else "unknown"
            if item.msg_size > 0:
                key = (item.comm_type.name, item.msg_size, group)
                count = count_dict[key]
                data.append((item.comm_type.name, item.msg_size, group, item.busbw, count))
    return data

def create_pie_chart_for_epoch(comm_type_counts: Dict[str, int]):
    pie = Pie()
    data = [(k, v) for k, v in comm_type_counts.items()]
    pie.add("", data)
    pie.set_global_opts(
        title_opts=opts.TitleOpts(title=f"CommType Counts"),
        legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical")
    )
    pie.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"))
    return pie

def create_scatter_chart(Type:str,datas: List):
    
    scatter = Scatter()
    if Type == "commtype":
        index = 0
        index_y = 2
        _title="Commtype_Scatter"
    elif Type == "group" :
        index = 2
        index_y = 0
        _title="Commgroup_Scatter"
    x_data = list(dict.fromkeys([item[index] for item in datas])) 
    scatter.add_xaxis(x_data)

    # colors = {group: generate_color() for group in groups}

    # Add y data to scatter with its size and color (commtype, size, group, busbw, count)    
    for data in datas:
        y = [None] * len(x_data)
        sizes = [None] * len(x_data)
        
        if data[index] in x_data:
            y[x_data.index(data[index])] = math.log(data[1], 2)
            sizes[x_data.index(data[index])] = data[1]
            scatter.add_yaxis(
                series_name=data[index_y],
                y_axis=y,
                symbol_size=30,  
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color=colors_group[data[index_y]]),  # set group color
                tooltip_opts=opts.TooltipOpts(
                    formatter=f"msg_size: {data[1]}<br>busbw: {data[3]}<br>count: {data[4]}"  
                )
            )

    # Set global options    
    max_y_value =math.ceil((max(math.log(data[1], 2) for data in datas)))
    min_y_value = min(math.log(data[1], 2) for data in datas)
    scatter.set_global_opts(
        title_opts=opts.TitleOpts(title=_title),
        xaxis_opts=opts.AxisOpts(
            name="Comm_type", 
            type_="category",
            name_location="middle",
            name_gap=30, 
            name_textstyle_opts=opts.TextStyleOpts(
                font_size="15px",  
                font_weight='bold'  
            )),  
        yaxis_opts=opts.AxisOpts(
            name="log(msg_size)", 
            min_=0, 
            max_=max_y_value,
            name_location="middle",
            name_gap=30, 
            name_textstyle_opts=opts.TextStyleOpts(
                font_size="15px",  
                font_weight='bold'  
            )),  
        legend_opts=opts.LegendOpts(
            type_="scroll",
            pos_left="right",
            orient="vertical",
        ),

    )

    return scatter 

def calculate_cdf_by_commtype(data: List[tuple[str, int, str, int, int]]) -> Dict[str, tuple[List[tuple[str, int, str, int, int]], np.ndarray]]:
    cdf_data = defaultdict(list)

    # Split data based on comm_type
    for item in data:
        comm_type = item[0]
        cdf_data[comm_type].append(item)

    # CDPF calculation
    cdf_result = {}
    for comm_type, items in cdf_data.items():
        items.sort(key=lambda x: x[1])
        msg_sizes = [math.log(item[1], 2) for item in items] 
        cdf = np.arange(1, len(msg_sizes) + 1) / len(msg_sizes)
        cdf_result[comm_type] = (items, cdf)

    return cdf_result
def create_cdf_chart_by_commtype(cdf_data: Dict[str, tuple[List[tuple[str, int, str, int, int]], np.ndarray]]) -> Line:
    line = Line()


    # Generate a CDF line for each comm_type
    for comm_type, (data, cdf) in cdf_data.items():
        
        x_data = [math.log(item[1], 2)  for item in data]
        y_data = cdf.tolist()
        msg_sizes = [item[1] for item in data]
        random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

        line.add_xaxis(x_data)
        line.add_yaxis(
            series_name=comm_type,
            y_axis=y_data,
            label_opts=opts.LabelOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(
            formatter=JsCode(
                    """
                    function(params) {
                        var msgSizes = %s;
                        return params.seriesName + '<br/>' + 
                               'msg_size: ' + msgSizes[params.dataIndex].toFixed(2) + '<br/>' +
                               'CDF: ' + params.value[1].toFixed(4);
                    }
                    """ % str(msg_sizes)
                ),
        ),
        )

    line.set_global_opts(
        title_opts=opts.TitleOpts(title="msg_size CDF"),
        xaxis_opts=opts.AxisOpts(
            type_="value",
            name="log(msg_size)",
            name_location="middle",
            name_gap=30,
            name_textstyle_opts=opts.TextStyleOpts(font_size=14, font_weight='bold')
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="CDF",
            name_location="middle",
            name_gap=30,
            name_textstyle_opts=opts.TextStyleOpts(font_size=14, font_weight='bold')
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(
            type_="scroll",
            pos_left="right",
            orient="vertical",
        ),
        
    )

    return line



def create_timeline_chart(epoch_data):
    def process_items(items):
        timeline_comp = []
        timeline_comm = []
        x_data = []
        current_time = 0 
        len_data = len(items) + 1
        x_data.append(f"{current_time:.3f}")
        for i,item in enumerate(items):
            if item.comm_type != CommType.epoch_end:
                # y = [None] * len_data
                start_time = current_time
                end_time = current_time + item._elapsed_time
                if i < len_data - 1 and item.comm_type == CommType.computation:  
                    y = [None] * (i + 2)
                    y[i] = 1
                    y[i+1] = 1
                    timeline_comp.append({
                        'value': y,
                        'stage': item.stage,
                        'elapsed_time': item._elapsed_time,
                        'comm_type':item.comm_type
                    })
                elif i < len_data - 1:
                    y = [None] * (i + 2)
                    y[i] = 2
                    y[i+1] = 2
                    timeline_comm.append({
                        'value': y,
                        'stage': item.stage,
                        'elapsed_time': item._elapsed_time,
                        'comm_type':item.comm_type
                    })
                x_data.append(f"{end_time:.3f}")
                current_time = end_time
        return timeline_comp,timeline_comm,x_data
    
    computation_timeline = []
    communication_timeline = []
    computation_timeline,communication_timeline,x_data = process_items(epoch_data)
    # Calculate computation time and communication time
    total_computation_time = sum(item['elapsed_time'] for item in computation_timeline)
    total_communication_time = sum(item['elapsed_time'] for item in communication_timeline)

    line = Line()
    line.add_xaxis(x_data)

    for comp_y in computation_timeline:
        line.add_yaxis(
            "Computation",
            comp_y['value'],
            is_connect_nones=False,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=5),  
            tooltip_opts=opts.TooltipOpts(
                    formatter=f"stage: {comp_y['stage']}<br>elapsed_time: {comp_y['elapsed_time']}<br>comm_type: {comp_y['comm_type']}"  
                ),
        )
    

    for comm_y in communication_timeline:
        
        line.add_yaxis(
            "communication",
            comm_y['value'],
            is_connect_nones=False,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=5),  
            tooltip_opts=opts.TooltipOpts(
                    formatter=f"stage: {comm_y['stage']}<br>elapsed_time: {comm_y['elapsed_time']}<br>comm_type: {comm_y['comm_type']}"  
                ),
        )


    line.set_global_opts(
        title_opts=opts.TitleOpts(title="Computation and Communication Timeline"),
        xaxis_opts=opts.AxisOpts(name="Time (ms)",type_ = "value"),#type_ = "value"
        yaxis_opts=opts.AxisOpts(name="Type", max_=3,axislabel_opts=opts.LabelOpts(is_show=False),axistick_opts=opts.AxisTickOpts(is_show=False)  
                                 ),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),

        datazoom_opts=[
            opts.DataZoomOpts(type_="slider", range_start=0, range_end=100),
            opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
        ],
        legend_opts=opts.LegendOpts(
            type_="scroll",
            pos_left="right",
            orient="vertical",
        ),
        
    )
    #comp—comm Pie
    pie = Pie()
    pie.add(
        "",
        [
            ("Computation", total_computation_time),
            ("Communication", total_communication_time)
        ],
        radius=["40%", "75%"],
    )
    pie.set_global_opts(
        title_opts=opts.TitleOpts(title="Computation vs Communication time Ratio"),
        legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical")
    )
    pie.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}:{d}%"))
    return line,pie
def extract_iteration(epoch_data):
    data_by_iter = []
    current_iter = []
    broadcast_count = 0

    for item in epoch_data:
        if item.comm_type == CommType.epoch_end:
            if current_iter:
                data_by_iter.append(current_iter)
            break

        if broadcast_count < 2:
            if item.comm_type == CommType.broadcast:
                broadcast_count += 1
            current_iter.append(item)
        else:
            if item.comm_type == CommType.broadcast:
                data_by_iter.append(current_iter)
                current_iter = [item]
                broadcast_count = 1
            else:
                current_iter.append(item)



    return data_by_iter

def create_ratio_pie(epoch_data):
    total_computation_time = total_communication_time = 0
    for item in epoch_data:
        if item.comm_type == CommType.computation:
            total_computation_time += item._elapsed_time
        elif item.comm_type != CommType.epoch_end:
            total_communication_time += item._elapsed_time

    total_ratio_pie = Pie()
    total_ratio_pie.add(
        "",
        [
            ("Computation", total_computation_time),
            ("Communication", total_communication_time)
        ],
        radius=["40%", "75%"],
    )
    total_ratio_pie.set_global_opts(
        title_opts=opts.TitleOpts(title="Overall Computation vs Communication time Ratio"),
        legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical")
    )
    total_ratio_pie.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
    return total_ratio_pie

def extract_ranks_data(raw_rows):
    domain_edges = defaultdict(lambda: defaultdict(int))
    domain_msg_total = defaultdict(int)
    domain_msg_count = defaultdict(int)
    for row in raw_rows:
        cg_raw = row.get('comm_group', 'None')
        if cg_raw == 'None' or cg_raw == '':
            continue
        group_name = cg_raw.split('.')[-1] if '.' in cg_raw else cg_raw
        msg_size = parse_msg_size(row.get('msg_size', '0'))
        ranks_str = row.get('ranks', '')
        if ranks_str:
            ranks = [int(r) for r in ranks_str.split(',') if r.strip()]
            for i in range(len(ranks)):
                for j in range(i + 1, len(ranks)):
                    edge = (min(ranks[i], ranks[j]), max(ranks[i], ranks[j]))
                    domain_edges[group_name][edge] += 1
        domain_msg_total[group_name] += msg_size
        domain_msg_count[group_name] += 1
    return {'edges': domain_edges, 'msg_total': domain_msg_total, 'msg_count': domain_msg_count}


def create_domain_flow_graph(domain_ranks):
    graph = Graph(init_opts=opts.InitOpts(width="100%", height="500px"))
    all_ranks = set()
    categories = []
    links = []
    group_colors = {'tp_group': '#376AB3', 'dp_group': '#87C0CA', 'pp_group': '#8cc540', 'ep_group': '#E8EDB9', 'ep_dp_group': '#f4a261', 'ep_tp_group': '#e76f51'}
    for group_name, edges in domain_ranks['edges'].items():
        color = group_colors.get(group_name, '#999999')
        if group_name not in [c['name'] for c in categories]:
            categories.append({'name': group_name, 'itemStyle': {'color': color}})
        for (src, dst), count in edges.items():
            all_ranks.add(src); all_ranks.add(dst)
            links.append({'source': str(src), 'target': str(dst), 'value': count, 'lineStyle': {'color': color, 'curveness': 0.15, 'opacity': 0.6}})
    nodes = [{'name': str(r), 'symbolSize': 20, 'category': 0} for r in sorted(all_ranks)]
    graph.add("", nodes, links, categories, repulsion=200, edge_length=150, is_draggable=True, label_opts=opts.LabelOpts(is_show=True, position="inside"))
    graph.set_global_opts(title_opts=opts.TitleOpts(title="Communication Domain: Rank Connectivity Graph"), legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"))
    return graph


def create_domain_msg_size_bar(domain_ranks):
    bar = Bar(init_opts=opts.InitOpts(width="100%", height="400px"))
    groups = list(domain_ranks['msg_total'].keys())
    totals = [domain_ranks['msg_total'][g] for g in groups]
    bar.add_xaxis(groups)
    bar.add_yaxis("Total Bytes", totals, label_opts=opts.LabelOpts(formatter="{c}"), itemstyle_opts=opts.ItemStyleOpts(color='#376AB3'))
    bar.set_global_opts(title_opts=opts.TitleOpts(title="Communication Domain: Total Message Size by Group"), xaxis_opts=opts.AxisOpts(name="Comm Group", axislabel_opts=opts.LabelOpts(rotate=30)), yaxis_opts=opts.AxisOpts(name="Total Bytes"))
    return bar


def visualize_output(filepath,only_workload:bool):

    log_items = read_csv_and_structure_data(filepath,only_workload)
    raw_rows = list(custom_csv_reader(filepath, only_workload))

    #pie
    data_by_epoch_comm = split_data_by_epoch(1,log_items) #only comm
    comm_type_counts = count_by_epoch(data_by_epoch_comm)
    pie_chart = create_pie_chart_for_epoch(comm_type_counts['epoch_0'])


    data = extract_data_from_log_items(data_by_epoch_comm['epoch_0'])
    #commtype Scatter
    effect_scatter_by_commtype = create_scatter_chart("commtype",data)

    #commtype cdf
    cdf_data = calculate_cdf_by_commtype(data)
    cdf_chart = create_cdf_chart_by_commtype(cdf_data)


    #group Scatter
    effect_scatter_by_group = create_scatter_chart("group",data)

    #ranks-based domain charts
    domain_ranks = extract_ranks_data(raw_rows)
    domain_graph = create_domain_flow_graph(domain_ranks)
    domain_bar = create_domain_msg_size_bar(domain_ranks)

    #comp-comm pattern
    data_by_epoch = split_data_by_epoch(0, log_items)

    timeline_charts = []
    ratio_pies = []
    all_ratio_pie = []
    if not only_workload:  
        all_iterations = extract_iteration(data_by_epoch['epoch_0'])
        for iteration in all_iterations:
            timeline_chart, ratio_pie = create_timeline_chart(iteration)
            timeline_charts.append(timeline_chart.dump_options())
            ratio_pies.append(ratio_pie.dump_options())
        all_ratio_pie = create_ratio_pie(data_by_epoch['epoch_0'])
    
    else:
        all_ratio_pie = None

    context = {
        'pie_chart_js': pie_chart.dump_options(),
        'scatter_by_commtype_js': effect_scatter_by_commtype.dump_options(),
        'cdf_chart_js': cdf_chart.dump_options(),
        'scatter_by_group_js': effect_scatter_by_group.dump_options(),
        'timeline_charts_js': json.dumps(timeline_charts),
        'ratio_pies_js': json.dumps(ratio_pies),
        'iteration_count': len(all_iterations) if not only_workload else 0,
        'all_ratio_pie':all_ratio_pie.dump_options() if all_ratio_pie else None,
        'domain_graph_js': domain_graph.dump_options(),
        'domain_bar_js': domain_bar.dump_options(),
    }

    # read Example.html
    with open('visualize/example.html', 'r', encoding='utf-8') as f:
        template = Template(f.read())


    rendered_html = template.render(**context)
    # write to file
    default_folder_path = 'results/visual_output' 
    if not os.path.exists(default_folder_path):
        os.makedirs(default_folder_path, exist_ok=True)
    filename = os.path.basename(filepath).split(".")[0]+'.html'
    output_file = os.path.join('results/visual_output',filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(rendered_html)

    print(f"Report generated:{output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: script.py <filepath> [only_workload]")
        sys.exit(1)

    filepath = sys.argv[1]
    flag = (len(sys.argv) > 2 and sys.argv[2] == 'only_workload')
    print(f'only workload flag is {flag}')
    visualize_output(filepath,flag)