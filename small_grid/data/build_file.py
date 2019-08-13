# -*- coding: utf-8 -*-
"""
build *.xml files for a small 6-intersection benchmark network
w/ the traffic dynamics modified from the following paper:

Ye, Bao-Lin, et al. "A hierarchical model predictive control approach for signal splits optimization
in large-scale urban road networks." IEEE Transactions on Intelligent Transportation Systems 17.8
(2016): 2182-2192.

network structure is in fig.2, traffic flow dynamics is in fig.4, turning matrix is in tab.II, other
simulation details are under section V.
@author: Tianshu Chu
"""
import numpy as np
import os
import xml.etree.cElementTree as ET

# FLOW_MULTIPLIER = 0.8
FLOW_MULTIPLIER = 1.0
SPEED_LIMIT = 20
L0, L1 = 200, 400
L0_end = 75

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def output_nodes(node):
    L2 = L0 / np.sqrt(2)
    L2_end = L0_end / np.sqrt(2)
    str_nodes = '<nodes>\n'
    # traffic light nodes w/ node 1 at (0, 0)
    str_nodes += node % ('nt1', 0, 0, 'traffic_light')
    str_nodes += node % ('nt2', L1, 0, 'traffic_light')
    str_nodes += node % ('nt3', L1, L0, 'traffic_light')
    str_nodes += node % ('nt4', L1, L1, 'traffic_light')
    str_nodes += node % ('nt5', L0, L1, 'traffic_light')
    str_nodes += node % ('nt6', 0, L1, 'traffic_light')
    # other nodes
    str_nodes += node % ('np1', 0, -L0_end, 'priority')
    str_nodes += node % ('np2', -L2_end, -L2_end, 'priority')
    str_nodes += node % ('np3', -L0_end, 0, 'priority')
    str_nodes += node % ('np4', L0_end + L1, 0, 'priority')
    str_nodes += node % ('np5', L1, -L0_end, 'priority')
    str_nodes += node % ('np6', L0_end + L1, L0, 'priority')
    str_nodes += node % ('np8', L0_end + L1, L1, 'priority')
    str_nodes += node % ('np9', L1, L0_end + L1, 'priority')
    str_nodes += node % ('np11', L0, L0_end + L1, 'priority')
    str_nodes += node % ('np12', -L0_end, L1, 'priority')
    str_nodes += node % ('np13', 0, L0_end + L1, 'priority')
    str_nodes += node % ('npc', L2, L2, 'priority')
    str_nodes += '</nodes>\n'
    return str_nodes


def output_road_types():
    str_types = '<types>\n'
    str_types += '  <type id="a" numLanes="1" speed="%.2f"/>\n' % SPEED_LIMIT
    str_types += '</types>\n'
    return str_types


def get_edge_str(edge, from_node, to_node):
    edge_id = '%s_%s' % (from_node, to_node)
    return edge % (edge_id, from_node, to_node)


def output_edges(edge):
    str_edges = '<edges>\n'
    # source roads
    for i in [1, 2, 3]:
        to_node = 'nt1'
        from_node = 'np' + str(i)
        str_edges += get_edge_str(edge, from_node, to_node)
    for i in [8, 9]:
        to_node = 'nt4'
        from_node = 'np' + str(i)
        str_edges += get_edge_str(edge, from_node, to_node)
    # network roads
    str_edges += get_edge_str(edge, 'nt1', 'nt2')
    str_edges += get_edge_str(edge, 'nt1', 'npc')
    str_edges += get_edge_str(edge, 'nt1', 'nt6')
    str_edges += get_edge_str(edge, 'npc', 'nt3')
    str_edges += get_edge_str(edge, 'npc', 'nt5')
    str_edges += get_edge_str(edge, 'nt5', 'nt6')
    str_edges += get_edge_str(edge, 'nt4', 'nt3')
    str_edges += get_edge_str(edge, 'nt4', 'nt5')
    str_edges += get_edge_str(edge, 'nt3', 'nt2')
    # sink roads
    for i in [12, 13]:
        from_node = 'nt6'
        to_node = 'np' + str(i)
        str_edges += get_edge_str(edge, from_node, to_node)
    for i in [4, 5]:
        from_node = 'nt2'
        to_node = 'np' + str(i)
        str_edges += get_edge_str(edge, from_node, to_node)
    str_edges += get_edge_str(edge, 'nt5', 'np11')
    str_edges += get_edge_str(edge, 'nt3', 'np6')
    str_edges += '</edges>\n'
    return str_edges


def get_con_str(con, from_node, cur_node, to_node):
    from_edge = '%s_%s' % (from_node, cur_node)
    to_edge = '%s_%s' % (cur_node, to_node)
    return con % (from_edge, to_edge)


def output_connections(con):
    str_cons = '<connections>\n'
    # cross nt1
    for i in [1, 2, 3]:
        for j in [2, 6]:
            from_node = 'np' + str(i)
            to_node = 'nt' + str(j)
            str_cons += get_con_str(con, from_node, 'nt1', to_node)
        str_cons += get_con_str(con, from_node, 'nt1', 'npc')
    # cross nt4
    for i in [8, 9]:
        for j in [3, 5]:
            from_node = 'np' + str(i)
            to_node = 'nt' + str(j)
            str_cons += get_con_str(con, from_node, 'nt4', to_node)
    # cross npc
    for i in [3, 5]:
        to_node = 'nt' + str(i)
        str_cons += get_con_str(con, 'nt1', 'npc', to_node)
    # cross nt2
    for i in [1, 3]:
        for j in [4, 5]:
            from_node = 'nt' + str(i)
            to_node = 'np' + str(j)
            str_cons += get_con_str(con, from_node, 'nt2', to_node)
    # cross nt6
    for i in [1, 5]:
        for j in [12, 13]:
            from_node = 'nt' + str(i)
            to_node = 'np' + str(j)
            str_cons += get_con_str(con, from_node, 'nt6', to_node)
    # cross nt3
    for from_node in ['npc', 'nt4']:
        for to_node in ['np6', 'nt2']:
            str_cons += get_con_str(con, from_node, 'nt3', to_node)
    # cross nt5
    for from_node in ['npc', 'nt4']:
        for to_node in ['np11', 'nt6']:
            str_cons += get_con_str(con, from_node, 'nt5', to_node)
    str_cons += '</connections>\n'
    return str_cons


def output_netconfig():
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <edge-files value="exp.edg.xml"/>\n'
    str_config += '    <node-files value="exp.nod.xml"/>\n'
    str_config += '    <type-files value="exp.typ.xml"/>\n'
    str_config += '    <tllogic-files value="exp.tll.xml"/>\n'
    str_config += '    <connection-files value="exp.con.xml"/>\n'
    str_config += '  </input>\n  <output>\n'
    str_config += '    <output-file value="exp.net.xml"/>\n'
    str_config += '  </output>\n</configuration>\n'
    return str_config


def output_flows(flow, num_car_hourly):
    prob = '%.2f' % (num_car_hourly / float(3600))
    flow1 = '  <flow id="mf_%s" departPos="random_free" begin="%d" end="%d" probability="' + \
            prob + '" type="type1">\n' + \
            '    <route edges="%s"/>\n  </flow>\n'
    routes = ['nt1_npc npc_nt5 nt5_np11',
              'nt1_npc npc_nt5 nt5_nt6 nt6_np12',
              'nt4_nt5 nt5_np11',
              'nt4_nt5 nt5_nt6 nt6_np12',
              'nt1_nt2 nt2_np4',
              'nt1_nt6 nt6_np13',
              'nt1_npc npc_nt3 nt3_np6',
              'nt1_npc npc_nt3 nt3_nt2 nt2_np5',
              'nt4_nt3 nt3_np6',
              'nt4_nt3 nt3_nt2 nt2_np5']
    cases = [(3, 4, 5), (0, 3, 4), (1, 2, 5), (4, 5, 9), (5, 6, 9), (4, 7, 8)]
    str_flows = '<routes>\n'
    str_flows += '  <vType id="type1" length="5" accel="5" decel="10"/>\n'

    # flows vary every 10min, with dim 5x12, 5 source links are x1, x2, x3, x8, x9
    # flows = [[450, 475, 500, 600, 625, 650, 700, 650, 625, 600, 500, 400],
    #          [300, 350, 400, 500, 525, 550, 575, 525, 500, 450, 350, 300],
    #          [475, 500, 550, 625, 725, 750, 800, 700, 675, 625, 550, 450],
    #          [575, 650, 700, 925, 950, 975, 1000, 950, 900, 750, 650, 550],
    #          [625, 700, 750, 825, 925, 950, 975, 900, 850, 800, 700, 600]]
    flows = [[500, 100, 700, 800, 550, 550, 100, 200, 250, 250, 400, 800],
             [600, 700, 100, 200, 50, 100, 1000, 500, 450, 150, 400, 200],
             [100, 400, 400, 200, 600, 550, 100, 500, 500, 800, 400, 200],
             [100, 200, 300, 300, 300, 400, 600, 600, 800, 500, 400, 300],
             [600, 400, 400, 600, 800, 400, 300, 300, 300, 200, 250, 250]]
    edges = ['%s_%s' % (x, 'nt1') for x in ['np1', 'np2', 'np3']] + \
            ['%s_%s' % (x, 'nt4') for x in ['np8', 'np9']]
    times = range(0, 7201, 1200)
    times1 = range(0, 7201, 600)
    for i in range(len(times) - 1):
        t_begin, t_end = times[i], times[i + 1]
        for c in cases[i]:
            name = str(c) + '_' + str(i)
            str_flows += flow1 % (name, t_begin, t_end, routes[c])
        for i0 in [i * 2, i * 2 + 1]:
            t_begin, t_end = times1[i0], times1[i0 + 1]
            for j in range(5):
                str_flows += flow % (str(j) + '_' + str(i0), edges[j], t_begin, t_end,
                                     int(flows[j][i0] * FLOW_MULTIPLIER))
    str_flows += '</routes>\n'
    return str_flows


def get_turn_str(from_edge, to_edges, to_probs):
    cur_str = '    <fromEdge id="%s">\n' % from_edge
    for to_edge, to_prob in zip(to_edges, to_probs):
        cur_str += '      <toEdge id="%s" probability="%.2f"/>\n' % (to_edge, to_prob)
    cur_str += '    </fromEdge>\n'
    return cur_str


def output_turns():
    str_turns = '<turns>\n'
    str_turns += '  <interval begin="0" end="7200">\n'
    # cross nt1
    from_edge = '%s_%s' % ('np1', 'nt1')
    to_edges = ['nt1_%s' % x for x in ['nt2', 'nt6', 'npc']]
    to_probs = [0.2, 0.5, 0.3]
    str_turns += get_turn_str(from_edge, to_edges, to_probs)
    from_edge = '%s_%s' % ('np2', 'nt1')
    to_probs = [0.15, 0.15, 0.7]
    str_turns += get_turn_str(from_edge, to_edges, to_probs)
    from_edge = '%s_%s' % ('np3', 'nt1')
    to_probs = [0.5, 0.15, 0.35]
    str_turns += get_turn_str(from_edge, to_edges, to_probs)
    # cross nt4
    from_edge = '%s_%s' % ('np8', 'nt4')
    to_edges = ['nt4_%s' % x for x in ['nt3', 'nt5']]
    to_probs = [0.4, 0.6]
    str_turns += get_turn_str(from_edge, to_edges, to_probs)
    from_edge = '%s_%s' % ('np9', 'nt4')
    to_probs = [0.6, 0.4]
    str_turns += get_turn_str(from_edge, to_edges, to_probs)
    # cross nt2
    from_edge = '%s_%s' % ('nt3', 'nt2')
    to_edges = ['nt2_np5']
    to_probs = [1.0]
    str_turns += get_turn_str(from_edge, to_edges, to_probs)
    from_edge = '%s_%s' % ('nt1', 'nt2')
    to_edges = ['nt2_np4']
    to_probs = [1.0]
    str_turns += get_turn_str(from_edge, to_edges, to_probs)
    # cross nt6
    from_edge = '%s_%s' % ('nt5', 'nt6')
    to_edges = ['nt6_np12']
    to_probs = [1.0]
    str_turns += get_turn_str(from_edge, to_edges, to_probs)
    from_edge = '%s_%s' % ('nt1', 'nt6')
    to_edges = ['nt6_np13']
    to_probs = [1.0]
    str_turns += get_turn_str(from_edge, to_edges, to_probs)
    # cross nt3
    from_edge = '%s_%s' % ('npc', 'nt3')
    to_edges = ['nt3_%s' % x for x in ['nt2', 'np6']]
    to_probs = [0.3, 0.7]
    str_turns += get_turn_str(from_edge, to_edges, to_probs)
    # cross nt5
    from_edge = '%s_%s' % ('npc', 'nt5')
    to_edges = ['nt5_%s' % x for x in ['nt6', 'np11']]
    to_probs = [0.3, 0.7]
    str_turns += get_turn_str(from_edge, to_edges, to_probs)
    str_turns += '  </interval>\n'
    # cross npc needs to be estimated based on flows
    # flows = [[450, 475, 500, 600, 625, 650, 700, 650, 625, 600, 500, 400],
    #          [300, 350, 400, 500, 525, 550, 575, 525, 500, 450, 350, 300],
    #          [475, 500, 550, 625, 725, 750, 800, 700, 675, 625, 550, 450]]
    flows = [[500, 100, 700, 800, 550, 550, 100, 200, 250, 250, 400, 800],
             [600, 700, 100, 200, 50, 100, 1000, 500, 450, 150, 400, 200],
             [100, 400, 400, 200, 600, 550, 100, 500, 500, 800, 400, 200]]
    times = range(0, 7201, 600)
    flows = np.array(flows)
    base_probs = np.array([[0.15, 0.15], [0.35, 0.35], [0.15, 0.2]])
    from_edge = '%s_%s' % ('nt1', 'npc')
    to_edges = ['npc_%s' % x for x in ['nt3', 'nt5']]
    for i in range(len(times) - 1):
        t_begin, t_end = times[i], times[i + 1]
        cur_prob = np.dot(flows[:, i].reshape(1, 3), base_probs)
        cur_prob = np.ravel(cur_prob)
        cur_prob /= np.sum(cur_prob)
        str_turns += '  <interval begin="%d" end="%d">\n' % (t_begin, t_end)
        str_turns += get_turn_str(from_edge, to_edges, list(cur_prob))
        str_turns += '  </interval>\n'
    sink_edges = []
    for i in [12, 13]:
        from_node = 'nt6'
        to_node = 'np' + str(i)
        sink_edges.append('%s_%s' % (from_node, to_node))
    for i in [4, 5]:
        from_node = 'nt2'
        to_node = 'np' + str(i)
        sink_edges.append('%s_%s' % (from_node, to_node))
    sink_edges.append('%s_%s' % ('nt5', 'np11'))
    sink_edges.append('%s_%s' % ('nt3', 'np6'))
    str_turns += '  <sink edges="%s"/>\n' % (' '.join(sink_edges))
    str_turns += '</turns>\n'
    return str_turns


def gen_rou_file(seed=None, thread=None, path=None, num_car_hourly=0):
    if thread is None:
        out_file = 'exp.rou.xml'
        flow_file = 'exp.raw.rou.xml'
    else:
        out_file = 'exp_%d.rou.xml' % int(thread)
        flow_file = 'exp_%d.raw.rou.xml' % int(thread)
    flow = '  <flow id="f_%s" from="%s" begin="%d" end="%d" vehsPerHour="%i" type="type1"/>\n'
    write_file(path + flow_file, output_flows(flow, num_car_hourly=num_car_hourly))
    files = [flow_file, 'exp.turns.xml', 'exp.net.xml', out_file]
    if path is not None:
        files = [path + f for f in files]
    flags = ['-r', '-t', '-n', '-o']
    command = 'jtrrouter'
    for a, b in zip(flags, files):
        command += ' ' + a + ' ' + b
    if seed is not None:
        command += ' --seed %d' % int(seed)
    os.system(command)
    # remove webpage loading
    tree = ET.ElementTree(file=files[-1])
    tree.getroot().attrib = {}
    tree.write(files[-1])
    sumocfg_file = path + ('exp_%d.sumocfg' % thread)
    write_file(sumocfg_file, output_config(thread=thread))
    return sumocfg_file


def output_config(thread=None):
    if thread is None:
        out_file = 'exp.rou.xml'
    else:
        out_file = 'exp_%d.rou.xml' % int(thread)
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="exp.net.xml"/>\n'
    str_config += '    <route-files value="%s"/>\n' % out_file
    str_config += '    <additional-files value="exp.add.xml"/>\n'
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="0"/>\n    <end value="7200"/>\n'
    str_config += '  </time>\n</configuration>\n'
    return str_config


def get_ild_str(from_node, to_node, ild_str, lane_i=0):
    edge = '%s_%s' % (from_node, to_node)
    return ild_str % (edge, lane_i, edge, lane_i)


def output_ild(ild):
    str_adds = '<additional>\n'
    for i in [1, 2, 3]:
        to_node = 'nt1'
        from_node = 'np' + str(i)
        str_adds += get_ild_str(from_node, to_node, ild)
    for i in [8, 9]:
        to_node = 'nt4'
        from_node = 'np' + str(i)
        str_adds += get_ild_str(from_node, to_node, ild)
    str_adds += get_ild_str('nt1', 'nt2', ild)
    str_adds += get_ild_str('nt1', 'npc', ild)
    str_adds += get_ild_str('nt1', 'nt6', ild)
    str_adds += get_ild_str('npc', 'nt3', ild)
    str_adds += get_ild_str('npc', 'nt5', ild)
    str_adds += get_ild_str('nt5', 'nt6', ild)
    str_adds += get_ild_str('nt4', 'nt3', ild)
    str_adds += get_ild_str('nt4', 'nt5', ild)
    str_adds += get_ild_str('nt3', 'nt2', ild)
    # for i in [12, 13]:
    #     from_node = 'nt6'
    #     to_node = 'np' + str(i)
    #     str_adds += get_ild_str(from_node, to_node, ild_in=ild_in)
    # for i in [4, 5]:
    #     from_node = 'nt2'
    #     to_node = 'np' + str(i)
    #     str_adds += get_ild_str(from_node, to_node, ild_in=ild_in)
    # str_adds += get_ild_str('nt5', 'np11', ild_in=ild_in)
    # str_adds += get_ild_str('nt3', 'np6', ild_in=ild_in)
    str_adds += '</additional>\n'
    return str_adds


def output_tls(tls, phase):
    str_adds = '<additional>\n'
    # cross 1 has 3 phases, other crosses have 2 phases
    three_phases = ['GGGrrrrrr', 'yyyrrrrrr', 'rrrGGGrrr',
                    'rrryyyrrr', 'rrrrrrGGG', 'rrrrrryyy']
    two_phases = ['GGrr', 'yyrr', 'rrGG', 'rryy']
    phase_duration = [30, 3]
    str_adds += tls % 'nt1'
    for k, p in enumerate(three_phases):
        str_adds += phase % (phase_duration[k % 2], p)
    str_adds += '  </tlLogic>\n'
    for i in range(2, 7):
        node = 'nt' + str(i)
        str_adds += tls % node
        for k, p in enumerate(two_phases):
            str_adds += phase % (phase_duration[k % 2], p)
        str_adds += '  </tlLogic>\n'
    str_adds += '</additional>\n'
    return str_adds


def main():
    # nod.xml file
    node = '  <node id="%s" x="%.2f" y="%.2f" type="%s"/>\n'
    write_file('./exp.nod.xml', output_nodes(node))

    # typ.xml file
    write_file('./exp.typ.xml', output_road_types())

    # edg.xml file
    edge = '  <edge id="%s" from="%s" to="%s" type="a"/>\n'
    write_file('./exp.edg.xml', output_edges(edge))

    # con.xml file
    con = '  <connection from="%s" to="%s" fromLane="0" toLane="0"/>\n'
    write_file('./exp.con.xml', output_connections(con))

    # tls.xml file
    tls = '  <tlLogic id="%s" programID="0" offset="0" type="static">\n'
    phase = '    <phase duration="%d" state="%s"/>\n'
    write_file('./exp.tll.xml', output_tls(tls, phase))

    # net config file
    write_file('./exp.netccfg', output_netconfig())

    # generate net.xml file
    os.system('netconvert -c exp.netccfg')

    # raw.rou.xml file
    flow = '  <flow id="f_%s" from="%s" begin="%d" end="%d" vehsPerHour="%d" type="type1"/>\n'
    write_file('./exp.raw.rou.xml', output_flows(flow, num_car_hourly=1000))

    # turns.xml file
    write_file('./exp.turns.xml', output_turns())

    # generate rou.xml file
    os.system('jtrrouter -t exp.turns.xml -n exp.net.xml -r exp.raw.rou.xml -o exp.rou.xml')

    # add.xml file
    ild = '  <laneAreaDetector file="ild.out" freq="1" id="%s_%d" lane="%s_%d" pos="-50" endPos="-1"/>\n'
    write_file('./exp.add.xml', output_ild(ild))

    # config file
    write_file('./exp.sumocfg', output_config())

if __name__ == '__main__':
    main()
