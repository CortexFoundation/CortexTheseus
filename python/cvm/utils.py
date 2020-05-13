import numpy as np
import logging
from . import symbol as _sym
from . import graph

def argmax(out):
    return np.argmax(out)

def classification_output(out, batch=1):
    batch_len = len(out) // batch
    assert batch_len * batch == len(out)
    for i in range(0, len(out), batch_len):
        tmp = out[i*batch_len:(i+1)*batch_len]

        print("\n*** Batch=%s ***" % i)
        print("Front 10 numbers: [%s]" % \
              " ".join([str(d) for d in tmp[:10]]))
        print("Last  10 numbers: [%s]" % \
              " ".join([str(d) for d in tmp[-10:]]))
        cat = argmax(tmp)
        print("Argmax output category: %d with possiblity %d" % \
              (cat, tmp[cat]))

def detection_output(out, batch=1):
    batch_len = len(out) // batch
    assert batch_len * batch == len(out)
    for i in range(0, len(out), batch_len):
        tmp = out[i*batch_len:(i+1)*batch_len]

        print("\n*** Batch=%s ***" % i)
        for i in range(0, len(tmp), 6):
           if tmp[i] == -1:
               print ("Detect object number: %d" % (i // 6))
               break
           print (tmp[i:i+6])

def load_model(sym_path, prm_path):
    with open(sym_path, "r") as f:
        json_str = f.read()
    with open(prm_path, "rb") as f:
        param_bytes = f.read()
    return json_str.encode("utf-8"), param_bytes

def load_np_data(data_path):
    data = np.load(data_path)
    return data.tobytes()

def topo_sort(symbol, logger=logging, with_deps=False):
    queue = []
    symbol_map = {}
    deps = {}
    dep_cnts = {}
    for s in symbol:
        symbol_map[s.attr('name')] = s
        queue.append(s)

    while queue:
        sym = queue.pop(0)
        name = sym.attr('name')
        childs = sym.get_children()
        if childs is None:
            dep_cnts[name] = 0
        else:
            # remove duplication dependency
            dep_cnts[name] = len({c.attr('name') for c in childs})
            for child in childs:
                child_name = child.attr('name')
                if child_name not in deps:
                    deps[child_name] = set()
                deps[child_name].add(name)
                if child_name not in symbol_map:
                    symbol_map[child_name] = child
                    queue.append(child)

    order = []
    reduce_flag = True
    while dep_cnts:
        if not reduce_flag:
            logger.critical("deps cannot reduce -> %s", dep_cnts)
            assert False

        remove = []
        reduce_flag = False
        for name in dep_cnts:
            if dep_cnts[name] == 0:
                order.append(symbol_map[name])
                remove.append(name)
                if name in deps:
                    for other in deps[name]:
                        dep_cnts[other] -= 1

                reduce_flag = True
        for name in remove:
            del dep_cnts[name]
    if with_deps:
        return order, deps
    else:
        return order

MULTIPYE_OUTS_NODE = [
    'get_valid_counts',
    # group's op_name is None
    'None',
]
def entry_id(sym):
    oindex = 0
    if sym.attr('op_name') in MULTIPYE_OUTS_NODE:
        # raise NotImplementedError("not implemented")
        graph = graph.create(sym)
        oindex = json.loads(graph.json())['heads'][0][1]
    return oindex

def node_entry(sym, graph):
    name = sym.attr('name')
    if name not in graph:
        raise RuntimeError("Unrecognized symbol:{} in graph" +
                " keys:{}".format(name, graph.keys()))
    return graph[name][entry_id(sym)]


def topo_visit(symbol, params, callback=None,
               logger=logging, **kwargs):
    graph = {}
    # params = {k:v[:] for k, v in params.items()}
    for op in topo_sort(symbol, logger=logger):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = op.get_children(), op.list_attr()
        if childs is not None:
            childs = [node_entry(c, graph) for c in childs]
            # op = get_op(op_name)(*childs, **attr, name=name)
            op = getattr(_sym, op_name)(*childs, **attr, name=name)

        if callback is not None:
            graph[name] = callback(
                    op, params=params, graph=graph, **kwargs)

        if graph.get(name, None) is None:
            graph[name] = op
    nodes = [node_entry(op, graph) for op in symbol]
    ret = nodes[0]
    if len(nodes) > 1:
        ret = getattr(_sym, "Group")(*nodes)
    return ret, params
