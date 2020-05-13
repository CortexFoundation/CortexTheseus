import cvm
from cvm import utils
from cvm import nd


def test_abs():
    x = cvm.sym.var('data', shape=(3), precision=8)
    y = cvm.sym.var('y', shape=(3), precision=8)
    x = x + y
    params = {'y': nd.array([-11, 5, 108])}
    x = cvm.sym.clip(x, a_min=-10, a_max=10)
    sym = cvm.sym.Group(x, y)
    op_name = y.attr("op_name")
    print("op name = ", op_name)

    def _print(sym, params, graph):
        print (sym.attr('op_name'), sym.attr('name'), graph.keys())

    sym, params = utils.topo_visit(sym, params, _print)
    return sym, params

def test_clip():
    x = cvm.sym.var('data', shape=(3), precision=16)
    y = cvm.sym.cvm_right_shift(x, shift_bit=8, precision=8)
    y = cvm.sym.cvm_clip(y, precision=8)
    return y, {}

def test_l2norm():
    x = cvm.sym.var('data', shape=(2,2,2), precision=8)
    params = {}
    sym = cvm.sym.l2norm(x, mode='channel')

    def _print(sym, params, graph):
        # print (sym.attr('op_name'), sym.attr('name'), graph.keys())
        name, op_name = sym.attr('name'), sym.attr('op_name')
        print("name: %20s,      op_name: %20s"%(name, op_name))

    sym, params = utils.topo_visit(sym, params, _print)
    return sym, params

if __name__ == "__main__":
    #  sym, params = test_abs()
    sym, params = test_clip()
    #  exit()
    graph, _ = cvm.graph.build(sym, params)
    # print (graph.json())

    json_str = graph.json()
    param_bytes = nd.save_param_dict(params)

    model = cvm.runtime.CVMAPILoadModel(
        json_str, param_bytes, cvm.gpu())
    data = nd.array([-30000, 1000, 23530], dtype="int32").as_runtime_input()
    print (len(data))
    out = cvm.runtime.CVMAPIInference(
        model, data)
    cvm.runtime.CVMAPIFreeModel(model)
    print (out)


