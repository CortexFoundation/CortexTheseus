import mxnet as mx
from mxnet import ndarray as nd

import logging
import math
import time

class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super(ColoredFormatter, self).__init__(fmt, datefmt, style)

        self.log_colors = {
            "DEBUG": "\033[38;5;111m",
            "INFO": "\033[38;5;47m",
            "WARNING": "\033[38;5;178m",
            "ERROR": "\033[38;5;196m",
            "CRITICAL": "\033[30;48;5;196m",
            "DEFAULT": "\033[38;5;15m",
            "RESET": "\033[0m"
        }

    def format(self, record):
        log_color = self.get_color(record.levelname)
        message = super(ColoredFormatter, self).format(record)
        message = log_color + message + self.log_colors["RESET"]
        return message

    def get_color(self, level_name):
        lname = level_name if level_name in self.log_colors else "DEFAULT"
        return self.log_colors[lname]


class FilterList(logging.Filter):
    """ Filter with logging module

        Filter rules as below:
            {allow|disable log name} > level no > keywords >
            {inheritance from parent log name} > by default filter
        TODO:
    """
    def __init__(self, default=False, allows=[], disables=[],
            keywords=[], log_level=logging.INFO):
        self.rules = {}
        self._internal_filter_rule = "_internal_filter_rule"
        self.log_level = log_level
        self.keywords = keywords

        self.rules[self._internal_filter_rule] = default
        for name in allows:
            splits = name.split(".")
            rules = self.rules
            for split in splits:
                if split not in rules:
                    rules[split] = {}
                rules = rules[split]

            rules[self._internal_filter_rule] = True

        for name in disables:
            splits = name.split(".")
            rules = self.rules
            for split in splits:
                if split not in rules:
                    rules[split] = {}
                rules = rules[split]

            rules[self._internal_filter_rule] = False

    def filter(self, record):
        rules = self.rules
        rv = rules[self._internal_filter_rule]

        splits = record.name.split(".")
        for split in splits:
            if split in rules:
                rules = rules[split]
                if self._internal_filter_rule in rules:
                    rv = rules[self._internal_filter_rule]
            else:
                if record.levelno >= self.log_level:
                    return True

                for keyword in self.keywords:
                    if keyword in record.getMessage():
                        return True

                return rv

        return rv

def log_init(level=logging.NOTSET):
    logging.basicConfig(level=level)
    formatter = ColoredFormatter(
            fmt="[ %(asctime)s %(name)s.%(levelname)s ] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")

    log_filter = FilterList(
                log_level=logging.DEBUG,
                default=False)
    for handler in logging.root.handlers:
        handler.addFilter(log_filter)
        handler.setFormatter(formatter)

def extend_fname(prefix, with_ext=False):
    ret = ["%s.json"%prefix, "%s.params"%prefix]
    if with_ext:
        ret.append("%s.ext"%prefix)
    return tuple(ret)

def load_model(sym_fname, param_fname, inputs, ctx=mx.gpu()):
    sym, params = mx.sym.load(sym_fname), nd.load(param_fname)
    net = mx.gluon.nn.SymbolBlock(sym, inputs)
    load_parameters(net, params, ctx=ctx)
    return net

def load_parameters(graph, params, prefix="", ctx=None, dtype=None):
    params_dict = graph.collect_params()
    params_dict.initialize(ctx=ctx)

    ret_params = {}
    for name in params_dict:
        split_name, uniq_name = name.split("_"), []
        [uniq_name.append(sname) for sname in split_name if sname not in uniq_name]
        param_name = "_".join(uniq_name)
        param_name = param_name[len(prefix):]
        assert param_name in params or name in params, \
            "param name(%s) with origin(%s) not exists in params dict(%s)" \
            %(param_name, name, params.keys())
        data = params[name] if name in params else params[param_name]
        data = data if dtype is None else data.astype(dtype)
        params_dict[name].set_data(data)
        ret_params[name] = data

    return ret_params

def multi_validate(base_func, data_iter, *comp_funcs,
        iter_num=10, logger=logging, batch_size=16):
    log_str = "Iteration: {:3d} | " + base_func.__name__ + ": {} | "
    for func in comp_funcs:
        log_str += func.__name__ + ": {} | "
    log_str += "Total Sample: {:5d}"

    total = 0
    for i in range(iter_num):
        data, label = data_iter()
        base_acc = base_func(data, label)
        comp_acc = [func(data, label) for func in comp_funcs]
        total += batch_size

        msg = log_str.format(i, base_acc, *comp_acc, total)
        logger.info(msg)

def multi_eval_accuracy(base_func, data_iter_func, *comp_funcs,
        iter_num=10, logger=logging):
    log_str = "Iteration: %3d | Accuracy: %5.2f%% | "
    for idx in range(len(comp_funcs)):
        log_str += comp_funcs[idx].__name__ + ": %5.2f%%, diff: %5.2f%% | "
    log_str += "Total Sample: %5d"

    acc, total = 0, 0
    comp_accs = [0 for _ in range(len(comp_funcs) * 2)]
    for i in range(iter_num):
        data, label = data_iter_func()

        res = base_func(data)
        res_comp = [func(data) for func in comp_funcs]

        for idx in range(res.shape[0]):
            res_label = res[idx].asnumpy().argmax()
            data_label = label[idx].asnumpy()
            acc += 1 if res_label == data_label else 0

            for fidx in range(len(comp_funcs)):
                res_comp_label = res_comp[fidx][idx].asnumpy().argmax()
                comp_accs[2*fidx] += 1 if res_comp_label == data_label else 0
                comp_accs[2*fidx+1] += 0 if res_label == res_comp_label else 1
            total += 1
        logger.info(log_str, i, 100.*acc/total,
                *[100.*acc/total for acc in comp_accs], total)

def eval_time_accuracy(base_func, data_iter_func, comp_func=None,
        iter_num=10, logger=logging):
    log_str = "Iteration: %3d | Accuracy: %5.2f%%(%.4f sec) | "
    log_str += comp_func.__name__ if comp_func else "NULL"
    log_str += ": %5.2f%%(%.4f sec), diff: %5.2f%% | "
    log_str += "Total Sample: %5d"

    acc, comp_acc, diff, total = 0, 0, 0, 0
    sec, comp_sec = 0, 0
    for i in range(iter_num):
        data, label = data_iter_func()

        start = time.time()
        res = base_func(data)
        sec += time.time() - start
        if comp_func is not None:
            start = time.time()
            res_comp = comp_func(data)
            comp_sec += time.time() - start

        for idx in range(res.shape[0]):
            res_label = res[idx].asnumpy().argmax()
            data_label = label[idx].asnumpy()
            acc += 1 if res_label == data_label else 0

            if comp_func is not None:
                res_comp_label = res_comp[idx].asnumpy().argmax()
                comp_acc += 1 if res_comp_label == data_label else 0
                diff += 0 if res_label == res_comp_label else 1

            total += 1

        logger.info(log_str, i, 100.*acc/total, sec/total,
            100.*comp_acc/total, comp_sec/total, 100.*diff/total, total)












