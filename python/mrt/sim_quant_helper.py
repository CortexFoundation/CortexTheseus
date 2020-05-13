import logging
import math
import json

import mxnet as mx
from mxnet import ndarray as nd

def load_sim_data(data, name, inputs_ext):
    return data * inputs_ext[name]['scale']
def load_real_data(data, name, inputs_ext):
    logger = logging.getLogger('log.data.load')
    data = load_sim_data(data, name, inputs_ext)
    return int_realize(data, inputs_ext[name]['target_bit'],
            logger=None)

def save_ext(fname, *infos, logger=logging):
    fout = open(fname, "w")
    for info in infos:
        try:
            json.dump(info, fout)
        except:
            logger.error("Json seralize invalid with data: %s", info)
        fout.write('\n')
def load_ext(fname):
    fin = open(fname, "r")
    infos = []
    for line in fin:
        infos.append(json.loads(line))
    return tuple(infos)

def get_sim_scale(alpha, target_bit):
    sim_max = 2 ** (target_bit - 1) - 1
    scale = 1 if alpha == 0 else sim_max / alpha
    return scale

def int_realize(data, target_bit, logger=logging):
    out = data.round()
    clip_range = 2 ** (target_bit - 1) - 1
    if logger and out.abs().max() > clip_range:
        logger.warn("quant out of range int%d with data=<%s,%s,%s>",
                target_bit,
                out.asnumpy().flatten()[0],
                out.max().asnumpy(),
                out.min().asnumpy())

    out = out.clip(a_min=-clip_range, a_max=clip_range)
    return out

def extract_float(number):
    sign, binary = float_bin(number, 24)
    dot_idx = binary.find('.')
    binary = binary.replace('.', '')
    use_idx = binary.rfind('1') + 1
    if use_idx == 0:
        return 0, 0
    sb = dot_idx - use_idx
    frac = sign * int(binary[:use_idx], 2)
    return frac, sb

def float_bin(number, places = 24):
    """Single precision float convert into binary
    """
    sign = 1 if number >= 0 else -1
    number = abs(number)
    whole, dec = int(number), number - int(number)
    res = bin(whole).lstrip('0b') + '.'
    if len(res) > places:
        return res
    for x in range(places - len(res) + 1):
        dec *= 2
        whole, dec = int(dec), dec - int(dec)
        res += str(whole)
    return sign, res

def cvm_float(number, bits=24):
    alpha = max((2 ** (bits - 1)) - 1, 1)
    bits -= 1
    assert number >= 0
    if number == 0:
        return 0, 0
    exp = 0
    while (number >= 1):
        number /= 2
        exp += 1
    while (number < 1):
        number *= 2
        exp -= 1
    while (bits > 1):
        if (int(number) == number):
            break
        number *= 2
        exp -= 1
        bits -= 1
    frac, sb = round(number), exp
    return min(frac, alpha), sb
