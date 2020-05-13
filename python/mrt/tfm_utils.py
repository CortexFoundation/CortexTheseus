import math
import logging

import mxnet as mx
from mxnet import ndarray as nd

from .tfm_base import OUT_KEY, N, MAX_BIT
from . import sim_quant_helper as sim
from . import sym_utils as sutils

def get_bit(opt):
    if isinstance(opt, nd.NDArray):
        opt = opt.abs().max().asscalar()
    return math.ceil(math.log2(math.fabs(opt)+1)) + 1

def get_bit_cnt(cnt):
    # get_bit_cnt (mrt) should be consistent with
    # GetReduceSumBit (cvm-runtime)
    assert isinstance(cnt, int) and cnt > 0, \
        "Error in get_bit_cnt, provided cnt: %s"%cnt
    prec = 0
    while cnt != 0:
        prec += 1
        cnt >>= 1
    return prec

def get_range(prec):
    return (2 ** (prec - 1)) - 1

def scale(threshold, precision):
    assert threshold >= 0
    if threshold == 0:
        return 1
    alpha = (2 ** (precision - 1)) - 1
    return alpha / threshold

def realize(X, sb, prec, name=None):
    name = name if name else N.n('realize')
    if sb == 0:
        sym = mx.sym.Custom(X, precision=prec,
                            name=name, op_type='cvm_clip')
    elif sb < 0:
        sym = mx.sym.Custom(X, shift_bit=-sb, precision=prec,
                            name=name, op_type='cvm_left_shift')
    else:
        sym = mx.sym.Custom(X, shift_bit=sb, precision=prec,
                            name=name, op_type='cvm_right_shift')
    return sym

def requant_operator(X, oprec, oscale=None, **kwargs):
    logger = logging.getLogger('log.mrt.realize')
    params, graph = kwargs['params'], kwargs['graph']
    shift_bits = kwargs['shift_bits']
    th_dict, precs = kwargs['th_dict'], kwargs['precs']
    xopn, xn = X.attr('op_name'), X.attr('name')

    if th_dict[xn] == 0:
        return X, 1, oscale if oscale else 1

    exactly = True if oscale else False
    oprec = precs[xn].get(kwargs['oname'], oprec)
    oscale = oscale if oscale else scale(th_dict[xn], oprec)
    iscale = kwargs['scales'][xn]
    iprec = precs[xn][OUT_KEY]

    sb = get_bit(th_dict[xn]*iscale) - oprec
    if sb > shift_bits:
        iprec -= sb
        X = realize(X, sb, iprec)
        iscale = iscale / (2**sb)

    if exactly or iprec > oprec:
        rescale = oscale / iscale
        bits = MAX_BIT - iprec
        frac, exp = sim.cvm_float(rescale, bits)
        sim_scale = frac * (2 ** exp)
        scale_err = abs((sim_scale - rescale) / rescale)
        if scale_err > 0.001:
            logger.warn(
                "Operator  %-20s name=%-40s quantize with sb=%s" +
                " scale=%s, error=%s",
                xopn, xn, sb, iscale, scale_err)
        oscale = iscale * frac * (2 ** exp)
        if frac > 1:
            var = sutils.nd_const(frac, graph, params)
            X = mx.sym.broadcast_mul(X, var, name=N.n("mrt_quantize_scale"))
        X = realize(X, -exp, oprec)
        logger.debug(
            "Operator  %-20s name=%-40s requantize with scale=%-16.8f<%d, %d>" +
            " iprec=%s, iscale=%-10.5f, oprec=%s, oscale=%-10.5f",
            xopn, xn, rescale, frac, exp, iprec, iscale, oprec, oscale)
    else:
        oscale = iscale
        logger.debug(
            "Operator  %-20s name=%-40s clip with iprec=%s, oprec=%s",
            xopn, xn, iprec, oprec)
    return X, oprec, oscale

def requant_parameter(wname, oprec, oscale=None, **kwargs):
    params, th_dict = kwargs['params'], kwargs['th_dict']
    logger = logging.getLogger('log.mrt.realize')
    Wn = N.n(wname)

    W = None
    if th_dict[wname] == 0:
        oprec, oscale = 1, 1
        shp = params[wname].shape
        params[Wn] = sutils.nd_zeros(shp)
        attr = {'precision': '1'}
        W = mx.sym.var(Wn, shape=shp, attr=attr)
    else:
        oprec = kwargs['precs'][wname].get(kwargs['oname'], oprec)
        oscale = oscale if oscale else scale(th_dict[wname], oprec)
        params[Wn] = sim.int_realize(
            params[wname].astype("float64") * oscale,
            oprec, logger=logger)
        attr = {'precision': str(oprec)}
        max_v = params[Wn].abs().max().asscalar()
        range_v = (2**(oprec-1)-1)
        assert max_v <= range_v,\
            "name:%s, max_v:%s, range_v:%s, oprec:%s"%\
            (wname, max_v, range_v, oprec)
        W = mx.sym.var(Wn, shape=params[Wn].shape, attr=attr)

    logger.debug(
        "Parameter th_dict=%-12.8f name=%-40s " + \
        "requantize with scale=%-16.8f to prec=%s",
        th_dict[wname], wname, oscale, oprec)

    return W, oprec, oscale

def requant(sym, oprec, oscale=None, **kwargs):
    if sutils.is_params(sym, kwargs['params']):
        return requant_parameter(sym.attr('name'), oprec, oscale, **kwargs)
    return requant_operator(sym, oprec, oscale, **kwargs)
