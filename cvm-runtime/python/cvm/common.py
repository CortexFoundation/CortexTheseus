from ._ctypes.context import *

def context(dev_type, dev_id=0):
    if isinstance(dev_type, str):
        dev_type = dev_type.split()[0]
        if dev_type not in CVMContext.STR2MASK:
            raise ValueError("Unknown device type %s" % dev_type)
        dev_type = CVMContext.STR2MASK[dev_type]
    return CVMContext(dev_type, dev_id)

def cpu(dev_id=0):
    return CVMContext(kDLCPU, dev_id)

def gpu(dev_id=0):
    return CVMContext(kDLGPU, dev_id)

def formal(dev_id=0):
    return CVMContext(kDLFORMAL, dev_id)

def opencl(dev_id=0):
    return CVMContext(kDLOPENCL, dev_id)

RuntimeDevAPIMap = {
    kDLCPU: 0,
    kDLGPU: 1,
    kDLFORMAL: 2,
    kDLOPENCL: 3,
}

def runtime_context(ctx):
    return CVMContext(
        RuntimeDevAPIMap[ctx.device_type],
        ctx.device_id)
