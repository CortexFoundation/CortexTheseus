import ctypes

kDLCPU = 1
kDLGPU = 2
kDLFORMAL = 3
kDLOPENCL = 4

class CVMContext(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int),
                ("device_id", ctypes.c_int)]
    MASK2STR = {
        kDLCPU : 'cpu',
        kDLGPU : 'gpu',
        kDLFORMAL : 'formal',
        kDLOPENCL : 'opencl',
    }
    STR2MASK = {
        'cpu' : kDLCPU,
        'gpu' : kDLGPU,
        'formal' : kDLFORMAL,
        'opencl' : kDLOPENCL,
    }
    def __init__(self, device_type, device_id):
        super(CVMContext, self).__init__()
        self.device_type = device_type
        self.device_id = device_id

    def __repr__(self):
        return "@{}({})".format(
            self.MASK2STR[self.device_type], self.device_id)

    def __str__(self):
        return repr(self)

