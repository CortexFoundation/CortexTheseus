import numpy as np
import sys
import subprocess
import json
def RpcCall(cmds):
    process = subprocess.Popen(cmds.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()
    try:
        return json.loads(out)['result']
    except Exception as e:
        print ('error', repr(e))
        return None
def Discrete(v):
    if abs(v - 127) < 30:
        v = 127
    else:
        if v < 127:
            v = 0
        else:
            v = 255
    return v
data = np.load('/home/tian/.cortex/storage/cerebro/' + sys.argv[1] + '/data').reshape(28, 28)
for i in range(28):
    for j in range(28):
        v = data[i][j] = Discrete(data[i][j])
        print ("%02x" % v, end='')
    print ()
sys.stdout.flush()
# exit(0)
querys = []
hex_result = bytearray.fromhex(RpcCall('bash ./getPixels.sh 0x426bfd00 latest')[2:])
cur_state = list(map(int, hex_result))
print ('len = ', len(cur_state), ' len(hex) = ', len(hex_result))
cur_state = np.array(cur_state[64: -16]).reshape(28, 28).astype(int)
for i in range(28):
    for j in range(28):
        print ("%02x" % cur_state[i][j], end='')
    print ()
sys.stdout.flush()

for i in range(28):
    for j in range(28):
        if abs(i - 14) > 14 or abs(j - 14) > 14:
            continue
        v = int(data[i][j])
        if v != cur_state[i][j].astype(int):
            querys.append("%064x%064x%064x" % (j, i, v)) ## 0 if v < 127 else 255))
stepsize = 10
querys = ["".join(querys[i:i+stepsize]) for i in range(0, len(querys), stepsize)]
header = '0x9223d64f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000040'
print ("#querys = %d" % (len(querys)), file=sys.stderr)
for x in querys:
    draw_query = "%s%064x%s"% (header, int(len(x) / 64), x)
    # print (draw_query)
    print (RpcCall('bash fillPixels.sh %s' % draw_query), file=sys.stderr)
    print (int(len(x) / 64), file=sys.stderr)
