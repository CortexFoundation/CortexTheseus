import numpy as np
import sys
data = np.load('/home/tian/data/ctxc-42-12/warehouse/' + sys.argv[1] + '/data')
print >> sys.stderr, data
for i in range(28):
    for j in range(28):
        v = int(data[0][i][j])
        print ("%064x%064x%064x" % (i,j ,v))
        print >>sys.stderr, "%02x" % v,
