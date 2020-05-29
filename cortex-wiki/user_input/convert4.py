import sys
from PIL import Image

img = Image.open(sys.argv[1])
img = img.resize((28,28))
img = img.load()

h = '0123456789abcdef'
s = ''
for i in range(28):
    for j in range(28):
        t = 0
        if type(img[j, i]) is int:
            t = img[j, i] // 2
        else:
            for k in img[j, i]:
                t += k
            t //= len(img[j, i]) * 2
        s += h[t // 16] + h[t % 16]
ret = []
for i in range(0, len(s), 64):
    if i <= len(s):
        e = i+64
        if e > len(s):
            e = len(s)
        subs = s[i:e]
        if len(subs) < 64:
            subs = subs + '0' * (64 - len(subs))
        ret.append('0x' + subs)
    else:
        ret.append('0x' + '0' * (len(s) - i + 64) + s[i:])

print('[' + ','.join(['"' + x + '"' for x in ret]) + ']')
