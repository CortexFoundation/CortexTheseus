# equihash 简介

# 概要

equihash是zcash中的挖矿算法。作为一种抗ASIC算法，它的内存开销比较高，验证解的正确性只需很少的计算量。本概要很大程度上参考了[1]。

equihash中解决的问题是一个generialized birthday problem (GBP)，具体内容如下：
1. 输入长度108字节的 block header，以及32字节的 nonce，合并为140字节的“work”
2. 将140字节的 work 与4个字节的 hash index 合并，进行 blake2b，一共产生$2^{20}$个字符串，每个50字节。将每个字符串对半拆开，各25字节，长度为200 bit，因此最终得到$2^{21}$个字符串。
3. 单个字符串长度200 bit，对应参数 N=200；21对应于 $N/(K+1)+1$，这里 K=9。
4. GBP的问题是，在这$2^{21}$个字符串中找到 $2^{K}=512$ 个字符串 ${X_1,X_2,...,X_{512}}$，使得他们有$$ X_1 \oplus X_2 ... \oplus X_{512} = 0, $$ 这里的$\oplus$是异或（XOR）运算。solution的长度为$21 \times 512 = 10752$ bit，也就是1344字节。
5. 得到解之后，将140字节的header，与3个字节的数值1344，以及1344字节的solution合并得到1487字节，进行两次sha256，再与target作比较。如果小于target，就得到一个最终的解。
6. Equihash对GBP的另一个改进是algorithm binding，通过特定的约束保证只有Wagner算法的结果复合条件，从而避开了low amortised的解法的影响[3]，这里不再深入。

# 算法

算法部分参考了[2]，xenoncat的实现。
算法一共有 $K+1=10$轮，每一轮处理20 bit。
* __stage 0__
blake2b，生成$2^{21}$个字符串XorWork0
* __stage 1__
对XorWork0，寻找第0-20个 bit XOR为0的pairs1，XOR之后的结果保存在XorWork1
* __stage 2__
对XorWork1，寻找第21-40个 bit XOR为0的pairs2， XOR之后保存在XorWork2 
* ...
* __stage 9__:
对XorWork8，寻找第160-180个 bit XOR为0的pairs9，XOR之后保存在XorWork9
* __stage 10__:
对XorWork9，寻找第181-200个 bit XOR为0的pairs10。

注意每一轮在上一轮的XOR结果中寻找XOR为0的pair，得到的是pair的pair。具体地说，第一轮中，pairs1中的每一个代表了2个index的组合，第二轮中，pairs2中的每一个代表了4个index的组合，以此类推。

在整个计算中，XOR结果XorWork不保留，只需要分配2个buffer。但每一轮的pairs会被保存到最后，在最后一轮中得到的pair可以通过反向查找还原出对应的512个index。

每一轮产生的pair数量大约为2 million（2965504），其中包含大量trivial或者error的解，例如重复出现两个相同的index，但是每一轮中基本不做处理。在最后一轮中，筛选出正确的解。从数学期望上说会剩下2个解，同时有3%的几率会丢失解。


寻找XOR为0的pair，意味着找到20个bit完全相同（colliding）的两个字符串。为了加速这一过程，一个优化是将前8个bit用bucket排序，需要256个bucket。之后剩下的12个bit排序后做一次scan，找出colliding的pair。

保存两个范围 $0 \sim 2965504 < 2^{22}$的index 需要44个bit，通过pair compression可以把pair压缩到32bit，从而可以用4字节整型保存，细节参考[2]。

# 参考文献
1. easy to follow description of the equihash algorithm
https://forum.z.cash/t/easy-to-follow-description-of-the-equihash-algorithm/12689/2


2. algorithm description
https://github.com/xenoncat/equihash-xenon/blob/master/notes/algorithm%20description.pdf

3. Equihash: Asymmetric Proof-of-Work Based on the Generalized Birthday Problem
https://www.cryptolux.org/images/b/b9/Equihash.pdf