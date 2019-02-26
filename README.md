# PoolMiner
1. [参考1](https://github.com/tromp/cuckoo/blob/master/doc/cuckoo.pdf?raw=true)
2. [参考2](https://github.com/tromp/cuckoo)


## 算法描述
1. 输入：4个64位的key，边的个数EdgeBits，要找的环长度L
2. 生成随机二分图：定义nonce=0\~2^EdgeBits 的32位无符号整数，计算每个nonce和key对应的siphash值（32位uint，0~2^EdgeBits），siphash(2*nonce)作为一条边的左节点，siphash(2*nonce+1)作为一条边的右节点。如此，得到一个有2^EdgeBits条边和2^(EdgeBits+1)个节点的二分图。siphash是一个生成伪随机数的函数。
3. 找固定长度L的环：在步骤2生成的二分图中查找是否有长度为L的环，如果找到，则返回该环中每条边对应的nonce作为输入key的一个解。

## 分析
1. 问题的规模是EdgeBits=29， L=42。这是一个比较大且稀疏的随机二分图，多数的边是无效的，因此tromp采用先删除无关边再找环的方法。
2. 删边的依据：先对二分图每条边的某一侧节点进行统计，统计每个点出现的次数，然后将统计次数为0或1的节点对应的边删除，因为如果一条边出现在一个环中，那么边对应的节点度大于1。
3. 对于删边之后的图采用并查集找长度为L的环。

## tromp删边方案1 [lean](https://github.com/tromp/cuckoo/blob/master/src/cuckoo/lean.cu)
1. cpu端循环调用cuda计数和删边的kernel，第2*i次对图的左边节点进行计数和删边，第2*i+1次对图的右边节点进行计数和删边
2. count_node_deg: 设cuda线程总数为nthreads，总的边数edges=2^29，每个线程生成edges/nthreads条边，同时对每条边进行计数。计数方法是在global memory中维护一个bitset，bitset的大小为edges/32 * 2，乘2的原因是每个节点需要2个bit来计数。每个线程生成边后将节点映射到bitset对应位置，然后进行atoimc对bitset进行计数。
3. kill_leaf_edges：同样的线程数，每个线程重新生成边，将节点映射到bitset判断第二个bit是否为1，如果为0则删除。一条边是否已经被删除也是通过维护一个bitset来标志，同样也是通过atomic来更新这个bitset。
4. 迭代指定次数后，将剩余的图返回给cpu端，cpu端进行并查集找环。
5. 该删边方案的优点在于删边过程不保存边的数据，每次重新计算siphash，只需要两个bitset的显存空间。

## tromp删边方案2 [mean](https://github.com/tromp/cuckoo/blob/master/src/cuckoo/mean.cu)
lean方案需要大量的对global memory进行atomic操作，会有很多的原子冲突，使得效率低下。mean的方法是将边数据保存下来，并对边进行分桶，同样分别按左边节点和右边节点分别进行分桶、计数、删边。tromp将节点按二进制分为X(6位)Y(6位)Z(17位），分别按X分桶，在对每个桶按Y分桶，然后对每个桶进行Z的计数，然后删边。

1. SeedA：线程规模：64\*64个block，一个block有256个thread，每个thread生成 edges/nthreads 条边，edges=2^29, nthreads为总的线程数。 每个block需要64\*32的shared memory，作为block生成边的临时存储区，每个线程通过atomic先在shared中按X分成到64个桶中，然后在对global进行atomic找到写入位置，把shared写入到global中。虽然这一步只分成64个桶，但是还是把global也分成64\*64个区块，只不过连续的64个块是一个桶， 这样落在一个桶里的可以根据blockId分散到64个区块中去，进一步减少冲突。 总结，SeedA生成所有边，并对边分成64个桶，每个桶又划分为64个子桶，所以有64\*64个桶。 
2. SeedB：SeedB是对SeedA生成每个桶进行按Y进行分桶，因此这里block还是64*64个，每个block对一个小桶进行按Y重新分配到新的桶中。同样先对边在shared中分桶，然后在写入到global中。SeedB的global冲突只有在SeedA中属于同一个桶的数据。
3. Round：SeedA和SeedB之后，同一个桶里的XY是一样的，那么只要统计一个桶里的Z出现的次数，然后进行删边。每个边需要2bit来计数，Z有17位，那么采用bitset需要(2^17)/16*sizeof(int)=32k空间，shared memory是足够的，因此每个block对一个桶进行计数，然后同步操作，然后判断每个边对应bitset的第二bit是否为1，为0的话丢弃，为1的话则根据这条边的另外一个节点的XY存储到global对应桶中。CPU端会调用指定次数的Round，使得边数被删除到一定数量。
4. Tail：在前面的步骤中会为每一个桶维护一个计数器，记录每个桶有多少条边。在这一步，需要把Round之后的每个桶数据拼接起来，这样在这步之后可以一次把数据拷贝会CPU端。
5. 将剩余的边数据考回CPU，进行找环操作。
6. mean方法比lean要快将近10倍，但是需要大量显存空间，其中存储图需要2^29*sizeof(int)*2=4G,分桶的时候需要额外的一个缓冲区4G，因此总共需要8G内存。tromp对此进行了优化，额外的缓冲区只需要2G：分桶按两次进行，一次对一半的边分桶，这样就只需要2G了。因此内存使用优化后需要6G内存。
