# Cortex Miner
1. [参考1](https://github.com/tromp/cuckoo/blob/master/doc/cuckoo.pdf?raw=true)
2. [参考2](https://github.com/tromp/cuckoo)


## cuckoo cycle二分图
1. 二分图是一个图中每个点被分成两个集合，并且任意一条边的两个点分属不同的两个集合。cuckoo cycle里，图的点数N，边数M，N=2\*M，两个集合的点数是一样的。下图是一个N=8，M=4的cuckoo cycle二分图：
<img src="https://github.com/mimblewimble/grin/blob/master/doc/pow/images/cuckoo_base_numbered_few_edges.png" />

2. cuckoo cycle二分图是采用siphash随机生成的，siphash的输入是4个64位的key和一个nonce。每个图对应4个key和2^N个nonce，根据每个nonce生成一个节点，两个连续的nonce生成的点组成一条边。如上图，假设这里的nonce是0\~8，并且siphash的值也是0\~8。这里的图没有环。
3. cuckoo cycle问题是在像上面这样规则生成的图里找固定长度L的环，假如在上图中增加几条边，我们可以得到一个环：
<img src="https://github.com/mimblewimble/grin/blob/master/doc/pow/images/cuckoo_base_numbered_more_edges_cycle.png" />

4. 随着图的规模增大，L的增大，问题的难度也随之增大。这样的图的稀疏度=M/N=1/2, 平均的度=1，找到一个长度为L的难度=1/L。那么我们现在要在N=30，M=29的图上找一个L=42的环，期望越快越好。

## 问题定义
1. 输入：4个64位的key，边的个数Edgebits=29，要找的环长度L
2. 生成随机二分图：N=2^(Edgebits+1)表示节点的个数，M=2^Edgebits表示边的数量, 图G(V,E), V={v0,...,vN-1}，E={e0,...,eM-1}，vi=siphash(i, key), ei=(v2\*j, v2\*j+1)。
3. 找固定长度L的环：在步骤2生成的二分图中查找是否有长度为L的环，如果找到，则返回该环中每条边对应的nonce作为输入key的一个解。

## 分析
1. 问题的规模是EdgeBits=29， L=42，稀疏度=1\/2，平均度=1，找长度L的环的难度=1\/L。这是一个比较大且稀疏的随机二分图，多数的边是无效的，因此tromp采用先删除无关边再找环的方法。
2. 删边的依据：先对二分图每条边的某一侧节点进行统计，统计每个点出现的次数，然后将统计次数为0或1的节点对应的边删除，因为如果一条边出现在一个环中，那么边对应的节点度大于1。
3. 对于删边之后的图采用并查集找长度为L的环。

## tromp删边方案1 [lean](https://github.com/tromp/cuckoo/blob/master/src/cuckoo/lean.cu)
1. cpu端循环调用cuda计数和删边的kernel，第2\*i次对图的左边节点进行计数和删边，第2\*i+1次对图的右边节点进行计数和删边
2. count_node_deg: 设cuda线程总数为nthreads，总的边数edges=2^29，每个线程生成edges/nthreads条边，同时对每条边进行计数。计数方法是在global memory中维护一个bitset，bitset的大小为edges/32 \* 2，乘2的原因是每个节点需要2个bit来计数。每个线程生成边后将节点映射到bitset对应位置，然后进行atoimc对bitset进行计数。
3. kill_leaf_edges：同样的线程数，每个线程重新生成边，将节点映射到bitset判断第二个bit是否为1，如果为0则删除。一条边是否已经被删除也是通过维护一个bitset来标志，同样也是通过atomic来更新这个bitset。
4. 迭代指定次数后，将剩余的图返回给cpu端，cpu端进行并查集找环。
5. 该删边方案的优点在于删边过程不保存边的数据，每次重新计算siphash，只需要两个bitset的显存空间。

## tromp删边方案2 [mean](https://github.com/tromp/cuckoo/blob/master/src/cuckoo/mean.cu)
lean方案需要大量的对global memory进行atomic操作，会有很多的原子冲突，使得效率低下。mean的方法是将边数据保存下来，并对边进行分桶，同样分别按左边节点和右边节点分别进行分桶、计数、删边。tromp将节点按二进制分为X(6位)Y(6位)Z(17位），分别按X分桶，在对每个桶按Y分桶，然后对每个桶进行Z的计数，然后删边。

GPU上对global memory的访问是昂贵的。像1080ti上每个SM有48KB的shared memory，可以利用起来进行访存优化。

1. SeedA：线程规模：64\*64个block，一个block有256个thread，每个thread生成 edges/nthreads 条边，edges=2^29, nthreads为总的线程数。 每个block需要64\*32的shared memory，作为block生成边的临时存储区，每个线程通过atomic先在shared中按X分成到64个桶中，然后在对global进行atomic找到写入位置，把shared写入到global中。虽然这一步只分成64个桶，但是还是把global也分成64\*64个区块，只不过连续的64个块是一个桶， 这样落在一个桶里的可以根据blockId分散到64个区块中去，进一步减少冲突。 总结，SeedA生成所有边，并对边分成64个桶，每个桶又划分为64个子桶，所以有64\*64个桶。 
2. SeedB：SeedB是对SeedA生成每个桶进行按Y进行分桶，因此这里block还是64*64个，每个block对一个小桶进行按Y重新分配到新的桶中。同样先对边在shared中分桶，然后在写入到global中。SeedB的global冲突只有在SeedA中属于同一个桶的数据。
3. Round：SeedA和SeedB之后，同一个桶里的XY是一样的，那么只要统计一个桶里的Z出现的次数，然后进行删边。每个边需要2bit来计数，Z有17位，那么采用bitset需要(2^17)/16*sizeof(int)=32k空间，shared memory是足够的，因此每个block对一个桶进行计数，然后同步操作，然后判断每个边对应bitset的第二bit是否为1，为0的话丢弃，为1的话则根据这条边的另外一个节点的XY存储到global对应桶中。CPU端会调用指定次数的Round，使得边数被删除到一定数量。
4. Tail：在前面的步骤中会为每一个桶维护一个计数器，记录每个桶有多少条边。在这一步，需要把Round之后的每个桶数据拼接起来，这样在这步之后可以一次把数据拷贝会CPU端。
<img src="https://github.com/CortexFoundation/PoolMiner/blob/zkh_dev/mean_case1.png" />

5. 将剩余的边数据考回CPU，进行找环操作。
6. mean方法比lean要快将近10倍，但是需要大量显存空间，其中存储图需要2^29*sizeof(int)*2=4G,分桶的时候需要额外的一个缓冲区4G，因此总共需要8G内存。tromp对此进行了优化，额外的缓冲区只需要2G：分桶按两次进行，一次对一半的边分桶，这样就只需要2G了。因此内存使用优化后需要6G内存。
