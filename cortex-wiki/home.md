<!-- TITLE: Official Cortex Wiki -->
<!-- SUBTITLE: Your Guide to the AI on Blockchain Ecosystem -->

# Cortex Overview
### What is Cortex?
Cortex is the first public blockchain capable of executing of AI algorithms and AI DApps on the blockchain. Cortex provides an AI platform for developers to upload their models on the blockchain and be incorporated into smart contracts. Instead of a black box, we can run AI models on the blockchain in a decentralized, immutable, and transparent manner − network consensus verifies every step of the AI inference.

The MainNet was launched at the end of June, 2019. The whitepaper can be found <a href="https://www.cortexlabs.ai/Cortex_AI_on_Blockchain_EN.pdf">here</a>.

To put it in context, blockchain started with bitcoin, a decentralized digital currency. Then entered Ethereum, which allows programming on top of the blockchain, namely the smart contract. Now Cortex builds on top of Ethereum to enable AI-powered smart contract. 
<img src="/uploads/hiearchy.png" style="width:450px; margin-top: 1%; margin-bottom: 1%; "/>
(The relationship between Cortex, Ethereum, and Bitcoin)

### How does Cortex enable on-chain AI?

Executing AI models on the blockchain is a difficult engineering problem that was unsolved before the Cortex team developed their own. 

The specific solutions involve building a blockchain whose virtual machine utilizes the GPU and applying a quantization scheme to ensure that the execution of deep learning models is deterministic. The technical details of the original solution, endorsed by the official MXNet team, can be read <a href="https://medium.com/apache-mxnet/quantizing-neural-network-models-in-mxnet-for-strict-consistency-on-blockchain-b5c950674866">here</a>. Below is a high-level explanation. 

Traditionally, there are two huge obstacles to AI inference on blockchain:

1) Nondeterministic behaviors of DNN models. (If you’re getting slightly different / nondeterministic inference results across different devices, there’s no way for network consensus to occur on the blockchain.)
2) Resource constraint across devices in a blockchain network.

The solution developed by Cortex, which involves simulated quantization and integer-only inference, has (1) eliminated the nondeterministic behaviors of DNN models without significant loss of accuracy (2) accelerated DNN models’ inference

These two achievements have made running non-trivial AI models on the blockchain possible. 

### Won't running AI on the blockchain be extremely expensive?
With the CVM coupled with the quantization method developed by the Cortex team, running AI on the blockchain has become cost-effective and realistic. The transaction fee is extremely low if the network is not super busy. 

### How does Cortex compare to conventional blockchains like Ethereum?
Right now, conventional blockchains like Ethereum have virtual machines that run on the CPU, which cannot **realistically execute non-trivial AI models.** To incorporate any sort of AI into an Ethereum Dapp, for example, would require running the AI models off-chain, which defeats the purpose of a smart contract. The <a href="https://www.cortexlabs.ai">Cortex team </a> addresses this problem by building a virtual machine that runs on the GPU. This allows AI models to execute directly on the blockchain, enabling **true AI smart contract.** In addition, the CVM is backward-compatible with the EVM, so Ethereum developers can easily migrate their applications onto to the Cortex blockchain and on top of that, incorporate AI models into them.  

### How does Cortex compare to other AI on Blockchain projects?
Cortex is the only project that allows the on-chain execution of AI models.

### Why do we want to run AI model on the blockchain?

The reasons for running AI programs on the blockchain are exactly the same as those for running "traditional" programs on the blockchain: transparency, integrity, immutability and censorship-resistance. 

A simple example to illustrate the benefits of running AI programs on the blockchain: When an autonomous car crash, we need to look into the AI inferences that lead to the crash to determine the liabilities. Now if the AI models have been run on the blockchain, we can easily verify the inference steps even if the whole car is obliterated since there is no single point of failure.

### What does “Decentralized AI Autonomous Ecosystem” mean? How does Cortex decentralize AI?  

To understand this question, we need to first understand the infrastructure of the Cortex blockchain.

We offer a blockchain platform for AI developers to upload their models and for Dapp developers to easily integrate these models in their Dapps (without the hassle of dealing with different protocols of different AI model service providers). A helpful analogy is to think of Cortex as a decentralized eBay for machine learning model API calls, where the sellers are the AI model providers and the buyers are the Dapp developers. Via smart contracts, AI model provider gets a portion of the transaction fee whenever the model is inferred. 

This creates an ecosystem where individual AI developers, not just big corporations, are incentivized to upload their AI models to the blockchain, and Dapp developers have access to the best AI models in the world and have the freedom to choose between them. Hence the decentralization. The competition within the ecosystem between AI model providers will naturally lead to the evolution of better and better AI models. Hence the term “autonomous ecosystem.”

### Sometimes you use the word “AI” and sometimes you use the word “machine learning”, what is the relationship between these two terms?

Machine learning is a subfield of AI and by far the most promising one in helping achieve better artificial intelligence. The basic idea of machine learning is to train machines to certain perform tasks without explicitly programming them. Nowadays, they have been used rather interchangeably due to the dominance of machine learning as a method of aritificial intelligence. 

## Cortex Virtual Machine (CVM)
The Cortex Virtual Machine (CVM), is ported from the Ethereum Virtual Machine (EVM) with added support for AI inference and AI contracts. The CVM is compatible with EVM and capable of running both ethereum smart contracts and AI smart contracts. 

The CVM has two layers: infer instructions and deterministic inference engine. 

Infer instructions allows models to be called in contracts through instruction sets, including Infer (code: 0xc0), InferArray (code: 0xc1). 

The deterministic inference engine is called Synapse or the CVM Executor. It guarantees the consistency of AI inference results in heterogeneous computing environments, without significantly compromising performance or accuracy. Synapse proposes a model-based fixed-point execution framework and a corresponding deterministic machine learning operator library. AI developers can train and quantize their models using MRT to be executable on the CVM. 

Link for further technical details: https://github.com/CortexFoundation/CortexTheseus/tree/dev/infernet

## Model Representation Tool (MRT)
MRT, short for Model Representation Tool, is a deterministic quantization framework developed by Cortex that enables model inference in the limited-resource and strictly deterministic environment of blockchain, ushering in a new generation of AI smart contracts. 

MRT is designed to convert floating point models supported by nnvm into fixed-point models executable on the CVM while preventing significant loss of precision. The quantization method reduces the output number field of all layers of the model to INT8 or INT32 to simulate the floating-point network and converts the operators involved in the floating-point operation into integer operators using fuse and rewrite. Quantization ensures no overflow and guarantees the deterministic outcome of the model execution.

Link for further technical details: https://github.com/CortexFoundation/tvm-cvm

## Endorphin
Endorphin in Cortex is similar to Gas in Ethereum. 

To prevent abuse of the Cortex network, a fee is charged for each computational step executed in a transaction. Endorphin is the unit that measures the computational effort required for every transaction made on Cortex. The sender of each transaction is required to include an endorphin limit and an endorphin price. (Transaction fee = endorphin limit * endorphin price) The higher the endorphin price, the more likely and quicker miners will execute and verify the transaction. 

For AI inference, generally speaking, the cost of the endorphin is proportional to the size of the AI model. Cortex also sets an upper bound of 1GB on the parameter size of the model, corresponding to up to about 2 billion Float32 parameters.

Unlike the traditional blockchains where all of the block rewards go to miners, on Cortex, a portion of the block reward goes to the model providers to incentivize them to optimize better models. 

## Storage Layer
Cortex uses a distributed file system based on DHT (Distributed Hash Table) as a storage layer solution to reduce network load and network transmission cost. Storage quota is treated as a resource on the Cortex chain. Each mined block provides a 64K byte storage quota. Users freely bid on the use of storage quota with transaction fees.

AI models and input data are treated as a special type of smart contract on the Cortex chain. Creators need to send a special transaction with a function call to the contract in order to advance its upload progress. Each transaction will increase the file upload progress by 512K bytes, consuming the corresponding storage quota. 

After the completion of the upload phase, the file preparation phase is entered. This phase lasts for 100 blocks (about 25 minutes), and at the end of it, the prepared files enter the mature phase and can be used by AI inference contracts.

The owner is responsible for broadcasting the file to the network to reach the entire distributed file system; otherwise, the network consensus will reject relevant contract calls.  

## Cortex Remix
Cortex Remix is a browser-based compiler and IDE for programming language Solidity. It is based on the Remix IDE. It supports the compilation and deployment of AI smart contracts as well as debugging transactions.

Cortex Remix mainly consists of two functional modules: compilation and deployment. 

The compilation module supports compilation and optimization of AI smart contracts. Complied abi, bytecode, and additional information are also displayed in this module.

The deployment module can help deploy AI smart contracts to the Cortex network with the support of Cortex Wallet, allowing for on-chain inference. 

## Cortex Use Cases
* **DeFi:** credit report, anti-fraud in decentralized exchanges, p2p financing, insurance, cryptocurrency lending
* **Gaming:** AI judge, player agent, NPC, assistant/coaching/education
* **Global Climate Action & Carbon Credit Management and Trading:** collects environmental data and puts on the blockchain for a transparent carbon pricing system
* **AI Governance:** stablecoins based on machine learning, sentiment analysis, decentralized decision making, malicious behavior detection, smart resource allocation
* **Others:** on-chain data mining, facial recognition, recommendation, chatbot, machine translation, voice synthesis, etc.

Games and game AIs (such as Fomo3D or sports betting) are the most likely to be the first mass-market, because they form the shortest closed loop. Fintech blockchain technologies, such as anti-fraud in decentralized exchanges, credit systems, lending and smart investment, will probably be the next biggest market, considering all their data will be stored on the blockchain. In addition, we can realize stablecoin model controlled by AI models on Cortex, and the on-chain inference process makes it much more transparent than other stablecoins such as USDT. Furthermore,  decentralized autonomous token distribution, decentralized anonymous advertisement recommendation engine, autonomous driving, native Cortex AI Dapps, and really any mass markets that involve AI will see use cases on Cortex. 

### Specific Use Case Examples 
#### Defi 
For example, a decentralized lending app can run an AI algorithm to determine your interest rate based on your personal credit history. The AI used to analyze your credit score is not a black box, but instead, every step of the AI inference is transparent to prevent discrimination and ensure fairness. 

#### Gaming
CryptoKitties would be much cuter, more life-like, and unique if they incorporated AI. Imagine these kitties moving dynamically and behaving uniquely depending on your personal experience interacting with them. While Ethereum is not able to execute AI models and allow for this user experience, this is something that Cortex can uniquely enable.

#### Insurance
Blockchain finds many use cases in the insurance industry, where immutability, fairness, openness, and transparency are in high demand. AI can help improve underwriting decisions, better manage risks, and prevent fraud. An insurance DAO powered by on-chain AI can bring us better, cheaper, fairer, and less bureaucratic insurance. 

#### Decentralized Uber
Almost every aspect of Uber involves AI, from matching drivers and riders, route optimization, driver onboarding to determining fares. Therefore, if we want to build a decentralized Uber, it is necessary to be able to run AI on the blockchain. 

#### Anti-fake AI
The emergence of deepfakes (AI-manipulated videos that are indistinguishable to the human eye) poses a significant threat to society. Social stability will inevitably suffer if video recordings can simply be dismissed as untrustworthy in court. Anti-fake AI algorithms (algorithms that detect whether a video has been tampered with) will run on the blockchain to ensure their transparency and fairness, especially if there were to be used in court. 

The bullet points and specific examples above are only use cases thought of by the Cortex team alone. It is almost certain that the community will conceive many more and better use cases for AI on the blockchain. After all, rarely anyone thought of the best use cases for the internet today when it was first invented.


## Developing on Cortex

### Why should I develop on Cortex?

Cortex is the only blockchain that can realistically execute AI programs. As a DApp developer, if you want to incorporate any sort of AI algorithms into your DApps without resorting to off-chain AI solutions, Cortex is currently the only place you can do so. As an AI developer, you can upload AI models onto the storage of Cortex, and whenever your model is called, you will be rewarded some CTXCs that come from part of the transaction fee. 

### What are some of the models that are currently on Cortex?
To start, Cortex has 23 models, trained with four datasets, that serve 7 different purposes. All models have been quantized using MRT, ready to be inferred on the Cortex Virtual Machine (CVM). Since MainNet Launch, the amount of AI models has increased to 27.
![Image 1](/uploads/image-1.jpg "Image 1")

### Where can I get started developing on Cortex?

### What is the process to upload AI models to Cortex? Are there tutorials / documentations?


## Mining

### Mining intro and Spec
Cortex uses Cuckoo Cycle for its proof of work algorithm. 

Cuckoo Cycle is a graph theory-based algorithm that is far less energy-intensive than most other CPU, GPU or ASIC-bound PoW algorithms. The goal is to lower mining requirements, ensuring true decentralization and laying the foundation for future scalability.

The difficulty adjusts dynamically so that on average, a block is produced every 15 seconds, i.e., 15s block time. This rate ensures the synchronization of the system state while preventing double-spend and history alteration unless an attacker possesses more than 51% of the network's mining power.

**Mining Minimum Requirements**

System: Linux Ubuntu 16.04+
GPU: Nvidia GPU with >=10.7GB GDRAM (1080ti, 2080ti, Titan V, etc.)
Space: 2TB  (the size of the blockchain increases over time)
CUDA version: 9.2+
CUDA driver: 396+
Compiler: Go 1.10+, GCC 5.4+
Other stats
Quota general: 64k per block (model uploading space)
Uploading network bandwidth: 1MB/s
Model mature: 100 blocks
Model size limit: 1GB
TPS: 25.4
Pre-allocation: 149792458
Total reward for mining: 150000000
Total supply: 299792458
Reward: 2.5 per block (half every 4 years)  =  8409600

Link for further technical details: https://github.com/CortexFoundation/CortexTheseus
Cortex Miner (official implementation): https://github.com/CortexFoundation/PoolMiner

### Incentives for running full nodes on Cortex?
Cortex is similar to Ethereum and Bitcoin in this regard. There are no direct incentives for running a full node since there is no way to verify; however, mining pools, exchanges and data analysis of on-chain data require the running of full nodes. Also, the belief in the decentralized AI on blockchain ecosystem leads people to run full nodes.

### What opeartion systems are supported?
Right now the official miner implementation only support Linux. However, developers can look into the <a href="https://github.com/CortexFoundation/PoolMiner">source code</a> and port it to their own operating systems. 

### No support for AMD, only for NVDIA?
Right now we have only officially tested 1080ti and 2080ti. You can mine with titan v and titan rtx but it wouldn't be too cost-effective. No support for amd yet but may come later. In general we need VRAM above 10.5G. The point of the Cuckoo cycle algorithm is to bind the solution time to memory bandwidth, making mining ASIC-resistant to ensure true decentralization.

### Hardware needed to run a Cortex full node?
A 1060 GPU or even a MX150 GPU laptop is enough. 

### Does your platform support WASM or is WASM compatible?
WASM compatibility is not the focus of the Cortex MainNet, and WASM programming is not directly supported yet. However, if it turns out to be adopted by the blockchain industry mainstream, we will work to support it. We pay very close attention to Ethereum's progress in this regard.



## Tokenomics 

### Current exchanges that have completed the token swap for CTXCs?

As of right now, Binance, BitForex, BitAsset and OKex are the only four exchanges that have completed CTXC token swaps for their users, meaning that the addresses on these exchanges are MainNet addresses. Warning: DO NOT try to send ERC20 CTXCs into a Cortex MainNet address nor send MainNet CTXCs into an ERC20 (Eth) address. If you don’t understand what this means, please reach out to us and we can help explain further. Our suggestions is that if you hold ERC20 CTXCs, you carry out a token swap following this guide <a href="https://medium.com/cortexlabs/cortex-mainnet-token-swap-announcement-c06769c40663">here</a>.

### How long does token swap last?
At least a year.

### How many CTXCs will be issued?
Cortex Coins will be classically capped 299,792,458 coins. 150,000,000 Cortex Coins will be issued by mining.

![Screen Shot 2019 07 02 At 5 01 46 Pm](/uploads/screen-shot-2019-07-02-at-5-01-46-pm.png "Screen Shot 2019 07 02 At 5 01 46 Pm")



## Community
### Technical Development Plan
Next up, there will be three major milestones that we aim to reach.

First, we will upgrade the CVM + MRT. Right now, MRT is just a deterministic quantization framework — we hope to turn it into a full-fledged programming language that provides a complete instruction set and better deterministic support. Meanwhile, we want to upgrade the CVM to support more AI models, specifically dynamic models.

Second, we want to scale by working on possible layer 1 or 2 solution. Our initial target is to increase the TPS to 1000. We will also improve our DOPS (deterministic operations per second), which currently is at 10G DOPS.

Third, we will work to improve the privacy of the AI models. We are researching cryptographic solutions in order to implement shielded on-chain AI inferences. The current thinking is to use zk-starks or zk-snarks as one of the possible layer 1 solutions and trusted computing as a possible layer 2 solution.

### Community Development Plan
Upon the MainNet Launch and the release of Cortex source code, the Cortex team is opening up the development to the decentralized open-source community around the world; however, the Cortex Foundation will continue to provide underlying support to the Cortex blockchain and its open-source ecosystem in many ways: we have established a developer forum and will start to organize online and offline workshops to educate AI developers in navigating the Cortex ecosystem, set up bounty programs, and further develop the open-source technical collaboration mechanism (establish open model libraries and datasets); etc.

In terms of the AI Dapp ecosystem, the Cortex Foundation will work with Dapp developers and companies around the world to help implement more AI Dapps on the Cortex chain.
Meanwhile, we will work on cross-chain support: As programs in other blockchains may require running AI on-chain for reliability and transparency, they will be able to execute the AI models on our chain and get the results returned back to their chain for further processing.

Furthermore, we will closely collaborate with academia and industry for research partnerships and publications. Our unique core solutions to implement on-chain AI inference have already gained support from the official MXNet team. There will only be more of such collaborations and partnerships.



## Further Reading & References
<strong>Cortex Wallet: </strong> https://www.cortexlabs.ai/wallet
**Block Explorer:** https://cerebro.cortexlabs.ai/
**Cortex Remix IDE:** https://cerebro.cortexlabs.ai/remix
**Full Nodes:** https://github.com/CortexFoundation/CortexTheseus
**Cortex Miner:** https://github.com/CortexFoundation/PoolMiner
**Token Swap:** https://medium.com/cortexlabs/cortex-mainnet-token-swap-announcement-c06769c40663?postPublishedType=repub
**Whitepaper:** https://www.cortexlabs.ai/Cortex_AI_on_Blockchain_EN.pdf
**Cortex Forum:** https://www.cortexlabs.ai/forum

**Medium:** https://medium.com/cortexlabs
**Website:** https://www.cortexlabs.ai
**Twitter:** https://twitter.com/CTXCBlockchain
**Github:** https://github.com/CortexFoundation
**Facebook:** https://www.facebook.com/CTXCBlockchain/
**Reddit:** https://www.reddit.com/r/Cortex_Official/


# Technical Documentations
**MainNet Github Repo:** https://github.com/CortexFoundation/CortexTheseus

# JSON RPC API

[JSON](http://json.org/) is a lightweight data-interchange format. It can represent numbers, strings, ordered sequences of values, and collections of name/value pairs.

[JSON-RPC](http://www.jsonrpc.org/specification) is a stateless, light-weight remote procedure call (RPC) protocol. Primarily this specification defines several data structures and the rules around their processing. It is transport agnostic in that the concepts can be used within the same process, over sockets, over HTTP, or in many various message passing environments. It uses JSON ([RFC 4627](http://www.ietf.org/rfc/rfc4627.txt)) as data format.


## JavaScript API

To talk to an Cortex node from inside a JavaScript application use the [web3.js](https://github.com/ctxc/web3.js) library, which gives a convenient interface for the RPC methods.


## JSON-RPC Endpoint

Default JSON-RPC endpoints:

| Client | URL |
|-------|:------------:|
| Go |http://localhost:8545 | 

### Go

You can start the HTTP JSON-RPC with the `--rpc` flag 
```bash
build/bin/cortex --rpc
```

change the default port (8545) and listing address (localhost) with:

```bash
build/bin/cortex --rpc --rpcaddr <ip> --rpcport <portnumber>
```

If accessing the RPC from a browser, CORS will need to be enabled with the appropriate domain set. Otherwise, JavaScript calls are limit by the same-origin policy and requests will fail:

```bash
build/bin/cortex --rpc --rpccorsdomain "http://localhost:3000"
```

The JSON RPC can also be started from the [console](https://github.com/ctxc/go-ctxc/wiki/JavaScript-Console) using the `admin.startRPC(addr, port)` command.


## JSON-RPC support

| | cpp-ctxc | go-ctxc | py-ctxc| parity | pantheon |
|-------|:------------:|:-----------:|:-----------:|:-----:|:-----:|
| JSON-RPC 1.0 | &#x2713; | | | | |
| JSON-RPC 2.0 | &#x2713; | &#x2713; | &#x2713; | &#x2713; | &#x2713; |
| Batch requests | &#x2713; |  &#x2713; |  &#x2713; | &#x2713; | &#x2713; |
| HTTP | &#x2713; | &#x2713; | &#x2713; | &#x2713; | &#x2713; |
| IPC | &#x2713; | &#x2713; | | &#x2713; | |
| WS | | &#x2713; | | &#x2713; | &#x2713; |

## HEX value encoding

At present there are two key datatypes that are passed over JSON: unformatted byte arrays and quantities. Both are passed with a hex encoding, however with different requirements to formatting:

When encoding **QUANTITIES** (integers, numbers): encode as hex, prefix with "0x", the most compact representation (slight exception: zero should be represented as "0x0"). Examples:
- 0x41 (65 in decimal)
- 0x400 (1024 in decimal)
- WRONG: 0x (should always have at least one digit - zero is "0x0")
- WRONG: 0x0400 (no leading zeroes allowed)
- WRONG: ff (must be prefixed 0x)

When encoding **UNFORMATTED DATA** (byte arrays, account addresses, hashes, bytecode arrays): encode as hex, prefix with "0x", two hex digits per byte. Examples:
- 0x41 (size 1, "A")
- 0x004200 (size 3, "\0B\0")
- 0x (size 0, "")
- WRONG: 0xf0f0f (must be even number of digits)
- WRONG: 004200 (must be prefixed 0x)

Currently [cpp-ctxc](https://github.com/ctxc/cpp-ctxc),[go-ctxc](https://github.com/ctxc/go-ctxc) and [parity](https://github.com/paritytech/parity) provide JSON-RPC communication over http and IPC (unix socket Linux and OSX/named pipes on Windows). Version 1.4 of go-ctxc, version 1.6 of Parity and version 0.8 of Pantheon onwards have websocket support.

## The default block parameter

The following methods have an extra default block parameter:

- [ctxc_getBalance](#ctxc_getbalance)
- [ctxc_getCode](#ctxc_getcode)
- [ctxc_getTransactionCount](#ctxc_gettransactioncount)
- [ctxc_getStorageAt](#ctxc_getstorageat)
- [ctxc_call](#ctxc_call)

When requests are made that act on the state of ctxc, the last default block parameter determines the height of the block.

The following options are possible for the defaultBlock parameter:

- `HEX String` - an integer block number
- `String "earliest"` for the earliest/genesis block
- `String "latest"` - for the latest mined block
- `String "pending"` - for the pending state/transactions

## Curl Examples Explained

The curl options below might return a response where the node complains about the content type, this is because the --data option sets the content type to application/x-www-form-urlencoded . If your node does complain, manually set the header by placing -H "Content-Type: application/json" at the start of the call.

The examples also do not include the URL/IP & port combination which must be the last argument given to curl e.x. 127.0.0.1:8545

## JSON-RPC methods

* [web3_clientVersion](#web3_clientversion)
* [web3_sha3](#web3_sha3)
* [net_version](#net_version)
* [net_peerCount](#net_peercount)
* [net_listening](#net_listening)
* [ctxc_protocolVersion](#ctxc_protocolversion)
* [ctxc_syncing](#ctxc_syncing)
* [ctxc_coinbase](#ctxc_coinbase)
* [ctxc_mining](#ctxc_mining)
* [ctxc_hashrate](#ctxc_hashrate)
* [ctxc_gasPrice](#ctxc_gasprice)
* [ctxc_accounts](#ctxc_accounts)
* [ctxc_blockNumber](#ctxc_blocknumber)
* [ctxc_getBalance](#ctxc_getbalance)
* [ctxc_getStorageAt](#ctxc_getstorageat)
* [ctxc_getTransactionCount](#ctxc_gettransactioncount)
* [ctxc_getBlockTransactionCountByHash](#ctxc_getblocktransactioncountbyhash)
* [ctxc_getBlockTransactionCountByNumber](#ctxc_getblocktransactioncountbynumber)
* [ctxc_getUncleCountByBlockHash](#ctxc_getunclecountbyblockhash)
* [ctxc_getUncleCountByBlockNumber](#ctxc_getunclecountbyblocknumber)
* [ctxc_getCode](#ctxc_getcode)
* [ctxc_sign](#ctxc_sign)
* [ctxc_sendTransaction](#ctxc_sendtransaction)
* [ctxc_sendRawTransaction](#ctxc_sendrawtransaction)
* [ctxc_call](#ctxc_call)
* [ctxc_estimateGas](#ctxc_estimategas)
* [ctxc_getBlockByHash](#ctxc_getblockbyhash)
* [ctxc_getBlockByNumber](#ctxc_getblockbynumber)
* [ctxc_getTransactionByHash](#ctxc_gettransactionbyhash)
* [ctxc_getTransactionByBlockHashAndIndex](#ctxc_gettransactionbyblockhashandindex)
* [ctxc_getTransactionByBlockNumberAndIndex](#ctxc_gettransactionbyblocknumberandindex)
* [ctxc_getTransactionReceipt](#ctxc_gettransactionreceipt)
* [ctxc_pendingTransactions](#ctxc_pendingtransactions)
* [ctxc_getUncleByBlockHashAndIndex](#ctxc_getunclebyblockhashandindex)
* [ctxc_getUncleByBlockNumberAndIndex](#ctxc_getunclebyblocknumberandindex)
* [ctxc_getCompilers](#ctxc_getcompilers)
* [ctxc_compileLLL](#ctxc_compilelll)
* [ctxc_compileSolidity](#ctxc_compilesolidity)
* [ctxc_compileSerpent](#ctxc_compileserpent)
* [ctxc_newFilter](#ctxc_newfilter)
* [ctxc_newBlockFilter](#ctxc_newblockfilter)
* [ctxc_newPendingTransactionFilter](#ctxc_newpendingtransactionfilter)
* [ctxc_uninstallFilter](#ctxc_uninstallfilter)
* [ctxc_getFilterChanges](#ctxc_getfilterchanges)
* [ctxc_getFilterLogs](#ctxc_getfilterlogs)
* [ctxc_getLogs](#ctxc_getlogs)
* [ctxc_getWork](#ctxc_getwork)
* [ctxc_submitWork](#ctxc_submitwork)
* [ctxc_submitHashrate](#ctxc_submithashrate)
* [ctxc_getProof](#ctxc_getproof)
* [db_putString](#db_putstring)
* [db_getString](#db_getstring)
* [db_putHex](#db_puthex)
* [db_getHex](#db_gctxcex) 
* [shh_post](#shh_post)
* [shh_version](#shh_version)
* [shh_newIdentity](#shh_newidentity)
* [shh_hasIdentity](#shh_hasidentity)
* [shh_newGroup](#shh_newgroup)
* [shh_addToGroup](#shh_addtogroup)
* [shh_newFilter](#shh_newfilter)
* [shh_uninstallFilter](#shh_uninstallfilter)
* [shh_getFilterChanges](#shh_getfilterchanges)
* [shh_getMessages](#shh_getmessages)

## JSON RPC API Reference

***

#### web3_clientVersion

Returns the current client version.

##### Parameters
none

##### Returns

`String` - The current client version.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":67}'

// Result
{
  "id":67,
  "jsonrpc":"2.0",
  "result": "Mist/v0.9.3/darwin/go1.4.1"
}
```

***

#### web3_sha3

Returns Keccak-256 (*not* the standardized SHA3-256) of the given data.

##### Parameters

1. `DATA` - the data to convert into a SHA3 hash.

##### Example Parameters
```js
params: [
  "0x68656c6c6f20776f726c64"
]
```

##### Returns

`DATA` - The SHA3 result of the given string.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"web3_sha3","params":["0x68656c6c6f20776f726c64"],"id":64}'

// Result
{
  "id":64,
  "jsonrpc": "2.0",
  "result": "0x47173285a8d7341e5e972fc677286384f802f8ef42a5ec5f03bbfa254cb01fad"
}
```

***

#### net_version

Returns the current network id.

##### Parameters
none

##### Returns

`String` - The current network id.
- `"1"`: Eth Mainnet
- `"2"`: Morden Testnet  (deprecated)
- `"3"`: Ropsten Testnet
- `"4"`: Rinkeby Testnet
- `"42"`: Kovan Testnet

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"net_version","params":[],"id":67}'

// Result
{
  "id":67,
  "jsonrpc": "2.0",
  "result": "3"
}
```

***

#### net_listening

Returns `true` if client is actively listening for network connections.

##### Parameters
none

##### Returns

`Boolean` - `true` when listening, otherwise `false`.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"net_listening","params":[],"id":67}'

// Result
{
  "id":67,
  "jsonrpc":"2.0",
  "result":true
}
```

***

#### net_peerCount

Returns number of peers currently connected to the client.

##### Parameters
none

##### Returns

`QUANTITY` - integer of the number of connected peers.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"net_peerCount","params":[],"id":74}'

// Result
{
  "id":74,
  "jsonrpc": "2.0",
  "result": "0x2" // 2
}
```

***

#### ctxc_protocolVersion

Returns the current ctxc protocol version.

##### Parameters
none

##### Returns

`String` - The current ctxc protocol version.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_protocolVersion","params":[],"id":67}'

// Result
{
  "id":67,
  "jsonrpc": "2.0",
  "result": "0x54"
}
```

***

#### ctxc_syncing

Returns an object with data about the sync status or `false`.


##### Parameters
none

##### Returns

`Object|Boolean`, An object with sync status data or `FALSE`, when not syncing:
  - `startingBlock`: `QUANTITY` - The block at which the import started (will only be reset, after the sync reached his head)
  - `currentBlock`: `QUANTITY` - The current block, same as ctxc_blockNumber
  - `highestBlock`: `QUANTITY` - The estimated highest block

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_syncing","params":[],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": {
    startingBlock: '0x384',
    currentBlock: '0x386',
    highestBlock: '0x454'
  }
}
// Or when not syncing
{
  "id":1,
  "jsonrpc": "2.0",
  "result": false
}
```

***

#### ctxc_coinbase

Returns the client coinbase address.


##### Parameters
none

##### Returns

`DATA`, 20 bytes - the current coinbase address.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_coinbase","params":[],"id":64}'

// Result
{
  "id":64,
  "jsonrpc": "2.0",
  "result": "0xc94770007dda54cF92009BFF0dE90c06F603a09f"
}
```

***

#### ctxc_mining

Returns `true` if client is actively mining new blocks.

##### Parameters
none

##### Returns

`Boolean` - returns `true` of the client is mining, otherwise `false`.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_mining","params":[],"id":71}'

// Result
{
  "id":71,
  "jsonrpc": "2.0",
  "result": true
}

```

***

#### ctxc_hashrate

Returns the number of hashes per second that the node is mining with.

##### Parameters
none

##### Returns

`QUANTITY` - number of hashes per second.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_hashrate","params":[],"id":71}'

// Result
{
  "id":71,
  "jsonrpc": "2.0",
  "result": "0x38a"
}

```

***

#### ctxc_gasPrice

Returns the current price per gas in wei.

##### Parameters
none

##### Returns

`QUANTITY` - integer of the current gas price in wei.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_gasPrice","params":[],"id":73}'

// Result
{
  "id":73,
  "jsonrpc": "2.0",
  "result": "0x09184e72a000" // 10000000000000
}
```

***

#### ctxc_accounts

Returns a list of addresses owned by client.


##### Parameters
none

##### Returns

`Array of DATA`, 20 Bytes - addresses owned by the client.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_accounts","params":[],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": ["0xc94770007dda54cF92009BFF0dE90c06F603a09f"]
}
```

***

#### ctxc_blockNumber

Returns the number of most recent block.

##### Parameters
none

##### Returns

`QUANTITY` - integer of the current block number the client is on.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_blockNumber","params":[],"id":1}'

// Result
{
  "id":83,
  "jsonrpc": "2.0",
  "result": "0xc94" // 1207
}
```

***

#### ctxc_getBalance

Returns the balance of the account of given address.

##### Parameters

1. `DATA`, 20 Bytes - address to check for balance.
2. `QUANTITY|TAG` - integer block number, or the string `"latest"`, `"earliest"` or `"pending"`, see the [default block parameter](#the-default-block-parameter)

##### Example Parameters
```js
params: [
   '0xc94770007dda54cF92009BFF0dE90c06F603a09f',
   'latest'
]
```

##### Returns

`QUANTITY` - integer of the current balance in wei.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getBalance","params":["0xc94770007dda54cF92009BFF0dE90c06F603a09f", "latest"],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0x0234c8a3397aab58" // 158972490234375000
}
```

***

#### ctxc_getStorageAt

Returns the value from a storage position at a given address. 

##### Parameters

1. `DATA`, 20 Bytes - address of the storage.
2. `QUANTITY` - integer of the position in the storage.
3. `QUANTITY|TAG` - integer block number, or the string `"latest"`, `"earliest"` or `"pending"`, see the [default block parameter](#the-default-block-parameter)

##### Returns

`DATA` - the value at this storage position.

##### Example
Calculating the correct position depends on the storage to retrieve. Consider the following contract deployed at `0x295a70b2de5e3953354a6a8344e616ed314d7251` by address `0x391694e7e0b0cce554cb130d723a9d27458f9298`.

```
contract Storage {
    uint pos0;
    mapping(address => uint) pos1;
    
    function Storage() {
        pos0 = 1234;
        pos1[msg.sender] = 5678;
    }
}
```

Retrieving the value of pos0 is straight forward:

```js
curl -X POST --data '{"jsonrpc":"2.0", "method": "ctxc_getStorageAt", "params": ["0x295a70b2de5e3953354a6a8344e616ed314d7251", "0x0", "latest"], "id": 1}' localhost:8545

{"jsonrpc":"2.0","id":1,"result":"0x00000000000000000000000000000000000000000000000000000000000004d2"}
```

Retrieving an element of the map is harder. The position of an element in the map is calculated with:
```js
keccack(LeftPad32(key, 0), LeftPad32(map position, 0))
```

This means to retrieve the storage on pos1["0x391694e7e0b0cce554cb130d723a9d27458f9298"] we need to calculate the position with:
```js
keccak(decodeHex("000000000000000000000000391694e7e0b0cce554cb130d723a9d27458f9298" + "0000000000000000000000000000000000000000000000000000000000000001"))
```
The gctxc console which comes with the web3 library can be used to make the calculation:
```js
> var key = "000000000000000000000000391694e7e0b0cce554cb130d723a9d27458f9298" + "0000000000000000000000000000000000000000000000000000000000000001"
undefined
> web3.sha3(key, {"encoding": "hex"})
"0x6661e9d6d8b923d5bbaab1b96e1dd51ff6ea2a93520fdc9eb75d059238b8c5e9"
```
Now to fetch the storage:
```js
curl -X POST --data '{"jsonrpc":"2.0", "method": "ctxc_getStorageAt", "params": ["0x295a70b2de5e3953354a6a8344e616ed314d7251", "0x6661e9d6d8b923d5bbaab1b96e1dd51ff6ea2a93520fdc9eb75d059238b8c5e9", "latest"], "id": 1}' localhost:8545

{"jsonrpc":"2.0","id":1,"result":"0x000000000000000000000000000000000000000000000000000000000000162e"}

```

***

#### ctxc_getTransactionCount

Returns the number of transactions *sent* from an address.


##### Parameters

1. `DATA`, 20 Bytes - address.
2. `QUANTITY|TAG` - integer block number, or the string `"latest"`, `"earliest"` or `"pending"`, see the [default block parameter](#the-default-block-parameter)

##### Example Parameters
```js
params: [
   '0xc94770007dda54cF92009BFF0dE90c06F603a09f',
   'latest' // state at the latest block
]
```

##### Returns

`QUANTITY` - integer of the number of transactions send from this address.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getTransactionCount","params":["0xc94770007dda54cF92009BFF0dE90c06F603a09f","latest"],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0x1" // 1
}
```

***

#### ctxc_getBlockTransactionCountByHash

Returns the number of transactions in a block from a block matching the given block hash.


##### Parameters

1. `DATA`, 32 Bytes - hash of a block.

##### Example Parameters
```js
params: [
   '0xb903239f8543d04b5dc1ba6579132b143087c68db1b2168786408fcbce568238'
]
```

##### Returns

`QUANTITY` - integer of the number of transactions in this block.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getBlockTransactionCountByHash","params":["0xc94770007dda54cF92009BFF0dE90c06F603a09f"],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0xc" // 11
}
```

***

#### ctxc_getBlockTransactionCountByNumber
> > 
Returns the number of transactions in a block matching the given block number.


##### Parameters

1. `QUANTITY|TAG` - integer of a block number, or the string `"earliest"`, `"latest"` or `"pending"`, as in the [default block parameter](#the-default-block-parameter).

##### Example Parameters
```js
params: [
   '0xe8', // 232
]
```

##### Returns

`QUANTITY` - integer of the number of transactions in this block.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getBlockTransactionCountByNumber","params":["0xe8"],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0xa" // 10
}
```

***

#### ctxc_getUncleCountByBlockHash

Returns the number of uncles in a block from a block matching the given block hash.


##### Parameters

1. `DATA`, 32 Bytes - hash of a block.

##### Example Parameters
```js
params: [
   '0xc94770007dda54cF92009BFF0dE90c06F603a09f'
]
```

##### Returns

`QUANTITY` - integer of the number of uncles in this block.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getUncleCountByBlockHash","params":["0xc94770007dda54cF92009BFF0dE90c06F603a09f"],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0xc" // 1
}
```

***

#### ctxc_getUncleCountByBlockNumber

Returns the number of uncles in a block from a block matching the given block number.


##### Parameters

1. `QUANTITY|TAG` - integer of a block number, or the string "latest", "earliest" or "pending", see the [default block parameter](#the-default-block-parameter).

```js
params: [
   '0xe8', // 232
]
```

##### Returns

`QUANTITY` - integer of the number of uncles in this block.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getUncleCountByBlockNumber","params":["0xe8"],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0x1" // 1
}
```

***

#### ctxc_getCode

Returns code at a given address.


##### Parameters

1. `DATA`, 20 Bytes - address.
2. `QUANTITY|TAG` - integer block number, or the string `"latest"`, `"earliest"` or `"pending"`, see the [default block parameter](#the-default-block-parameter).

##### Example Parameters
```js
params: [
   '0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b',
   '0x2'  // 2
]
```

##### Returns

`DATA` - the code from the given address.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getCode","params":["0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b", "0x2"],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0x600160008035811a818181146012578301005b601b6001356025565b8060005260206000f25b600060078202905091905056"
}
```

***

#### ctxc_sign

The sign method calculates an Eth specific signature with: `sign(keccak256("\x19Eth Signed Message:\n" + len(message) + message)))`.

By adding a prefix to the message makes the calculated signature recognisable as an Eth specific signature. This prevents misuse where a malicious DApp can sign arbitrary data (e.g. transaction) and use the signature to impersonate the victim.

**Note** the address to sign with must be unlocked. 

##### Parameters
account, message

1. `DATA`, 20 Bytes - address.
2. `DATA`, N Bytes - message to sign.

##### Returns

`DATA`: Signature

##### Example

```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_sign","params":["0x9b2055d370f73ec7d8a03e965129118dc8f5bf83", "0xdeadbeaf"],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0xa3f20717a250c2b0b729b7e5becbff67fdaef7e0699da4de7ca5895b02a170a12d887fd3b17bfdce3481f10bea41f45ba9f709d39ce8325427b57afcfc994cee1b"
}
```

An example how to use solidity ecrecover to verify the signature calculated with `ctxc_sign` can be found [here](https://gist.github.com/bas-vk/d46d83da2b2b4721efb0907aecdb7ebd). The contract is deployed on the testnet Ropsten and Rinkeby.

***

#### ctxc_sendTransaction

Creates new message call transaction or a contract creation, if the data field contains code.

##### Parameters

1. `Object` - The transaction object
  - `from`: `DATA`, 20 Bytes - The address the transaction is send from.
  - `to`: `DATA`, 20 Bytes - (optional when creating new contract) The address the transaction is directed to.
  - `gas`: `QUANTITY`  - (optional, default: 90000) Integer of the gas provided for the transaction execution. It will return unused gas.
  - `gasPrice`: `QUANTITY`  - (optional, default: To-Be-Determined) Integer of the gasPrice used for each paid gas
  - `value`: `QUANTITY`  - (optional) Integer of the value sent with this transaction
  - `data`: `DATA`  - The compiled code of a contract OR the hash of the invoked method signature and encoded parameters. For details see [Eth Contract ABI](https://github.com/ctxc/wiki/wiki/Eth-Contract-ABI)
  - `nonce`: `QUANTITY`  - (optional) Integer of a nonce. This allows to overwrite your own pending transactions that use the same nonce.

##### Example Parameters
```js
params: [{
  "from": "0xb60e8dd61c5d32be8058bb8eb970870f07233155",
  "to": "0xd46e8dd67c5d32be8058bb8eb970870f07244567",
  "gas": "0x76c0", // 30400
  "gasPrice": "0x9184e72a000", // 10000000000000
  "value": "0x9184e72a", // 2441406250
  "data": "0xd46e8dd67c5d32be8d46e8dd67c5d32be8058bb8eb970870f072445675058bb8eb970870f072445675"
}]
```

##### Returns

`DATA`, 32 Bytes - the transaction hash, or the zero hash if the transaction is not yet available.

Use [ctxc_getTransactionReceipt](#ctxc_gettransactionreceipt) to get the contract address, after the transaction was mined, when you created a contract.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_sendTransaction","params":[{see above}],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0xe670ec64341771606e55d6b4ca35a1a6b75ee3d5145a99d05921026d1527331"
}
```

***

#### ctxc_sendRawTransaction

Creates new message call transaction or a contract creation for signed transactions.

##### Parameters

1. `DATA`, The signed transaction data.

##### Example Parameters
```js
params: ["0xd46e8dd67c5d32be8d46e8dd67c5d32be8058bb8eb970870f072445675058bb8eb970870f072445675"]
```

##### Returns

`DATA`, 32 Bytes - the transaction hash, or the zero hash if the transaction is not yet available.

Use [ctxc_getTransactionReceipt](#ctxc_gettransactionreceipt) to get the contract address, after the transaction was mined, when you created a contract.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_sendRawTransaction","params":[{see above}],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0xe670ec64341771606e55d6b4ca35a1a6b75ee3d5145a99d05921026d1527331"
}
```

***

#### ctxc_call

Executes a new message call immediately without creating a transaction on the block chain.


##### Parameters

1. `Object` - The transaction call object
  - `from`: `DATA`, 20 Bytes - (optional) The address the transaction is sent from.
  - `to`: `DATA`, 20 Bytes  - The address the transaction is directed to.
  - `gas`: `QUANTITY`  - (optional) Integer of the gas provided for the transaction execution. ctxc_call consumes zero gas, but this parameter may be needed by some executions.
  - `gasPrice`: `QUANTITY`  - (optional) Integer of the gasPrice used for each paid gas
  - `value`: `QUANTITY`  - (optional) Integer of the value sent with this transaction
  - `data`: `DATA`  - (optional) Hash of the method signature and encoded parameters. For details see [Eth Contract ABI](https://github.com/ctxc/wiki/wiki/Eth-Contract-ABI)
2. `QUANTITY|TAG` - integer block number, or the string `"latest"`, `"earliest"` or `"pending"`, see the [default block parameter](#the-default-block-parameter)

##### Returns

`DATA` - the return value of executed contract.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_call","params":[{see above}],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0x"
}
```

***

#### ctxc_estimateGas

Generates and returns an estimate of how much gas is necessary to allow the transaction to complete. The transaction will not be added to the blockchain. Note that the estimate may be significantly more than the amount of gas actually used by the transaction, for a variety of reasons including EVM mechanics and node performance.

##### Parameters

See [ctxc_call](#ctxc_call) parameters, expect that all properties are optional. If no gas limit is specified gctxc uses the block gas limit from the pending block as an upper bound. As a result the returned estimate might not be enough to executed the call/transaction when the amount of gas is higher than the pending block gas limit.

##### Returns

`QUANTITY` - the amount of gas used.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_estimateGas","params":[{see above}],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0x5208" // 21000
}
```

***

#### ctxc_getBlockByHash

Returns information about a block by hash.


##### Parameters

1. `DATA`, 32 Bytes - Hash of a block.
2. `Boolean` - If `true` it returns the full transaction objects, if `false` only the hashes of the transactions.

##### Example Parameters
```js
params: [
   '0xe670ec64341771606e55d6b4ca35a1a6b75ee3d5145a99d05921026d1527331',
   true
]
```

##### Returns

`Object` - A block object, or `null` when no block was found:

  - `number`: `QUANTITY` - the block number. `null` when its pending block.
  - `hash`: `DATA`, 32 Bytes - hash of the block. `null` when its pending block.
  - `parentHash`: `DATA`, 32 Bytes - hash of the parent block.
  - `nonce`: `DATA`, 8 Bytes - hash of the generated proof-of-work. `null` when its pending block.
  - `sha3Uncles`: `DATA`, 32 Bytes - SHA3 of the uncles data in the block.
  - `logsBloom`: `DATA`, 256 Bytes - the bloom filter for the logs of the block. `null` when its pending block.
  - `transactionsRoot`: `DATA`, 32 Bytes - the root of the transaction trie of the block.
  - `stateRoot`: `DATA`, 32 Bytes - the root of the final state trie of the block.
  - `receiptsRoot`: `DATA`, 32 Bytes - the root of the receipts trie of the block.
  - `miner`: `DATA`, 20 Bytes - the address of the beneficiary to whom the mining rewards were given.
  - `difficulty`: `QUANTITY` - integer of the difficulty for this block.
  - `totalDifficulty`: `QUANTITY` - integer of the total difficulty of the chain until this block.
  - `extraData`: `DATA` - the "extra data" field of this block.
  - `size`: `QUANTITY` - integer the size of this block in bytes.
  - `gasLimit`: `QUANTITY` - the maximum gas allowed in this block.
  - `gasUsed`: `QUANTITY` - the total used gas by all transactions in this block.
  - `timestamp`: `QUANTITY` - the unix timestamp for when the block was collated.
  - `transactions`: `Array` - Array of transaction objects, or 32 Bytes transaction hashes depending on the last given parameter.
  - `uncles`: `Array` - Array of uncle hashes.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getBlockByHash","params":["0xe670ec64341771606e55d6b4ca35a1a6b75ee3d5145a99d05921026d1527331", true],"id":1}'

// Result
{
"id":1,
"jsonrpc":"2.0",
"result": {
    "number": "0x1b4", // 436
    "hash": "0xe670ec64341771606e55d6b4ca35a1a6b75ee3d5145a99d05921026d1527331",
    "parentHash": "0x9646252be9520f6e71339a8df9c55e4d7619deeb018d2a3f2d21fc165dde5eb5",
    "nonce": "0xe04d296d2460cfb8472af2c5fd05b5a214109c25688d3704aed5484f9a7792f2",
    "sha3Uncles": "0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347",
    "logsBloom": "0xe670ec64341771606e55d6b4ca35a1a6b75ee3d5145a99d05921026d1527331",
    "transactionsRoot": "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421",
    "stateRoot": "0xd5855eb08b3387c0af375e9cdb6acfc05eb8f519e419b874b6ff2ffda7ed1dff",
    "miner": "0x4e65fda2159562a496f9f3522f89122a3088497a",
    "difficulty": "0x027f07", // 163591
    "totalDifficulty":  "0x027f07", // 163591
    "extraData": "0x0000000000000000000000000000000000000000000000000000000000000000",
    "size":  "0x027f07", // 163591
    "gasLimit": "0x9f759", // 653145
    "gasUsed": "0x9f759", // 653145
    "timestamp": "0x54e34e8e" // 1424182926
    "transactions": [{...},{ ... }] 
    "uncles": ["0x1606e5...", "0xd5145a9..."]
  }
}
```

***

#### ctxc_getBlockByNumber

Returns information about a block by block number.

##### Parameters

1. `QUANTITY|TAG` - integer of a block number, or the string `"earliest"`, `"latest"` or `"pending"`, as in the [default block parameter](#the-default-block-parameter).
2. `Boolean` - If `true` it returns the full transaction objects, if `false` only the hashes of the transactions.

##### Example Parameters
```js
params: [
   '0x1b4', // 436
   true
]
```

##### Returns

See [ctxc_getBlockByHash](#ctxc_getblockbyhash)

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getBlockByNumber","params":["0x1b4", true],"id":1}'
```

Result see [ctxc_getBlockByHash](#ctxc_getblockbyhash)

***

#### ctxc_getTransactionByHash

Returns the information about a transaction requested by transaction hash.


##### Parameters

1. `DATA`, 32 Bytes - hash of a transaction

##### Example Parameters
```js
params: [
   "0x88df016429689c079f3b2f6ad39fa052532c56795b733da78a91ebe6a713944b"
]
```

##### Returns

`Object` - A transaction object, or `null` when no transaction was found:

  - `blockHash`: `DATA`, 32 Bytes - hash of the block where this transaction was in. `null` when its pending.
  - `blockNumber`: `QUANTITY` - block number where this transaction was in. `null` when its pending.
  - `from`: `DATA`, 20 Bytes - address of the sender.
  - `gas`: `QUANTITY` - gas provided by the sender.
  - `gasPrice`: `QUANTITY` - gas price provided by the sender in Wei.
  - `hash`: `DATA`, 32 Bytes - hash of the transaction.
  - `input`: `DATA` - the data send along with the transaction.
  - `nonce`: `QUANTITY` - the number of transactions made by the sender prior to this one.
  - `to`: `DATA`, 20 Bytes - address of the receiver. `null` when its a contract creation transaction.
  - `transactionIndex`: `QUANTITY` - integer of the transaction's index position in the block. `null` when its pending.
  - `value`: `QUANTITY` - value transferred in Wei.
  - `v`: `QUANTITY` - ECDSA recovery id
  - `r`: `QUANTITY` - ECDSA signature r
  - `s`: `QUANTITY` - ECDSA signature s

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getTransactionByHash","params":["0x88df016429689c079f3b2f6ad39fa052532c56795b733da78a91ebe6a713944b"],"id":1}'

// Result
{
  "jsonrpc":"2.0",
  "id":1,
  "result":{
    "blockHash":"0x1d59ff54b1eb26b013ce3cb5fc9dab3705b415a67127a003c3e61eb445bb8df2",
    "blockNumber":"0x5daf3b", // 6139707
    "from":"0xa7d9ddbe1f17865597fbd27ec712455208b6b76d",
    "gas":"0xc350", // 50000
    "gasPrice":"0x4a817c800", // 20000000000
    "hash":"0x88df016429689c079f3b2f6ad39fa052532c56795b733da78a91ebe6a713944b",
    "input":"0x68656c6c6f21",
    "nonce":"0x15", // 21
    "to":"0xf02c1c8e6114b1dbe8937a39260b5b0a374432bb",
    "transactionIndex":"0x41", // 65
    "value":"0xf3dbb76162000", // 4290000000000000
    "v":"0x25", // 37
    "r":"0x1b5e176d927f8e9ab405058b2d2457392da3e20f328b16ddabcebc33eaac5fea",
    "s":"0x4ba69724e8f69de52f0125ad8b3c5c2cef33019bac3249e2c0a2192766d1721c"
  }
}
```

***

#### ctxc_getTransactionByBlockHashAndIndex

Returns information about a transaction by block hash and transaction index position.


##### Parameters

1. `DATA`, 32 Bytes - hash of a block.
2. `QUANTITY` - integer of the transaction index position.

##### Example Parameters
```js
params: [
   '0xe670ec64341771606e55d6b4ca35a1a6b75ee3d5145a99d05921026d1527331',
   '0x0' // 0
]
```

##### Returns

See [ctxc_getTransactionByHash](#ctxc_gettransactionbyhash)

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getTransactionByBlockHashAndIndex","params":["0xc6ef2fc5426d6ad6fd9e2a26abeab0aa2411b7ab17f30a99d3cb96aed1d1055b", "0x0"],"id":1}'
```

Result see [ctxc_getTransactionByHash](#ctxc_gettransactionbyhash)

***

#### ctxc_getTransactionByBlockNumberAndIndex

Returns information about a transaction by block number and transaction index position.


##### Parameters

1. `QUANTITY|TAG` - a block number, or the string `"earliest"`, `"latest"` or `"pending"`, as in the [default block parameter](#the-default-block-parameter).
2. `QUANTITY` - the transaction index position.

##### Example Parameters
```js
params: [
   '0x29c', // 668
   '0x0' // 0
]
```

##### Returns

See [ctxc_getTransactionByHash](#ctxc_gettransactionbyhash)

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getTransactionByBlockNumberAndIndex","params":["0x29c", "0x0"],"id":1}'
```

Result see [ctxc_getTransactionByHash](#ctxc_gettransactionbyhash)

***

#### ctxc_getTransactionReceipt

Returns the receipt of a transaction by transaction hash.

**Note** That the receipt is not available for pending transactions.


##### Parameters

1. `DATA`, 32 Bytes - hash of a transaction

##### Example Parameters
```js
params: [
   '0xb903239f8543d04b5dc1ba6579132b143087c68db1b2168786408fcbce568238'
]
```

##### Returns

`Object` - A transaction receipt object, or `null` when no receipt was found:

  - `transactionHash `: `DATA`, 32 Bytes - hash of the transaction.
  - `transactionIndex`: `QUANTITY` - integer of the transaction's index position in the block.
  - `blockHash`: `DATA`, 32 Bytes - hash of the block where this transaction was in.
  - `blockNumber`: `QUANTITY` - block number where this transaction was in.
  - `from`: `DATA`, 20 Bytes - address of the sender.
  - `to`: `DATA`, 20 Bytes - address of the receiver. null when it's a contract creation transaction.
  - `cumulativeGasUsed `: `QUANTITY ` - The total amount of gas used when this transaction was executed in the block.
  - `gasUsed `: `QUANTITY ` - The amount of gas used by this specific transaction alone.
  - `contractAddress `: `DATA`, 20 Bytes - The contract address created, if the transaction was a contract creation, otherwise `null`.
  - `logs`: `Array` - Array of log objects, which this transaction generated.
  - `logsBloom`: `DATA`, 256 Bytes - Bloom filter for light clients to quickly retrieve related logs.
  
It also returns _either_ :

  - `root` : `DATA` 32 bytes of post-transaction stateroot (pre Byzantium)
  - `status`: `QUANTITY` either `1` (success) or `0` (failure) 


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getTransactionReceipt","params":["0xb903239f8543d04b5dc1ba6579132b143087c68db1b2168786408fcbce568238"],"id":1}'

// Result
{
"id":1,
"jsonrpc":"2.0",
"result": {
     transactionHash: '0xb903239f8543d04b5dc1ba6579132b143087c68db1b2168786408fcbce568238',
     transactionIndex:  '0x1', // 1
     blockNumber: '0xb', // 11
     blockHash: '0xc6ef2fc5426d6ad6fd9e2a26abeab0aa2411b7ab17f30a99d3cb96aed1d1055b',
     cumulativeGasUsed: '0x33bc', // 13244
     gasUsed: '0x4dc', // 1244
     contractAddress: '0xb60e8dd61c5d32be8058bb8eb970870f07233155', // or null, if none was created
     logs: [{
         // logs as returned by getFilterLogs, etc.
     }, ...],
     logsBloom: "0x00...0", // 256 byte bloom filter
     status: '0x1'
  }
}
```

***

#### ctxc_pendingTransactions

Returns the pending transactions list.

##### Parameters
none

##### Returns

`Array` - A list of pending transactions.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_pendingTransactions","params":[],"id":1}'

// Result
{
"id":1,
"jsonrpc":"2.0",
"result": [{ 
    blockHash: '0x0000000000000000000000000000000000000000000000000000000000000000',
    blockNumber: null,
    from: '0x28bdb9c230f4d5e45435e4d006326ee32e46cb31',
    gas: '0x204734',
    gasPrice: '0x4a817c800',
    hash: '0x8dfa6a59307a490d672494a171feee09db511f05e9c097e098edc2881f9ca4f6',
    input: '0x6080604052600',
    nonce: '0x12',
    to: null,
    transactionIndex: '0x0',
    value: '0x0',
    v: '0x3d',
    r: '0xaabc9ddafffb2ae0bac4107697547d22d9383667d9e97f5409dd6881ce08f13f',
    s: '0x69e43116be8f842dcd4a0b2f760043737a59534430b762317db21d9ac8c5034' 
   },....,{ 
    blockHash: '0x0000000000000000000000000000000000000000000000000000000000000000',
    blockNumber: null,
    from: '0x28bdb9c230f4d5e45435e4d006326ee32e487b31',
    gas: '0x205940',
    gasPrice: '0x4a817c800',
    hash: '0x8e4340ea3983d86e4b6c44249362f716ec9e09849ef9b6e3321140581d2e4dac',
    input: '0xe4b6c4424936',
    nonce: '0x14',
    to: null,
    transactionIndex: '0x0',
    value: '0x0',
    v: '0x3d',
    r: '0x1ec191ef20b0e9628c4397665977cbe7a53a263c04f6f185132b77fa0fd5ca44',
    s: '0x8a58e00c63e05cfeae4f1cf19f05ce82079dc4d5857e2cc281b7797d58b5faf' 
   }]
}
```

***

#### ctxc_getUncleByBlockHashAndIndex

Returns information about a uncle of a block by hash and uncle index position.


##### Parameters


1. `DATA`, 32 Bytes - hash a block.
2. `QUANTITY` - the uncle's index position.

```js
params: [
   '0xc6ef2fc5426d6ad6fd9e2a26abeab0aa2411b7ab17f30a99d3cb96aed1d1055b',
   '0x0' // 0
]
```

##### Returns

See [ctxc_getBlockByHash](#ctxc_getblockbyhash)

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getUncleByBlockHashAndIndex","params":["0xc6ef2fc5426d6ad6fd9e2a26abeab0aa2411b7ab17f30a99d3cb96aed1d1055b", "0x0"],"id":1}'
```

Result see [ctxc_getBlockByHash](#ctxc_getblockbyhash)

**Note**: An uncle doesn't contain individual transactions.

***

#### ctxc_getUncleByBlockNumberAndIndex

Returns information about a uncle of a block by number and uncle index position.


##### Parameters

1. `QUANTITY|TAG` - a block number, or the string `"earliest"`, `"latest"` or `"pending"`, as in the [default block parameter](#the-default-block-parameter).
2. `QUANTITY` - the uncle's index position.

##### Example Parameters
```js
params: [
   '0x29c', // 668
   '0x0' // 0
]
```

##### Returns

See [ctxc_getBlockByHash](#ctxc_getblockbyhash)

**Note**: An uncle doesn't contain individual transactions.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getUncleByBlockNumberAndIndex","params":["0x29c", "0x0"],"id":1}'
```

Result see [ctxc_getBlockByHash](#ctxc_getblockbyhash)

***

#### ctxc_getCompilers (DEPRECATED)

Returns a list of available compilers in the client.

##### Parameters
none

##### Returns

`Array` - Array of available compilers.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getCompilers","params":[],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": ["solidity", "lll", "serpent"]
}
```

***

#### ctxc_compileSolidity (DEPRECATED)

Returns compiled solidity code.

##### Parameters

1. `String` - The source code.

##### Example Parameters
```js
params: [
   "contract test { function multiply(uint a) returns(uint d) {   return a * 7;   } }",
]
```

##### Returns

`DATA` - The compiled source code.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_compileSolidity","params":["contract test { function multiply(uint a) returns(uint d) {   return a * 7;   } }"],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": {
      "code": "0x605880600c6000396000f3006000357c010000000000000000000000000000000000000000000000000000000090048063c6888fa114602e57005b603d6004803590602001506047565b8060005260206000f35b60006007820290506053565b91905056",
      "info": {
        "source": "contract test {\n   function multiply(uint a) constant returns(uint d) {\n       return a * 7;\n   }\n}\n",
        "language": "Solidity",
        "languageVersion": "0",
        "compilerVersion": "0.9.19",
        "abiDefinition": [
          {
            "constant": true,
            "inputs": [
              {
                "name": "a",
                "type": "uint256"
              }
            ],
            "name": "multiply",
            "outputs": [
              {
                "name": "d",
                "type": "uint256"
              }
            ],
            "type": "function"
          }
        ],
        "userDoc": {
          "methods": {}
        },
        "developerDoc": {
          "methods": {}
        }
      }

}
```

***

#### ctxc_compileLLL (DEPRECATED)

Returns compiled LLL code.

##### Parameters

1. `String` - The source code.

##### Example Parameters
```js
params: [
   "(returnlll (suicide (caller)))",
]
```

##### Returns

`DATA` - The compiled source code.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_compileLLL","params":["(returnlll (suicide (caller)))"],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0x603880600c6000396000f3006001600060e060020a600035048063c6888fa114601857005b6021600435602b565b8060005260206000f35b600081600702905091905056" // the compiled source code
}
```

***

#### ctxc_compileSerpent (DEPRECATED)

Returns compiled serpent code.

##### Parameters

1. `String` - The source code.

##### Example Parameters
```js
params: [
   "/* some serpent */",
]
```

##### Returns

`DATA` - The compiled source code.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_compileSerpent","params":["/* some serpent */"],"id":1}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0x603880600c6000396000f3006001600060e060020a600035048063c6888fa114601857005b6021600435602b565b8060005260206000f35b600081600702905091905056" // the compiled source code
}
```

***

#### ctxc_newFilter

Creates a filter object, based on filter options, to notify when the state changes (logs).
To check if the state has changed, call [ctxc_getFilterChanges](#ctxc_getfilterchanges).

##### A note on specifying topic filters:
Topics are order-dependent. A transaction with a log with topics [A, B] will be matched by the following topic filters:
* `[]` "anything"
* `[A]` "A in first position (and anything after)"
* `[null, B]` "anything in first position AND B in second position (and anything after)"
* `[A, B]` "A in first position AND B in second position (and anything after)"
* `[[A, B], [A, B]]` "(A OR B) in first position AND (A OR B) in second position (and anything after)"

##### Parameters

1. `Object` - The filter options:
  - `fromBlock`: `QUANTITY|TAG` - (optional, default: `"latest"`) Integer block number, or `"latest"` for the last mined block or `"pending"`, `"earliest"` for not yet mined transactions.
  - `toBlock`: `QUANTITY|TAG` - (optional, default: `"latest"`) Integer block number, or `"latest"` for the last mined block or `"pending"`, `"earliest"` for not yet mined transactions.
  - `address`: `DATA|Array`, 20 Bytes - (optional) Contract address or a list of addresses from which logs should originate.
  - `topics`: `Array of DATA`,  - (optional) Array of 32 Bytes `DATA` topics. Topics are order-dependent. Each topic can also be an array of DATA with "or" options.

##### Example Parameters
```js
params: [{
  "fromBlock": "0x1",
  "toBlock": "0x2",
  "address": "0x8888f1f195afa192cfee860698584c030f4c9db1",
  "topics": ["0x000000000000000000000000a94f5374fce5edbc8e2a8697c15331677e6ebf0b", null, ["0x000000000000000000000000a94f5374fce5edbc8e2a8697c15331677e6ebf0b", "0x0000000000000000000000000aff3454fce5edbc8cca8697c15331677e6ebccc"]]
}]
```

##### Returns

`QUANTITY` - A filter id.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_newFilter","params":[{"topics":["0x0000000000000000000000000000000000000000000000000000000012341234"]}],"id":73}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0x1" // 1
}
```

***

#### ctxc_newBlockFilter

Creates a filter in the node, to notify when a new block arrives.
To check if the state has changed, call [ctxc_getFilterChanges](#ctxc_getfilterchanges).

##### Parameters
None

##### Returns

`QUANTITY` - A filter id.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_newBlockFilter","params":[],"id":73}'

// Result
{
  "id":1,
  "jsonrpc":  "2.0",
  "result": "0x1" // 1
}
```

***

#### ctxc_newPendingTransactionFilter

Creates a filter in the node, to notify when new pending transactions arrive.
To check if the state has changed, call [ctxc_getFilterChanges](#ctxc_getfilterchanges).

##### Parameters
None

##### Returns

`QUANTITY` - A filter id.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_newPendingTransactionFilter","params":[],"id":73}'

// Result
{
  "id":1,
  "jsonrpc":  "2.0",
  "result": "0x1" // 1
}
```

***

#### ctxc_uninstallFilter

Uninstalls a filter with given id. Should always be called when watch is no longer needed.
Additonally Filters timeout when they aren't requested with [ctxc_getFilterChanges](#ctxc_getfilterchanges) for a period of time.


##### Parameters

1. `QUANTITY` - The filter id.

##### Example Parameters
```js
params: [
  "0xb" // 11
]
```

##### Returns

`Boolean` - `true` if the filter was successfully uninstalled, otherwise `false`.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_uninstallFilter","params":["0xb"],"id":73}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": true
}
```

***

#### ctxc_getFilterChanges

Polling method for a filter, which returns an array of logs which occurred since last poll.


##### Parameters

1. `QUANTITY` - the filter id.

##### Example Parameters
```js
params: [
  "0x16" // 22
]
```

##### Returns

`Array` - Array of log objects, or an empty array if nothing has changed since last poll.

- For filters created with `ctxc_newBlockFilter` the return are block hashes (`DATA`, 32 Bytes), e.g. `["0x3454645634534..."]`.
- For filters created with `ctxc_newPendingTransactionFilter ` the return are transaction hashes (`DATA`, 32 Bytes), e.g. `["0x6345343454645..."]`.
- For filters created with `ctxc_newFilter` logs are objects with following params:

  - `removed`: `TAG` - `true` when the log was removed, due to a chain reorganization. `false` if its a valid log.
  - `logIndex`: `QUANTITY` - integer of the log index position in the block. `null` when its pending log.
  - `transactionIndex`: `QUANTITY` - integer of the transactions index position log was created from. `null` when its pending log.
  - `transactionHash`: `DATA`, 32 Bytes - hash of the transactions this log was created from. `null` when its pending log.
  - `blockHash`: `DATA`, 32 Bytes - hash of the block where this log was in. `null` when its pending. `null` when its pending log.
  - `blockNumber`: `QUANTITY` - the block number where this log was in. `null` when its pending. `null` when its pending log.
  - `address`: `DATA`, 20 Bytes - address from which this log originated.
  - `data`: `DATA` - contains the non-indexed arguments of the log.
  - `topics`: `Array of DATA` - Array of 0 to 4 32 Bytes `DATA` of indexed log arguments. (In *solidity*: The first topic is the *hash* of the signature of the event (e.g. `Deposit(address,bytes32,uint256)`), except you declared the event with the `anonymous` specifier.)

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getFilterChanges","params":["0x16"],"id":73}'

// Result
{
  "id":1,
  "jsonrpc":"2.0",
  "result": [{
    "logIndex": "0x1", // 1
    "blockNumber":"0x1b4", // 436
    "blockHash": "0x8216c5785ac562ff41e2dcfdf5785ac562ff41e2dcfdf829c5a142f1fccd7d",
    "transactionHash":  "0xdf829c5a142f1fccd7d8216c5785ac562ff41e2dcfdf5785ac562ff41e2dcf",
    "transactionIndex": "0x0", // 0
    "address": "0x16c5785ac562ff41e2dcfdf829c5a142f1fccd7d",
    "data":"0x0000000000000000000000000000000000000000000000000000000000000000",
    "topics": ["0x59ebeb90bc63057b6515673c3ecf9438e5058bca0f92585014eced636878c9a5"]
    },{
      ...
    }]
}
```

***

#### ctxc_getFilterLogs

Returns an array of all logs matching filter with given id.


##### Parameters

1. `QUANTITY` - The filter id.

##### Example Parameters
```js
params: [
  "0x16" // 22
]
```

##### Returns

See [ctxc_getFilterChanges](#ctxc_getfilterchanges)

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getFilterLogs","params":["0x16"],"id":74}'
```

Result see [ctxc_getFilterChanges](#ctxc_getfilterchanges)

***

#### ctxc_getLogs

Returns an array of all logs matching a given filter object.

##### Parameters

1. `Object` - The filter options:
  - `fromBlock`: `QUANTITY|TAG` - (optional, default: `"latest"`) Integer block number, or `"latest"` for the last mined block or `"pending"`, `"earliest"` for not yet mined transactions.
  - `toBlock`: `QUANTITY|TAG` - (optional, default: `"latest"`) Integer block number, or `"latest"` for the last mined block or `"pending"`, `"earliest"` for not yet mined transactions.
  - `address`: `DATA|Array`, 20 Bytes - (optional) Contract address or a list of addresses from which logs should originate.
  - `topics`: `Array of DATA`,  - (optional) Array of 32 Bytes `DATA` topics. Topics are order-dependent. Each topic can also be an array of DATA with "or" options.
  - `blockhash`:  `DATA`, 32 Bytes - (optional) With the addition of EIP-234 (Gctxc >= v1.8.13 or Parity >= v2.1.0), `blockHash` is a new filter option which restricts the logs returned to the single block with the 32-byte hash `blockHash`.  Using `blockHash` is equivalent to `fromBlock` = `toBlock` = the block number with hash `blockHash`.  If `blockHash` is present in the filter criteria, then neither `fromBlock` nor `toBlock` are allowed.

##### Example Parameters
```js
params: [{
  "topics": ["0x000000000000000000000000a94f5374fce5edbc8e2a8697c15331677e6ebf0b"]
}]
```

##### Returns

See [ctxc_getFilterChanges](#ctxc_getfilterchanges)

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getLogs","params":[{"topics":["0x000000000000000000000000a94f5374fce5edbc8e2a8697c15331677e6ebf0b"]}],"id":74}'
```

Result see [ctxc_getFilterChanges](#ctxc_getfilterchanges)

***

#### ctxc_getWork

Returns the hash of the current block, the seedHash, and the boundary condition to be met ("target").

##### Parameters
none

##### Returns

`Array` - Array with the following properties:
  1. `DATA`, 32 Bytes - current block header pow-hash
  2. `DATA`, 32 Bytes - the seed hash used for the DAG.
  3. `DATA`, 32 Bytes - the boundary condition ("target"), 2^256 / difficulty.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getWork","params":[],"id":73}'

// Result
{
  "id":1,
  "jsonrpc":"2.0",
  "result": [
      "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
      "0x5EED00000000000000000000000000005EED0000000000000000000000000000",
      "0xd1ff1c01710000000000000000000000d1ff1c01710000000000000000000000"
    ]
}
```

***

#### ctxc_submitWork

Used for submitting a proof-of-work solution.


##### Parameters

1. `DATA`, 8 Bytes - The nonce found (64 bits)
2. `DATA`, 32 Bytes - The header's pow-hash (256 bits)
3. `DATA`, 32 Bytes - The mix digest (256 bits)

##### Example Parameters
```js
params: [
  "0x0000000000000001",
  "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
  "0xD1FE5700000000000000000000000000D1FE5700000000000000000000000000"
]
```

##### Returns

`Boolean` - returns `true` if the provided solution is valid, otherwise `false`.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0", "method":"ctxc_submitWork", "params":["0x0000000000000001", "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef", "0xD1GE5700000000000000000000000000D1GE5700000000000000000000000000"],"id":73}'

// Result
{
  "id":73,
  "jsonrpc":"2.0",
  "result": true
}
```

***

#### ctxc_submitHashrate

Used for submitting mining hashrate.


##### Parameters

1. `Hashrate`, a hexadecimal string representation (32 bytes) of the hash rate 
2. `ID`, String - A random hexadecimal(32 bytes) ID identifying the client

##### Example Parameters
```js
params: [
  "0x0000000000000000000000000000000000000000000000000000000000500000",
  "0x59daa26581d0acd1fce254fb7e85952f4c09d0915afd33d3886cd914bc7d283c"
]
```

##### Returns

`Boolean` - returns `true` if submitting went through succesfully and `false` otherwise.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0", "method":"ctxc_submitHashrate", "params":["0x0000000000000000000000000000000000000000000000000000000000500000", "0x59daa26581d0acd1fce254fb7e85952f4c09d0915afd33d3886cd914bc7d283c"],"id":73}'

// Result
{
  "id":73,
  "jsonrpc":"2.0",
  "result": true
}
```

***

#### ctxc_getProof

Returns the account- and storage-values of the specified account including the Merkle-proof.

##### getProof-Parameters

1. `DATA`, 20 bytes - address of the account or contract
2. `ARRAY`, 32 Bytes - array of storage-keys which should be proofed and included. See ctxc_getStorageAt
3. `QUANTITY|TAG` - integer block number, or the string "latest" or "earliest", see the default block parameter


##### Example Parameters
```
params: ["0x1234567890123456789012345678901234567890",["0x0000000000000000000000000000000000000000000000000000000000000000","0x0000000000000000000000000000000000000000000000000000000000000001"],"latest"]
```

##### getProof-Returns

Returns
`Object` - A account object:

`balance`: `QUANTITY` - the balance of the account. See ctxc_getBalance

`codeHash`: `DATA`, 32 Bytes - hash of the code of the account. For a simple Account without code it will return "0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"

`nonce`: `QUANTITY`, - nonce of the account. See ctxc_getTransactionCount

`storageHash`: `DATA`, 32 Bytes - SHA3 of the StorageRoot. All storage will deliver a MerkleProof starting with this rootHash.

`accountProof`: `ARRAY` - Array of rlp-serialized MerkleTree-Nodes, starting with the stateRoot-Node, following the path of the SHA3 (address) as key.

`storageProof`: `ARRAY` - Array of storage-entries as requested. Each entry is a object with these properties:

`key`: `QUANTITY` - the requested storage key
`value`: `QUANTITY` - the storage value
`proof`: `ARRAY` - Array of rlp-serialized MerkleTree-Nodes, starting with the storageHash-Node, following the path of the SHA3 (key) as path.

##### getProof-Example
```
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"ctxc_getProof","params":["0x1234567890123456789012345678901234567890",["0x0000000000000000000000000000000000000000000000000000000000000000","0x0000000000000000000000000000000000000000000000000000000000000001"],"latest"],"id":1}' -H "Content-type:application/json" http://localhost:8545

// Result
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "address": "0x1234567890123456789012345678901234567890",
    "accountProof": [
      "0xf90211a090dcaf88c40c7bbc95a912cbdde67c175767b31173df9ee4b0d733bfdd511c43a0babe369f6b12092f49181ae04ca173fb68d1a5456f18d20fa32cba73954052bda0473ecf8a7e36a829e75039a3b055e51b8332cbf03324ab4af2066bbd6fbf0021a0bbda34753d7aa6c38e603f360244e8f59611921d9e1f128372fec0d586d4f9e0a04e44caecff45c9891f74f6a2156735886eedf6f1a733628ebc802ec79d844648a0a5f3f2f7542148c973977c8a1e154c4300fec92f755f7846f1b734d3ab1d90e7a0e823850f50bf72baae9d1733a36a444ab65d0a6faaba404f0583ce0ca4dad92da0f7a00cbe7d4b30b11faea3ae61b7f1f2b315b61d9f6bd68bfe587ad0eeceb721a07117ef9fc932f1a88e908eaead8565c19b5645dc9e5b1b6e841c5edbdfd71681a069eb2de283f32c11f859d7bcf93da23990d3e662935ed4d6b39ce3673ec84472a0203d26456312bbc4da5cd293b75b840fc5045e493d6f904d180823ec22bfed8ea09287b5c21f2254af4e64fca76acc5cd87399c7f1ede818db4326c98ce2dc2208a06fc2d754e304c48ce6a517753c62b1a9c1d5925b89707486d7fc08919e0a94eca07b1c54f15e299bd58bdfef9741538c7828b5d7d11a489f9c20d052b3471df475a051f9dd3739a927c89e357580a4c97b40234aa01ed3d5e0390dc982a7975880a0a089d613f26159af43616fd9455bb461f4869bfede26f2130835ed067a8b967bfb80",
      "0xf90211a0395d87a95873cd98c21cf1df9421af03f7247880a2554e20738eec2c7507a494a0bcf6546339a1e7e14eb8fb572a968d217d2a0d1f3bc4257b22ef5333e9e4433ca012ae12498af8b2752c99efce07f3feef8ec910493be749acd63822c3558e6671a0dbf51303afdc36fc0c2d68a9bb05dab4f4917e7531e4a37ab0a153472d1b86e2a0ae90b50f067d9a2244e3d975233c0a0558c39ee152969f6678790abf773a9621a01d65cd682cc1be7c5e38d8da5c942e0a73eeaef10f387340a40a106699d494c3a06163b53d956c55544390c13634ea9aa75309f4fd866f312586942daf0f60fb37a058a52c1e858b1382a8893eb9c1f111f266eb9e21e6137aff0dddea243a567000a037b4b100761e02de63ea5f1fcfcf43e81a372dafb4419d126342136d329b7a7ba032472415864b08f808ba4374092003c8d7c40a9f7f9fe9cc8291f62538e1cc14a074e238ff5ec96b810364515551344100138916594d6af966170ff326a092fab0a0d31ac4eef14a79845200a496662e92186ca8b55e29ed0f9f59dbc6b521b116fea090607784fe738458b63c1942bba7c0321ae77e18df4961b2bc66727ea996464ea078f757653c1b63f72aff3dcc3f2a2e4c8cb4a9d36d1117c742833c84e20de994a0f78407de07f4b4cb4f899dfb95eedeb4049aeb5fc1635d65cf2f2f4dfd25d1d7a0862037513ba9d45354dd3e36264aceb2b862ac79d2050f14c95657e43a51b85c80",
      "0xf90171a04ad705ea7bf04339fa36b124fa221379bd5a38ffe9a6112cb2d94be3a437b879a08e45b5f72e8149c01efcb71429841d6a8879d4bbe27335604a5bff8dfdf85dcea00313d9b2f7c03733d6549ea3b810e5262ed844ea12f70993d87d3e0f04e3979ea0b59e3cdd6750fa8b15164612a5cb6567cdfb386d4e0137fccee5f35ab55d0efda0fe6db56e42f2057a071c980a778d9a0b61038f269dd74a0e90155b3f40f14364a08538587f2378a0849f9608942cf481da4120c360f8391bbcc225d811823c6432a026eac94e755534e16f9552e73025d6d9c30d1d7682a4cb5bd7741ddabfd48c50a041557da9a74ca68da793e743e81e2029b2835e1cc16e9e25bd0c1e89d4ccad6980a041dda0a40a21ade3a20fcd1a4abb2a42b74e9a32b02424ff8db4ea708a5e0fb9a09aaf8326a51f613607a8685f57458329b41e938bb761131a5747e066b81a0a16808080a022e6cef138e16d2272ef58434ddf49260dc1de1f8ad6dfca3da5d2a92aaaadc58080",
      "0xf851808080a009833150c367df138f1538689984b8a84fc55692d3d41fe4d1e5720ff5483a6980808080808080808080a0a319c1c415b271afc0adcb664e67738d103ac168e0bc0b7bd2da7966165cb9518080"
    ],
    "balance": "0x0",
    "codeHash": "0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
    "nonce": "0x0",
    "storageHash": "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421",
    "storageProof": [
      {
        "key": "0x0000000000000000000000000000000000000000000000000000000000000000",
        "value": "0x0",
        "proof": []
      },
      {
        "key": "0x0000000000000000000000000000000000000000000000000000000000000001",
        "value": "0x0",
        "proof": []
      }
    ]
  }
}
```

***

#### db_putString

Stores a string in the local database.

**Note** this function is deprecated and will be removed in the future.

##### Parameters

1. `String` - Database name.
2. `String` - Key name.
3. `String` - String to store.

##### Example Parameters
```js
params: [
  "testDB",
  "myKey",
  "myString"
]
```

##### Returns

`Boolean` - returns `true` if the value was stored, otherwise `false`.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"db_putString","params":["testDB","myKey","myString"],"id":73}'

// Result
{
  "id":1,
  "jsonrpc":"2.0",
  "result": true
}
```

***

#### db_getString

Returns string from the local database.

**Note** this function is deprecated and will be removed in the future.

##### Parameters

1. `String` - Database name.
2. `String` - Key name.

##### Example Parameters
```js
params: [
  "testDB",
  "myKey",
]
```

##### Returns

`String` - The previously stored string.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"db_getString","params":["testDB","myKey"],"id":73}'

// Result
{
  "id":1,
  "jsonrpc":"2.0",
  "result": "myString"
}
```

***

#### db_putHex

Stores binary data in the local database.

**Note** this function is deprecated and will be removed in the future.


##### Parameters

1. `String` - Database name.
2. `String` - Key name.
3. `DATA` - The data to store.

##### Example Parameters
```js
params: [
  "testDB",
  "myKey",
  "0x68656c6c6f20776f726c64"
]
```

##### Returns

`Boolean` - returns `true` if the value was stored, otherwise `false`.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"db_putHex","params":["testDB","myKey","0x68656c6c6f20776f726c64"],"id":73}'

// Result
{
  "id":1,
  "jsonrpc":"2.0",
  "result": true
}
```

***

#### db_getHex

Returns binary data from the local database.

**Note** this function is deprecated and will be removed in the future.


##### Parameters

1. `String` - Database name.
2. `String` - Key name.

##### Example Parameters
```js
params: [
  "testDB",
  "myKey",
]
```

##### Returns

`DATA` - The previously stored data.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"db_getHex","params":["testDB","myKey"],"id":73}'

// Result
{
  "id":1,
  "jsonrpc":"2.0",
  "result": "0x68656c6c6f20776f726c64"
}
```

***

#### shh_version

Returns the current whisper protocol version.

##### Parameters
none

##### Returns

`String` - The current whisper protocol version

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"shh_version","params":[],"id":67}'

// Result
{
  "id":67,
  "jsonrpc": "2.0",
  "result": "2"
}
```

***

#### shh_post

Sends a whisper message.

##### Parameters

1. `Object` - The whisper post object:
  - `from`: `DATA`, 60 Bytes - (optional) The identity of the sender.
  - `to`: `DATA`, 60 Bytes - (optional) The identity of the receiver. When present whisper will encrypt the message so that only the receiver can decrypt it.
  - `topics`: `Array of DATA` - Array of `DATA` topics, for the receiver to identify messages.
  - `payload`: `DATA` - The payload of the message.
  - `priority`: `QUANTITY` - The integer of the priority in a range from ... (?).
  - `ttl`: `QUANTITY` - integer of the time to live in seconds.

##### Example Parameters
```js
params: [{
  from: "0x04f96a5e25610293e42a73908e93ccc8c4d4dc0edcfa9fa872f50cb214e08ebf61a03e245533f97284d442460f2998cd41858798ddfd4d661997d3940272b717b1",
  to: "0x3e245533f97284d442460f2998cd41858798ddf04f96a5e25610293e42a73908e93ccc8c4d4dc0edcfa9fa872f50cb214e08ebf61a0d4d661997d3940272b717b1",
  topics: ["0x776869737065722d636861742d636c69656e74", "0x4d5a695276454c39425154466b61693532"],
  payload: "0x7b2274797065223a226d6",
  priority: "0x64",
  ttl: "0x64",
}]
```

##### Returns

`Boolean` - returns `true` if the message was send, otherwise `false`.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"shh_post","params":[{"from":"0xc931d93e97ab07fe42d923478ba2465f2..","topics": ["0x68656c6c6f20776f726c64"],"payload":"0x68656c6c6f20776f726c64","ttl":0x64,"priority":0x64}],"id":73}'

// Result
{
  "id":1,
  "jsonrpc":"2.0",
  "result": true
}
```

***

#### shh_newIdentity

Creates new whisper identity in the client.

##### Parameters
none

##### Returns

`DATA`, 60 Bytes - the address of the new identiy.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"shh_newIdentity","params":[],"id":73}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0xc931d93e97ab07fe42d923478ba2465f283f440fd6cabea4dd7a2c807108f651b7135d1d6ca9007d5b68aa497e4619ac10aa3b27726e1863c1fd9b570d99bbaf"
}
```

***

#### shh_hasIdentity

Checks if the client hold the private keys for a given identity.


##### Parameters

1. `DATA`, 60 Bytes - The identity address to check.

##### Example Parameters
```js
params: [  "0x04f96a5e25610293e42a73908e93ccc8c4d4dc0edcfa9fa872f50cb214e08ebf61a03e245533f97284d442460f2998cd41858798ddfd4d661997d3940272b717b1"
]
```

##### Returns

`Boolean` - returns `true` if the client holds the privatekey for that identity, otherwise `false`.


##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"shh_hasIdentity","params":["0x04f96a5e25610293e42a73908e93ccc8c4d4dc0edcfa9fa872f50cb214e08ebf61a03e245533f97284d442460f2998cd41858798ddfd4d661997d3940272b717b1"],"id":73}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": true
}
```

***

#### shh_newGroup

Creates a new group.

##### Parameters
none

##### Returns

`DATA`, 60 Bytes - the address of the new group.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"shh_newGroup","params":[],"id":73}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": "0xc65f283f440fd6cabea4dd7a2c807108f651b7135d1d6ca90931d93e97ab07fe42d923478ba2407d5b68aa497e4619ac10aa3b27726e1863c1fd9b570d99bbaf"
}
```

***

#### shh_addToGroup

Adds a whisper identity to the group.

##### Parameters

1. `DATA`, 60 Bytes - The identity address to add to a group.

##### Example Parameters
```js
params: [ "0x04f96a5e25610293e42a73908e93ccc8c4d4dc0edcfa9fa872f50cb214e08ebf61a03e245533f97284d442460f2998cd41858798ddfd4d661997d3940272b717b1"
]
```

##### Returns

`Boolean` - returns `true` if the identity was successfully added to the group, otherwise `false`.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"shh_addToGroup","params":["0x04f96a5e25610293e42a73908e93ccc8c4d4dc0edcfa9fa872f50cb214e08ebf61a03e245533f97284d442460f2998cd41858798ddfd4d661997d3940272b717b1"],"id":73}'

// Result
{
  "id":1,
  "jsonrpc": "2.0",
  "result": true
}
```

***

#### shh_newFilter

Creates filter to notify, when client receives whisper message matching the filter options.


##### Parameters

1. `Object` - The filter options:
  - `to`: `DATA`, 60 Bytes - (optional) Identity of the receiver. *When present it will try to decrypt any incoming message if the client holds the private key to this identity.*
  - `topics`: `Array of DATA` - Array of `DATA` topics which the incoming message's topics should match.  You can use the following combinations:
    - `[A, B] = A && B`
    - `[A, [B, C]] = A && (B || C)`
    - `[null, A, B] = ANYTHING && A && B` `null` works as a wildcard

##### Example Parameters
```js
params: [{
   "topics": ['0x12341234bf4b564f'],
   "to": "0x04f96a5e25610293e42a73908e93ccc8c4d4dc0edcfa9fa872f50cb214e08ebf61a03e245533f97284d442460f2998cd41858798ddfd4d661997d3940272b717b1"
}]
```

##### Returns

`QUANTITY` - The newly created filter.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"shh_newFilter","params":[{"topics": ['0x12341234bf4b564f'],"to": "0x2341234bf4b2341234bf4b564f..."}],"id":73}'

// Result
{
  "id":1,
  "jsonrpc":"2.0",
  "result": "0x7" // 7
}
```

***

#### shh_uninstallFilter

Uninstalls a filter with given id. Should always be called when watch is no longer needed.
Additonally Filters timeout when they aren't requested with [shh_getFilterChanges](#shh_getfilterchanges) for a period of time.


##### Parameters

1. `QUANTITY` - The filter id.

##### Example Parameters
```js
params: [
  "0x7" // 7
]
```

##### Returns

`Boolean` - `true` if the filter was successfully uninstalled, otherwise `false`.

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"shh_uninstallFilter","params":["0x7"],"id":73}'

// Result
{
  "id":1,
  "jsonrpc":"2.0",
  "result": true
}
```

***

#### shh_getFilterChanges

Polling method for whisper filters. Returns new messages since the last call of this method.

**Note** calling the [shh_getMessages](#shh_getmessages) method, will reset the buffer for this method, so that you won't receive duplicate messages.


##### Parameters

1. `QUANTITY` - The filter id.

##### Example Parameters
```js
params: [
  "0x7" // 7
]
```

##### Returns

`Array` - Array of messages received since last poll:

  - `hash`: `DATA`, 32 Bytes (?) - The hash of the message.
  - `from`: `DATA`, 60 Bytes - The sender of the message, if a sender was specified.
  - `to`: `DATA`, 60 Bytes - The receiver of the message, if a receiver was specified.
  - `expiry`: `QUANTITY` - Integer of the time in seconds when this message should expire (?).
  - `ttl`: `QUANTITY` -  Integer of the time the message should float in the system in seconds (?).
  - `sent`: `QUANTITY` -  Integer of the unix timestamp when the message was sent.
  - `topics`: `Array of DATA` - Array of `DATA` topics the message contained.
  - `payload`: `DATA` - The payload of the message.
  - `workProved`: `QUANTITY` - Integer of the work this message required before it was send (?).

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"shh_getFilterChanges","params":["0x7"],"id":73}'

// Result
{
  "id":1,
  "jsonrpc":"2.0",
  "result": [{
    "hash": "0x33eb2da77bf3527e28f8bf493650b1879b08c4f2a362beae4ba2f71bafcd91f9",
    "from": "0x3ec052fc33..",
    "to": "0x87gdf76g8d7fgdfg...",
    "expiry": "0x54caa50a", // 1422566666
    "sent": "0x54ca9ea2", // 1422565026
    "ttl": "0x64", // 100
    "topics": ["0x6578616d"],
    "payload": "0x7b2274797065223a226d657373616765222c2263686...",
    "workProved": "0x0"
    }]
}
```

***

#### shh_getMessages

Get all messages matching a filter. Unlike `shh_getFilterChanges` this returns all messages.

##### Parameters

1. `QUANTITY` - The filter id.

##### Example Parameters
```js
params: [
  "0x7" // 7
]
```

##### Returns

See [shh_getFilterChanges](#shh_getfilterchanges)

##### Example
```js
// Request
curl -X POST --data '{"jsonrpc":"2.0","method":"shh_getMessages","params":["0x7"],"id":73}'
```

Result see [shh_getFilterChanges](#shh_getfilterchanges)







