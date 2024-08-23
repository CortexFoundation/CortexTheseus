// Copyright 2019 The go-ethereum Authors
// This file is part of the CortexFoundation library.
//
// The CortexFoundation library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexFoundation library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexFoundation library. If not, see <http://www.gnu.org/licenses/>.

package params

import "github.com/CortexFoundation/CortexTheseus/common"

// MainnetBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the main Cortex network.
var MainnetBootnodes = []string{
	"enode://953226ae83e451fd78fe37bbedc55892f3e3407ae2f0c6a043ff35699fa8d45829a2490b05ce8c5de5f0b9af1b67fd5c1701eea7573a587abe0a7f8e93e5c694@35.178.69.64:40404",   // UK ipfs4
	"enode://78f3feb1781f61a6a52abeaeeefeddec5b52383d0b1d0cf2a14c92ba90dc60ff010f4489b9ca3bd103ae6867cc679fea1b8558bbd515a96e828a0c7b52869bb5@47.95.215.37:40404",   // HZ testnet
	"enode://881f6548ee68e087ec157e479b96046660b96385c6fc4fe5d00d65404b725e49c4b16f85a6c1e606412f5caf82a581266b7dc1568d0cec3ea738a5589fdd3d01@47.242.33.115:37566",  // HK centos
	"enode://7b6bc170ccc16f63e032b9531f4fe86dd5547a8a67ca57452966e5851450681867f53c064058f61af4ce808fbdafc027b6995c14b25c04de1b082e94ceb9a400@52.76.52.58:40404",    // SG node1
	"enode://375f719c2e31e10d592716397026a0c32fa6dff47c8261e0caed343d01ce5b77d81a3eb350532accd04832dd6beb240c40c40e13aa56188bbb5ac7b77a45f4a5@47.91.147.37:11223",   // HK b6
	"enode://3436bac8cf1109ceffc76e1bbcb4f0ce9f420a9928d59f5bca1887bd2e310877b0f1b8aefa91e0ae62f438081c6f7b0cc7fec6d627b622ac8a5763c9c8153ae6@47.75.211.148:40404",  // HK cere
	"enode://22fb39a2861385cc6f19d29ed1cada4c439e0eecc40f5e74aa2f3446b5499947645ae9fd65f0a4bffc294426ca6cea96d453ebd11794dc80d764824e9362caf9@139.196.32.192:40404", // SH bxxxxxx4
}

var V5Bootnodes = []string{
	"enr:-J64QIWO2L7v59o9sbwQJUt1F1Ixw_uCJP9HxxD3YoPPsrjeTVXEVcy4Gp_8cjdKuXFfeo8pCPAbTP8eHxlWvkS9f8qGAX90e893hGN0eGPHxoR1STAggIJpZIJ2NIJpcIQjskVAiXNlY3AyNTZrMaEClTImroPkUf14_je77cVYkvPjQHri8MagQ_81aZ-o1FiDdGNwgp3Ug3VkcIKd1A",
	"enr:-J64QB44lRspwf2S8yYRxJ44_yYzRddn0SicdVjDy8NgEcBudCw8ar1VK9aMQh_RVxcFEeW2ajuwEPM0LiBLCdTytEuGAYNpEWxHhGN0eGPHxoR1STAggIJpZIJ2NIJpcIQvX9cliXNlY3AyNTZrMaEDePP-sXgfYaalKr6u7v7d7FtSOD0LHQzyoUySupDcYP-DdGNwgp3Ug3VkcIKd1A",
	"enr:-J64QK4ZPVhaEV-IRu_FILqtKrmhkYg3ZEgOARbv1lkmU6VdaJUmzgdHhYUEPJuBZWH12rtIm-a4gzFWAbz5hz48YTiGAYOEyveMhGN0eGPHxoR1STAggIJpZIJ2NIJpcIQv8iFziXNlY3AyNTZrMaEDiB9lSO5o4IfsFX5Hm5YEZmC5Y4XG_E_l0A1lQEtyXkmDdGNwgpK-g3VkcIKSvg",
	"enr:-J64QBxjZqSlYanUFtWlfX3cYeFbXOl4HkfbNp0TjxyGDNeVN_akhgu38a4PvgYX9w2hTxRYy_vbxidRKvP4Kf7JdlyGAYPPUS9rhGN0eGPHxoR1STAggIJpZIJ2NIJpcIQ0TDQ6iXNlY3AyNTZrMaECe2vBcMzBb2PgMrlTH0_obdVUeopnyldFKWblhRRQaBiDdGNwgp3Ug3VkcIKd1A",
	"enr:-Jq4QKjOfpLnpJ8kzoYfVkb4VCQBL_ZxSS2239z3rfDPawPDU5jOwP_aBgXfNR3jI_dmeIe7xQkSJohM2Gxow7ffvc-CBaqEY3R4Y8fGhHVJMCCAgmlkgnY0gmlwhH8AAAGJc2VjcDI1NmsxoQM3X3GcLjHhDVknFjlwJqDDL6bf9HyCYeDK7TQ9Ac5bd4N0Y3CCK9eDdWRwgivX",
	"enr:-J64QAApph5w17sgoSOyZMm4m4uzqXzj_9BiQqaJXMQaAgEsLKHYN9B2EK8OfaVjWS13Fp-xLm7_lIJJ7Xnjo_eO_raGAYEetOrnhGN0eGPHxoR1STAggIJpZIJ2NIJpcIR_AAABiXNlY3AyNTZrMaECNDa6yM8RCc7_x24bvLTwzp9CCpko1Z9byhiHvS4xCHeDdGNwgp3Ug3VkcIKd1A",
	"enr:-J64QIUs0ONRt3X_D-iQ5IT_ah_iB3Skjq3o0XHciGJdGM_OIqeiC0B7-G_EmBJXprHnbN3-N_nvpbFK0DhEZnM3OJmGAYRGAWIYhGN0eGPHxoR1STAggIJpZIJ2NIJpcISLxCDAiXNlY3AyNTZrMaEDIvs5ooYThcxvGdKe0craTEOeDuzED150qi80RrVJmUeDdGNwgp3Ug3VkcIKd1A",
}

// TestnetBootnodes are the enode URLs of the P2P bootstrap nodes running on the
// Bernard test network.
var BernardBootnodes = []string{
	"enode://cba70f4f25fa9ab33d3a92e1e5350d4e821775fc015223c77487bbe05976c7d9ec0da7f778e97ecfde208cce28e4ca7ab22b131df87419809d44ef317efc8a90@47.88.214.96:37566",
}

var DoloresBootnodes = []string{
	"enode://3bba372ec3d25442c20031064bfabb472ab7dae72d447860388797336dc35f6012c7ced2f2946508a0297d0237c7dc42d41c88e7d7ca7a82a288fa46299bda88@47.89.178.175:37566",
}

// DiscoveryV5Bootnodes are the enode URLs of the P2P bootstrap nodes for the
// experimental RLPx v5 topic-discovery network.
////var DiscoveryV5Bootnodes = []string{
//}

// const dnsPrefix = "enrtree://AKLET737XA6CY7T4QAPFJCUZRZ46EFAGZIV6LOAGKTG45RRZSUUWI@"
const dnsPrefix = "enrtree://AKLET737XA6CY7T4QAPFJCUZRZ46EFAGZIV6LOAGKTG45RRZSUUWI@"

func KnownDNSNetwork(genesis common.Hash, protocol string) string {
	var net string
	switch genesis {
	case MainnetGenesisHash:
		net = "mainnet"
	case DoloresGenesisHash:
		net = "dolores"
	case BernardGenesisHash:
		net = "bernard"
	}
	return dnsPrefix + protocol + "." + net + ".coinbag.org"
}
