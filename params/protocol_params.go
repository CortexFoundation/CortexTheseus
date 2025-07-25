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

import (
	"github.com/CortexFoundation/CortexTheseus/common"
	"math/big"
)

const (
	//all configs should not be changed
	GasLimitBoundDivisor uint64 = 1024               // The bound divisor of the gas limit, used in update calculations.
	MinGasLimit          uint64 = 8000000            // Minimum the gas limit may ever be.
	MaxGasLimit          uint64 = 0x7fffffffffffffff // Maximum the gas limit (2^63-1).
	GenesisGasLimit      uint64 = MinGasLimit        // Gas limit of the Genesis block.
	MinerGasFloor        uint64 = MinGasLimit
	MinerGasCeil         uint64 = 160000000

	MaxTxGas uint64 = 1 << 24 // Maximum transaction gas limit after eip-7825 (16,777,216).

	MaximumExtraDataSize  uint64 = 32      // Maximum size extra data may be after Genesis.
	ExpByteGas            uint64 = 10      // Times ceil(log256(exponent)) for the EXP instruction.
	SloadGas              uint64 = 50      // Multiplied by the number of 32-byte words that are copied (round up) for any *COPY operation and added.
	CallValueTransferGas  uint64 = 9000    // Paid for CALL when the value transfer is non-zero.
	CallNewAccountGas     uint64 = 25000   // Paid for CALL when the destination address didn't exist prior.
	TxGas                 uint64 = 21000   // Per transaction not creating a contract. NOTE: Not payable on data of calls between transactions.
	TxGasContractCreation uint64 = 53000   // Per transaction that creates a contract. NOTE: Not payable on data of calls between transactions.
	UploadGas             uint64 = 277777  //555555
	TxDataZeroGas         uint64 = 4       // Per byte of data attached to a transaction that equals zero. NOTE: Not payable on data of calls between transactions.
	QuadCoeffDiv          uint64 = 512     // Divisor for the quadratic particle of the memory cost equation.
	SstoreSetGas          uint64 = 20000   // Once per SLOAD operation.
	LogDataGas            uint64 = 8       // Per byte in a LOG* operation's data.
	CallStipend           uint64 = 2300    // Free gas given at beginning of call.
	CallInferGas          uint64 = 1000000 // Base gas for call infer
	InferOpsPerGas        uint64 = 20000   // 1 gas infer 10000 ops

	Sha3Gas         uint64 = 30    // Once per SHA3 operation.
	Sha3WordGas     uint64 = 6     // Once per word of the SHA3 operation's data.
	SstoreResetGas  uint64 = 5000  // Once per SSTORE operation if the zeroness changes from zero.
	SstoreClearGas  uint64 = 5000  // Once per SSTORE operation if the zeroness doesn't change.
	SstoreRefundGas uint64 = 15000 // Once per SSTORE operation if the zeroness changes to zero.

	NetSstoreNoopGas  uint64 = 200   // Once per SSTORE operation if the value doesn't change.
	NetSstoreInitGas  uint64 = 20000 // Once per SSTORE operation from clean zero.
	NetSstoreCleanGas uint64 = 5000  // Once per SSTORE operation from clean non-zero.
	NetSstoreDirtyGas uint64 = 200   // Once per SSTORE operation from dirty.

	NetSstoreClearRefund      uint64 = 15000 // Once per SSTORE operation for clearing an originally existing storage slot
	NetSstoreResetRefund      uint64 = 4800  // Once per SSTORE operation for resetting to the original non-zero value
	NetSstoreResetClearRefund uint64 = 19800 // Once per SSTORE operation for resetting to the original zero value

	SstoreSentryGasEIP2200            uint64 = 2300  // Minimum gas required to be present for an SSTORE call, not consumed
	SstoreSetGasEIP2200               uint64 = 20000 // Once per SSTORE operation from clean zero to non-zero
	SstoreResetGasEIP2200             uint64 = 5000  // Once per SSTORE operation from clean non-zero to something else
	SstoreClearsScheduleRefundEIP2200 uint64 = 15000 // Once per SSTORE operation for clearing an originally existing storage slot

	ColdAccountAccessCostEIP2929 = uint64(2600) // COLD_ACCOUNT_ACCESS_COST
	ColdSloadCostEIP2929         = uint64(2100) // COLD_SLOAD_COST
	WarmStorageReadCostEIP2929   = uint64(100)  // WARM_STORAGE_READ_COST

	// In EIP-2200: SstoreResetGas was 5000.
	// In EIP-2929: SstoreResetGas was changed to '5000 - COLD_SLOAD_COST'.
	// In EIP-3529: SSTORE_CLEARS_SCHEDULE is defined as SSTORE_RESET_GAS + ACCESS_LIST_STORAGE_KEY_COST
	// Which becomes: 5000 - 2100 + 1900 = 4800
	SstoreClearsScheduleRefundEIP3529 uint64 = SstoreResetGasEIP2200 - ColdSloadCostEIP2929 + TxAccessListStorageKeyGas

	JumpdestGas   uint64 = 1     // Refunded gas, once per SSTORE operation if the zeroness changes to zero.
	EpochDuration uint64 = 30000 // Duration between proof-of-work epochs.

	CallGas                   uint64 = 40    // Once per CALL operation & message call transaction.
	CreateDataGas             uint64 = 20    //200
	CallCreateDepth           uint64 = 1024  // Maximum depth of call/create stack.
	ExpGas                    uint64 = 10    // Once per EXP instruction
	LogGas                    uint64 = 375   // Per LOG* operation.
	CopyGas                   uint64 = 3     //
	StackLimit                uint64 = 1024  // Maximum size of VM stack allowed.
	TierStepGas               uint64 = 0     // Once per operation, for a selection of them.
	LogTopicGas               uint64 = 375   // Multiplied by the * of the LOG*, per LOG transaction. e.g. LOG0 incurs 0 * c_txLogTopicGas, LOG4 incurs 4 * c_txLogTopicGas.
	CreateGas                 uint64 = 32000 // Once per CREATE operation & contract-creation transaction.
	Create2Gas                uint64 = 32000 // Once per CREATE2 operation
	SelfdestructRefundGas     uint64 = 24000 // Refunded following a selfdestruct operation.
	SuicideRefundGas          uint64 = 24000 // Refunded following a suicide operation.
	MemoryGas                 uint64 = 3     // Times the address of the (highest referenced byte in memory + 1). NOTE: referencing happens on read, write and in instructions such as RETURN and CALL.
	TxDataNonZeroGasFrontier  uint64 = 68    // Per byte of data attached to a transaction that is not equal to zero. NOTE: Not payable on data of calls between transactions.
	TxDataNonZeroGasEIP2028   uint64 = 16    // Per byte of non zero data attached to a transaction after EIP 2028 (part in Istanbul)
	TxAccessListAddressGas    uint64 = 2400  // Per address specified in EIP 2930 access list
	TxAccessListStorageKeyGas uint64 = 1900  // Per storage key specified in EIP 2930 access list

	// These have been changed during the course of the chain
	CallGasFrontier              uint64 = 40  // Once per CALL operation & message call transaction.
	CallGasEIP150                uint64 = 700 // Static portion of gas for CALL-derivates after EIP 150 (Tangerine)
	BalanceGasFrontier           uint64 = 20  // The cost of a BALANCE operation
	BalanceGasEIP150             uint64 = 400 // The cost of a BALANCE operation after Tangerine
	BalanceGasEIP1884            uint64 = 700 // The cost of a BALANCE operation after EIP 1884 (part of Istanbul)
	ExtcodeSizeGasFrontier       uint64 = 20  // Cost of EXTCODESIZE before EIP 150 (Tangerine)
	ExtcodeSizeGasEIP150         uint64 = 700 // Cost of EXTCODESIZE after EIP 150 (Tangerine)
	SloadGasFrontier             uint64 = 50
	SloadGasEIP150               uint64 = 200
	SloadGasEIP1884              uint64 = 800  // Cost of SLOAD after EIP 1884 (part of Istanbul)
	SloadGasEIP2200              uint64 = 800  // Cost of SLOAD after EIP 2200 (part of Istanbul)
	ExtcodeHashGasConstantinople uint64 = 400  // Cost of EXTCODEHASH (introduced in Constantinople)
	ExtcodeHashGasEIP1884        uint64 = 700  // Cost of EXTCODEHASH after EIP 1884 (part in Istanbul)
	SelfdestructGasEIP150        uint64 = 5000 // Cost of SELFDESTRUCT post EIP 150 (Tangerine)

	// EXP has a dynamic portion depending on the size of the exponent
	ExpByteFrontier uint64 = 10 // was set to 10 in Frontier
	ExpByteEIP158   uint64 = 50 // was raised to 50 during Eip158 (Spurious Dragon)

	// Extcodecopy has a dynamic AND a static cost. This represents only the
	// static portion of the gas. It was changed during EIP 150 (Tangerine)
	ExtcodeCopyBaseFrontier uint64 = 20
	ExtcodeCopyBaseEIP150   uint64 = 700

	// CreateBySelfdestructGas is used when the refunded account is one that does
	// not exist. This logic is similar to call.
	// Introduced in Tangerine Whistle (Eip 150)
	CreateBySelfdestructGas uint64 = 25000

	MaxCodeSize     = 24576           // Maximum bytecode to permit for a contract
	MaxInitCodeSize = 2 * MaxCodeSize // Maximum initcode to permit in a creation transaction and create instructions
	//MaxRawSize  = 384 * 1024

	// Precompiled contract gas prices

	EcrecoverGas        uint64 = 3000 // Elliptic curve sender recovery gas price
	Sha256BaseGas       uint64 = 60   // Base price for a SHA256 operation
	Sha256PerWordGas    uint64 = 12   // Per-word price for a SHA256 operation
	Ripemd160BaseGas    uint64 = 600  // Base price for a RIPEMD160 operation
	Ripemd160PerWordGas uint64 = 120  // Per-word price for a RIPEMD160 operation
	IdentityBaseGas     uint64 = 15   // Base price for a data copy operation
	IdentityPerWordGas  uint64 = 3    // Per-work price for a data copy operation

	Bn256AddGasByzantium             uint64 = 500    // Byzantium gas needed for an elliptic curve addition
	Bn256AddGasIstanbul              uint64 = 150    // Gas needed for an elliptic curve addition
	Bn256ScalarMulGasByzantium       uint64 = 40000  // Byzantium gas needed for an elliptic curve scalar multiplication
	Bn256ScalarMulGasIstanbul        uint64 = 6000   // Gas needed for an elliptic curve scalar multiplication
	Bn256PairingBaseGasByzantium     uint64 = 100000 // Byzantium base price for an elliptic curve pairing check
	Bn256PairingBaseGasIstanbul      uint64 = 45000  // Base price for an elliptic curve pairing check
	Bn256PairingPerPointGasByzantium uint64 = 80000  // Byzantium per-point price for an elliptic curve pairing check
	Bn256PairingPerPointGasIstanbul  uint64 = 34000  // Per-point price for an elliptic curve pairing check

	Bls12381G1AddGas          uint64 = 375   // Price for BLS12-381 elliptic curve G1 point addition
	Bls12381G1MulGas          uint64 = 12000 // Price for BLS12-381 elliptic curve G1 point scalar multiplication
	Bls12381G2AddGas          uint64 = 600   // Price for BLS12-381 elliptic curve G2 point addition
	Bls12381G2MulGas          uint64 = 22500 // Price for BLS12-381 elliptic curve G2 point scalar multiplication
	Bls12381PairingBaseGas    uint64 = 37700 // Base gas price for BLS12-381 elliptic curve pairing check
	Bls12381PairingPerPairGas uint64 = 32600 // Per-point pair gas price for BLS12-381 elliptic curve pairing check
	Bls12381MapG1Gas          uint64 = 5500  // Gas price for BLS12-381 mapping field element to G1 operation
	Bls12381MapG2Gas          uint64 = 23800 // Gas price for BLS12-381 mapping field element to G2 operation

	P256VerifyGas uint64 = 6900 // secp256r1 elliptic curve signature verifier gas price

	// The Refund Quotient is the cap on how much of the used gas can be refunded. Before EIP-3529,
	// up to half the consumed gas could be refunded. Redefined as 1/5th in EIP-3529
	RefundQuotient        uint64 = 2
	RefundQuotientEIP3529 uint64 = 5

	BlobTxBytesPerFieldElement         = 32      // Size in bytes of a field element
	BlobTxFieldElementsPerBlob         = 4096    // Number of field elements stored in a single data blob
	BlobTxBlobGasPerBlob               = 1 << 17 // Gas consumption of a single data blob (== blob byte size)
	BlobTxMinBlobGasprice              = 1       // Minimum gas price for data blobs
	BlobTxBlobGaspriceUpdateFraction   = 3338477 // Controls the maximum rate of change for blob gas price
	BlobTxPointEvaluationPrecompileGas = 50000   // Gas price for the point evaluation precompile.

	BlobTxTargetBlobGasPerBlock = 3 * BlobTxBlobGasPerBlob // Target consumable blob gas for data blobs per block (for 1559-like pricing)
	MaxBlobGasPerBlock          = 6 * BlobTxBlobGasPerBlob // Maximum consumable blob gas for data blobs per block

	MaxBlockSize = 8_388_608 // maximum size of an RLP-encoded block
)

// Bls12381G1MultiExpDiscountTable is the gas discount table for BLS12-381 G1 multi exponentiation operation
var Bls12381G1MultiExpDiscountTable = [128]uint64{1000, 949, 848, 797, 764, 750, 738, 728, 719, 712, 705, 698, 692, 687, 682, 677, 673, 669, 665, 661, 658, 654, 651, 648, 645, 642, 640, 637, 635, 632, 630, 627, 625, 623, 621, 619, 617, 615, 613, 611, 609, 608, 606, 604, 603, 601, 599, 598, 596, 595, 593, 592, 591, 589, 588, 586, 585, 584, 582, 581, 580, 579, 577, 576, 575, 574, 573, 572, 570, 569, 568, 567, 566, 565, 564, 563, 562, 561, 560, 559, 558, 557, 556, 555, 554, 553, 552, 551, 550, 549, 548, 547, 547, 546, 545, 544, 543, 542, 541, 540, 540, 539, 538, 537, 536, 536, 535, 534, 533, 532, 532, 531, 530, 529, 528, 528, 527, 526, 525, 525, 524, 523, 522, 522, 521, 520, 520, 519}

// Bls12381G2MultiExpDiscountTable is the gas discount table for BLS12-381 G2 multi exponentiation operation
var Bls12381G2MultiExpDiscountTable = [128]uint64{1000, 1000, 923, 884, 855, 832, 812, 796, 782, 770, 759, 749, 740, 732, 724, 717, 711, 704, 699, 693, 688, 683, 679, 674, 670, 666, 663, 659, 655, 652, 649, 646, 643, 640, 637, 634, 632, 629, 627, 624, 622, 620, 618, 615, 613, 611, 609, 607, 606, 604, 602, 600, 598, 597, 595, 593, 592, 590, 589, 587, 586, 584, 583, 582, 580, 579, 578, 576, 575, 574, 573, 571, 570, 569, 568, 567, 566, 565, 563, 562, 561, 560, 559, 558, 557, 556, 555, 554, 553, 552, 552, 551, 550, 549, 548, 547, 546, 545, 545, 544, 543, 542, 541, 541, 540, 539, 538, 537, 537, 536, 535, 535, 534, 533, 532, 532, 531, 530, 530, 529, 528, 528, 527, 526, 526, 525, 524, 524}

var (
	// SystemAddress is where the system-transaction is sent from as per EIP-4788
	SystemAddress = common.HexToAddress("0xfffffffffffffffffffffffffffffffffffffffe")

	DifficultyBoundDivisor_2   = big.NewInt(2) // The bound divisor of the difficulty, used in the update calculations.
	DifficultyBoundDivisor_256 = big.NewInt(256)
	DifficultyBoundDivisor_512 = big.NewInt(512)

	GenesisDifficulty = big.NewInt(512) // Difficulty of the Genesis block.
	MinimumDifficulty = big.NewInt(2)   // The minimum that the difficulty may ever be.

	MeanDifficultyBoundDivisor = big.NewInt(1024)

	HighDifficultyBoundDivisor = big.NewInt(2048) // The bound divisor of the difficulty, used in the update calculations.

	DurationLimit = big.NewInt(13) // The decision boundary on the blocktime duration used to determine whether difficulty should go up or not.

	// For Internal Test
	//CTXC_TOP = big.NewInt(0).Mul(big.NewInt(15000), big.NewInt(1000000000000000000))
	//CTXC_INIT = big.NewInt(0).Mul(big.NewInt(0), big.NewInt(1000000000000000000))
	//CTXC_MINING = big.NewInt(0).Mul(big.NewInt(15000), big.NewInt(1000000000000000000))

	// For Mainnet
	// |CTXC_TOP|:    Total Amount of Cortex Coin(CTXC) is lightspeed in vacuum: 299792458 m/s
	CTXC_TOP = big.NewInt(0).Mul(big.NewInt(299_792_458), big.NewInt(1000000000000000000))
	// |CTXC_INIT|:   For Pre-Allocated CTXCs before Mainnet launch
	CTXC_INIT = big.NewInt(0).Mul(big.NewInt(149_792_458), big.NewInt(1000000000000000000))
	// |CTXC_MINING|: For mining
	CTXC_MINING = big.NewInt(0).Mul(big.NewInt(150_000_000), big.NewInt(1000000000000000000))

	CTXC_F1 = big.NewInt(0).Mul(big.NewInt(20_486_540), big.NewInt(1000000000000000000))
	CTXC_F2 = big.NewInt(0).Mul(big.NewInt(21_285_544), big.NewInt(1000000000000000000))
)

const (
	SeedingBlks = 6   // TESTING: for torrent seed spreading
	MatureBlks  = 100 // Blocks between model uploading tx and model ready for use.
	// For the full node to synchronize the models
	BernardMatureBlks = 10                  // TESTING: For the full node to synchronize the models, in bernard testnet
	DoloresMatureBlks = 1                   // TESTING: For the full node to synchronize the models, in dolores testnet
	ExpiredBlks       = 1000000000000000000 // TESTING: Model expire blocks. Not effective. 8409600

	PER_UPLOAD_BYTES       uint64 = 1 * 512 * 1024     // Step of each progress update about how many bytes per upload tx
	DEFAULT_UPLOAD_BYTES   uint64 = 0                  // Default upload bytes
	MODEL_MIN_UPLOAD_BYTES        = 0                  // Minimum size of a model
	MODEL_MAX_UPLOAD_BYTES uint64 = 1024 * 1024 * 1024 // Maximum size of a model
	MODEL_GAS_LIMIT        uint64 = 20000              // Max gas limit for a model inference's reward to the author
	MODEL_GAS_UP_LIMIT     uint64 = 400000

	//CONFIRM_TIME   = -60                 // TESTING:* time.Second block should be protected past this time
	//CONFIRM_BLOCKS = 12                  // TESTING

	BLOCK_QUOTA         = 65536       // Upon the generation of a new valid block, 64kB file quota is added to the network. Empty blocks also count.
	Bernard_BLOCK_QUOTA = 65536       // for bernard
	Dolores_BLOCK_QUOTA = 65536 * 128 // for dolores
)
