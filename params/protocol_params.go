// Copyright 2015 The CortexFoundation Authors
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

import "math/big"

const (
	//all configs should not be changed
	GasLimitBoundDivisor uint64 = 1024        // The bound divisor of the gas limit, used in update calculations.
	MinGasLimit          uint64 = 8000000     // Minimum the gas limit may ever be.
	GenesisGasLimit      uint64 = MinGasLimit // Gas limit of the Genesis block.
	MinerGasFloor        uint64 = MinGasLimit
	MinerGasCeil         uint64 = 160000000

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
	JumpdestGas               uint64 = 1     // Refunded gas, once per SSTORE operation if the zeroness changes to zero.
	EpochDuration             uint64 = 30000 // Duration between proof-of-work epochs.
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
	SuicideRefundGas          uint64 = 24000 // Refunded following a suicide operation.
	MemoryGas                 uint64 = 3     // Times the address of the (highest referenced byte in memory + 1). NOTE: referencing happens on read, write and in instructions such as RETURN and CALL.
	TxDataNonZeroGas          uint64 = 68    // Per byte of data attached to a transaction that is not equal to zero. NOTE: Not payable on data of calls between transactions.

	MaxCodeSize = 24576 // Maximum bytecode to permit for a contract
	//MaxRawSize  = 384 * 1024

	// Precompiled contract gas prices

	EcrecoverGas            uint64 = 3000   // Elliptic curve sender recovery gas price
	Sha256BaseGas           uint64 = 60     // Base price for a SHA256 operation
	Sha256PerWordGas        uint64 = 12     // Per-word price for a SHA256 operation
	Ripemd160BaseGas        uint64 = 600    // Base price for a RIPEMD160 operation
	Ripemd160PerWordGas     uint64 = 120    // Per-word price for a RIPEMD160 operation
	IdentityBaseGas         uint64 = 15     // Base price for a data copy operation
	IdentityPerWordGas      uint64 = 3      // Per-work price for a data copy operation
	ModExpQuadCoeffDiv      uint64 = 20     // Divisor for the quadratic particle of the big int modular exponentiation
	Bn256AddGas             uint64 = 500    // Gas needed for an elliptic curve addition
	Bn256ScalarMulGas       uint64 = 40000  // Gas needed for an elliptic curve scalar multiplication
	Bn256PairingBaseGas     uint64 = 100000 // Base price for an elliptic curve pairing check
	Bn256PairingPerPointGas uint64 = 80000  // Per-point price for an elliptic curve pairing check
)

var (
	DifficultyBoundDivisor = big.NewInt(2)   // The bound divisor of the difficulty, used in the update calculations.
	GenesisDifficulty      = big.NewInt(512) // Difficulty of the Genesis block.
	MinimumDifficulty      = big.NewInt(2)   // The minimum that the difficulty may ever be.

	MeanDifficultyBoundDivisor = big.NewInt(1024)

	HighDifficultyBoundDivisor = big.NewInt(2048) // The bound divisor of the difficulty, used in the update calculations.

	DurationLimit = big.NewInt(13) // The decision boundary on the blocktime duration used to determine whether difficulty should go up or not.

	// For Internal Test
	//CTXC_TOP = big.NewInt(0).Mul(big.NewInt(15000), big.NewInt(1000000000000000000))
	//CTXC_INIT = big.NewInt(0).Mul(big.NewInt(0), big.NewInt(1000000000000000000))
	//CTXC_MINING = big.NewInt(0).Mul(big.NewInt(15000), big.NewInt(1000000000000000000))

	// For Mainnet
	// |CTXC_TOP|:    Total Amount of Cortex Coin(CTXC) is lightspeed in vacuum: 299792458 m/s
	CTXC_TOP = big.NewInt(0).Mul(big.NewInt(299792458), big.NewInt(1000000000000000000))
	// |CTXC_INIT|:   For Pre-Allocated CTXCs before Mainnet launch
	CTXC_INIT = big.NewInt(0).Mul(big.NewInt(149792458), big.NewInt(1000000000000000000))
	// |CTXC_MINING|: For mining
	CTXC_MINING = big.NewInt(0).Mul(big.NewInt(150000000), big.NewInt(1000000000000000000))
)

const (
	SeedingBlks = 6   // TESTING: for torrent seed spreading
	MatureBlks  = 100 // Blocks between model uploading tx and model ready for use.
	// For the full node to synchronize the models
	BernardMatureBlks = 10                  // TESTING: For the full node to synchronize the models, in dolores testnet
	DoloresMatureBlks = 1                  // TESTING: For the full node to synchronize the models, in dolores testnet
	ExpiredBlks       = 1000000000000000000 // TESTING: Model expire blocks. Not effective. 8409600

	PER_UPLOAD_BYTES       uint64 = 1 * 512 * 1024     // Step of each progress update about how many bytes per upload tx
	DEFAULT_UPLOAD_BYTES   uint64 = 0                  // Default upload bytes
	MODEL_MIN_UPLOAD_BYTES        = 0                  // Minimum size of a model
	MODEL_MAX_UPLOAD_BYTES uint64 = 1024 * 1024 * 1024 // Maximum size of a model
	MODEL_GAS_LIMIT        uint64 = 20000              // Max gas limit for a model inference's reward to the author
	MODEL_GAS_UP_LIMIT uint64 = 400000

	//CONFIRM_TIME   = -60                 // TESTING:* time.Second block should be protected past this time
	//CONFIRM_BLOCKS = 12                  // TESTING

	BLOCK_QUOTA = 65536 // Upon the generation of a new valid block, 64kB file quota is added to the network. Empty blocks also count.
	Bernard_BLOCK_QUOTA = 65536				// for bernard
	Dolores_BLOCK_QUOTA = 65536 * 128 // for dolores
)
