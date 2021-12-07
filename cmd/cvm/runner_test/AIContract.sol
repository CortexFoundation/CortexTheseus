// SPDX-License-Identifier: GPL-3.0

pragma solidity >=0.7.0 <0.9.0;

/**
 * @title Infer_0870
 */
contract AIContract {
    uint256[] input_data;
    uint256[] infer_output = new uint256[](uint256((1 * 10 + 31) >> 5));

    constructor() {
      input_data = new uint256[]((1 * 3 * 32 * 32 + 31) >> 5);
      Infer(address(0x5A4A06AC80E44E2239977E309884c654B223a3B8),address(0xe296Ecd28970e38411cdC3C2a045107c4Bd53eBB));
      InferArray(address(0x5A4A06AC80E44E2239977E309884c654B223a3B8));
    }

    function Infer(address model, address input) public view returns (uint256) {
      // feed data in input to model and store the output in infer_output
      infer(model, input, infer_output);
      return infer_output[0];
    }

    function InferArray(address model) public view returns (uint256) {
      // feed data in input_data to model and store the output in infer_output
      inferArray(model, input_data, infer_output);
      return infer_output[0];
    }
}
