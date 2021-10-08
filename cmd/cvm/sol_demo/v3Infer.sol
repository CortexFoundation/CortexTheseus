// SPDX-License-Identifier: MIT
pragma solidity >=0.6.0 <0.9.0;

contract AIContract {

  // bytes infer_output = new uint256[](((1 * 10 + 31) >> 5) << 5);
  uint256[] infer_output = new uint256[](1);
  uint256[] input_data;
  
  constructor() {
      input_data = new uint256[]((1 * 28 * 28 + 31) >> 5);
      Infer(address(0x5A4A06AC80E44E2239977E309884c654B223a3B8),address(0xe296Ecd28970e38411cdC3C2a045107c4Bd53eBB));
      InferArray(address(0x5A4A06AC80E44E2239977E309884c654B223a3B8));
  }
  
  function Infer(address model, address input) public view returns (uint256) {
    // feed data in input to model and store the output in infer_output
    return infer(model, input, infer_output);
  }
  
  function InferArray(address model) public view returns (uint256) {
    // feed data in input_data to model and store the output in infer_output
    uint256[] memory output2 = new uint256[](1);
    return inferArray(model, input_data, output2);
  }
}
