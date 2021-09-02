// SPDX-License-Identifier: MIT
pragma solidity >=0.6.0 <0.9.0;

contract AIContract {

  // bytes infer_output = new uint256[](((1 * 10 + 31) >> 5) << 5);
  uint256[] infer_output = new uint256[](1);
  uint256[] input_data;
  
  constructor() {
      input_data = new uint256[]((1 * 28 * 28 + 31) >> 5);
      Infer(address(0x0000000000000000000000000000000000001013),address(0x0000000000000000000000000000000000002013));
      InferArray(address(0x0000000000000000000000000000000000001013));
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
