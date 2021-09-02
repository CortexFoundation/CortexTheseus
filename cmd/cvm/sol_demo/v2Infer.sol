// SPDX-License-Identifier: MIT
pragma solidity >=0.6.0 <0.9.0;
          
// Creating a contract
contract ConstructorInfer{

    // infer_array input is bytes storage(defined by ctxc-solc-v2)
    bytes public input_array;

    constructor() {
        input_array = new bytes(1 * 28 * 28);
        Infer(address(0x0000000000000000000000000000000000001013),address(0x0000000000000000000000000000000000002013));
        InferArray(address(0x0000000000000000000000000000000000001013));
    }

    function Infer(address model_addr, address input_addr) public returns (uint256) {
        // feed data in input_addr, feed model into model_addr, and store the output in infer_output
        uint256 infer_output = infer(model_addr, input_addr);
        return infer_output;
    }

    function InferArray(address model_addr) public returns (uint256) {
        // feed data in input_array, feed model into model_addr, and store the output in infer_output
        uint256 infer_output = inferArray(model_addr, input_array);
        return infer_output;
   }

}
