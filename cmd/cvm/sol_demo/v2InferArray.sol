// SPDX-License-Identifier: MIT
pragma solidity >=0.6.0 <0.9.0;
          
// Creating a contract
contract ConstructorInferArray{

    // infer_array input is bytes storage(defined by ctxc-solc-v2)
    bytes public input_array;

    constructor() {
        input_array = new bytes(1 * 28 * 28);
        infer_lfj();
    }

    function infer_lfj() public returns(uint256){
        address model = address(0x0000000000000000000000000000000000001013);
        uint256 infer_output = InferArray(model);
        return infer_output;
    }

    function InferArray(address model_addr) public returns (uint256) {
        // feed data in input_array, feed model into model_addr, and store the output in infer_output
        uint256 infer_output = inferArray(model_addr, input_array);
        return infer_output;
   }
}
