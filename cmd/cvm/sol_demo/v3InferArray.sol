// SPDX-License-Identifier: MIT
pragma solidity >=0.6.0 <0.9.0;
          
// Creating a contract
contract v3InferArray{

    // infer_array input is bytes storage(defined by ctxc-solc-v2)
    bytes public input_array;

    constructor() {
        input_array = new bytes(1 * 28 * 28);
        infer_lfj();
    }

    function infer_lfj() public view returns(bool){
        address model = address(0x0000000000000000000000000000000000001013);
        return InferArray(model);
    }

    function InferArray(address model_addr) public view returns (bool) {
        // feed data in input_array, feed model into model_addr, and store the output in infer_output
        bytes memory output = new bytes(1);
        return inferArray(model_addr, input_array, output);
   }
}
