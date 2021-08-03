// SPDX-License-Identifier: MIT
pragma solidity >=0.6.0 <0.9.0;
          
// Creating a contract
contract ConstructorInfer{

    bytes5 public e1 = 0x1313131313;
    bytes5 public e2 = 0x3131313131;

    // infer_array input is bytes storage(defined by ctxc-solc-v2)
    bytes public input_array;

    constructor() {
        e2 = 0x9898989898;
        getValue();
        infer_lfj();
    }

    function infer_lfj() public returns(uint256){
        address model = address(0x0000000000000000000000000000000000001013);
        address input = address(0x0000000000000000000000000000000000002013);

        uint256 infer_output = infer(model, input);
        return infer_output;
    }

    function getValue() public pure returns (bytes3) {
        return 0x838383;
    }      
  
    function Infer(address model_addr, address input_addr)  public returns (uint256) {
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
