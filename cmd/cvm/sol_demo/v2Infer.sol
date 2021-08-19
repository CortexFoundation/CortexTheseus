// SPDX-License-Identifier: MIT
pragma solidity >=0.6.0 <0.9.0;
          
// Creating a contract
contract ConstructorInfer{

    constructor() {
        infer_lfj();
    }

    function infer_lfj() public returns(uint256){
        address model = address(0x0000000000000000000000000000000000001013);
        address input = address(0x0000000000000000000000000000000000002013);

        uint256 infer_output = infer(model, input);
        return infer_output;
    }
  
    function Infer(address model_addr, address input_addr)  public returns (uint256) {
        // feed data in input_addr, feed model into model_addr, and store the output in infer_output
        uint256 infer_output = infer(model_addr, input_addr);
        return infer_output;
    }
}
