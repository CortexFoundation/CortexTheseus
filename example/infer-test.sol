pragma solidity ^0.4.18;

contract InferTest {
    function makeInfer(address x, address y)  returns (uint256 ret) {
        address model =  x;
        assembly {
ret := infer(x, y)
        }
    }
} 
