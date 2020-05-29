# Cortex AI Smart Contract Guide
Example of inferring digit recognition model on Cortex blockchain (model address: **0x420bD8d2FE73514E9b77fdBc78c0306Fd8a12866**).
### Create a contract
```javascript
pragma solidity ^0.4.18;

contract Infer {

}
```
### Add basic variables
```javascript
    uint256[] public input_data;
    address modelAddr = 0x420bD8d2FE73514E9b77fdBc78c0306Fd8a12866;
    uint256 public currentInferResult;
```
### The built-in Infer function
```javascript
    // feed data in input_data to model and store the output in infer_output
    inferArray(model, input_data, infer_output);
```
### Create our function which calls the Infer function 
```javascript
    function DigitRecognitionInfer() public {
        uint256[] memory output = new uint256[](uint256((1 * 10 + 31) >> 5));
        inferArray(modelAddr, input_data, output);
        currentInferResult = (output[0] >> (256 - 32)) & ((1 << 32) - 1);
        currentInferResult = currentInferResult % 10;
    }
```
### Overall code
```javascript
pragma solidity ^0.4.18;

contract Infer {
    uint256[] public input_data;
    address modelAddr = 0x420bD8d2FE73514E9b77fdBc78c0306Fd8a12866;
    uint256 public currentInferResult;
    
    constructor() public {
        input_data = new uint256[]((1 * 3 * 32 * 32 + 31) >> 5);
    }
    
    function GenerateRandomInput() public {
        input_data[0] = uint(sha256(now));
        for(uint i = 1; i < input_data.length; ++i) {
          input_data[i] = uint(sha256(input_data[i - 1]));
        }
    }
    
    function DigitRecognitionInfer() public {
        uint256[] memory output = new uint256[](uint256((1 * 10 + 31) >> 5));
        inferArray(modelAddr, input_data, output);
        currentInferResult = (output[0] >> (256 - 32)) & ((1 << 32) - 1);
        currentInferResult = currentInferResult % 10;
    }
    
    function NewDigitRecognitionInfer(uint256[] imgData) public {
        uint256[] memory output = new uint256[](uint256((1 * 10 + 31) >> 5));
        input_data = imgData;
        inferArray(modelAddr, input_data, output);
        currentInferResult = (output[0] >> (256 - 32)) & ((1 << 32) - 1);
        currentInferResult = currentInferResult % 10;
    }
}
```
