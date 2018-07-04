pragma solidity ^0.4.18;
contract Owned {
    function owned() public { owner = msg.sender; }
    address owner;

    modifier onlyOwner {
        require(
                msg.sender == owner,
                "Only owner can call this function."
               );
        _;
    }
}

contract AIContract is Owned {
    mapping (address => uint256) reward;
    uint256 targetLabel_;

    // update target, e.g. label:530 is bird
    function SetTargetLabel(uint256 targetLabel) onlyOwner public {
        targetLabel_ = targetLabel;
    }

    function Call(address inputData) public returns (uint256 ret) {
        uint256 result = makeInfer(inputData);
        address caller =  msg.sender;
        if (result == targetLabel_)
            reward[caller] += 2**16;
        else
            reward[caller] += 2**0;
    }

    // internal function for calling operation infer
	function makeInfer(address input) private constant returns (uint256 ret) {
        // model address
        address model =  0x0000000000000000000000000000000000001001; 
        assembly {
            ret := infer(model, input)
        }
	}
}
