pragma solidity ^0.4.18;
contract AIContract {
    event D(uint32 indexed i, uint32 indexed j, uint8 indexed v);
    bytes bmp1;

    constructor() public {
        bmp1 = new bytes(1 * 28 * 28);
        bmp1[1] = 1;
    }

	function makeInfer() public returns (uint256 ret) {
        address model =  0xdb50fb9fb84270199e932725cdf59e3b9bb79dee; 
        ret = inferArray(model, bmp1);
	}

    function Update(uint32 i, uint32 j, uint8 v) public {
        bmp1[i * 28 + j] = byte(v);
        emit D(i, j, v); 
    }

    function Reset() public {
        for (uint32 i = 0; i < bmp1.length; i++)
            bmp1[i] = byte(0);
    }

    function get() public returns (uint256) {
        return 1;
    }
}
