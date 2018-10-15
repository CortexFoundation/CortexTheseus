pragma solidity ^0.4.18;
contract AIContract {
    event DebugInfo(bytes indexed data);
    bytes bmp1;
    bytes bmp2;
	function makeInfer() public returns (uint256 ret) {
        bmp1 = new bytes(1 * 28 * 28);
        bmp2 = new bytes(3 * 224 * 224);
        for (uint32 i = 0; i < bmp2.length; i++)
            bmp2[i] = byte(48 + i % 10);
        emit DebugInfo(bytes(bmp2));
        address model =  0x0000000000000000000000000000000000001001; 
        ret = inferArray(model, bmp1);
	}
}
