pragma solidity ^0.4.18;
contract AIContract {
    bytes mnist_data;
    uint256[10] balance;
	function makeInfer() public returns(int8[]) {
        address model =  0x0000000000000000000000000000000000001001;
        mnist_data = new bytes(1 * 28 * 28);
        bytes memory l1out = new bytes(1 * 10);
        //int8[] memory l1out = new int8[](10);
        nnforward(model, mnist_data, l1out);
        // softmax(l1out, l1out);
        // return l1out;
	}
}
