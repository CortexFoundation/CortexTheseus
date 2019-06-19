pragma solidity ^0.4.18;
contract AIContract {
  uint256[] input_data;
  function set(uint256[] storage data, uint256 idx, uint8 value) private {
    uint256 x = idx / 32;
    uint256 y = idx % 32;
    require(x < data.length);
    data[x] |= (uint256(value) << ((31 - y) * 8));
  }

  function set32(uint256[] storage data, uint256 idx, uint32 value) private {
    uint256 x = idx / 8;
    uint256 y = idx % 8;
    require(x < data.length);
    data[x] |= (uint256(value) << ((7 - y) * 32));
  }

  function get(uint256[] data, uint256 idx) private returns (uint8){
    uint256 x = idx / 32;
    uint256 y = idx % 32;
    require(x < data.length);
    return uint8((data[x] >> ((31 - y) * 8)) % 256);
  }

  function argmax_int8(uint256[] data) private returns (uint256) {
    uint256 max_id = 0;
    int8 max_v = -128;
    for (uint i = 0; i < data.length; i++) {
      for (uint y = 0; y < 32; y++) {
        uint256 label = i * 32 + y;
        int8 value = int8(get(data, label));
        if (value > max_v) {
          max_id = label;
          max_v = value;
        }
      }
    }
    return max_id;
  }

  function call_yolo(address model) public returns (uint256) {
    input_data = new uint256[]((1 * 3 * 416 * 416 + 31) >> 5);
    uint256[] memory output = new uint256[](uint256((1 * 60 + 7) >> 3));
    inferArray(model, input_data, output);
    return output[0];
  }

  function init_cifar() public returns (uint256) {
    input_data = new uint256[]((1 * 3 * 32 * 32 + 31) >> 5);
  }

  function call_cifar(address model) public returns (uint256) {
    uint256[] memory output = new uint256[](uint256((1 * 10 + 31) >> 3));
    inferArray(model, input_data, output);
    return output[0];
  }

  function call(address model, address input) public returns(uint256) {
    uint256[] memory output = new uint256[](uint256((1 * 60 + 7) >> 3));
    infer(model, input, output);
  }

  function make_detection() public returns(uint256) {
    address model =  0x0000000000000000000000000000000000001003;
    address input =  0x0000000000000000000000000000000000002003;
    uint256[] memory output = new uint256[](uint256((1 * 240 + 31) >> 5));
    infer(model, input, output);
    uint256 first_obj_target = (output[0] >> (256 - 32)) & ((1 << 32) - 1);
    return first_obj_target;
  }

  function make_predict_trec() public returns(uint256) {
    address model =  0x0000000000000000000000000000000000001004;
    address input =  0x0000000000000000000000000000000000002004;
    uint256[] memory output = new uint256[](uint256((1 * 6 + 31) >> 5));
    infer(model, input, output);
    return output[0];
  }

  function make_predict_trec_inferarray() public returns(uint256) {
    address model =  0x0000000000000000000000000000000000001004;
    input_data = new uint256[]((38 * 1 + 7) >> 3);
    uint32[38] memory org_data;
    org_data[0] = 11;
    org_data[1] = 17;
    org_data[2] = 15;
    org_data[3] = 5;
    org_data[4] = 40;
    org_data[5] = 245;
    org_data[6] = 108;
    org_data[7] = 16;
    org_data[8] = 1257;
    org_data[9] = 9;
    org_data[10] = 57;
    org_data[11] = 1203;
    org_data[12] = 16;
    org_data[13] = 84;
    org_data[14] = 3716;
    org_data[15] = 4;
    for (uint256 i = 0; i < 38; i++) {
      if (org_data[i] == 0)
        break;
      set32(input_data, i, org_data[i]);
    }
    uint256[] memory output = new uint256[](uint256((1 * 6 + 31) >> 5));
    inferArray(model, input_data, output);
    return output[0];
  }

  function makeInferArray() public returns(uint256) {
    // cifar10 model with shape 3 * 32 * 32
    address model =  0x0000000000000000000000000000000000001003;
    input_data = new uint256[]((1 * 3 * 416 * 416 + 31) >> 5);
    input_data[0] = 0x0001020304050607080910111213141516171819202122232425262728293031;
    // for (uint256 i = 0; i < 3 * 32 * 32; i++) {
    //   set(input_data, uint256(i), uint8(i));
    // }
    uint256[] memory output = new uint256[](uint256((1 * 28 + 31) >> 5));
    // output[0] = 0x0001020304050607080910111213141516171819202122232425262728293031;
    inferArray(model, input_data, output);
    // uint256 label = argmax_int8(output);
    return (output[0] >> (256 - 32)) & ((1 << 32) - 1);
  }
}
