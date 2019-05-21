#include <iostream>
#include <stdint.h>
using namespace std;

inline int32_t broadcast_o_index(int* oshape, int odim, int& o_index){
    if(o_index == -1){
        o_index = 0;
        return o_index;
    }
    int tmp_o_index = o_index;
    for(int i = 0; i < odim; i++){
        int idx = odim - 1 - i;
        int ovar = tmp_o_index % oshape[idx];
        if(ovar + 1 != oshape[idx]){
            o_index += 1;
            break;
        }
        tmp_o_index /= oshape[idx];
    }
    return o_index;
}
inline int32_t broadcast_i_index(int* oshape, int o_index, int* ishape, int idim){
    int index = 0;
    int allIndex = 0;
    for(int i = 0; i < idim; i++){
        int idx = idim - 1 - i;
        int ovar = o_index % oshape[idx];
        if(ovar < ishape[idx]){
            index += i == 0 ? ovar : allIndex * ovar;
        }else if(ishape[idx] == 1){
        }else{
        }
        allIndex = (i == 0 ? ishape[idim-1] : allIndex * ishape[idx]);
        o_index /= oshape[idx];
    }
    return index;
}

void print(int index, int *shape){
    int tmpi[4];
    for(int i = 0; i < 4; i++){
        int idx = 4 - 1 - i;
        tmpi[idx] = index % shape[idx];
        index /= shape[idx];
    }
    for(int i = 0; i < 4; i++){
        cout << tmpi[i] << " ";
    }
    cout << endl;
}
int main(){
    int cshape[4] = {3, 1, 3 ,3};
    int ashape[4] = {3, 1, 3, 1};
    int bshape[4] = {1, 1, 1, 3};
    int o_index = -1;
    for(int i = 0; i < 3*1*3*3; i++){
        o_index = broadcast_o_index(cshape, 4, o_index);
        cout << "o_index: ";
        print(o_index, cshape);
        int a_index = broadcast_i_index(cshape, o_index, ashape, 4);
        cout << "a_index: ";
        print(a_index, ashape);
        int b_index = broadcast_i_index(cshape, o_index, bshape, 4);
    }
    return 0;
}
