#include <iostream>
using namespace std;

int main(){
    int ndim = 2;
    int ishape[] = {2,2};
    int n = 1;
    for(int i = 0; i < ndim; i++){
        n *= ishape[i];
    }
    int *input = new int[n];
    for(int i = 0; i < n; i++){
        input[i] = i+1;
    }
    int repeat = 2;
    int axis = 1;
    int oshape[] = {2, 2*2};
    int *output = new int[n*repeat];

    for(int i = 0; i < n*repeat; i++){
        int o_i = i, in_i = 0, shapeSize = 0;
        for(int j = ndim-1; j >= 0; j--){
            int col = o_i % oshape[j];
            o_i /= oshape[j];
            int tmpcol = col;
            if(j == axis) tmpcol = col / repeat;
            in_i += (j == ndim-1 ? tmpcol : tmpcol * shapeSize);
            shapeSize = (j == ndim-1 ? ishape[j] : shapeSize * ishape[j]);
        }
        output[i] = input[in_i];
        cout << output[i] << " ";
    }
    cout << endl;
    return 0;
}
