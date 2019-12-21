#include <iostream>

struct Array{
    int n;
    int *shape;
    int *data;
    Array(int n, int *shape, int *data){
        this->n = n;
        this->shape = shape;
        this->data = data;
    }
};

int main(){
    int aN = 2 * 2 * 2 * 2;
    int bN = 2 * 1 * 2 * 2;
    int cN = 2 * 4 * 2 * 2;

    int aShape[4] = {2, 2, 2, 2};
    int bShape[4] = {2, 1, 2, 2};
    int cShape[4] = {2, 4, 2, 2};

    int *a = new int[aN];
    int *b = new int[bN];
    int *c = new int[cN];
    int *d = new int[bN];

    for(int i = 0; i < aN; i++){
        a[i] = 1;
    }
    for(int i = 0; i < bN; i++){
        b[i] = 2;
    }
    for(int i = 0; i < bN; i++){
        d[i] = 3;
    }


    Array args[3] = {Array(aN, aShape, a), Array(bN, bShape, b), Array(bN, bShape, d)};
    int axis = 1;

    for(int i = 0; i < cN; i++){
        int32_t o_i = i, in_i = 0, in_i2 = 0, shapeSize = 0;
        for(int j = 3; j >= 0; j--){
            int32_t col = o_i % cShape[j];
            o_i /= cShape[j];
            int32_t tmpcol = col;
            if(j == axis){
                int32_t allShapeSize = 0;
                for(int k = 0; k < 3; k++){
                    tmpcol = col - allShapeSize;
 //                   DLTensor *input = args[k];
                    Array input = args[k];
                    allShapeSize += input.shape[axis];
                    if(col < allShapeSize){
                        in_i = k;
                        break;
                    }
                }
            }
            in_i2 += (j == 3 ? tmpcol : tmpcol * shapeSize);
            Array input = args[in_i];
            shapeSize = (j == 3 ? input.shape[j] : shapeSize * input.shape[j]);
        }
        Array input = args[in_i];
//        int32_t *input_data = static_cast<int32_t*>(input->data);
        c[i] = input.data[in_i2];
        std::cout << c[i] << " ";
    }

}
