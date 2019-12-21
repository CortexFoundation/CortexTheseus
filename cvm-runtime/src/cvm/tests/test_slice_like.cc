#include <iostream>
using namespace std;

int main(){
   int ndim = 2;
   int ishape[] = {3,4};
   int n = 1;
   for(int i = 0; i < ndim; i++){
       n *= ishape[i];
   }
   int *input = new int[n];
   for(int i = 0; i < n; i++){
       input[i] = i+1;
   }

   int shape_like[] = {2,3};
   int axis[] = {1};

   int oshape[] = {3, 3};
   int on = 1;
   for(int i = 0; i < ndim; i++){
    on *= oshape[i];
   }
   int *output = new int[on];

   for(int i = 0; i < on; i++){
       int o_i = i, in_i = 0, shapeSize = 0;
       for(int j = ndim-1; j >= 0; j--){
           int col = o_i % oshape[j];
           o_i /= oshape[j];
           in_i += (j == ndim-1 ? col : col * shapeSize);
           shapeSize = (j == ndim-1 ? ishape[j] : shapeSize * ishape[j]);
       }
       output[i] = input[in_i];
       cout << output[i] << " ";
   }
   cout << endl;
}
