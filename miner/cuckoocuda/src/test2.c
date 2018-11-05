#include <stdio.h>

int g(int *x){
	*x = 100;
}
int f(int * a){
	g(&a[0]);
}
int main(){
	int a[2] = {1,2};
	f(a);
	printf("%d\n", a[0]);
	return 0;	
}
