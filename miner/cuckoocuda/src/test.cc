#include <stdio.h>
#include "blake2.h"
int main() 
{
	char hdrkey[32];
	char header[40];
	blake2b((void *)&(hdrkey[0]), sizeof(hdrkey), (const void *)header, 40, 0, 0);
	for (int i = 0; i < 32; i++) {
		printf("%d ", hdrkey[i]);
	}
	return 0;
}
