#include<stdlib.h>
#include<stdio.h>
#include <math.h>
#include<sys/time.h>

__global__ void MatrixTranspose(float *a,float *b,int nx, int ny){
int ix = threadIdx.x+ blockIdx.x*blockDim.x;
int iy = threadIdx.y+ blockIdx.y*blockDim.y;
int idx = ix*ny + iy;
int odx= iy*nx + ix;

if((ix<nx)&&(iy<ny)){
			b[odx]=a[idx];
	}

}


__global__ void MatrixMul(float *a,float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

__global__ void MatAdd(float *A, float *B, float *C, int nx, int ny){
int ix = threadIdx.x+ blockIdx.x*blockDim.x;
int iy = threadIdx.y+ blockIdx.y*blockDim.y;
int idx = ix*ny + iy;

if((ix<nx)&&(iy<ny)){
			C[idx]=A[idx]+B[idx];
	}

}

__global__ void Mul(float *A, float *B, float *C, int nx, int ny){
int ix = threadIdx.x+ blockIdx.x*blockDim.x;
int iy = threadIdx.y+ blockIdx.y*blockDim.y;
int idx = ix*ny + iy;

if((ix<nx)&&(iy<ny)){
			C[idx]=A[idx]*B[idx];
	}
}

__global__ void div(float *A, float *B, float *C, int nx, int ny){
int ix = threadIdx.x+ blockIdx.x*blockDim.x;
int iy = threadIdx.y+ blockIdx.y*blockDim.y;
int idx = ix*ny + iy;

if((ix<nx)&&(iy<ny)){
			C[idx]=A[idx]/B[idx];
	}
}

__global__ void MatSub(float *A, float *B, float *C, int nx, int ny){
int ix = threadIdx.x+ blockIdx.x*blockDim.x;
int iy = threadIdx.y+ blockIdx.y*blockDim.y;
int idx = ix*ny + iy;

if((ix<nx)&&(iy<ny)){
			C[idx]=A[idx]-B[idx];
	}
}

double getTimeStamp(){
struct timeval tv;
gettimeofday(&tv, NULL);
return (double) tv.tv_usec/1000000+ tv.tv_sec;
}

struct matStruct{
float *m;
int x;
int y;
};
typedef struct matStruct matrix;

matrix setup_matrix(int x,int y)
{
matrix p;
p.m= (float *)malloc(x*y*sizeof(float *));
p.x=x;
p.y=y;

return p;
}
matrix transpose(matrix A){
	matrix C;
	C=setup_matrix(A.y,A.x);
	float *d_A, *d_C;
	cudaMalloc((void **) &d_A, ((A.x*A.y)*sizeof(float)));
	cudaMalloc((void **) &d_C, ((C.x*C.y)*sizeof(float)));
	cudaMemcpy(d_A,A.m, (A.x*A.y)*sizeof(float), cudaMemcpyHostToDevice );
	dim3 block(32,32);
	dim3 grid(1,1);
	MatrixTranspose<<<grid,block>>>(d_A,d_C,A.x,A.y);
	cudaMemcpy(C.m,d_C,(C.x*C.y)*sizeof(float), cudaMemcpyDeviceToHost);
	return C;
}



matrix matmul(matrix A, matrix B){
	if ((A.y==B.x))
	{	matrix C;
		C=setup_matrix(A.x,B.y);
		float *d_A, *d_B, *d_C;
		
				
		cudaMalloc((void **) &d_A, ((A.x*A.y)*sizeof(float)));
		cudaMalloc((void **) &d_B, ((B.x*B.y)*sizeof(float)));
		cudaMalloc((void **) &d_C, ((C.x*C.y)*sizeof(float)));
		cudaMemcpy(d_A,A.m, (A.x*A.y)*sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy(d_B,B.m, (B.x*B.y)*sizeof(float), cudaMemcpyHostToDevice );
		printf("%d %d\n",C.x,C.y);
		dim3 block(32,32);
		dim3 grid(1,1);
		MatrixMul<<<grid,block>>>(d_A,d_B,d_C,A.x,A.y,B.y);
		
		cudaMemcpy(C.m,d_C,(C.x*C.y)*sizeof(float), cudaMemcpyDeviceToHost);
		return C;

		
	}
	else{
	printf("Error:Vector Sum failed incompatible sizes");
	matrix C;
	C=setup_matrix(A.x,A.y);
	return C;
	}
	
}


matrix add_mat(matrix A, matrix B){
	if ((A.x==B.x)&&(A.y==B.y))
	{	matrix C;
		C=setup_matrix(A.x,A.y);
		float *d_A, *d_B, *d_C;
		float *h_dC;
		int bytes=(A.x*A.y)*sizeof(float);
		cudaError_t status3 = cudaMallocHost((void**)&h_dC, bytes);
		
		cudaMalloc((void **) &d_A, bytes);
		cudaMalloc((void **) &d_B, bytes);
		cudaMalloc((void **) &d_C,bytes);
		cudaMemcpy(d_A,A.m, bytes, cudaMemcpyHostToDevice );
		cudaMemcpy(d_B,B.m, bytes, cudaMemcpyHostToDevice );
		
		dim3 block(A.x,A.y);
		dim3 grid(1,1);
		MatAdd<<<grid,block>>>(d_A,d_B,d_C, A.x,A.y);
		cudaDeviceSynchronize();
		cudaMemcpy(C.m,d_C,bytes, cudaMemcpyDeviceToHost);
		
		return C;

		
	}
	else{
	printf("Error:Vector Sum failed incompatible sizes");
	matrix C;
	C=setup_matrix(A.x,A.y);
	return C;
	}
	
}

matrix multiply(matrix A, matrix B){
	if ((A.x==B.x)&&(A.y==B.y))
	{	matrix C;
		C=setup_matrix(A.x,A.y);
		float *d_A, *d_B, *d_C;
		float *h_dC;
		int bytes=(A.x*A.y)*sizeof(float);
		cudaError_t status3 = cudaMallocHost((void**)&h_dC, bytes);
		
		cudaMalloc((void **) &d_A, bytes);
		cudaMalloc((void **) &d_B, bytes);
		cudaMalloc((void **) &d_C,bytes);
		cudaMemcpy(d_A,A.m, bytes, cudaMemcpyHostToDevice );
		cudaMemcpy(d_B,B.m, bytes, cudaMemcpyHostToDevice );
		
		dim3 block(A.x,A.y);
		dim3 grid(1,1);
		Mul<<<grid,block>>>(d_A,d_B,d_C, A.x,A.y);
		cudaDeviceSynchronize();
		cudaMemcpy(C.m,d_C,bytes, cudaMemcpyDeviceToHost);
		return C;

		
	}
	else{
	printf("Error:Vector Sum failed incompatible sizes");
	matrix C;
	C=setup_matrix(A.x,A.y);
	return C;
	}
	
}

matrix sub_mat(matrix A, matrix B){
	if ((A.x==B.x)&&(A.y==B.y))
	{	matrix C;
		C=setup_matrix(A.x,A.y);
		float *d_A, *d_B, *d_C;
		float *h_dC;
		int bytes=(A.x*A.y)*sizeof(float);
		cudaError_t status3 = cudaMallocHost((void**)&h_dC, bytes);
		
		cudaMalloc((void **) &d_A, bytes);
		cudaMalloc((void **) &d_B, bytes);
		cudaMalloc((void **) &d_C,bytes);
		cudaMemcpy(d_A,A.m, bytes, cudaMemcpyHostToDevice );
		cudaMemcpy(d_B,B.m, bytes, cudaMemcpyHostToDevice );
		
		dim3 block(A.x,A.y);
		dim3 grid(1,1);
		MatSub<<<grid,block>>>(d_A,d_B,d_C, A.x,A.y);
		cudaDeviceSynchronize();
		cudaMemcpy(C.m,d_C,bytes, cudaMemcpyDeviceToHost);
		return C;

		
	}
	else{
	printf("Error:Vector Sum failed incompatible sizes");
	matrix C;
	C=setup_matrix(A.x,A.y);
	return C;
	}
	
}

matrix divide(matrix A, matrix B){
	if ((A.x==B.x)&&(A.y==B.y))
	{	matrix C;
		C=setup_matrix(A.x,A.y);
		float *d_A, *d_B, *d_C;
		float *h_dC;
		int bytes=(A.x*A.y)*sizeof(float);
		cudaError_t status3 = cudaMallocHost((void**)&h_dC, bytes);
		
		cudaMalloc((void **) &d_A, bytes);
		cudaMalloc((void **) &d_B, bytes);
		cudaMalloc((void **) &d_C,bytes);
		cudaMemcpy(d_A,A.m, bytes, cudaMemcpyHostToDevice );
		cudaMemcpy(d_B,B.m, bytes, cudaMemcpyHostToDevice );
		
		dim3 block(A.x,A.y);
		dim3 grid(1,1);
		div<<<grid,block>>>(d_A,d_B,d_C, A.x,A.y);
		cudaDeviceSynchronize();
		cudaMemcpy(C.m,d_C,bytes, cudaMemcpyDeviceToHost);
		return C;

		
	}
	else{
	printf("Error:Vector Sum failed incompatible sizes");
	matrix C;
	C=setup_matrix(A.x,A.y);
	return C;
	}
	
}


int main( int argc, char *argv[])
{
if (argc !=3){
	printf("Error: wrong number of args\n");
	exit(0);
}
int nx= atoi(argv[1]);
int ny = atoi( argv[2]);
matrix a1,b1,c1;
a1=setup_matrix(3,3);
b1=setup_matrix(3,6);

for(int i=0; i<a1.x; i++)
{for(int j=0;j<a1.y;j++)
{a1.m[a1.y*i+j]=2;
printf("%f ",a1.m[a1.y*i+j]);
}
printf("\n");
}
for(int i=0; i<b1.x; i++)
{for(int j=0;j<b1.y;j++)
{b1.m[b1.y*i+j]=1;
printf("%f ",b1.m[b1.y*i+j]);
}
printf("\n");
}
c1=matmul(a1,b1);

for(int i=0; i<c1.x; i++)
{for(int j=0;j<c1.y;j++)
{printf("%f ",c1.m[c1.y*i+j]);
}
printf("\n");
}

}
