// CUDA
#ifdef __CUDACC__
  #define GLOBAL
  #define KERNEL extern "C" __global__
// OpenCL
#else
  #define GLOBAL __global
  #define KERNEL __kernel
#endif

#define MODULUS_P 17 // 一个大质数，可以根据需要更改

KERNEL void add(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
      result[i] = a[i] + b[i];
    }
}
KERNEL void subtract(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        result[i] = a[i] - b[i];
    }
}

KERNEL void multiply(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        result[i] = a[i] * b[i];
    }
}

KERNEL void divide(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        if (b[i] != 0) { 
            result[i] = a[i] / b[i];
        } else {
            result[i] = 0; 
        }
    }
}
KERNEL void add_field(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        result[i] = (a[i] + b[i]) % MODULUS_P;
    }
}

KERNEL void subtract_field(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        result[i] = (a[i] >= b[i]) ? (a[i] - b[i]) : (MODULUS_P + a[i] - b[i]);
        result[i] %= MODULUS_P;
    }
}

KERNEL void multiply_field(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        result[i] = (a[i] * b[i]) % MODULUS_P;
    }
}
// Function to compute Extended Euclidean Algorithm
__device__ void extended_gcd(uint a, uint b, int* lastx, int* lasty) {
    int x = 0, y = 1, temp;
    *lastx = 1, *lasty = 0;
    while (b != 0) {
        uint quotient = a / b;

        temp = b;
        b = a % b;
        a = temp;

        temp = x;
        x = *lastx - quotient * x;
        *lastx = temp;

        temp = y;
        y = *lasty - quotient * y;
        *lasty = temp;
    }
}

// Function to find modular inverse using Extended Euclidean Algorithm
__device__ uint mod_inverse(uint a, int p) {
    int x, y;
    extended_gcd(a, p, &x, &y);


    
    return (x % p + p) % p;
}

KERNEL void divide_field(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        if (b[i] != 0) {
            uint inv = mod_inverse(b[i], int(MODULUS_P));

            result[i] = (a[i] * inv) % MODULUS_P;
        } else {
            result[i] = 0;  // Division by zero yields zero
        }
    }
}
