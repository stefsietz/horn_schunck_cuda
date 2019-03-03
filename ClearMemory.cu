
#ifndef RESAMPLE_BILINEAR
	#define RESAMPLE_BILINEAR

    #if __CUDACC_VER_MAJOR__ >= 9
        #include <cuda_fp16.h>
    #endif
	#include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
	#include "PrGPU/KernelSupport/KernelMemory.h"

	#if GF_DEVICE_TARGET_DEVICE
		GF_KERNEL_FUNCTION(kClearMem,
			((GF_PTR(float4))(destImg)),
			((float)(R))
			((float)(G))
			((float)(B))
			((float)(A))
			((int)(destPitch))
			((int)(in16f))
			((unsigned int)(outWidth))
			((unsigned int)(outHeight)),
			((uint2)(inXY)(KERNEL_XY)))
		{
			float4  dest;


			if (inXY.x >= outWidth || inXY.y >= outHeight) return;

			dest.x = B;
			dest.y = G;
			dest.z = R;
			dest.w = A;

			WriteFloat4(dest, destImg, inXY.y * destPitch + inXY.x, !!in16f);
		}
	#endif

	#if __NVCC__
		void ClearMem_CUDA (
			float *destBuf,
			float r,
			float g,
			float b,
			float a,
			int destPitch,
			int	is16f,
			unsigned int width,
			unsigned int height)
		{
			dim3 blockDim (16, 16, 1);
			dim3 gridDim ( (width + blockDim.x - 1)/ blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1 );		

			kClearMem <<< gridDim, blockDim, 0 >>> ((float4*) destBuf, r, g, b, a, destPitch, is16f, width, height );

			cudaDeviceSynchronize();
		}
	#endif //GF_DEVICE_TARGET_HOST

#endif //SDK_CROSS_DISSOLVE
