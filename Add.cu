
#ifndef RESAMPLE_BILINEAR
	#define RESAMPLE_BILINEAR

    #if __CUDACC_VER_MAJOR__ >= 9
        #include <cuda_fp16.h>
    #endif
	#include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
	#include "PrGPU/KernelSupport/KernelMemory.h"

	#if GF_DEVICE_TARGET_DEVICE
		GF_KERNEL_FUNCTION(kAdd,
			((GF_PTR(float4))(inImg1))
			((GF_PTR(float4))(inImg2))
			((GF_PTR(float4))(destImg)),
			((int)(inPitch))
			((int)(destPitch))
			((float)(input2Mult))
			((int)(in16f))
			((unsigned int)(outWidth))
			((unsigned int)(outHeight)),
			((uint2)(inXY)(KERNEL_XY)))
		{
			float4  color1, color2, dest;


			if (inXY.x >= outWidth || inXY.y >= outHeight) return;

			color1 = ReadFloat4(inImg1, inXY.y * inPitch + inXY.x, !!in16f);
			color2 = ReadFloat4(inImg2, inXY.y * inPitch + inXY.x, !!in16f);

			dest.x = color1.x + color2.x * input2Mult;
			dest.y = color1.y + color2.y * input2Mult;
			dest.z = color1.z + color2.z * input2Mult;
			dest.w = color1.w;

			WriteFloat4(dest, destImg, inXY.y * destPitch + inXY.x, !!in16f);
		}
	#endif

	#if __NVCC__
		void Add_CUDA (
			float *inBuf1,
			float *inBuf2,
			float *destBuf,
			int inPitch,
			int destPitch,
			float input2Mult,
			int	is16f,
			unsigned int width,
			unsigned int height)
		{
			dim3 blockDim (16, 16, 1);
			dim3 gridDim ( (width + blockDim.x - 1)/ blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1 );		

			kAdd << < gridDim, blockDim, 0 >> > ((float4*)inBuf1, (float4*)inBuf2, (float4*)destBuf, inPitch, destPitch, input2Mult, is16f, width, height);

			cudaDeviceSynchronize();
		}
	#endif //GF_DEVICE_TARGET_HOST

#endif //SDK_CROSS_DISSOLVE
