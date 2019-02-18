
#ifndef RESAMPLE_BILINEAR
	#define RESAMPLE_BILINEAR

    #if __CUDACC_VER_MAJOR__ >= 9
        #include <cuda_fp16.h>
    #endif
	#include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
	#include "PrGPU/KernelSupport/KernelMemory.h"

	#if GF_DEVICE_TARGET_DEVICE
		GF_KERNEL_FUNCTION(kFrameDiff,
		((GF_PTR(float4))(inImg))
		((GF_PTR(float4))(nextImg))
		((GF_PTR(float4))(destImg)),
		((int)(inPitch))
		((int)(destPitch))
		((int)(in16f))
		((unsigned int)(outWidth))
		((unsigned int)(outHeight)),
		((uint2)(inXY)(KERNEL_XY)))
		{
			float4  color, nextColor, dest;


			if (inXY.x >= outWidth || inXY.y >= outHeight) return;

			color = ReadFloat4(inImg, inXY.y * inPitch + inXY.x, !!in16f);
			nextColor = ReadFloat4(nextImg, inXY.y * inPitch + inXY.x, !!in16f);

			dest.x = nextColor.x - color.x;
			dest.y = nextColor.y - color.y;
			dest.z = nextColor.z - color.z;
			dest.w = color.w;

			WriteFloat4(dest, destImg, inXY.y * destPitch + inXY.x, !!in16f);
		}
	#endif

#if __NVCC__
		void FrameDiff_CUDA(
			float *inBuf,
			float *nextBuf,
			float *destBuf,
			int inPitch,
			int destPitch,
			int	is16f,
			unsigned int width,
			unsigned int height)
		{
			dim3 blockDim(16, 16, 1);
			dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

			kFrameDiff << < gridDim, blockDim, 0 >> > ((float4*)inBuf, (float4*)nextBuf, (float4*)destBuf, inPitch, destPitch, is16f, width, height);

			cudaDeviceSynchronize();
		}
#endif //GF_DEVICE_TARGET_HOST

#endif //SDK_CROSS_DISSOLVE
