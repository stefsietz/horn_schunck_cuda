
#ifndef RESAMPLE_BILINEAR
	#define RESAMPLE_BILINEAR

    #if __CUDACC_VER_MAJOR__ >= 9
        #include <cuda_fp16.h>
    #endif
	#include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
	#include "PrGPU/KernelSupport/KernelMemory.h"

	#if GF_DEVICE_TARGET_DEVICE
		GF_KERNEL_FUNCTION(kHSAlphaStep,
			((GF_PTR(float4))(inImgUVAvg))
			((GF_PTR(float4))(inImgX))
			((GF_PTR(float4))(inImgY))
			((GF_PTR(float4))(inImgT))
			((GF_PTR(float4))(destImg)),
			((int)(inPitch))
			((int)(destPitch))
			((int)(alpha))
			((int)(in16f))
			((unsigned int)(outWidth))
			((unsigned int)(outHeight)),
			((uint2)(inXY)(KERNEL_XY)))
		{
			float4  uvAvg, dest;
			float dX, dY, dT;

			if (inXY.x >= outWidth || inXY.y >= outHeight) return;

			dX = ReadFloat4(inImgX, inXY.y * inPitch + inXY.x, !!in16f).x;
			dY = ReadFloat4(inImgY, inXY.y * inPitch + inXY.x, !!in16f).x;
			dT = ReadFloat4(inImgT, inXY.y * inPitch + inXY.x, !!in16f).x;
			uvAvg = ReadFloat4(inImgUVAvg, inXY.y * inPitch + inXY.x, !!in16f);

			dest.x = uvAvg.x - (dX*((dX * uvAvg.x) + (dY * uvAvg.y) + dT)) / (alpha* alpha + dX * dX + dY * dY);
			dest.y = uvAvg.y - (dY*((dX * uvAvg.x) + (dY * uvAvg.y) + dT)) / (alpha*alpha + dX*dX + dY*dY);
			dest.z = 0;
			dest.w = 1.0;

			dest.x = dest.x > 10000 ? 0 : dest.x;
			dest.y = dest.y > 10000 ? 0 : dest.y;
			dest.x = dest.x < -10000 ? 0 : dest.x;
			dest.y = dest.y < -10000 ? 0 : dest.y;

			WriteFloat4(dest, destImg, inXY.y * destPitch + inXY.x, !!in16f);
		}
	#endif

	#if __NVCC__
		void HSAlphaStep_CUDA (
			float *inBufUVAvg,
			float *inBufX,
			float *inBufY,
			float *inBufT,
			float *destBuf,
			int inPitch,
			int destPitch,
			int alpha,
			int	is16f,
			unsigned int width,
			unsigned int height)
		{
			dim3 blockDim (16, 16, 1);
			dim3 gridDim ( (width + blockDim.x - 1)/ blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1 );		

			kHSAlphaStep <<< gridDim, blockDim, 0 >>> ((float4*)inBufUVAvg, (float4*)inBufX, (float4*)inBufY, (float4*)inBufT, (float4*) destBuf, inPitch, destPitch, alpha, is16f, width, height );

			cudaDeviceSynchronize();
		}
	#endif //GF_DEVICE_TARGET_HOST

#endif //SDK_CROSS_DISSOLVE
