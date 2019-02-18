
#ifndef RESAMPLE_BILINEAR
	#define RESAMPLE_BILINEAR

    #if __CUDACC_VER_MAJOR__ >= 9
        #include <cuda_fp16.h>
    #endif
	#include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
	#include "PrGPU/KernelSupport/KernelMemory.h"

	#if GF_DEVICE_TARGET_DEVICE
		GF_TEXTURE_GLOBAL(float4, inSrcTexture, GF_DOMAIN_NATURAL, GF_RANGE_NATURAL_CUDA, GF_EDGE_CLAMP, GF_FILTER_LINEAR)

		GF_KERNEL_FUNCTION(kConvolve3x3,
			((GF_TEXTURE_TYPE(float4))(GF_TEXTURE_NAME(inSrcTexture)))
			((GF_PTR(float4))(destImg))
			((GF_PTR(float))(kernelBuf)),
			((int)(kernelRadius))
			((int)(destPitch))
			((int)(in16f))
			((unsigned int)(outWidth))
			((unsigned int)(outHeight)),
			((uint2)(outXY)(KERNEL_XY)))
		{
			float4  dest;


			if (outXY.x >= outWidth || outXY.y >= outHeight) return;

			float4 color;
			color.x = 0;
			color.y = 0;
			color.z = 0;
			color.w = 1.0;
			int i = 0;
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					float kernelVal = kernelBuf[i++];;
					float4 texVal = GF_READTEXTURE(GF_TEXTURE_NAME(inSrcTexture), (outXY.x + x*kernelRadius + 0.5) / outWidth, (outXY.y + y*kernelRadius + 0.5) / outHeight);
					color.x += texVal.x*kernelVal;
					color.y += texVal.y*kernelVal;
					color.z += texVal.z*kernelVal;
				}
			}
			dest = color;

			WriteFloat4(dest, destImg, outXY.y * destPitch + outXY.x, !!in16f);
		}
	#endif

	#if __NVCC__
		void Convolve3x3_CUDA (	
			cudaTextureObject_t inSrcTexture,
			float *destBuf,
			float *kernelBuf,
			int kernelRadius,
			int destPitch,
			int	is16f,
			unsigned int width,
			unsigned int height)
		{
			dim3 blockDim (16, 16, 1);
			dim3 gridDim ( (width + blockDim.x - 1)/ blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1 );		

			kConvolve3x3 <<< gridDim, blockDim, 0 >>> ( inSrcTexture, (float4*) destBuf, kernelBuf, kernelRadius, destPitch, is16f, width, height );

			cudaDeviceSynchronize();
		}
	#endif //GF_DEVICE_TARGET_HOST

#endif //SDK_CROSS_DISSOLVE
