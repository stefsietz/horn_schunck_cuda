
#ifndef RESAMPLE_BILINEAR
	#define RESAMPLE_BILINEAR

    #if __CUDACC_VER_MAJOR__ >= 9
        #include <cuda_fp16.h>
    #endif
	#include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
	#include "PrGPU/KernelSupport/KernelMemory.h"

	#if GF_DEVICE_TARGET_DEVICE
		GF_TEXTURE_GLOBAL(float4, inSrcTexture, GF_DOMAIN_NATURAL, GF_RANGE_NATURAL_CUDA, GF_EDGE_CLAMP, GF_FILTER_LINEAR)

		GF_KERNEL_FUNCTION(kResampleBilinear,
			((GF_TEXTURE_TYPE(float4))(GF_TEXTURE_NAME(inSrcTexture)))
			((GF_PTR(float4))(destImg)),
			((unsigned int)(destPitch))
			((unsigned int)(in16f))
			((unsigned int)(outWidth))
			((unsigned int)(outHeight)),
			((uint2)(outXY)(KERNEL_XY)))
		{
			float4  dest;


			if (outXY.x >= outWidth || outXY.y >= outHeight) return;

			float4 color = GF_READTEXTURE(GF_TEXTURE_NAME(inSrcTexture), (outXY.x+ 0.5) / outWidth, (outXY.y + 0.5) / outHeight);

			dest = color;

			WriteFloat4(dest, destImg, outXY.y * destPitch + outXY.x, !!in16f);
		}
	#endif

	#if __NVCC__
		void ResampleBilinear_CUDA (	
			cudaTextureObject_t inSrcTexture,
			float *destBuf,
			unsigned int destPitch,
			unsigned int is16f,
			unsigned int width,
			unsigned int height)
		{
			dim3 blockDim (16, 16, 1);
			dim3 gridDim ( (width + blockDim.x - 1)/ blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1 );		

			kResampleBilinear <<< gridDim, blockDim, 0 >>> ( inSrcTexture, (float4*) destBuf, destPitch, is16f, width, height );

			cudaDeviceSynchronize();
		}
	#endif //GF_DEVICE_TARGET_HOST

#endif //SDK_CROSS_DISSOLVE
