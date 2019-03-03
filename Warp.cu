
#ifndef RESAMPLE_BILINEAR
	#define RESAMPLE_BILINEAR

    #if __CUDACC_VER_MAJOR__ >= 9
        #include <cuda_fp16.h>
    #endif
	#include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
	#include "PrGPU/KernelSupport/KernelMemory.h"

	#if GF_DEVICE_TARGET_DEVICE
		GF_TEXTURE_GLOBAL(float4, inSrcTexture, GF_DOMAIN_NATURAL, GF_RANGE_NATURAL_CUDA, GF_EDGE_CLAMP, GF_FILTER_LINEAR)
		GF_TEXTURE_GLOBAL(float4, inWarpTexture, GF_DOMAIN_NATURAL, GF_RANGE_NATURAL_CUDA, GF_EDGE_CLAMP, GF_FILTER_LINEAR)

		GF_KERNEL_FUNCTION(kWarp,
			((GF_TEXTURE_TYPE(float4))(GF_TEXTURE_NAME(inSrcTexture)))
			((GF_TEXTURE_TYPE(float4))(GF_TEXTURE_NAME(inWarpTexture)))
			((GF_PTR(float4))(destImg)),
			((unsigned int)(destPitch))
			((float)(scaleFac))
			((unsigned int)(in16f))
			((unsigned int)(flip))
			((unsigned int)(outWidth))
			((unsigned int)(outHeight)),
			((uint2)(outXY)(KERNEL_XY)))
		{
			float4  color, warpColor, dest;


			if (outXY.x >= outWidth || outXY.y >= outHeight) return;

			warpColor = GF_READTEXTURE(GF_TEXTURE_NAME(inWarpTexture), (outXY.x + 0.5) / outWidth, (outXY.y + 0.5) / outHeight);
			warpColor.x *= scaleFac;
			warpColor.y *= scaleFac;
			warpColor.z *= scaleFac;

			if(!!flip)
				color = GF_READTEXTURE(GF_TEXTURE_NAME(inSrcTexture), (outXY.x - warpColor.x + 0.5) / outWidth, (outXY.y - warpColor.y + 0.5) / outHeight);
			else
				color = GF_READTEXTURE(GF_TEXTURE_NAME(inSrcTexture), (outXY.x + 0.5) / outWidth, (outXY.y + 0.5) / outHeight);

			dest = color;
			dest.x *= 1;
			dest.y *= 1;
			dest.z *= 1;
			dest.w = 1;

			//dest.x += warpColor.x;
			//dest.y += warpColor.y;
			//dest.z += warpColor.z;

			//warpColor.x = (warpColor.x > 50 ? 50 : warpColor.x);
			//warpColor.y = (warpColor.y > 50 ? 50 : warpColor.y);
			//warpColor.x = (warpColor.x < -50 ? -50 : warpColor.x);
			//warpColor.y = (warpColor.y < -50 ? -50 : warpColor.y);

			int index = (outXY.y) * destPitch + outXY.x ;
			if(!flip)
				index = (outXY.y + (int)warpColor.y) * destPitch + outXY.x + warpColor.x;
			//index = (outXY.y) * destPitch + outXY.x;
			if (index > outWidth * outHeight - 1)
				index = (outXY.y) * destPitch + outXY.x;
			WriteFloat4(dest, destImg, index, !!in16f);
		}
	#endif

	#if __NVCC__
		void Warp_CUDA (	
			cudaTextureObject_t inSrcTexture,
			cudaTextureObject_t inWarpTexture,
			float *destBuf,
			unsigned int destPitch,
			float scaleFac,
			unsigned int is16f,
			unsigned int flip,
			unsigned int width,
			unsigned int height)
		{
			dim3 blockDim (16, 16, 1);
			dim3 gridDim ( (width + blockDim.x - 1)/ blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1 );		

			kWarp <<< gridDim, blockDim, 0 >>> ( inSrcTexture, inWarpTexture,(float4*) destBuf, destPitch, scaleFac, is16f, flip, width, height );

			cudaDeviceSynchronize();
		}
	#endif //GF_DEVICE_TARGET_HOST

#endif //SDK_CROSS_DISSOLVE
