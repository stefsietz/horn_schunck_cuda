/*
** ADOBE CONFIDENTIAL
**
** Copyright 2013 Adobe
** All Rights Reserved.
**
** NOTICE: Adobe permits you to use, modify, and distribute this file in
** accordance with the terms of the Adobe license agreement accompanying
** it. If you have received this file from a source other than Adobe,
** then your use, modification, or distribution of it requires the prior
** written permission of Adobe.
*/

/*
	SDK_CrossDissolve_GPU.cpp
	
	Revision History
		
	Version		Change													Engineer	Date
	=======		======													========	======
	1.0 		Created	with OpenCL render path							shoeg		8/5/2013
	1.1			Work around a crasher in CC								zlam		9/10/2013
	1.5			Fix SDK sample to handle 16f, added in 7.2				shoeg		4/23/2014
	2.0			Integrating CUDA render path generously provided by		zlam		1/20/2015
				Rama Hoetzlein from nVidia
    2.0.1       Fixed custom build steps for CUDA on Windows            zlam        5/6/2017
*/

#include "ResampleBilinear.cl.h"
#include "Convolve3x3.cl.h"
#include "Rgb2Gray.cl.h"
#include "FrameDiff.cl.h"
#include "SDK_CrossDissolve.h"
#include "PrGPUFilterModule.h"
#include "PrSDKVideoSegmentProperties.h"

#if _WIN32
#include <CL/cl.h>
#else
#include <OpenCL/cl.h>
#endif
#include <cuda_runtime.h>
#include <math.h>

//  CUDA KERNEL 
//  * See SDK_CrossDissolve.cu
extern void ResampleBilinear_CUDA ( cudaTextureObject_t texObj, float *destBuf, unsigned int destPitch, unsigned int is16bit, unsigned int width, unsigned int height);
extern void Convolve3x3_CUDA(cudaTextureObject_t texObj, float *destBuf, float *kernelBuf, int kernelRadius, int destPitch, int is16bit, unsigned int width, unsigned int height);
extern void Rgb2Gray_CUDA(float *inBuf, float *destBuf, int inPitch, int destPitch, int is16bit, unsigned int width, unsigned int height);
extern void FrameDiff_CUDA(float *inBuf, float *nextBuf, float *destBuf, int inPitch, int destPitch, int is16bit, unsigned int width, unsigned int height);

static cl_kernel sKernelCache[4];

/*
**
*/
class SDK_CrossDissolve :
	public PrGPUFilterBase
{
public:
	prSuiteError InitializeCUDA ()
	{
		// Nothing to do here. CUDA Kernel statically linked

		return suiteError_NoError;
		}

	prSuiteError InitializeOpenCL ()
		{
		if (mDeviceIndex > sizeof(sKernelCache) / sizeof(cl_kernel))  	{			
			return suiteError_Fail;		// Exceeded max device count
		}
        if (!mTransitionSuite)
        {
            // Running in PPro 7.0 in GPU mode, there is a crasher related to GetFrameDependencies
            // (error message: VideoFrameFactory.cpp-510)
            // So only use GPU rendering path in 7.1 and later (when TransitionSuite exists)
			return suiteError_Fail;
        }

		mCommandQueue = (cl_command_queue)mDeviceInfo.outCommandQueueHandle;

		// Load and compile the kernel - a real plugin would cache binaries to disk
		mKernel = sKernelCache[mDeviceIndex];
		if (!mKernel)
		{
			cl_int result = CL_SUCCESS;
			size_t size = strlen(kResampleBilinear_OpenCLString);
			char const* kKernelStrings = &kResampleBilinear_OpenCLString[0];
			cl_context context = (cl_context)mDeviceInfo.outContextHandle;
			cl_device_id device = (cl_device_id)mDeviceInfo.outDeviceHandle;
			cl_program program = clCreateProgramWithSource(context, 1, &kKernelStrings, &size, &result);
			if (result != CL_SUCCESS)
			{
				return suiteError_Fail;
			}

			result = clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0);
			if (result != CL_SUCCESS)
			{
				return suiteError_Fail;
			}

			mKernel = clCreateKernel(program, "CrossDissolveKernel", &result);
			if (result != CL_SUCCESS)
			{
				return suiteError_Fail;
			}

			sKernelCache[mDeviceIndex] = mKernel;
		}

		return suiteError_NoError;
	}


	virtual prSuiteError Initialize( PrGPUFilterInstance* ioInstanceData )
	{
		PrGPUFilterBase::Initialize(ioInstanceData);

		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)	
			return InitializeCUDA();			

		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL)
			return InitializeOpenCL();			

		return suiteError_Fail;			// GPUDeviceFramework unknown
	}

	prSuiteError GetFrameDependencies(
		const PrGPUFilterRenderParams* inRenderParams,
		csSDK_int32* ioQueryIndex,
		PrGPUFilterFrameDependency* outFrameRequirements)
	{
		PrTime clipStartTime = inRenderParams->inSequenceTime - inRenderParams->inClipTime;
		

		if (ioQueryIndex[0] == 0) {
			outFrameRequirements->outDependencyType = PrGPUDependency_InputFrame;
			outFrameRequirements->outTrackID = GetParam(1, inRenderParams->inClipTime).mInt32;
			outFrameRequirements->outSequenceTime = clipStartTime + inRenderParams->inClipTime /2;
			ioQueryIndex[0]++;
		}
		else {
			outFrameRequirements->outDependencyType = PrGPUDependency_InputFrame;
			outFrameRequirements->outTrackID = GetParam(1, inRenderParams->inClipTime).mInt32;
			outFrameRequirements->outSequenceTime = clipStartTime + inRenderParams->inClipTime /2 + inRenderParams->inRenderTicksPerFrame;
		}

		return suiteError_NoError;
	}


	prSuiteError Render(
		const PrGPUFilterRenderParams* inRenderParams,
		const PPixHand* inFrames,
		csSDK_size_t inFrameCount,
		PPixHand* outFrame)
	{
		float progress;

		// Initial steps are independent of CUDA and OpenCL

		if (inFrameCount < 3 || (!inFrames[1]) || (!inFrames[2]))
		{
			return suiteError_Fail;
		}

		// read the parameters
		int flip = GetParam(SDK_CROSSDISSOLVE_FLIP, inRenderParams->inClipTime).mBool;

		PPixHand properties = inFrames[1];

		csSDK_uint32 index = 0;
		mGPUDeviceSuite->GetGPUPPixDeviceIndex(properties, &index);

		// Get pixel format
		PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
		mPPixSuite->GetPixelFormat(properties, &pixelFormat);
		int is16f = pixelFormat != PrPixelFormat_GPU_BGRA_4444_32f;

		// Get width & height
		prRect bounds = {};
		mPPixSuite->GetBounds(properties, &bounds);
		int width = bounds.right - bounds.left;
		int height = bounds.bottom - bounds.top;

		csSDK_uint32 parNumerator = 0;
		csSDK_uint32 parDenominator = 0;
		mPPixSuite->GetPixelAspectRatio(properties, &parNumerator, &parDenominator);
		
		prFieldType fieldType = 0;
		mPPix2Suite->GetFieldOrder(properties, &fieldType);

		// Create a frame to process output
		mGPUDeviceSuite->CreateGPUPPix(
			index,
			pixelFormat,
			width,
			height,
			parNumerator,
			parDenominator,
			fieldType,
			outFrame);

		if (!outFrame)
		{
			return suiteError_Fail;
		}

		// Get incoming data
		void* incomingFrameData = 0;
		csSDK_int32 incomingRowBytes = 0;
		if (inFrames[1])
		{
			mGPUDeviceSuite->GetGPUPPixData(inFrames[1], &incomingFrameData);
			mPPixSuite->GetRowBytes(inFrames[1], &incomingRowBytes);
		}

		// Get incoming data
		void* nextFrameData = 0;
		if (inFrames[2])
		{
			mGPUDeviceSuite->GetGPUPPixData(inFrames[2], &nextFrameData);
		}

		// Get dest data
		int incomingPitch = incomingRowBytes / GetGPUBytesPerPixel(pixelFormat);
		void* destFrameData = 0;
		csSDK_int32 destRowBytes = 0;
		mGPUDeviceSuite->GetGPUPPixData(*outFrame, &destFrameData);
		mPPixSuite->GetRowBytes(*outFrame, &destRowBytes);
		int destPitch = destRowBytes / GetGPUBytesPerPixel(pixelFormat);

		if (!incomingFrameData || !nextFrameData)
		{
			return suiteError_Fail;
		}

		// Start CUDA or OpenCL specific code

		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA) {
			int downscale = 1;
			int scaleFac = pow(2, downscale);
				
			float* incomingBuffer = (float* )incomingFrameData;	
			float* nextFrameBuffer = (float*)nextFrameData;

			float* intermediateBuffer1;
			mGPUDeviceSuite->AllocateDeviceMemory(0, (int)(sizeof(float) * 4 * width * height / scaleFac / scaleFac), (void**)&intermediateBuffer1);

			float* intermediateBuffer2;
			mGPUDeviceSuite->AllocateDeviceMemory(0, (int)(sizeof(float) * 4 * width * height / scaleFac / scaleFac), (void**)&intermediateBuffer2);

			float* intermediateBuffer3;
			mGPUDeviceSuite->AllocateDeviceMemory(0, (int)(sizeof(float) * 4 * width * height / scaleFac / scaleFac), (void**)&intermediateBuffer3);

			float* intermediateBuffer4;
			mGPUDeviceSuite->AllocateDeviceMemory(0, (int)(sizeof(float) * 4 * width * height / scaleFac / scaleFac), (void**)&intermediateBuffer4);

			float* destBuffer = (float*)destFrameData;

			Downsample(incomingBuffer, intermediateBuffer1, destPitch, width, height, width, height, downscale, is16f);
			Downsample(nextFrameBuffer, intermediateBuffer3, destPitch, width, height, width, height, downscale, is16f);
			Rgb2Gray(intermediateBuffer1, intermediateBuffer2, incomingPitch / scaleFac, destPitch/ scaleFac, width / scaleFac, height / scaleFac, is16f);
			Rgb2Gray(intermediateBuffer3, intermediateBuffer4, incomingPitch / scaleFac, destPitch / scaleFac, width / scaleFac, height / scaleFac, is16f);
			Gaussian3x3(intermediateBuffer2, intermediateBuffer1, destPitch / scaleFac, width / scaleFac, height / scaleFac, is16f);
			Gaussian3x3(intermediateBuffer4, intermediateBuffer3, destPitch / scaleFac, width / scaleFac, height / scaleFac, is16f);

			//DiffY(intermediateBuffer2, intermediateBuffer1, destPitch / scaleFac, width / scaleFac, height / scaleFac, is16f);
			FrameDiff(intermediateBuffer1, intermediateBuffer3, intermediateBuffer2, incomingPitch / scaleFac, destPitch / scaleFac, width / scaleFac, height / scaleFac, is16f);
			Downsample(intermediateBuffer2, destBuffer, destPitch, width / scaleFac, height / scaleFac, width, height, 0, is16f);

			cudaFree(intermediateBuffer1);
			cudaFree(intermediateBuffer2);
			cudaFree(intermediateBuffer3);
			cudaFree(intermediateBuffer4);

	
			if ( cudaPeekAtLastError() != cudaSuccess) 			
			{
				return suiteError_Fail;
			}

		} else {
			// OpenCL device pointers
			cl_mem incomingBuffer = (cl_mem)incomingFrameData;
			cl_mem destBuffer = (cl_mem)destFrameData;

			// Set the arguments
			clSetKernelArg(mKernel, 0, sizeof(cl_mem), &incomingBuffer);
			clSetKernelArg(mKernel, 1, sizeof(cl_mem), &destBuffer);
			clSetKernelArg(mKernel, 2, sizeof(unsigned int), &incomingPitch);
			clSetKernelArg(mKernel, 3, sizeof(unsigned int), &destPitch);
			clSetKernelArg(mKernel, 4, sizeof(int), &is16f);
			clSetKernelArg(mKernel, 5, sizeof(unsigned int), &width);
			clSetKernelArg(mKernel, 6, sizeof(unsigned int), &height);

			// Launch the kernel
			size_t threadBlock[2] = { 16, 16 };
			size_t grid[2] = { RoundUp(width, threadBlock[0]), RoundUp(height, threadBlock[1] )};

			cl_int result = clEnqueueNDRangeKernel(
				mCommandQueue,
				mKernel,
				2,
				0,
				grid,
				threadBlock,
				0,
				0,
				0);

			if ( result != CL_SUCCESS )	
				return suiteError_Fail;
		}
		return suiteError_NoError;
	}

	void DiffY(float* incomingFrameData, float* destFrameData, int destPitch, int width, int height, int is16f) {

		float kernel[9] = {
			-1.0 / 4, -1.0 / 2, -1.0 / 4,
			0, 0, 0,
			1.0 / 4, 1.0 / 2, 1.0 / 4
		};

		Diff(incomingFrameData, destFrameData, kernel, destPitch, width, height, is16f);
	}

	void DiffX(float* incomingFrameData, float* destFrameData, int destPitch, int width, int height, int is16f) {

		float kernel[9] = {
			-1.0 / 4, 0, 1.0 / 4,
			-1.0 / 2, 0, 1.0 / 2,
			-1.0 / 4, 0, 1.0 / 4
		};

		Diff(incomingFrameData, destFrameData, kernel, destPitch, width, height, is16f);
	}

	void Diff(float* incomingFrameData, float* destFrameData, float* kernel, int destPitch, int width, int height, int is16f) {

		cudaChannelFormatDesc channelDesc =
			cudaCreateChannelDesc<float4>();

		cudaArray* cuArrayIn;
		cudaMallocArray(&cuArrayIn, &channelDesc, width, height);

		cudaMemcpyToArray(cuArrayIn, 0, 0, incomingFrameData, width*height * 16,
			cudaMemcpyDeviceToDevice);

		// Specify texture
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArrayIn;

		// Specify texture object parameters
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeMirror;
		texDesc.addressMode[1] = cudaAddressModeMirror;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		// Create texture object
		cudaTextureObject_t texObj = 0;
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

		float* kernelBuffer;
		mGPUDeviceSuite->AllocateDeviceMemory(0, sizeof(float) * 9, (void**)&kernelBuffer);
		cudaMemcpy(kernelBuffer, &kernel[0], sizeof(float) * 9,
			cudaMemcpyHostToDevice);

		// Launch CUDA kernel
		Convolve3x3_CUDA(
			texObj,
			destFrameData,
			kernelBuffer,
			1,
			destPitch,
			is16f,
			width,
			height);

		cudaDestroyTextureObject(texObj);

		// Free device memory
		cudaFreeArray(cuArrayIn);
	}

	void Gaussian3x3(float* incomingFrameData, float* destFrameData, int destPitch, int width, int height, int is16f) {

		cudaChannelFormatDesc channelDesc =
			cudaCreateChannelDesc<float4>();

		cudaArray* cuArrayIn;
		cudaMallocArray(&cuArrayIn, &channelDesc, width, height);

		cudaMemcpyToArray(cuArrayIn, 0, 0, incomingFrameData, width*height * 16,
			cudaMemcpyDeviceToDevice);

		// Specify texture
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArrayIn;

		// Specify texture object parameters
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeMirror;
		texDesc.addressMode[1] = cudaAddressModeMirror;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		// Create texture object
		cudaTextureObject_t texObj = 0;
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

		float kernel[9] = {
			1.0 / 16, 1.0 / 8, 1.0 / 16,
			1.0 / 8, 1.0 / 4, 1.0 / 8,
			1.0 / 16, 1.0 / 8, 1.0 / 16
		};

		float* kernelBuffer;
		mGPUDeviceSuite->AllocateDeviceMemory(0, sizeof(float) * 9, (void**)&kernelBuffer);
		cudaMemcpy(kernelBuffer, &kernel[0], sizeof(float) * 9,
			cudaMemcpyHostToDevice);

		// Launch CUDA kernel
		Convolve3x3_CUDA(
			texObj,
			destFrameData,
			kernelBuffer,
			1,
			destPitch,
			is16f,
			width,
			height);

		cudaDestroyTextureObject(texObj);

		// Free device memory
		cudaFreeArray(cuArrayIn);
	}

	void FrameDiff(float* incomingFrameData, float* nextFrameData, float* destFrameData, int inPitch, int destPitch, int width, int height, int is16f) {

		// Launch CUDA kernel
		FrameDiff_CUDA(
			incomingFrameData,
			nextFrameData,
			destFrameData,
			inPitch,
			destPitch,
			is16f,
			width,
			height);
	}

	void Rgb2Gray(float* incomingFrameData, float* destFrameData, int inPitch, int destPitch, int width, int height, int is16f) {

		// Launch CUDA kernel
		Rgb2Gray_CUDA(
			incomingFrameData,
			destFrameData,
			inPitch,
			destPitch,
			is16f,
			width,
			height);
	}

	void Downsample(float* incomingFrameData, float* destFrameData, int destPitch, int inWidth, int inHeight, int outWidth, int outHeight, int levels, int is16f) {

		cudaChannelFormatDesc channelDesc =
			cudaCreateChannelDesc<float4>();

		cudaArray* cuArrayIn;
		cudaMallocArray(&cuArrayIn, &channelDesc, inWidth, inHeight);

		cudaMemcpyToArray(cuArrayIn, 0, 0, incomingFrameData, inWidth*inHeight * 16,
			cudaMemcpyDeviceToDevice);

		// Specify texture
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArrayIn;

		// Specify texture object parameters
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeMirror;
		texDesc.addressMode[1] = cudaAddressModeMirror;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		// Create texture object
		cudaTextureObject_t texObj = 0;
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

		float* tempOutBuffer;
		mGPUDeviceSuite->AllocateDeviceMemory(0, (int)(sizeof(float) * 4 * outWidth * outHeight / 4), (void**)&tempOutBuffer);

		int scale = 1;
		for (int i = 1; i <= levels; i++) {
			scale = pow(2, i);

			// Launch CUDA kernel
			ResampleBilinear_CUDA(
				texObj,
				tempOutBuffer,
				destPitch / scale,
				is16f,
				outWidth / scale,
				outHeight / scale);

			cudaFreeArray(cuArrayIn);
			cudaMallocArray(&cuArrayIn, &channelDesc, outWidth / scale, outHeight / scale);
			cudaDestroyTextureObject(texObj);
			cudaMemcpyToArray(cuArrayIn, 0, 0, tempOutBuffer, (int)(sizeof(float) * 4 * outWidth * outHeight /scale /scale),
				cudaMemcpyDeviceToDevice);
			resDesc.res.array.array = cuArrayIn;
			cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
		}

		float* destBuffer = (float*)destFrameData;
		// Launch CUDA kernel
		ResampleBilinear_CUDA(
			texObj,
			destBuffer,
			destPitch / scale,
			is16f,
			outWidth / scale,
			outHeight / scale );

		cudaDestroyTextureObject(texObj);

		// Free device memory
		cudaFreeArray(cuArrayIn);
		cudaFree(tempOutBuffer);
	}

private:
	// CUDA


	// OpenCL
	cl_command_queue mCommandQueue;
	cl_kernel mKernel;
};


DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<SDK_CrossDissolve>)
