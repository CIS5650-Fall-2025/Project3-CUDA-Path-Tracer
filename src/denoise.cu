// denoise.cu
#include "denoise.h"

// Copy input data (color, normal, albedo) into OIDN's GPU buffers
__global__ void CopyToOIDNInputs(
    int pixelCount,
    const glm::vec3* colorIn,       // from dev_image
    const glm::vec3* albedoIn,      // from dev_albedo
    const glm::vec3* normalIn,      // from dev_normal
    glm::vec3* oidnColor,           // to OIDN color buffer
    glm::vec3* oidnNormal,          // to OIDN normal buffer
    glm::vec3* oidnAlbedo           // to OIDN albedo buffer
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pixelCount) {
        oidnColor[i] = colorIn[i];                         // no need to normalize
        oidnNormal[i] = glm::normalize(normalIn[i]);
        oidnAlbedo[i] = glm::normalize(albedoIn[i]);
    }
}

// Copy denoised result from OIDN back into our final image buffer
__global__ void CopyFromOIDNOutput(
    int pixelCount,
    const glm::vec3* oidnColor,     // from OIDN color buffer
    glm::vec3* finalImageOut        // to dev_denoised_image
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pixelCount) {
        finalImageOut[i] = oidnColor[i];
    }
}

void setupOIDN(DenoiserState& state, int width, int height)
{
    if (state.oidnCudaDevice) return;

    const int pixelCount = width * height;

    // Create an OIDN device that runs on the GPU using CUDA
    state.oidnCudaDevice = oidn::newDevice(oidn::DeviceType::CUDA);
    state.oidnCudaDevice.commit();  // Finalize device creation

    // Allocate GPU buffers for the inputs and output (one pixel = glm::vec3)
    state.oidnColorBuffer = state.oidnCudaDevice.newBuffer(pixelCount * sizeof(glm::vec3));   // stores noisy image & output
    state.oidnNormalBuffer = state.oidnCudaDevice.newBuffer(pixelCount * sizeof(glm::vec3));   // stores surface normals
    state.oidnAlbedoBuffer = state.oidnCudaDevice.newBuffer(pixelCount * sizeof(glm::vec3));   // stores base color (albedo)

    // Create a main denoising filter using the "RT" model (for path-traced images)
    state.oidnMainDenoiser = state.oidnCudaDevice.newFilter("RT");  // "RT" = Ray Tracing model

    // Create debug filters to visualize albedo/normal buffers directly
    state.oidnDebugAlbedoDenoiser = state.oidnCudaDevice.newFilter("RT");
    state.oidnDebugAlbedoDenoiser.setImage("albedo", state.oidnAlbedoBuffer, oidn::Format::Float3, width, height);
    state.oidnDebugAlbedoDenoiser.setImage("output", state.oidnAlbedoBuffer, oidn::Format::Float3, width, height);
    state.oidnDebugAlbedoDenoiser.commit();

    state.oidnDebugNormalDenoiser = state.oidnCudaDevice.newFilter("RT");
    state.oidnDebugNormalDenoiser.setImage("normal", state.oidnNormalBuffer, oidn::Format::Float3, width, height);
    state.oidnDebugNormalDenoiser.setImage("output", state.oidnNormalBuffer, oidn::Format::Float3, width, height);
    state.oidnDebugNormalDenoiser.commit();

    // Configure the main denoising filter with color, normal, and albedo inputs
    state.oidnMainDenoiser.setImage("color", state.oidnColorBuffer, oidn::Format::Float3, width, height);   // noisy input
    state.oidnMainDenoiser.setImage("normal", state.oidnNormalBuffer, oidn::Format::Float3, width, height); // guides
    state.oidnMainDenoiser.setImage("albedo", state.oidnAlbedoBuffer, oidn::Format::Float3, width, height); // guides
    state.oidnMainDenoiser.setImage("output", state.oidnColorBuffer, oidn::Format::Float3, width, height);  // write result back to colorBuffer

    // Settings for the denoiser
    state.oidnMainDenoiser.set("hdr", true);       // Input is in HDR (float values may be > 1.0)
    state.oidnMainDenoiser.set("cleanAux", true);  // Albedo and normal inputs are clean (not noisy)

    // Finalize the filter setup before running it
    state.oidnMainDenoiser.commit();

    std::cout << "[OIDN] Setup complete with resolution: " << width << " x " << height << std::endl;

}


void runDenoisingPipeline(
    DenoiserState& state,
    glm::vec3* devColor,
    glm::vec3* devNormal,
    glm::vec3* devAlbedo,
    int pixelCount,
    int iteration, 
    glm::vec3* devDenoisedColor)
{
    const int blockSize = 128;
    const int numBlocks = (pixelCount + blockSize - 1) / blockSize;

    glm::vec3* oidnColorPtr = (glm::vec3*)state.oidnColorBuffer.getData();
    glm::vec3* oidnNormalPtr = (glm::vec3*)state.oidnNormalBuffer.getData();
    glm::vec3* oidnAlbedoPtr = (glm::vec3*)state.oidnAlbedoBuffer.getData();

    CopyToOIDNInputs <<<numBlocks, blockSize>>> (
        pixelCount, devColor, devAlbedo, devNormal,
        oidnColorPtr, oidnNormalPtr, oidnAlbedoPtr);

    cudaDeviceSynchronize();

    state.oidnDebugAlbedoDenoiser.execute();
    state.oidnDebugNormalDenoiser.execute();


    state.oidnMainDenoiser.execute();

    cudaDeviceSynchronize();

    // Copy OIDN output to framebuffer for display
    CopyFromOIDNOutput <<<numBlocks, blockSize>>> (pixelCount, oidnColorPtr, devDenoisedColor);
}

void cleanupOIDN(DenoiserState& state)
{
    state.oidnColorBuffer.release();
    state.oidnNormalBuffer.release();
    state.oidnAlbedoBuffer.release();
    state.oidnDebugNormalDenoiser.release();
    state.oidnDebugAlbedoDenoiser.release();
    state.oidnMainDenoiser.release();
    state.oidnCudaDevice.release();
}
