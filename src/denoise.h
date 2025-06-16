#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>
#include "./thirdparty/oidn-2.3.0.x64.windows/include/OpenImageDenoise/oidn.hpp"


struct DenoiserState {
    // Denoising device (runs on CUDA backend)
    oidn::DeviceRef oidnCudaDevice;

    // Buffers for noisy inputs
    oidn::BufferRef oidnColorBuffer;
    oidn::BufferRef oidnNormalBuffer;
    oidn::BufferRef oidnAlbedoBuffer;

    // Main filter that performs full denoising
    oidn::FilterRef oidnMainDenoiser;

    oidn::FilterRef oidnDebugNormalDenoiser;
    oidn::FilterRef oidnDebugAlbedoDenoiser;
};


__global__ void CopyToOIDNInputs(
    int pixelCount,
    const glm::vec3* colorIn,       // from dev_image
    const glm::vec3* albedoIn,      // from dev_albedo
    const glm::vec3* normalIn,      // from dev_normal
    glm::vec3* oidnColor,           // to OIDN color buffer
    glm::vec3* oidnNormal,          // to OIDN normal buffer
    glm::vec3* oidnAlbedo           // to OIDN albedo buffer
);

__global__ void CopyFromOIDNOutput(
    int pixelCount,
    const glm::vec3* oidnColor,     // from OIDN color buffer
    glm::vec3* finalImageOut        // to dev_denoised_image
);


// Initializes OIDN and allocates necessary buffers
void setupOIDN(DenoiserState& state, int width, int height);

// Runs denoising on the given device buffers
void runDenoisingPipeline(
    DenoiserState& state,
    glm::vec3* devColor,
    glm::vec3* devNormal,
    glm::vec3* devAlbedo,
    int pixelCount,
    int iteration,
    glm::vec3* devDenoisedColor);

// Releases OIDN resources
void cleanupOIDN(DenoiserState& state);


