#include <iostream>
#include <cstring>
#include <type_traits>
#include <stb_image.h>

#include "texture.h"

template<typename T>
bool loadTexture(const std::string& filename, HostTexture<T>& outTex) {
    //stbi_set_flip_vertically_on_load(true);
    stbi_set_flip_vertically_on_load(false);

    int width = 0, height = 0, channels = 0;

    // Branch based on type
    if constexpr (std::is_same<T, unsigned char>::value) {
        unsigned char* pixels = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);
        if (!pixels) {
            std::cerr << "Failed to load 8-bit texture: " << filename << "\n";
            std::cerr << "stb_image error: " << stbi_failure_reason() << "\n";
            return false;
        }

        size_t totalPixels = static_cast<size_t>(width) * height * 4;
        outTex.data = new unsigned char[totalPixels];
        std::memcpy(outTex.data, pixels, totalPixels * sizeof(unsigned char));
        stbi_image_free(pixels);
    }
    else if constexpr (std::is_same<T, float>::value) {
        float* pixels = stbi_loadf(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);
        if (!pixels) {
            std::cerr << "Failed to load HDR texture: " << filename << "\n";
            std::cerr << "stb_image error: " << stbi_failure_reason() << "\n";
            return false;
        }

        size_t totalPixels = static_cast<size_t>(width) * height * 4;
        outTex.data = new float[totalPixels];
        std::memcpy(outTex.data, pixels, totalPixels * sizeof(float));
        stbi_image_free(pixels);
    }
    else {
        static_assert(!std::is_same<T, T>::value, "Unsupported texture type.");
    }

    // Populate metadata
    outTex.width = width;
    outTex.height = height;
    outTex.channels = 4; // STBI_rgb_alpha ensures this

    return true;
}

// Frees host texture memory
template<typename T>
void freeTexture(HostTexture<T>& tex) {
    delete[] tex.data;
    tex.data = nullptr;
    tex.width = tex.height = tex.channels = 0;
}

// Explicit template instantiations (required in a .cpp file)
template bool loadTexture<unsigned char>(const std::string&, HostTexture<unsigned char>&);
template bool loadTexture<float>(const std::string&, HostTexture<float>&);
template void freeTexture<unsigned char>(HostTexture<unsigned char>&);
template void freeTexture<float>(HostTexture<float>&);
