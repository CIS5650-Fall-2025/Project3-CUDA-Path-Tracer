#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

using namespace std;

class image {
public:
    int width;
    int height;
    glm::vec3* pixels;


    image(int x, int y);
    image(std::string filePath, float gamma = 1.f);
    ~image();
    void setPixel(int x, int y, const glm::vec3 &pixel);
    // Default 3 channels
    void savePNG(const std::string &baseFilename);
    void savePNG(const std::string& baseFilename, int channels);
    void saveHDR(const std::string &baseFilename);

    size_t byteSize() const {
        return sizeof(glm::vec3) * width * height;
    }

    glm::vec3* data() const {
        return pixels;
    }

    glm::vec3 getValue(int x, int y) const
    {
        return pixels[y * width + x];
    }

    glm::vec3 linearSample(const glm::vec2& uv) const
    {
        float u = uv.x, v = uv.y;
        float x = u * (width - 1), y = v * (height - 1);
        int lx = x, ux = x + 1 >= width ? lx : lx + 1;
        int ly = y, uy = y + 1 >= height ? ly : ly + 1;

        float fx = glm::fract(x), fy = glm::fract(y);
        glm::vec3 vll = getValue(lx, ly), vul = getValue(ux, ly), vlu = getValue(lx, uy), vuu = getValue(ux, uy);
        glm::vec3 p1 = glm::mix(getValue(lx, ly), getValue(ux, ly), fx);
        glm::vec3 p2 = glm::mix(getValue(lx, uy), getValue(ux, uy), fx);
        return glm::mix(p1, p2, fy);
    }
};

class DevTexObj
{
public:
    int width, height;
    glm::vec3* data;

    DevTexObj() = default;

    DevTexObj(image* img, glm::vec3 *devData)
    {
        width = img->width;
        height = img->height;
        data = devData;
    }

    __host__ __device__ glm::vec3 getValue(int x, int y)
    {
        return data[y * width + x];
    }

    __host__ __device__ glm::vec3 linearSample(const glm::vec2 &_uv)
    {
        //float u = uv.x, v = uv.y;
        //float x = u * (width - 1), y = v * (height - 1);
        //int lx = x, ux = x + 1 >= width ? lx : lx + 1;
        //int ly = y, uy = y + 1 >= height ? ly : ly + 1;

        //float fx = glm::fract(x), fy = glm::fract(y);
        //volatile float cll = getValue(lx, ly).x, cul = getValue(ux, ly).x, clu = getValue(lx, uy).x, cuu = getValue(ux, uy).x;
        //glm::vec3 p1 = glm::mix(getValue(lx, ly), getValue(ux, ly), fx);
        //glm::vec3 p2 = glm::mix(getValue(lx, uy), getValue(ux, uy), fx);
        //return glm::mix(p1, p2, fy);

        const float Eps = FLT_MIN;
        glm::vec2 uv = glm::fract(_uv);

        float fx = uv.x * (width - Eps) + .5f;
        float fy = uv.y * (height - Eps) + .5f;

        int ix = glm::fract(fx) > .5f ? fx : fx - 1;
        ix += width * (ix < 0);

        int iy = glm::fract(fy) > .5f ? fy : fy - 1;
        iy += height * (iy < 0);

        int ux = ix + 1;
        ux -= width * (ux >= width);

        int uy = iy + 1;
        uy -= height * (uy >= height);

        float lx = glm::fract(fx + .5f);
        float ly = glm::fract(fy + .5f);

        glm::vec3 c1 = glm::mix(getValue(ix, iy), getValue(ux, iy), lx);
        glm::vec3 c2 = glm::mix(getValue(ix, uy), getValue(ux, uy), lx);
        return glm::mix(c1, c2, ly);
    }
};

class DevTexSampler
{
public:
    DevTexObj* tex;
    glm::vec3 fixedVal;

    DevTexSampler() :tex(nullptr), fixedVal(0) {};

    DevTexSampler(DevTexObj *_tex) :tex(_tex) {}

    DevTexSampler(const glm::vec3& val) :fixedVal(val), tex(nullptr) {}

    DevTexSampler(float val) :fixedVal(glm::vec3(val)), tex(nullptr) {}

    __host__ __device__ glm::vec3 linearSample(const glm::vec2 &uv) const
    {
        if (!this->tex)
        {
            return fixedVal;
        }
        return tex->linearSample(uv);
    }
};