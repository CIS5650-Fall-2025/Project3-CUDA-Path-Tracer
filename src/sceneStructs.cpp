#include "sceneStructs.h"
#include <fstream>

void Camera::serialize(std::ofstream& ofs)
{
    ofs.write(reinterpret_cast<const char*>(&resolution), sizeof(resolution));
    ofs.write(reinterpret_cast<const char*>(&position), sizeof(position));
    ofs.write(reinterpret_cast<const char*>(&lookAt), sizeof(lookAt));
    ofs.write(reinterpret_cast<const char*>(&view), sizeof(view));
    ofs.write(reinterpret_cast<const char*>(&up), sizeof(up));
    ofs.write(reinterpret_cast<const char*>(&right), sizeof(right));
    ofs.write(reinterpret_cast<const char*>(&fov), sizeof(fov));
    ofs.write(reinterpret_cast<const char*>(&pixelLength), sizeof(pixelLength));
    ofs.write(reinterpret_cast<const char*>(&aperture), sizeof(aperture));
    ofs.write(reinterpret_cast<const char*>(&exposure), sizeof(exposure));
}

void Camera::deserialize(std::ifstream& ifs)
{
    ifs.read(reinterpret_cast<char*>(&resolution), sizeof(resolution));
    ifs.read(reinterpret_cast<char*>(&position), sizeof(position));
    ifs.read(reinterpret_cast<char*>(&lookAt), sizeof(lookAt));
    ifs.read(reinterpret_cast<char*>(&view), sizeof(view));
    ifs.read(reinterpret_cast<char*>(&up), sizeof(up));
    ifs.read(reinterpret_cast<char*>(&right), sizeof(right));
    ifs.read(reinterpret_cast<char*>(&fov), sizeof(fov));
    ifs.read(reinterpret_cast<char*>(&pixelLength), sizeof(pixelLength));
    ifs.read(reinterpret_cast<char*>(&aperture), sizeof(aperture));
    ifs.read(reinterpret_cast<char*>(&exposure), sizeof(exposure));
}


void RenderState::serialize(std::ofstream& ofs)
{

    // Serialize camera
    camera.serialize(ofs);

    // Serialize render state properties
    ofs.write(reinterpret_cast<const char*>(&iterations), sizeof(iterations));
    ofs.write(reinterpret_cast<const char*>(&traceDepth), sizeof(traceDepth));

    // Serialize image
    size_t imageSize = image.size();
    ofs.write(reinterpret_cast<const char*>(&imageSize), sizeof(imageSize));
    ofs.write(reinterpret_cast<const char*>(image.data()), imageSize * sizeof(glm::vec3));

    // Serialize image name (store string size first)
    size_t nameLength = imageName.size();
    ofs.write(reinterpret_cast<const char*>(&nameLength), sizeof(nameLength));
    ofs.write(imageName.c_str(), nameLength);
}

void RenderState::deserialize(std::ifstream& ifs)
{
    // Deserialize camera
    camera.deserialize(ifs);

    // Deserialize render state properties
    ifs.read(reinterpret_cast<char*>(&iterations), sizeof(iterations));
    ifs.read(reinterpret_cast<char*>(&traceDepth), sizeof(traceDepth));

    // Deserialize image
    size_t imageSize;
    ifs.read(reinterpret_cast<char*>(&imageSize), sizeof(imageSize));
    ifs.read(reinterpret_cast<char*>(image.data()), imageSize * sizeof(glm::vec3));

    // Deserialize image name
    size_t nameLength;
    ifs.read(reinterpret_cast<char*>(&nameLength), sizeof(nameLength));
    imageName.resize(nameLength);
    ifs.read(&imageName[0], nameLength);
}
