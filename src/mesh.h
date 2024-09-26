#pragma once

#include <stb_image.h>  
#include <stb_image_write.h>  
#include <tiny_gltf.h> 

#include <string>

namespace mesh {
	void loadGLTF(std::string filename);
	tinygltf::Model LoadModel(const std::string& filepath);
}