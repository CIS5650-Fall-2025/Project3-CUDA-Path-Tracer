#pragma once

#include <stb_image.h>  
#include <stb_image_write.h>  
#include <tiny_gltf.h> 

#include <string>

namespace mesh {
	void loadgltf(std::string filename);
}