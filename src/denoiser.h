#pragma once
#include <OpenImageDenoise/oidn.hpp>
#include "utilities.h"

void denosier(int width, int height, uchar4* colorBuffer, uchar4* outputBuffer, uchar4* albedoBuffer, uchar4* normalBuffer)
{
	oidn::DeviceRef device = oidn::newDevice();
	device.commit();

	oidn::FilterRef filter = device.newFilter("RT");

	filter.setImage("color", colorBuffer, oidn::Format::Float3, width, height);
	filter.setImage("output", outputBuffer, oidn::Format::Float3, width, height);
	filter.setImage("albedo", albedoBuffer, oidn::Format::Float3, width, height);
	filter.setImage("normal", normalBuffer, oidn::Format::Float3, width, height);
	filter.commit();

	filter.execute();

	const char* errorMessage;
	if (device.getError(errorMessage) != oidn::Error::None) {
		std::cerr << "Error: " << errorMessage << std::endl;
	}


}

