//#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include <set>


#include "main.h"
#include "preview.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include "selection_state.h"   // holds `extern SelectionState selection;`


GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

std::string currentTimeString()
{
	time_t now;
	time(&now);
	char buf[sizeof "0000-00-00_00-00-00z"];
	strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
	return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures()
{
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
	GLfloat vertices[] = {
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f,  1.0f,
		-1.0f,  1.0f,
	};

	GLfloat texcoords[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
	const char* attribLocations[] = { "Position", "Texcoords" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	//glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1)
	{
		glUniform1i(location, 0);
	}

	return program;
}

void deletePBO(GLuint* pbo)
{
	if (pbo)
	{
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

void deleteTexture(GLuint* tex)
{
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

void cleanupCuda()
{
	if (pbo)
	{
		deletePBO(&pbo);
	}
	if (displayImage)
	{
		deleteTexture(&displayImage);
	}
}

void initCuda()
{
	cudaGLSetGLDevice(0);

	// Clean up on program exit
	atexit(cleanupCuda);
}

void initPBO()
{
	// set up vertex data parameter
	int num_texels = width * height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;

	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &pbo);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(pbo);
}

void errorCallback(int error, const char* description)
{
	fprintf(stderr, "%s\n", description);
}

bool init()
{
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit())
	{
		exit(EXIT_FAILURE);
	}

	window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	// Set up GL context
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		return false;
	}
	printf("Opengl Version:%s\n", glGetString(GL_VERSION));
	//Set up ImGui

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	io = &ImGui::GetIO(); (void)io;
	ImGui::StyleColorsLight();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 120");

	// Initialize other stuff
	initVAO();
	initTextures();
	initCuda();
	initPBO();
	GLuint passthroughProgram = initShader();

	glUseProgram(passthroughProgram);
	glActiveTexture(GL_TEXTURE0);

	return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
	imguiData = guiData;

}


void updateGeomTransform(Geom& g) {
	g.transform = utilityCore::buildTransformationMatrix(
		g.translation, g.rotation, g.scale);
	g.inverseTransform = glm::inverse(g.transform);
	g.invTranspose = glm::transpose(g.inverseTransform);
};


void RenderImGui()
{
	mouseOverImGuiWinow = io->WantCaptureMouse;

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	static float f = 0.0f;
	static int counter = 0;


	//////////////////////////////////////////////////////////////////////////
	ImGui::Begin("Path Tracer Analytics");
	ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

	ImGui::Checkbox("Enable Antialiasing", &imguiData->EnableAntialiasing);
	ImGui::Checkbox("Enable Denoise", &imguiData->EnableDenoise);
	ImGui::Checkbox("Enable Ray Compaction", &imguiData->EnableRayCompaction);
	ImGui::Checkbox("Enable Material Sort", &imguiData->EnableMaterialSort);

	if (ImGui::CollapsingHeader("Depth of Field")) {
		ImGui::Checkbox("Enable DoF", &imguiData->EnableDOF);

		float rangeFactor = 2.0f; // Allow 2 times above and 2 times below

		float minLens = std::max(0.0f, imguiData->InitialLensRadius / rangeFactor);
		float maxLens = imguiData->InitialLensRadius * rangeFactor;

		float minFocal = std::max(0.0f, imguiData->InitialFocalDist / rangeFactor);
		float maxFocal = imguiData->InitialFocalDist * rangeFactor;


		ImGui::SliderFloat("Lens Radius", &imguiData->LensRadius, minLens, maxLens, "%.3f");
		ImGui::SliderFloat("Focal Distance", &imguiData->FocalDist, minFocal, maxFocal, "%.2f");
	}

	if (ImGui::Button("Reset DOF")) {
		imguiData->LensRadius = imguiData->InitialLensRadius;
		imguiData->FocalDist = imguiData->InitialFocalDist;
	}

	// Selection / Highlight toggle
	if (ImGui::Checkbox("Selection Mode (Q)", &selection.enabled)) {
		if (!selection.enabled) {
			removeHighlightShell(scene);
			selection.pickedID = -1;
			selection.previousPickedID = -1;
		}
		selection.changed = true;
	}

	ImGui::End();

	//////////////////////////////////////////////////////////////////////////
	ImGui::Begin("Object Editor");

	if (selection.pickedID >= 0 && selection.pickedID < scene->geoms.size()) {
		ImGui::Text("Selected Object ID: %d", selection.pickedID);

		Geom& picked_obj = scene->geoms[selection.pickedID];
		int currentMatID = picked_obj.materialid;
		ImGui::Text("Current Material: %d", currentMatID);

		// Show current material type
		Material& curr = scene->materials[currentMatID];
		MaterialType mt = deduceType(curr);
		const char* typeName = "Unknown";
		switch (mt) {
		case MaterialType::Diffuse:    typeName = "Diffuse"; break;
		case MaterialType::Emissive:   typeName = "Emissive"; break;
		case MaterialType::Metallic:   typeName = "Metallic"; break;
		case MaterialType::Dielectric: typeName = "Dielectric"; break;
		}
		ImGui::Text("Current Type: %s", typeName);

		// Build dropdown entries, one per unique deduced type (first occurrence wins)
		struct Entry { std::string label; int matID; MaterialType type; };
		static int lastPickedID = -1;
		static int chosenEntry = -1;

		std::vector<Entry> entries;
		std::set<MaterialType> seenTypes;


		//skip highlight shell
		for (int i = 1; i < scene->materials.size(); ++i) {
			MaterialType t = deduceType(scene->materials[i]);

			if (t == MaterialType::Unknown) continue; // skip unknowns

			if (t == MaterialType::Environment) continue; 

			if (seenTypes.insert(t).second) { // first time seeing this type
				std::string label = " ";
				switch (t) {
				case MaterialType::Diffuse:    label += "Diffuse"; break;
				case MaterialType::Emissive:   label += "Emissive"; break;
				case MaterialType::Metallic:   label += "Metallic"; break;
				case MaterialType::Dielectric: label += "Dielectric"; break;
				default: label += "Unknown"; break;
				}
				entries.push_back({ label, i, t });
			}
		}

		// Convert labels for ImGui
		std::vector<const char*> items;
		for (auto& e : entries) items.push_back(e.label.c_str());

		// If selected object changed or entries list changed, resync chosenEntry
		if (selection.pickedID != lastPickedID || chosenEntry >= (int)entries.size()) {
			chosenEntry = -1; // force reinit
			lastPickedID = selection.pickedID;
		}

		// Initialize or resync chosenEntry to reflect current object's type
		if (chosenEntry < 0 && !entries.empty()) {
			MaterialType currentType = deduceType(scene->materials[currentMatID]);
			for (int ei = 0; ei < entries.size(); ++ei) {
				if (entries[ei].type == currentType) {
					chosenEntry = ei;
					break;
				}
			}
			if (chosenEntry < 0)
				chosenEntry = 0;
		}

		if (!entries.empty()) {
			if (ImGui::Combo("Swap Material", &chosenEntry, items.data(), (int)items.size())) {
				int selectedMatID = entries[chosenEntry].matID;
				if (selectedMatID >= 0 && selectedMatID < scene->materials.size()) {
					scene->geoms[selection.pickedID].materialid = selectedMatID;
					selection.changed = true;
				}
			}
		}
		else {
			ImGui::Text("No valid materials to choose from.");
		}


		//////// Color Editing ////////////////////////
		// count how many geoms share this material
		int sharedCount = 0;
		for (const auto& g : scene->geoms) {
			if (g.materialid == currentMatID) ++sharedCount;
		}

		// if more than one uses it, clone so we have a unique material for this object
		if (sharedCount > 1) {
			Material original = scene->materials[currentMatID];
			scene->materials.push_back(original);
			int newMatID = (int)scene->materials.size() - 1;
			scene->geoms[selection.pickedID].materialid = newMatID;
			currentMatID = newMatID; // switch to the new one
		}


		// Show whether it has a texture
		bool hasAlbedoTex = (curr.albedoMapTex.texObj != 0);
		ImGui::Separator();
		ImGui::Text("Albedo (base) color:");

		if (hasAlbedoTex) {
			ImGui::TextDisabled("(albedo texture present, color tints texture)");
		}

		// store current color in a temp array for imgui
		float color[3] = { curr.color.r, curr.color.g, curr.color.b };

		// let user edit; always enable so they can tint even with a texture
		if (ImGui::ColorEdit3("base color", color)) {
			curr.color = glm::vec3(color[0], color[1], color[2]);
			selection.changed = true;
		}

		if (mt == MaterialType::Metallic) {
			// Roughness slider
			if (ImGui::SliderFloat("Roughness", &curr.roughness, 0.0f, 1.0f)) {
				selection.changed = true;
			}
			
			// Specular color
			float specColor[3] = { curr.specular.color.r, curr.specular.color.g, curr.specular.color.b };
			if (ImGui::ColorEdit3("Specular Color", specColor)) {
				curr.specular.color = glm::vec3(specColor[0], specColor[1], specColor[2]);
				selection.changed = true;
			}

		}
		else if (mt == MaterialType::Dielectric) {

			// IOR
			if (ImGui::InputFloat("Index of Refraction", &curr.indexOfRefraction)) {
				selection.changed = true;
			}
		}
		else if (mt == MaterialType::Emissive) {
			if (ImGui::InputFloat("Emittance", &curr.emittance)) {
				selection.changed = true;
			}
		}

		ImGui::Separator();


		// Translation
		std::string transLabel = "Translation##obj" + std::to_string(selection.pickedID);
		if (ImGui::DragFloat3(transLabel.c_str(), &picked_obj.translation.x, 0.1f)) {
			updateGeomTransform(picked_obj);
			selection.changed = true;
		}

		// Rotation (display in degrees for usability; internal storage is radians)
		glm::vec3 rot_deg = glm::degrees(picked_obj.rotation); // convert stored radians to degrees
		if (ImGui::DragFloat3("Rotation (deg)", &rot_deg.x, 1.0f)) {
			picked_obj.rotation = glm::radians(rot_deg); // convert edited degrees back to radians
			updateGeomTransform(picked_obj);

			selection.changed = true;
		}

		// Scale
		std::string scaleLabel = "Scale##obj" + std::to_string(selection.pickedID);
		if (ImGui::DragFloat3(scaleLabel.c_str(), &picked_obj.scale.x, 0.01f)) {
			updateGeomTransform(picked_obj);
			selection.changed = true;
		}

	}
	else {
		ImGui::Text("No object selected.");
	}

	ImGui::End();
	//////////////////////////////////////////////////////////////////////////


	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
	return mouseOverImGuiWinow;
}

void mainLoop()
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		runCuda();

		string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
		glfwSetWindowTitle(window, title.c_str());
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);

		// Binding GL_PIXEL_UNPACK_BUFFER back to default
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

		// Render ImGui Stuff
		RenderImGui();

		glfwSwapBuffers(window);
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
}
