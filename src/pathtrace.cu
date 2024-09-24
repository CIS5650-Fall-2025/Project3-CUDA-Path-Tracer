#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

// include the device pointer header in the thrust library
#include <thrust/device_ptr.h>

// include the sort header in the thrust library
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

// define the BVH_ACCELERATION macro to accelerate ray-triangle intersection
#define BVH_ACCELERATION

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

// declare the condition buffer for compacting the path segments
static bool* condition_buffer;

// declare an additional buffer for storing the output path segments
static PathSegment* path_segment_buffer;

// declare the intersection key buffer that stores the material types
static int* intersection_key_buffer;

// declare the path segment key buffer that stores the material types
static int* path_segment_key_buffer;

// declare the bounding sphere buffer that stores all the bounding spheres
static bounding_sphere_data* bounding_sphere_buffer;

// declare the vertex buffer that stores all the vertices
static vertex_data* vertex_buffer;

// declare the pixel buffer that stores all the pixels
static glm::vec4* pixel_buffer;

// declare the texture buffer that stores all the textures
static texture_data* texture_buffer;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // allocate the condition buffer
    cudaMalloc(
        reinterpret_cast<void**>(&condition_buffer), 
        pixelcount * sizeof(bool)
    );

    // allocate the path segment buffer
    cudaMalloc(
        reinterpret_cast<void**>(&path_segment_buffer),
        pixelcount * sizeof(PathSegment)
    );

    // allocate the intersection key buffer
    cudaMalloc(
        reinterpret_cast<void**>(&intersection_key_buffer),
        pixelcount * sizeof(int)
    );

    // allocate the path segment key buffer
    cudaMalloc(
        reinterpret_cast<void**>(&path_segment_key_buffer),
        pixelcount * sizeof(int)
    );

    // allocate the bounding sphere buffer
    cudaMalloc(
        reinterpret_cast<void**>(&bounding_sphere_buffer),
        scene->bounding_spheres.size() * sizeof(bounding_sphere_data)
    );

    // copy all the bounding sphere data to the bounding sphere buffer
    cudaMemcpy(
        reinterpret_cast<void*>(bounding_sphere_buffer),
        reinterpret_cast<void*>(scene->bounding_spheres.data()),
        scene->bounding_spheres.size() * sizeof(bounding_sphere_data),
        cudaMemcpyHostToDevice
    );

    // allocate the vertex buffer
    cudaMalloc(
        reinterpret_cast<void**>(&vertex_buffer),
        scene->vertices.size() * sizeof(vertex_data)
    );

    // copy all the vertex data to the vertex buffer
    cudaMemcpy(
        reinterpret_cast<void*>(vertex_buffer),
        reinterpret_cast<void*>(scene->vertices.data()),
        scene->vertices.size() * sizeof(vertex_data),
        cudaMemcpyHostToDevice
    );

    // allocate the pixel buffer
    cudaMalloc(
        reinterpret_cast<void**>(&pixel_buffer),
        scene->pixels.size() * sizeof(glm::vec4)
    );

    // copy all the pixel data to the pixel buffer
    cudaMemcpy(
        reinterpret_cast<void*>(pixel_buffer),
        reinterpret_cast<void*>(scene->pixels.data()),
        scene->pixels.size() * sizeof(glm::vec4),
        cudaMemcpyHostToDevice
    );

    // allocate the texture buffer
    cudaMalloc(
        reinterpret_cast<void**>(&texture_buffer),
        scene->textures.size() * sizeof(texture_data)
    );

    // copy all the texture data to the texture buffer
    cudaMemcpy(
        reinterpret_cast<void*>(texture_buffer),
        reinterpret_cast<void*>(scene->textures.data()),
        scene->textures.size() * sizeof(texture_data),
        cudaMemcpyHostToDevice
    );

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    
    // free the condition buffer
    cudaFree(reinterpret_cast<void*>(condition_buffer));

    // free the path segment buffer
    cudaFree(reinterpret_cast<void*>(path_segment_buffer));

    // free the intersection key buffer
    cudaFree(reinterpret_cast<void*>(intersection_key_buffer));

    // free the path segment key buffer
    cudaFree(reinterpret_cast<void*>(path_segment_key_buffer));

    // free the bounding sphere buffer
    cudaFree(reinterpret_cast<void*>(bounding_sphere_buffer));

    // free the vertex buffer
    cudaFree(reinterpret_cast<void*>(vertex_buffer));

    // free the pixel buffer
    cudaFree(reinterpret_cast<void*>(pixel_buffer));

    // free the texture buffer
    cudaFree(reinterpret_cast<void*>(texture_buffer));

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // create a random number generator
        thrust::default_random_engine generator {
            makeSeededRandomEngine(x, y, 0)
        };

        // declare a new distribution
        thrust::uniform_real_distribution<float> distribution (0.0f, 1.0f);

        // generate two random numbers between -0.5f and 0.5f
        const float random_x {distribution(generator) * 0.5f + 0.5f};
        const float random_y {distribution(generator) * 0.5f + 0.5f};

        // compute the horizontal offset
        const float offset_x {
            static_cast<float>(x) - static_cast<float>(cam.resolution.x) * 0.5f + random_x
        };

        // compute the vertical offset
        const float offset_y {
            static_cast<float>(y) - static_cast<float>(cam.resolution.y) * 0.5f + random_y
        };

        // create the ray direction equal to the camera's view vector
        glm::vec3 direction {cam.view};

        // update the horizontal direction
        direction -= cam.right * cam.pixelLength.x * offset_x;

        // update the vertical direction
        direction -= cam.up * cam.pixelLength.y * offset_y;

        // store the output direction
        segment.ray.direction = glm::normalize(direction);

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// declare the kernel function that samples a texture
__host__ __device__ glm::vec4 sample(const glm::vec2 coordinate,
                                     const texture_data texture,
                                     const glm::vec4* pixels) {

    // compute the target position to sample
    const float x {coordinate.x * static_cast<int>(texture.width)};
    const float y {(1.0f - coordinate.y) * static_cast<int>(texture.height)};

    // compute the positions below and above the target position
    const float floor_x {glm::floor(x)};
    const float floor_y {glm::floor(y)};
    const float ceil_x {glm::ceil(x)};
    const float ceil_y {glm::ceil(y)};

    // compute the interpolation factors
    const float factor_x {x - floor_x};
    const float factor_y {y - floor_y};

    // sample the four pixels near the target position
    const glm::vec4 floor_x_floor_y_pixel {
        pixels[texture.index + static_cast<int>(floor_y) * texture.width + static_cast<int>(floor_x)]
    };
    const glm::vec4 floor_x_ceil_y_pixel {
        pixels[texture.index + static_cast<int>(ceil_y) * texture.width + static_cast<int>(floor_x)]
    };
    const glm::vec4 ceil_x_floor_y_pixel {
        pixels[texture.index + static_cast<int>(floor_y) * texture.width + static_cast<int>(ceil_x)]
    };
    const glm::vec4 ceil_x_ceil_y_pixel {
        pixels[texture.index + static_cast<int>(ceil_y) * texture.width + static_cast<int>(ceil_x)]
    };

    // interpolate between the sampled pixels
    return glm::mix(
        glm::mix(floor_x_floor_y_pixel, floor_x_ceil_y_pixel, factor_y),
        glm::mix(ceil_x_floor_y_pixel, ceil_x_ceil_y_pixel, factor_y), 
        factor_x
    );
}

// declare the kernal function that detects intersections
__global__ void detect(const int depth, const int workload,
                       const int vertex_count,
                       const int bounding_sphere_count,
                       const int geometry_count,
                       const PathSegment* path_segments,
                       const vertex_data* vertices,
                       const bounding_sphere_data* bounding_spheres,
                       const Geom* geometries,
                       ShadeableIntersection* intersections) {
    
    // compute the thread index
    const unsigned int index {blockIdx.x * blockDim.x + threadIdx.x};

    // avoid execution when the index is out of range
    if (index >= workload) {
        return;
    }

    // acquire the current path segment
    const PathSegment path_segment {path_segments[index]};

    // declare a variable for the current geometry
    Geom geometry;

    // declare a variable for the intersection distance
    float distance;

    // declare a variable for the temporary intersection point
    glm::vec3 temporary_intersection_point;

    // declare a variable for the temporary intersection normal
    glm::vec3 temporary_intersection_normal;

    // declare a variable necessary for function calls
    bool condition;

    // declare a variable for the minimal intersection distance
    float minimal_distance {FLT_MAX};

    // declare a variable for the material index at the intersection
    int material_index {-1};

    // declare a variable for the intersection normal
    glm::vec3 intersection_normal;

    // declare a variable for the intersection tangent
    glm::vec3 intersection_tangent;

    // declare a variable for the intersection texture coordinate
    glm::vec2 intersection_coordinate;

    // iterate through all the geometries
    for (int geometry_index {0}; geometry_index < geometry_count; geometry_index += 1) {

        // acquire the current geometry
        geometry = geometries[geometry_index];

        // perform intersection test for the cube geometry type
        if (geometry.type == CUBE) {
            distance = boxIntersectionTest(
                geometry, path_segment.ray,
                temporary_intersection_point,
                temporary_intersection_normal,
                condition
            );

            // perform intersection test for the sphere geometry type
        } else if (geometry.type == SPHERE) {
            distance = sphereIntersectionTest(
                geometry, path_segment.ray,
                temporary_intersection_point,
                temporary_intersection_normal,
                condition
            );
        }

        // store the result if the distance is valid and closer than the minimal distance
        if (0.0f < distance && distance < minimal_distance) {

            // store the material index
            material_index = geometry.materialid;

            // store the intersection normal
            intersection_normal = temporary_intersection_normal;

            // update the minimal distance
            minimal_distance = distance;
        }
    }

    // declare a variable for the index of the first vertex in the intersected triangle
    int triangle_vertex_index {-1};

    // perform the BVH-based intersection tests if the target macro is defined
#   if defined(BVH_ACCELERATION)

    // acquire the origin of the ray
    const glm::vec3 origin {path_segment.ray.origin};

    // acquire the direction of the ray
    const glm::vec3 direction {glm::normalize(path_segment.ray.direction)};

    // declare an array to store all the traversed indices
    int traversed_indices[32];

    // declare a variable to store the number of traversed indices in the array
    int traversed_index_count {0};

    // declare a variable for the index of the current bounding sphere
    int current_bounding_sphere_index {0};

    // declare a variable for the index of the previous bounding sphere
    int previous_bounding_sphere_index {-1};

    // declare a variable for the current bounding sphere
    bounding_sphere_data bounding_sphere;

    // invalidate the current bounding sphere index when there is no bounding spheres
    if (bounding_sphere_count == 0) {
        current_bounding_sphere_index = -1;
    }

    // perform the BVH-based intersection tests
    while (current_bounding_sphere_index != -1) {

        // acquire the current bounding sphere
        bounding_sphere = bounding_spheres[current_bounding_sphere_index];

        // return to the parent bounding sphere if the second child has been processed
        if (previous_bounding_sphere_index == bounding_sphere.child_indices[1]) {

            // exit the loop when there is no return
            if (traversed_index_count == 0) {
                break;

                // delete the last element in the stack
            } else {
                traversed_index_count -= 1;

                // update the previous bounding sphere index
                previous_bounding_sphere_index = current_bounding_sphere_index;

                // update the current bounding sphere index
                current_bounding_sphere_index = traversed_indices[traversed_index_count];
            }

            // process the second child if the first child has been processed
        } else if (previous_bounding_sphere_index == bounding_sphere.child_indices[0]) {

            // push the index of the current bounding sphere to the stack
            traversed_indices[traversed_index_count] = current_bounding_sphere_index;
            traversed_index_count += 1;

            // update the previous bounding sphere index
            previous_bounding_sphere_index = current_bounding_sphere_index;

            // update the current bounding sphere index
            current_bounding_sphere_index = bounding_sphere.child_indices[1];

            // process the current bounding sphere
        } else {

            // compute the vector from the ray's origin to the center of the bounding sphere
            const glm::vec3 vector {
                bounding_sphere.center - origin
            };

            // compute the projection factor
            const float factor {
                glm::dot(vector, direction)
            };

            // compute the projected point
            const glm::vec3 point {
                origin + direction * factor
            };

            // compute the distance between the projected point to the center of the bounding sphere
            const float bounding_sphere_distance {
                glm::distance(point, bounding_sphere.center) - bounding_sphere.radius
            };

            // perform the naive intersection tests when the bounding sphere contains triangles
            if (bounding_sphere_distance <= 0.0f && bounding_sphere.count > 0) {

                // compute the start index of the vertex
                const int start_index {bounding_sphere.index};

                // compute the end index of the vertex
                const int end_index {bounding_sphere.index + bounding_sphere.count * 3};

                // iterate through all the triangles contained inside the current bounding sphere
                for (int vertex_index {start_index}; vertex_index < end_index; vertex_index += 3) {

                    // perform the ray-triangle intersection test
                    distance = intersect(
                        path_segment.ray,
                        vertices[vertex_index + 0].point,
                        vertices[vertex_index + 1].point,
                        vertices[vertex_index + 2].point
                    );

                    // store the result if the distance is valid and closer than the minimal distance
                    if (0.0f < distance && distance < minimal_distance) {

                        // store the index of the first vertex in the intersected triangle
                        triangle_vertex_index = vertex_index;

                        // update the minimal distance
                        minimal_distance = distance;
                    }
                }
            }

            // return to the previous bounding sphere
            if (bounding_sphere_distance > 0.0f || bounding_sphere.count > 0) {

                // exit the loop when there is no return
                if (traversed_index_count == 0) {
                    break;

                    // delete the last element in the stack
                } else {
                    traversed_index_count -= 1;

                    // update the previous bounding sphere index
                    previous_bounding_sphere_index = current_bounding_sphere_index;

                    // update the current bounding sphere index
                    current_bounding_sphere_index = traversed_indices[traversed_index_count];
                }

                // process the first child
            } else {

                // push the index of the current bounding sphere to the stack
                traversed_indices[traversed_index_count] = current_bounding_sphere_index;
                traversed_index_count += 1;

                // update the previous bounding sphere index
                previous_bounding_sphere_index = current_bounding_sphere_index;

                // update the current bounding sphere index
                current_bounding_sphere_index = bounding_sphere.child_indices[0];
            }
        }
    }
#   else

    // perform the naive intersection tests when the target macro is undefined
    for (int vertex_index {0}; vertex_index < vertex_count; vertex_index += 3) {

        // perform the ray-triangle intersection test
        distance = intersect(
            path_segment.ray,
            vertices[vertex_index + 0].point,
            vertices[vertex_index + 1].point,
            vertices[vertex_index + 2].point
        );

        // store the result if the distance is valid and closer than the minimal distance
        if (0.0f < distance && distance < minimal_distance) {

            // store the index of the first vertex in the intersected triangle
            triangle_vertex_index = vertex_index;

            // update the minimal distance
            minimal_distance = distance;
        }
    }
#   endif

    // process the intersected triangle if it exists
    if (triangle_vertex_index != -1) {

        // store the material index
        material_index = vertices[triangle_vertex_index].material_index;

        // compute the intersection point
        const glm::vec3 intersection_point {
            path_segment.ray.origin + path_segment.ray.direction * minimal_distance
        };

        // compute the barycentric weights
        const glm::vec3 weights {compute(
            intersection_point,
            vertices[triangle_vertex_index + 0].point,
            vertices[triangle_vertex_index + 1].point,
            vertices[triangle_vertex_index + 2].point
        )};

        // compute and store the intersection normal
        intersection_normal = vertices[triangle_vertex_index + 0].normal * weights.x;
        intersection_normal += vertices[triangle_vertex_index + 1].normal * weights.y;
        intersection_normal += vertices[triangle_vertex_index + 2].normal * weights.z;

        // compute and store the intersection tangent
        intersection_tangent = vertices[triangle_vertex_index + 0].tangent * weights.x;
        intersection_tangent += vertices[triangle_vertex_index + 1].tangent * weights.y;
        intersection_tangent += vertices[triangle_vertex_index + 2].tangent * weights.z;

        // compute and store the intersection texture coordinate
        intersection_coordinate = vertices[triangle_vertex_index + 0].coordinate * weights.x;
        intersection_coordinate += vertices[triangle_vertex_index + 1].coordinate * weights.y;
        intersection_coordinate += vertices[triangle_vertex_index + 2].coordinate * weights.z;
    }

    // invalidate the intersection if no material was hit by the ray
    if (material_index == -1) {
        intersections[index].t = -1.0f;

        // store the intersection data otherwise
    } else {

        // store the minimal distance
        intersections[index].t = minimal_distance;

        // store the index of the material
        intersections[index].materialId = material_index;

        // store the intersection normal
        intersections[index].surfaceNormal = intersection_normal;

        // store the intersection tangent
        intersections[index].tangent = intersection_tangent;

        // store the intersection texture coordinate
        intersections[index].coordiante = intersection_coordinate;
    }
}

// declare the kernal function that classifies the materials before sorting
__global__ void classify(const int workload,
                         const ShadeableIntersection* intersections,
                         const Material* materials,
                         int* intersection_keys,
                         int* path_segment_keys) {

    // compute the thread index
    const unsigned int index {blockIdx.x * blockDim.x + threadIdx.x};

    // avoid execution when the index is out of range
    if (index >= workload) {
        return;
    }

    // acquire the current material
    const Material material {materials[intersections[index].materialId]};

    // declare the material type
    int type;

    // compute the type of the material
    if (material.hasReflective == 1.0f) {

        // specify the type for the mirror material
        type = 1;

        // specify the type for the reflective material
    } else if (material.hasReflective > 0.0f) {
        type = 2;

        // specify the type for the purely refractive material
    } else if (material.hasRefractive == 1.0f) {
        type = 3;

        // specify the type for the emissive material
    } else if (material.emittance > 0.0f) {
        type = -1;

        // specify the type for the diffuse material
    } else {
        type = 0;
    }

    // store the type to the key buffers
    intersection_keys[index] = type;
    path_segment_keys[index] = type;
}

// declare the kernel function that shares the rays
__global__ void shade(const int iteration, const int workload,
                      const ShadeableIntersection* intersections,
                      const texture_data* textures,
                      const glm::vec4* pixels,
                      const Material* materials,
                      PathSegment* path_segments) {

    // compute the thread index
    const unsigned int index {blockIdx.x * blockDim.x + threadIdx.x};

    // avoid execution when the index is out of range
    if (index >= workload) {
        return;
    }

    // acquire the current intersection
    const ShadeableIntersection intersection {intersections[index]};

    // invalidate the path segment when the intersection is invalid
    if (intersection.t <= 0) {

        // update the color of the path segment to black
        path_segments[index].color = glm::vec3(0.0f);

        // invalidate the current path segment
        path_segments[index].remainingBounces = 0;

        // exit the kernel function
        return;
    }

    // acquire the current material
    const Material material {materials[intersection.materialId]};

    // update the path segment when the material is a light source
    if (material.emittance > 0.0f) {

        // update the color of the path segment
        path_segments[index].color *= material.color * material.emittance;

        // invalidate the current path segment after it reaches a light
        path_segments[index].remainingBounces = 0;

        // handle other material types when the remaining bounces is greater than one
    } else if (path_segments[index].remainingBounces > 1) {
        
        // create a new random number generator
        thrust::default_random_engine generator {
            makeSeededRandomEngine(iteration, index, path_segments[index].remainingBounces)
        };

        // compute the point of intersection
        const glm::vec3 point {
            path_segments[index].ray.origin + path_segments[index].ray.direction * intersection.t
        };

        // acquire the color of the material
        glm::vec3 color {material.color};

        // overwrite the color if the material is using a diffuse texture
        if (material.diffuse_texture_index > -1) {

            // sample the texture
            const glm::vec4 pixel {sample(
                intersection.coordiante,
                textures[material.diffuse_texture_index], 
                pixels
            )};

            // pass through the surface when the texture has transparency
            if (pixel.w < 1.0f) {

                // generate a random decimal
                thrust::uniform_real_distribution<float> distribution (0.0f, 1.0f);
                const float random_decimal {distribution(generator)};

                // pass through the surface when the transparency is low
                if (pixel.w < random_decimal) {

                    // update the ray's origin
                    path_segments[index].ray.origin = point + path_segments[index].ray.direction * 0.01f;

                    // exit the function
                    return;
                }
            }

            // overwrite the color
            color = glm::vec3(pixel.x, pixel.y, pixel.z);

            // perform gamma correction
            color = glm::pow(color, glm::vec3(1.0f / 2.2f));
        }

        // acquire the normal vector at the intersection point
        glm::vec3 normal {intersection.surfaceNormal};

        // overwrite the normal if the material is using a normal texture
        if (material.normal_texture_index > -1) {

            // sample the texture
            const glm::vec4 pixel {sample(
                intersection.coordiante,
                textures[material.normal_texture_index], 
                pixels
            )};

            // acquire the tangent vector at the intersection point
            const glm::vec3 tangent {intersection.tangent};

            // compute the bitangent vector
            const glm::vec3 bitangent {glm::cross(normal, tangent)};

            // construct the tangent-bitangent-normal matrix
            const glm::mat3 matrix {
                tangent, bitangent, normal
            };

            // overwrite the normal
            normal = matrix * glm::vec3(
                pixel.x,
                pixel.y,
                pixel.z
            );
        }

        // perform ray scattering
        scatter(
            color, point, normal, material, generator, 
            path_segments[index]
        );

        // invalidate the path segment otherwise
    } else {
        path_segments[index].color = glm::vec3(0.0f);
    }
}

// declare the kernal function that determines the conditions and also transfers inputs to outputs
__global__ void determine_and_transfer(const int workload,
                                       const PathSegment* input_path_segments,
                                       PathSegment* output_path_segments,
                                       bool* conditions) {

    // compute the thread index
    const unsigned int index {blockIdx.x * blockDim.x + threadIdx.x};

    // avoid execution when the index is out of range
    if (index >= workload) {
        return;
    }

    // acquire the input path segment
    const PathSegment input_path_segment {input_path_segments[index]};

    // mark the condition as true when the path segment is valid
    if (input_path_segment.remainingBounces > 0) {
        conditions[index] = true;
        return;
    }

    // transfer the input to the output when the path segment is invalid
    output_path_segments[input_path_segment.pixelIndex] = input_path_segment;

    // mark the condition as false when the path segment is invalid
    conditions[index] = false;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        // detect intersections
        detect<<<numblocksPathSegmentTracing, blockSize1d>>>(
            depth, num_paths,
            static_cast<int>(hst_scene->vertices.size()),
            static_cast<int>(hst_scene->bounding_spheres.size()),
            static_cast<int>(hst_scene->geoms.size()),
            dev_paths, vertex_buffer, bounding_sphere_buffer,
            dev_geoms, dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;
        
        // classify the intersections based on the material types
        classify<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths, dev_intersections, dev_materials,
            intersection_key_buffer, path_segment_key_buffer
        );

        // wait until completion
        cudaDeviceSynchronize();

        // declare the thrust vectors for sorting
        thrust::device_ptr<int> intersection_keys (intersection_key_buffer);
        thrust::device_ptr<int> path_segment_keys (path_segment_key_buffer);
        thrust::device_ptr<ShadeableIntersection> intersections (dev_intersections);
        thrust::device_ptr<PathSegment> path_segments (dev_paths);

        // perform sorting in parallel
        thrust::sort_by_key(
            intersection_keys, 
            intersection_keys + num_paths, 
            intersections
        );
        thrust::sort_by_key(
            path_segment_keys, 
            path_segment_keys + num_paths, 
            path_segments
        );

        // wait until completion
        cudaDeviceSynchronize();
        
        // perform shading with textures
        shade<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter, num_paths, dev_intersections,
            texture_buffer, pixel_buffer, dev_materials,
            dev_paths
        );
        
        // wait until completion
        cudaDeviceSynchronize();

        // determine the conditions and transfer the terminated path segments
        determine_and_transfer<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths, dev_paths, path_segment_buffer, condition_buffer
        );

        // wait until completion
        cudaDeviceSynchronize();

        // declare the thrust vectors for compaction
        thrust::device_ptr<PathSegment> input_pointer (dev_paths);
        thrust::device_ptr<bool> condition_pointer (condition_buffer);
        thrust::device_ptr<PathSegment> output_pointer;

        // perform a compaction on the input path segments
        output_pointer = thrust::remove_if(
            input_pointer, input_pointer + num_paths, condition_pointer,
            thrust::logical_not<bool>()
        );

        // wait until completion
        cudaDeviceSynchronize();

        // compute the new number of paths
        num_paths = output_pointer - input_pointer;

        // exit the loop when the number of paths is zero
        if (num_paths == 0) {
            iterationComplete = true;
        }

        // exit the loop when the maximum depth is reached
        if (depth == traceDepth) {
            iterationComplete = true;
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // gather the data in the path segment buffer for the output image
    finalGather<<<(pixelcount + blockSize1d - 1) / blockSize1d, blockSize1d>>>(
        pixelcount, dev_image, path_segment_buffer
    );

    // wait until completion
    cudaDeviceSynchronize();

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
