#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float meshIntersectionTest(
    Geom geom,
    Ray ray,
    glm::vec3* dev_meshes_positions,
    uint16_t* dev_meshes_indices,
    glm::vec3* dev_meshes_normals,
    glm::vec2* dev_meshes_uvs,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside
) {
    float t_min = FLT_MAX;
    bool hit = false;

    for (int i = geom.indexOffset; i < geom.indexOffset + geom.indexCount; i += 3) {
        glm::vec3 v0 = glm::vec3(geom.transform * glm::vec4(dev_meshes_positions[geom.offset+dev_meshes_indices[i]], 1.0f));
        glm::vec3 v1 = glm::vec3(geom.transform * glm::vec4(dev_meshes_positions[geom.offset+dev_meshes_indices[i+1]], 1.0f));
        glm::vec3 v2 = glm::vec3(geom.transform * glm::vec4(dev_meshes_positions[geom.offset+dev_meshes_indices[i+2]], 1.0f));

        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;

        glm::vec3 h = glm::cross(ray.direction, edge2);
        float a = glm::dot(edge1, h);

        if (fabs(a) < 1e-8f) {
            continue;
        }

        float f = 1.0f / a;
        glm::vec3 s = ray.origin - v0;
        float u = f * glm::dot(s, h);

        if (u < 0.0f || u > 1.0f) {
            continue;
        }

        glm::vec3 q = glm::cross(s, edge1);
        float v = f * glm::dot(ray.direction, q);

        if (v < 0.0f || u + v > 1.0f) {
            continue;
        }

        float t = f * glm::dot(edge2, q);

        if (t > 0.0001f && t < t_min) {
            t_min = t;
            intersectionPoint = ray.origin + t * ray.direction;

            glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(geom.transform)));
            normal = glm::normalize(normalMatrix * glm::cross(edge1, edge2));

            outside = glm::dot(ray.direction, normal) < 0.0f;

            if (!outside) {
                normal = -normal;
            }

            hit = true;
        }
    }

    if (!hit) {
        return -1.0f;
    }

    return t_min;
}

__host__ __device__ bool intersectMeshBVH(
    const Geom& mesh,              // The mesh geometry
    const BVHNode* meshBVH,        // The mesh's BVH nodes
    int nodeIdx,                   // The root node index of the mesh BVH
    const Ray& ray,                // The ray to intersect
    glm::vec3* dev_meshes_positions, // Mesh positions
    uint16_t* dev_meshes_indices,  // Mesh triangle indices
    glm::vec3* dev_meshes_normals, // Mesh vertex normals
    glm::vec3& intersectionPoint,  // Output intersection point
    glm::vec3& normal,             // Output normal at intersection
    bool& outside,                 // Whether the intersection is on the outside
    float& t_min                   // Minimum intersection distance
) {
    const BVHNode& node = meshBVH[nodeIdx];
    float tMin = 0.0f, tMax = FLT_MAX;

    // Test if the ray intersects the node's bounding box
    if (!node.bounds.intersect(ray, tMin, tMax)) {
        return false;  // No intersection with this node
    }

    // If this is a leaf node, test the triangles within this node
    if (node.isLeaf) {
        bool hit = false;

        // Iterate over all triangles in this leaf node
        for (int i = node.start; i < node.end; ++i) {
            int triIdx = mesh.triangleIndices[i];  // Get the index of the triangle

            // Retrieve the triangle's vertices
            glm::vec3 v0 = dev_meshes_positions[mesh.offset + dev_meshes_indices[triIdx * 3]];
            glm::vec3 v1 = dev_meshes_positions[mesh.offset + dev_meshes_indices[triIdx * 3 + 1]];
            glm::vec3 v2 = dev_meshes_positions[mesh.offset + dev_meshes_indices[triIdx * 3 + 2]];

            // Apply the mesh's transformation to the vertices
            v0 = glm::vec3(mesh.transform * glm::vec4(v0, 1.0f));
            v1 = glm::vec3(mesh.transform * glm::vec4(v1, 1.0f));
            v2 = glm::vec3(mesh.transform * glm::vec4(v2, 1.0f));

            // Perform ray-triangle intersection using Möller-Trumbore algorithm
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 h = glm::cross(ray.direction, edge2);
            float a = glm::dot(edge1, h);

            // If the determinant is near zero, the ray is parallel to the triangle
            if (fabs(a) < 1e-8f) continue;

            float f = 1.0f / a;
            glm::vec3 s = ray.origin - v0;
            float u = f * glm::dot(s, h);

            // The intersection lies outside the triangle
            if (u < 0.0f || u > 1.0f) continue;

            glm::vec3 q = glm::cross(s, edge1);
            float v = f * glm::dot(ray.direction, q);

            // The intersection lies outside the triangle
            if (v < 0.0f || u + v > 1.0f) continue;

            // Compute the distance along the ray to the intersection point
            float t = f * glm::dot(edge2, q);

            // Check if this intersection is closer and positive
            if (t > 0.0001f && t < t_min) {
                t_min = t;
                intersectionPoint = ray.origin + t * ray.direction;
                normal = glm::normalize(glm::cross(edge1, edge2));

                // Determine if the ray is hitting the front or back of the triangle
                outside = glm::dot(ray.direction, normal) < 0.0f;
                if (!outside) {
                    normal = -normal;  // Flip the normal if inside the object
                }

                hit = true;
            }
        }

        return hit;
    }

    // If this is an internal node, recursively check the children
    bool hitLeft = intersectMeshBVH(mesh, meshBVH, node.left, ray, dev_meshes_positions, dev_meshes_indices, dev_meshes_normals, intersectionPoint, normal, outside, t_min);
    bool hitRight = intersectMeshBVH(mesh, meshBVH, node.right, ray, dev_meshes_positions, dev_meshes_indices, dev_meshes_normals, intersectionPoint, normal, outside, t_min);

    return hitLeft || hitRight;
}


__host__ __device__ bool intersectBVH(
    const BVHNode* bvhNodes,       // The BVH nodes (top-level BVH for meshes)
    const Ray& ray,                // The ray to intersect
    int nodeIdx,                   // Index of the current BVH node
    const Geom* geoms,             // Array of all meshes in the scene
    glm::vec3* dev_meshes_positions, // Mesh positions
    uint16_t* dev_meshes_indices,  // Mesh indices
    glm::vec3* dev_meshes_normals, // Mesh normals
    int& hit_geom_index,           // Output: index of the hit geometry
    glm::vec3& intersectionPoint,  // Output intersection point
    glm::vec3& normal,             // Output normal at intersection
    bool& outside,                 // Whether the intersection is on the outside of the mesh
    float& t_min                   // Minimum intersection distance (output)
) {
    const BVHNode& node = bvhNodes[nodeIdx];
    float tMin = 0.0f, tMax = FLT_MAX;

    // Test if the ray intersects the current node's bounding box
    if (!node.bounds.intersect(ray, tMin, tMax)) {
        return false;  // No intersection with this node
    }

    bool hit = false;

    // If this node is a leaf, we need to check the meshes contained within it
    if (node.isLeaf) {
        // Loop over all meshes in the leaf node
        for (int i = node.start; i < node.end; ++i) {
            const Geom& mesh = geoms[i];  // Get the mesh

            // Perform ray-mesh BVH intersection for the current mesh
            glm::vec3 tmp_intersect_point;
            glm::vec3 tmp_normal;
            bool tmp_outside;
            float tmp_t_min = t_min;  // Local t_min for this mesh

            bool hitMesh = intersectMeshBVH(
                mesh,                         // The current mesh
                mesh.meshBVH,                 // The mesh's BVH on the GPU
                mesh.bvhRoot,                 // The root of the mesh's BVH
                ray,                          // The ray to intersect
                dev_meshes_positions,         // Mesh positions
                dev_meshes_indices,           // Mesh indices
                dev_meshes_normals,           // Mesh normals
                tmp_intersect_point,          // Output intersection point
                tmp_normal,                   // Output normal at intersection
                tmp_outside,                  // Whether the intersection is outside
                tmp_t_min                     // Minimum intersection distance
            );

            // Check if this is the closest hit so far
            if (hitMesh && tmp_t_min < t_min) {
                hit = true;
                t_min = tmp_t_min;                // Update the minimum distance
                hit_geom_index = i;               // Update the hit geometry index
                intersectionPoint = tmp_intersect_point;  // Update intersection point
                normal = tmp_normal;              // Update normal at intersection
                outside = tmp_outside;            // Update outside flag
            }
        }

        return hit;
    }

    // If this is an internal node, traverse its children
    bool hitLeft = intersectBVH(bvhNodes, ray, node.left, geoms, dev_meshes_positions, dev_meshes_indices, dev_meshes_normals, hit_geom_index, intersectionPoint, normal, outside, t_min);
    bool hitRight = intersectBVH(bvhNodes, ray, node.right, geoms, dev_meshes_positions, dev_meshes_indices, dev_meshes_normals, hit_geom_index, intersectionPoint, normal, outside, t_min);

    return hitLeft || hitRight;
}
