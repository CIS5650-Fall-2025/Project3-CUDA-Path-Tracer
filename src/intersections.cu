#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray q;

    //Instead of transforming the box to world space →
    // Transform the ray into the box’s local space.
    //box.inverseTransform is a matrix that brings world coordinates into the box’s local object space.
    //objectSpacePosition = inverseTransform × worldSpacePosition
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n; //when the ray finally fully entered the box.
    glm::vec3 tmax_n; //when the ray will exit the box.

    //Loop over 3 axes: xyz=0 (x-axis), xyz=1 (y-axis), xyz=2 (z-axis)
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2); //entry time for this axis
            float tb = glm::max(t1, t2); //exit time for this axis
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1; //normal of the first hit surface
            if (ta > 0 && ta > tmin)
            {
                //For the ray to still be inside the box, it must still be inside all slabs.
                tmin = ta; // take the largest entry time so far
                tmin_n = n;
            }
            if (tb < tmax)
            {
                //The ray exits the box as soon as you exit any axis slab → so we take the minimum of all tb across axes.
                tmax = tb;  // take the smallest exit time so far
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true; //ray hit the box from outside
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false; //ray hit the box from side
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
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
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

    return glm::length(r.origin - intersectionPoint);
}



__host__ __device__ bool aabbIntersectionTest(
    const Ray& ray,
    const AABB& box,
    float& tMin,
    float& tMax)
{
    // Use a large initial interval
    tMin = -FLT_MAX;
    tMax = FLT_MAX;

    // Unrolled version for x-axis
    {
        float dir = ray.direction.x;
        float invD = fabsf(dir) > 1e-8f ? 1.0f / dir : 1e8f;  // avoid division by zero

        float t0 = (box.min.x - ray.origin.x) * invD;
        float t1 = (box.max.x - ray.origin.x) * invD;

        if (dir < 0.0f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        tMin = fmaxf(tMin, t0);
        tMax = fminf(tMax, t1);

        if (tMax < tMin) return false;
    }

    // Unrolled version for y-axis
    {
        float dir = ray.direction.y;
        float invD = fabsf(dir) > 1e-8f ? 1.0f / dir : 1e8f;

        float t0 = (box.min.y - ray.origin.y) * invD;
        float t1 = (box.max.y - ray.origin.y) * invD;

        if (dir < 0.0f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        tMin = fmaxf(tMin, t0);
        tMax = fminf(tMax, t1);

        if (tMax < tMin) return false;
    }

    // Unrolled version for z-axis
    {
        float dir = ray.direction.z;
        float invD = fabsf(dir) > 1e-8f ? 1.0f / dir : 1e8f;

        float t0 = (box.min.z - ray.origin.z) * invD;
        float t1 = (box.max.z - ray.origin.z) * invD;

        if (dir < 0.0f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        tMin = fmaxf(tMin, t0);
        tMax = fminf(tMax, t1);

        if (tMax < tMin) return false;
    }

    return true;
}




__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside) {

    // Transform ray into object local space
    Ray q;
    q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t_hit = 1e38f;  // initialize with large value

    Triangle closestTri;
    glm::vec2 closestUV; // to store the final interpolated UV
    bool hit = false;

    for (int i = 0; i < mesh.num_triangles; i++) {
        Triangle tri = mesh.triangles[i];

        // tuv = (u, v, t), where:
        // t — intersection distance
        // u and v — barycentric coordinates
        // Any point inside the triangle can be represented as:
        //    P = (1−u−v) * v0 + u * v1 + v * v2
        glm::vec3 tuv;

        if (glm::intersectRayTriangle(
            q.origin, q.direction,
            tri.v0, tri.v1, tri.v2,
            tuv))
        {
            // store t, u, v if needed
            if (tuv[2] > 0 && tuv[2] < t_hit) {
                t_hit = tuv[2]; //local space
                closestTri = tri;
                hit = true;

                // Barycentric to UV interpolation
                float u = tuv.x;
                float v = tuv.y;
                closestUV =
                    (1.0f - u - v) * tri.uv0 +
                    u * tri.uv1 +
                    v * tri.uv2;
            }
        }
    }

    if (hit) {
        // calculate local intersection point
        glm::vec3 local_intersection = q.origin + t_hit * q.direction;

        // compute geometric normal from triangle vertices:
        glm::vec3 e1 = closestTri.v1 - closestTri.v0;
        glm::vec3 e2 = closestTri.v2 - closestTri.v0;
        glm::vec3 n = glm::normalize(glm::cross(e1, e2));

        // transform intersection point and normal back to world space
        intersectionPoint = multiplyMV(mesh.transform, glm::vec4(local_intersection, 1.0f));

        // The correct way to transform normals is by applying the inverse transpose of
        // the linear part of the transformation matrix.
        normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(n, 0.0f)));

        // assign interpolated UV
        uv = closestUV;

        // calculate outside flag
        outside = glm::dot(r.direction, normal) < 0.0f ? true : false;

        return glm::length(r.origin - intersectionPoint);
    }
    else {
        return -1.0f; // no hit
    }
}





__host__ __device__ float meshIntersectionTest_WithMeshBVH(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t_hit = 1e38f;
    Triangle closestTri;
    glm::vec2 closestUV;
    bool hit = false;

    // Stack-based BVH traversal
    int stack[64];
    int stackIdx = 0;

    // FIXED: the last node is the root (post-order build)
    stack[stackIdx++] = mesh.num_BVHNodes - 1;

    while (stackIdx > 0) {

        //pop a node off the stack and check if the ray intersects its AABB
        int nodeIdx = stack[--stackIdx];
        const BVHNode& node = mesh.bvhNodes[nodeIdx];

        float tMin, tMax;

        //If it doesn’t intersect, skip the node
        if (!aabbIntersectionTest(q, node.bound, tMin, tMax)) continue;

        //If it does intersect: Push children (if it’s an internal node); Test all triangles (if it’s a leaf node)
        if (node.isLeaf) {
            for (int i = node.start; i < node.end; ++i) {
                const Triangle& tri = mesh.triangles[i];
                glm::vec3 tuv;

                if (glm::intersectRayTriangle(q.origin, q.direction, tri.v0, tri.v1, tri.v2, tuv)) {
                    if (tuv[2] > 0 && tuv[2] < t_hit) {
                        t_hit = tuv[2];

                        float u = tuv[0]; //barycentric coordinates 
                        float v = tuv[1];

                        // calculating the exact UV coordinate at the point where the ray hits
                        closestUV =
                            (1.f - u - v) * tri.uv0 +
                            u * tri.uv1 +
                            v * tri.uv2;

                        //uv = glm::vec2(u, v);  // incorrect, giving the shader barycentric values,
                        closestTri = tri;
                        hit = true;
                    }
                }
            }
        }
        else {
            stack[stackIdx++] = node.left;
            stack[stackIdx++] = node.right;
        }
    }

    if (hit) {
        glm::vec3 local_intersection = q.origin + t_hit * q.direction;

        glm::vec3 e1 = closestTri.v1 - closestTri.v0;
        glm::vec3 e2 = closestTri.v2 - closestTri.v0;
        glm::vec3 n = glm::normalize(glm::cross(e1, e2));

        intersectionPoint = multiplyMV(mesh.transform, glm::vec4(local_intersection, 1.0f));
        normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(n, 0.0f)));

        uv = closestUV;
        outside = glm::dot(r.direction, normal) < 0.0f;

        return glm::length(r.origin - intersectionPoint);
    }

    return -1.0f;
}


