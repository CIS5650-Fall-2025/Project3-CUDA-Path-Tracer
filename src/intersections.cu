#include "intersections.h"
#include "scene.h"

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

__host__ __device__ float triangleIntersectionTest(
    Geom geom_triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    Triangle* dev_tris,
    int tri_index) 
{

    glm::vec3 edge1, edge2, h, s, q;
    float a, f, u, v;

    auto triangle = dev_tris[tri_index];

    //multiply ray by inv transform to make it an easy intersection test?
    glm::vec3 ro = multiplyMV(geom_triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(geom_triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 p0 = triangle.v0.pos;
    glm::vec3 p1 = triangle.v1.pos;
    glm::vec3 p2 = triangle.v2.pos;

    edge1 = p1 - p0;
    edge2 = p2 - p0;
    h = glm::cross(rd, edge2);
    a = glm::dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
        outside = true;
        return INFINITY;    // This ray is parallel to this triangle.
    }
    f = 1.0f / a;
    s = ro - p0;
    u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        outside = true;
        return INFINITY;
    }
    q = glm::cross(s, edge1);
    v = f * glm::dot(rd, q);
    if (v < 0.0 || u + v > 1.0) {
        outside = true;
        return INFINITY;
    }
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * dot(edge2, q);
    if (t > EPSILON) {
        outside = false;

        glm::vec3 objspaceIntersection = getPointOnRay(r, t);

        intersectionPoint = multiplyMV(geom_triangle.transform, glm::vec4(objspaceIntersection, 1.f));
        normal = glm::normalize(multiplyMV(geom_triangle.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
        return t;
    }
    else {
        // This means that there is a line intersection but not a ray intersection.
        outside = true;
        return INFINITY;
    }
    return INFINITY;
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    Triangle* dev_tris,
    int num_tris)
{
    float min_t = INFINITY;

    //multiply ray by inv transform to make it an easy intersection test
    glm::vec3 ro = r.origin;
    glm::vec3 rd = r.direction;

    for (int i = 0; i < num_tris; i++) {
        Triangle& triangle = dev_tris[i];
        glm::vec3 p0 = triangle.v0.pos;
        glm::vec3 p1 = triangle.v1.pos;
        glm::vec3 p2 = triangle.v2.pos;
        glm::vec3 out_pos;

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        glm::vec3 h = glm::cross(rd, edge2);
        float a = glm::dot(edge1, h);
        if (a > -EPSILON && a < EPSILON) {
            continue;    // This ray is parallel to this triangle.
        }
        float f = 1.0f / a;
        glm::vec3 s = ro - p0;
        float u = f * glm::dot(s, h);
        if (u < 0.0 || u > 1.0) {
            continue;
        }
        glm::vec3 q = glm::cross(s, edge1);
        float v = f * glm::dot(rd, q);
        if (v < 0.0 || u + v > 1.0) {
            continue;
        }
        // At this stage we can compute t to find out where the intersection point is on the line.
        float t = f * dot(edge2, q);
        if (t > EPSILON) {
            outside = false;

            glm::vec3 objspaceIntersection = getPointOnRay(r, t);

            if (t < min_t) {
                //intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
                //normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
                intersectionPoint = objspaceIntersection;;
                normal = triangle.v0.nor;
                min_t = t;
            }
        }
        else {
            // This means that there is a line intersection but not a ray intersection.
            continue;
        }
    }
    return min_t;
}

__host__ __device__ bool intersectAABB(const Ray& ray, const glm::vec3 bmin, const glm::vec3 bmax, float& t_out) {
    glm::vec3 ro = ray.origin, rd = ray.direction;
    float tx1 = (bmin.x - ro.x) / rd.x, tx2 = (bmax.x - ro.x) / rd.x;
    float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
    float ty1 = (bmin.y - ro.y) / rd.y, ty2 = (bmax.y - ro.y) / rd.y;
    tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
    float tz1 = (bmin.z - ro.z) / rd.z, tz2 = (bmax.z - ro.z) / rd.z;
    tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
    t_out = tmin;
    return tmax >= tmin && tmax > 0;
}

__host__ __device__ bool intersectTri(Ray& ray, const Triangle& tri, float& out_float)
{
    const glm::vec3 edge1 = tri.v1.pos - tri.v0.pos;
    const glm::vec3 edge2 = tri.v2.pos - tri.v0.pos;
    const glm::vec3 h = glm::cross(ray.direction, edge2);
    const float a = dot(edge1, h);
    if (a > -0.0001f && a < 0.0001f) return false; // ray parallel to triangle
    const float f = 1 / a;
    const glm::vec3 s = ray.origin - tri.v0.pos;
    const float u = f * glm::dot(s, h);
    if (u < 0 || u > 1) return false;
    const glm::vec3 q = glm::cross(s, edge1);
    const float v = f * glm::dot(ray.direction, q);
    if (v < 0 || u + v > 1) return false;
    const float t = f * glm::dot(edge2, q);
    if (t > 0.0001f) {
        out_float = t;
        return true;
    }
    return false;
}

__host__ __device__ float bvhIntersectionTest(
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    BVHNode* bvhNodes,
    Triangle* mesh_triangles,
    int num_tris) 
{
    //start at root
    int nodeIdx = 0;
    //use stack for iterative traversal, depth is max expected depth
    int stack[16];
    //use first level to indicate completion
    int ptr = 1;
    stack[ptr] = nodeIdx;

    float iter_min_t, global_min_t = INFINITY;
  
    while (ptr > 0) {
        const int curr_idx = stack[--ptr];
        BVHNode& node = bvhNodes[curr_idx];

        if (!intersectAABB(r, node.aabb.bmin, node.aabb.bmax, iter_min_t) || iter_min_t > global_min_t) {
            continue;
        }
        if (node.isLeaf())
        {
            for (unsigned int i = 0; i < node.triCount; i++) {
                int curr_tri_idx = node.leftFirst + i;
                Triangle& curr_tri = mesh_triangles[curr_tri_idx];
                float curr_t;
                if (!intersectTri(r, curr_tri, curr_t)) {
                    continue;
                }

                if (curr_t < global_min_t) {
                    global_min_t = curr_t;
                    intersectionPoint = getPointOnRay(r, global_min_t);
                    normal = curr_tri.v0.nor;
                    outside = false;
                }
            }
        }
        else
        {
            stack[ptr++] = node.leftFirst;
            stack[ptr++] = node.leftFirst + 1;
        }
    }

    return global_min_t;
}