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

__host__ __device__
void BVHVolumeIntersectionTest(
    bbox& bbox,
    const Ray& r,
    bool& hit,
    float& t) {
    
    glm::vec3& bboxMin = bbox.min;
    glm::vec3& bboxMax = bbox.max;
    float tmin = 0.0f;  // Start with tmin at 0
    float tmax = INFINITY;

    // Test if ray origin is inside the box
    bool inside = true;
    for (int i = 0; i < 3; ++i) {
        if (r.origin[i] < bboxMin[i] || r.origin[i] > bboxMax[i]) {
            inside = false;
            break;
        }
    }

    // If ray origin is inside the box, set hit to true and t to 0
    if (inside) {
        hit = true;
        t = 0.0f;
        return;
    }

    for (int i = 0; i < 3; ++i) {
        float invD = 1.0f / r.direction[i];
        float t0 = (bboxMin[i] - r.origin[i]) * invD;
        float t1 = (bboxMax[i] - r.origin[i]) * invD;

        if (invD < 0.0f) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }

        tmin = fmaxf(t0, tmin);
        tmax = fminf(t1, tmax);

        if (tmax <= tmin) {
            hit = false;
            t = -1.0f;
            return;
        }
    }

    hit = true;
    t = tmin;
}

__host__ __device__
void BVHHitTestRecursive(
    const Ray& ray,
    const int nodeIndex,
    bvhNode* bvhNodes,

    glm::vec3* vertices,
    glm::ivec3* faceIndices, // for access vertices
    glm::vec3* faceNormals,
    int* faceIndicesBVH, // for access faceIndex

    const bool selfHitCheck,

    float& t_min,
    int& faceIndexHit,
    bool& hit){
    /*
        Check if hit any triangle in the BVH.
        1. check if ray intersects with current node's bbox
        2. if so, check if it's a leaf node or not. If yes check with triangles in the leaf node
        3. if it's not a leaf node, check if both children nodes are hit
        4. if both children nodes are hit, recurse on the closer,
           if after closer, farther still possible, then recurse on farther
        5. if only one child node is hit, recurse on that node
        6. if no child nodes are hit, return
    */
    
    bvhNode* node = &bvhNodes[nodeIndex];
    bbox& selfBbox = bvhNodes[nodeIndex].bbox;
    // 1. check if ray intersects with current if haven't checked yet
    if(!selfHitCheck){
        float selfHitT;
        bool selfHit;
        BVHVolumeIntersectionTest(selfBbox, ray, selfHit, selfHitT);
        if(!selfHit){
            return;
        }
    }

    // 2. check if it's a leaf node or not
    if(node->is_leaf){
        // check if ray intersects with the triangles in the leaf node
        float t;
        for(int i = node->startIndex; i < node->startIndex + node->size; ++i){
            int faceIndex = faceIndicesBVH[i];
            glm::ivec3& face = faceIndices[faceIndex];
            glm::vec3& v0 = vertices[face.x];
            glm::vec3& v1 = vertices[face.y];
            glm::vec3& v2 = vertices[face.z];

            glm::vec3 baryPosition;
            if (glm::intersectRayTriangle(ray.origin, ray.direction, v0, v1, v2, baryPosition)) {
                t = baryPosition.z;
                glm::vec3 faceNormal = faceNormals[faceIndex];
                if (glm::dot(ray.direction, faceNormal) >= 0) { // different direction
                    continue;
                }
                if (t > 0.0f && t < t_min) {
                    t_min = t;
                    faceIndexHit = faceIndex;
                    hit = true;
                }
            }
        }
    }else{
        // 3. if it's not a leaf node, check if both children nodes are hit
        float t_left, t_right;
        bool hit_left, hit_right;
        bbox leftBBox = bvhNodes[node->left].bbox;
        bbox rightBBox = bvhNodes[node->right].bbox;
        BVHVolumeIntersectionTest(leftBBox, ray, hit_left, t_left);
        BVHVolumeIntersectionTest(rightBBox, ray, hit_right, t_right);
        // 4. if both children nodes are hit, recurse on the closer
        if(hit_left && hit_right){
            float t_close =    t_left <= t_right ? t_left : t_right;
            int index_closer = t_left <= t_right ? node->left : node->right;
            float t_far =        t_left >= t_right ? t_left : t_right;
            int index_farther =  t_left >= t_right ? node->left : node->right;
            // recurse on closer
            bool closerRecursiveHit, fartherRecursiveHit;
            BVHHitTestRecursive(ray, index_closer, bvhNodes, 
                    vertices, faceIndices, faceNormals, faceIndicesBVH, 
                    true, t_min, faceIndexHit, closerRecursiveHit);
            // if a. closer hit but farther t is better, recurse on farther as well
            //    b. closer didn't hit, recurse on farther
            bool closeHitButFartherBetter = closerRecursiveHit && t_far < t_min;
            bool checkFarther = closeHitButFartherBetter || !closerRecursiveHit;
            if(checkFarther){
                BVHHitTestRecursive(ray, index_farther, bvhNodes, 
                vertices, faceIndices, faceNormals, faceIndicesBVH, 
                true, t_min, faceIndexHit, fartherRecursiveHit);
                hit = closerRecursiveHit || fartherRecursiveHit;
            }else{
                // return closer recursive result
                hit = closerRecursiveHit;
                return;
            }
        }else if (hit_left){
            BVHHitTestRecursive(ray, node->left, bvhNodes, 
                    vertices, faceIndices, faceNormals, faceIndicesBVH, 
                    true, t_min, faceIndexHit, hit);
        }else if (hit_right){
            BVHHitTestRecursive(ray, node->right, bvhNodes, 
                    vertices, faceIndices, faceNormals, faceIndicesBVH, 
                    true, t_min, faceIndexHit, hit);
        }else{
            // no child nodes are hit
            return;
        }
        
    }

}
    
