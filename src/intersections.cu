#include "intersections.h"

__device__ float boxIntersectionTest(
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

__device__ float sphereIntersectionTest(
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

__device__ float intersectRayWithBoundingBox(const glm::vec3& boxMin, const glm::vec3& boxMax, const Ray& ray) {
    float tmin = -1e38f;
    float tmax = 1e38f;

    for (int i = 0; i < 3; ++i) {
        float invD = 1.0f / ray.direction[i];
        float t0 = (boxMin[i] - ray.origin[i]) * invD;
        float t1 = (boxMax[i] - ray.origin[i]) * invD;

        if (invD < 0.0f) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }

        if (t0 > tmin) tmin = t0;
        if (t1 < tmax) tmax = t1;

        if (tmax < tmin) return -1;
    }

	return tmin;
}

__device__ float meshIntersectionTest(
    Geom geom,
    const Triangle* triangles,
    const glm::vec3* vertices,
    const glm::vec3* normals,
    const Mesh& mesh,
    int rootNodeIndex,
	const BvhNode* bvhNodes,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    int& hitTriangleIndex,
    glm::vec2& baryCoords,
    int* nodeStack) {

	// Each thread has its own stack, so read/write destination in shared memory needs to be offset.
	int offset = (threadIdx.x * MAX_BVH_DEPTH);

    // Transform the ray into object space
    Ray rt;
    rt.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    rt.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));
	float t = -1;
	float tMin = FLT_MAX;

	int stackIndex = 0;
	nodeStack[offset + (stackIndex++)] = rootNodeIndex; // note postfix increment
    
    // Test all triangles within the mesh
    while (stackIndex > 0) {
		BvhNode node = bvhNodes[nodeStack[offset + (--stackIndex)]]; // note prefix decrement
         
		// If the ray does not intersect the bounding box, or previous nodes have found closer intersections, skip this node
		float tBox = intersectRayWithBoundingBox(node.min, node.max, rt);
		if (tBox < 0 || tBox > tMin) {
			continue;
		}

		// If the node is a leaf node, test the triangles
        if (node.leftChild == -1 && node.rightChild == -1) {
            for (int i = node.trianglesStartIdx; i < node.trianglesStartIdx + node.numTriangles; ++i) {
                const Triangle& triangle = triangles[i];

                glm::vec3 v0 = vertices[mesh.vertStartIndex + triangle.attributeIndex[0]];
                glm::vec3 v1 = vertices[mesh.vertStartIndex + triangle.attributeIndex[1]];
                glm::vec3 v2 = vertices[mesh.vertStartIndex + triangle.attributeIndex[2]];

                glm::vec3 n0 = normals[mesh.vertStartIndex + triangle.attributeIndex[0]];
                glm::vec3 n1 = normals[mesh.vertStartIndex + triangle.attributeIndex[1]];
                glm::vec3 n2 = normals[mesh.vertStartIndex + triangle.attributeIndex[2]];

                glm::vec3 barycentricCoord;

                if (!glm::intersectRayTriangle(rt.origin, rt.direction, v0, v1, v2, barycentricCoord)) {
                    continue;
                }

                // Calculate the intersection point in world space
		        t = barycentricCoord.z;
                if (t >= tMin) continue;

                tMin = t;
                hitTriangleIndex = i;
                intersectionPoint = getPointOnRay(r, t);
                baryCoords = glm::vec2(barycentricCoord.x, barycentricCoord.y);

                // Interpolate the normal
                normal = glm::normalize(n0 * (1.0f - barycentricCoord.x - barycentricCoord.y) + n1 * barycentricCoord.x + n2 * barycentricCoord.y);

                // Transform the normal into to world space
                normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normal, 0.0f)));

                // Determine if the intersection is outside
                outside = glm::dot(rt.direction, normal) < 0;
            }
        }
        else {
			nodeStack[offset + (stackIndex++)] = node.leftChild;  // note postfix increment
			nodeStack[offset + (stackIndex++)] = node.rightChild;
        }
    }

	return tMin / glm::length(rt.direction);
}
