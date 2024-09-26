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
    Geom& geom,
	Triangle* triangles,
	int numTriangles,
	Ray r,
	glm::vec3& intersectionPoint,
	glm::vec3& normal,
	bool& outside)
{
	float t = FLT_MAX;
    glm::vec3 baryCoords;
	bool tempOutside = false;
	outside = false;
	// transform ray to object space
	glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = multiplyMV3(glm::transpose(glm::mat3(geom.transform)), r.direction); // transpose of model matrix
	int triangleIdx = -1;
	for (int i = geom.triangleStartIdx; i < geom.triangleEndIdx; i++)
	{
        Triangle& triangle = triangles[i];
        glm::vec3 tempBaryCoords(0.);
		bool intersect = glm::intersectRayTriangle(ro, rd, triangle.vertices[0], triangle.vertices[1], triangle.vertices[2], tempBaryCoords);
        if (intersect && tempBaryCoords.z < t)
        {
			t = tempBaryCoords.z;
			baryCoords = tempBaryCoords;
            outside = true;
			triangleIdx = i;
        }
	}

	// object space to world space

    // normal
	normal = glm::normalize(baryCoords.x * triangles[triangleIdx].normals[0] + baryCoords.y * triangles[triangleIdx].normals[1] + baryCoords.z * triangles[triangleIdx].normals[2]);
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normal, 0.f)));

	// intersection point
	intersectionPoint = r.origin + t * r.direction;

	return t == FLT_MAX ? -1 : t;
}

__device__ float meshIntersectionMoller(Geom& geom, const Ray& ray, const Triangle* triangles, glm::vec3& normal)
{
	float mint = FLT_MAX;
	for (int i = geom.triangleStartIdx; i < geom.triangleEndIdx; i++)
	{
		const Triangle& triangle = triangles[i];
		float t = triangle.intersect(ray);
		if (t > 0 && t < mint)
		{
			mint = t;
			normal = triangle.getNormal();
		}
	}
	return mint;
}