#include "intersections.h"

__host__ __device__ float aabbIntersectionTest(glm::vec3 min, glm::vec3 max, const Ray& r, glm::vec3& intersectionPoint,
                                               glm::vec3& normal, bool& outside) {
  float tmin = -FLT_MAX;
  float tmax = FLT_MAX;
  glm::vec3 tmin_n(0.0f);
  glm::vec3 tmax_n(0.0f);
  outside = true;

  for (int i = 0; i < 3; i++) {
    if (r.direction[i] != 0.0f) {
      float t1 = (min[i] - r.origin[i]) / r.direction[i];
      float t2 = (max[i] - r.origin[i]) / r.direction[i];

      float tNear = glm::min(t1, t2);
      float tFar = glm::max(t1, t2);

      glm::vec3 nNear(0.0f);
      glm::vec3 nFar(0.0f);

      nNear[i] = (t1 < t2) ? -1.0f : 1.0f;
      nFar[i] = -nNear[i];

      if (tNear > tmin) {
        tmin = tNear;
        tmin_n = nNear;
      }
      if (tFar < tmax) {
        tmax = tFar;
        tmax_n = nFar;
      }
      if (tmin > tmax) {
        return -1.f;
      }
    } else {
      if (r.origin[i] < min[i] || r.origin[i] > max[i]) {
        return -1.0f;
      }
    }
  }

  float t;
  if (tmin >= 0.0f) {
    t = tmin;
    normal = tmin_n;
  } else if (tmax >= 0.0f) {
    t = tmax;
    normal = tmax_n;
    outside = false;
  } else {
    return -1.0f;
  }

  intersectionPoint = r.origin + t * r.direction;
  return t;
}

__host__ __device__ float boxIntersectionTest(Geom box, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal,
                                              bool& outside) {
  Ray q;
  q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
  q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

  float tmin = -1e38f;
  float tmax = 1e38f;
  glm::vec3 tmin_n;
  glm::vec3 tmax_n;
  for (int xyz = 0; xyz < 3; ++xyz) {
    float qdxyz = q.direction[xyz];
    /*if (glm::abs(qdxyz) > 0.00001f)*/
    {
      float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
      float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
      float ta = glm::min(t1, t2);
      float tb = glm::max(t1, t2);
      glm::vec3 n;
      n[xyz] = t2 < t1 ? +1 : -1;
      if (ta > 0 && ta > tmin) {
        tmin = ta;
        tmin_n = n;
      }
      if (tb < tmax) {
        tmax = tb;
        tmax_n = n;
      }
    }
  }

  if (tmax >= tmin && tmax > 0) {
    outside = true;
    if (tmin <= 0) {
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

__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal,
                                                 bool& outside) {
  float radius = .5;

  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

  Ray rt;
  rt.origin = ro;
  rt.direction = rd;

  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
  if (radicand < 0) {
    return -1;
  }

  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;

  float t = 0;
  if (t1 < 0 && t2 < 0) {
    return -1;
  } else if (t1 > 0 && t2 > 0) {
    t = min(t1, t2);
    outside = true;
  } else {
    t = max(t1, t2);
    outside = false;
  }

  glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

  intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
  normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
  if (!outside) {
    normal = -normal;
  }

  return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool triangleIntersection(const Ray& r, const Triangle& tri, float& t, float& u, float& v) {
  glm::vec3 edge1 = tri.vertices[1] - tri.vertices[0];
  glm::vec3 edge2 = tri.vertices[2] - tri.vertices[0];
  glm::vec3 h = glm::cross(r.direction, edge2);
  float a = glm::dot(edge1, h);
  if (fabs(a) < EPSILON) {
    return false;
  }

  float f = 1.0f / a;
  glm::vec3 s = r.origin - tri.vertices[0];
  u = f * glm::dot(s, h);
  if (u < 0.0f || u > 1.0f) {
    return false;
  }

  glm::vec3 q = glm::cross(s, edge1);
  v = f * glm::dot(r.direction, q);
  if (v < 0.0f || u + v > 1.0f) {
    return false;
  }

  t = f * glm::dot(edge2, q);
  if (t > EPSILON) {
    return true;
  }

  return false;
}

__host__ __device__ float meshIntersectionTest(const Geom& meshGeom, const Triangle* dev_triangles, const Ray& r,
                                               bool enableBVC, glm::vec3& intersectionPoint, glm::vec3& normal,
                                               bool& outside) {
  if (enableBVC) {
    glm::vec3 min = meshGeom.boundingBoxMin;
    glm::vec3 max = meshGeom.boundingBoxMax;
    glm::vec3 tempIntersectionPoint, tempNormal;
    bool tempOutside;
    float t = aabbIntersectionTest(min, max, r, tempIntersectionPoint, tempNormal, tempOutside);
    if (t < 0.0f) {
      return -1.0f;
    }
  }

  float t_min = FLT_MAX;
  bool hit = false;

  for (int i = 0; i < meshGeom.meshTriCount; i++) {
    Triangle tri = dev_triangles[meshGeom.meshTriStartIdx + i];

    glm::vec3 ro = multiplyMV(meshGeom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(meshGeom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray objRay;
    objRay.origin = ro;
    objRay.direction = rd;

    float t, u, v;
    bool intersects = triangleIntersection(objRay, tri, t, u, v);
    if (intersects && t < t_min && t > 1e-4f) {
      t_min = t;
      hit = true;

      glm::vec3 objIntersection = objRay.origin + t * objRay.direction;
      glm::vec3 objNormal = glm::normalize((1.0f - u - v) * tri.normals[0] + u * tri.normals[1] + v * tri.normals[2]);

      intersectionPoint = multiplyMV(meshGeom.transform, glm::vec4(objIntersection, 1.0f));
      normal = glm::normalize(multiplyMV(meshGeom.invTranspose, glm::vec4(objNormal, 0.0f)));

      outside = glm::dot(r.direction, normal) < 0.0f;
      if (!outside) {
        normal = -normal;
      }
    }
  }

  if (hit) {
    return glm::length(r.origin - intersectionPoint);
  } else {
    return -1.0f;
  }
}
