// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <examples/ceres/types.h>
#include <examples/ceres/math.h>

namespace Ceres {
class ReprojectionError : public ceres::SizedCostFunction<3, 15, 3> {
public:
  ReprojectionError(const Vector2 &measurement, Scalar sqrt_weight)
      : m_measurement(measurement), m_sqrt_weight(sqrt_weight) {}

  ~ReprojectionError() {}

  virtual bool Evaluate(Scalar const *const *parameters, Scalar *residuals,
                        Scalar **jacobians) const override {
    Eigen::Map<const Matrix<3, 4>> extrinsics(parameters[0]);
    Eigen::Map<const Matrix<3, 1>> intrinsics(parameters[0] + 12);
    Eigen::Map<const Matrix<3, 1>> point(parameters[1]);
    Eigen::Map<Vector3> error(residuals);

    const Scalar delta = 1e-6;
    const Scalar delta_squared = 1e-12;

    Vector3 dist;
    Vector3 ray;
    Vector3 rotated_ray;

    Scalar radical_squared;
    Scalar error_squared_norm;
    Scalar dist_dot_rotated_ray;
    Scalar rescaled_dist_squared_norm_plus_delta_squared;
    Scalar sqrt_dist_squared_norm_plus_delta_squared;

    const auto &measurement = m_measurement;
    const auto &sqrt_weight = m_sqrt_weight;

    radical_squared = measurement.squaredNorm();

    ray.template head<2>() = measurement;
    ray[2] = intrinsics[0] + intrinsics[1] * radical_squared +
             intrinsics[2] * radical_squared * radical_squared;
    rotated_ray.noalias() = extrinsics.template leftCols<3>() * ray;

    dist = point - extrinsics.col(3);
    dist_dot_rotated_ray = dist.dot(rotated_ray);
    rescaled_dist_squared_norm_plus_delta_squared =
        dist.squaredNorm() + delta_squared;
    sqrt_dist_squared_norm_plus_delta_squared =
        sqrt(rescaled_dist_squared_norm_plus_delta_squared);
    rescaled_dist_squared_norm_plus_delta_squared +=
        delta * sqrt_dist_squared_norm_plus_delta_squared;
    error = rotated_ray;
    error -=
        (dist_dot_rotated_ray / rescaled_dist_squared_norm_plus_delta_squared) *
        dist;
    error *= sqrt_weight;

    if (jacobians != nullptr) {
      Eigen::Map<Eigen::Matrix<Scalar, 3, 15, Eigen::RowMajor>> jac_c(
          jacobians[0]);
      Eigen::Map<Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>> jac_pnt(
          jacobians[1]);

      auto jac_rot = jac_c.leftCols<9>();
      auto jac_lin = jac_c.middleCols<3>(9);
      auto jac_int = jac_c.middleCols<3>(12);

      Matrix<3, 3> jac_ray = Matrix<3, 3>::Identity();
      jac_ray.leftCols<3>().noalias() -=
          dist * dist.transpose() /
          rescaled_dist_squared_norm_plus_delta_squared;
      jac_rot.middleCols<3>(0) = jac_ray * sqrt_weight * ray[0];
      jac_rot.middleCols<3>(3) = jac_ray * sqrt_weight * ray[1];
      jac_rot.middleCols<3>(6) = jac_ray * sqrt_weight * ray[2];

      Scalar rescaled_sqrt_weight_over_dist_squared_norm =
          sqrt_weight / rescaled_dist_squared_norm_plus_delta_squared;
      Scalar rescale = rescaled_sqrt_weight_over_dist_squared_norm *
                       (delta / sqrt_dist_squared_norm_plus_delta_squared + 2) /
                       rescaled_dist_squared_norm_plus_delta_squared;
      jac_lin.setZero();
      jac_lin.diagonal().array() =
          dist_dot_rotated_ray * rescaled_sqrt_weight_over_dist_squared_norm;
      jac_lin.noalias() += rescaled_sqrt_weight_over_dist_squared_norm * dist *
                           rotated_ray.transpose();
      jac_lin.noalias() -=
          (rescale * dist_dot_rotated_ray) * dist * dist.transpose();

      jac_int.col(0) = extrinsics.col(2);
      jac_int.col(0) -= dist.dot(extrinsics.col(2)) /
                        (rescaled_dist_squared_norm_plus_delta_squared)*dist;
      jac_int.col(0) *= sqrt_weight;
      jac_int.col(1) = jac_int.col(0) * radical_squared;
      jac_int.col(2) = jac_int.col(1) * radical_squared;

      jac_pnt = -jac_lin;
    }

    return true;
  }

private:
  Vector2 m_measurement;
  Scalar m_sqrt_weight;
};

struct AutoDiffReprojectionError
{
public:
    AutoDiffReprojectionError(const Vector2 &measurement, Scalar sqrt_weight)
      : m_measurement(measurement), m_sqrt_weight(sqrt_weight) {}

    template <typename T>
    bool operator()(
        const T* const camera,
        const T* const point_input,
        T* residuals) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 4>> extrinsics(camera);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> intrinsics(camera + 12);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> point(point_input);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> error(residuals);

        const T delta = T(1e-6);
        const T delta_squared = T(1e-12);

        Eigen::Matrix<T, 3, 1> dist;
        Eigen::Matrix<T, 3, 1> ray;
        Eigen::Matrix<T, 3, 1> rotated_ray;

        T radical_squared;
        T error_squared_norm;
        T dist_dot_rotated_ray;
        T rescaled_dist_squared_norm_plus_delta_squared;
        T sqrt_dist_squared_norm_plus_delta_squared;
        T lambda;

        radical_squared = T(m_measurement.squaredNorm());

        ray[0] = T(m_measurement[0]);
        ray[1] = T(m_measurement[1]);
        ray[2] = intrinsics[0] + intrinsics[1] * radical_squared +
                intrinsics[2] * radical_squared * radical_squared;
        rotated_ray.noalias() = extrinsics.template leftCols<3>() * ray;

        dist = point - extrinsics.col(3);
        dist_dot_rotated_ray = dist.dot(rotated_ray);
        rescaled_dist_squared_norm_plus_delta_squared =
            dist.squaredNorm() + delta_squared;
        sqrt_dist_squared_norm_plus_delta_squared =
            sqrt(rescaled_dist_squared_norm_plus_delta_squared);
        rescaled_dist_squared_norm_plus_delta_squared +=
            delta * sqrt_dist_squared_norm_plus_delta_squared;
        lambda = dist_dot_rotated_ray / rescaled_dist_squared_norm_plus_delta_squared;

        error = rotated_ray;
        error -=
            lambda * dist;
        error *= T(m_sqrt_weight);

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const Vector2 &measurement, Scalar sqrt_weight)
    {
        return new ceres::AutoDiffCostFunction<AutoDiffReprojectionError, 3, 15, 3>(
                        new AutoDiffReprojectionError(measurement, sqrt_weight));
    }

    Eigen::Vector2d m_measurement;
    double m_sqrt_weight;
};

} // namespace Ceres

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 10 parameters. 4 for rotation, 3 for
// translation, 1 for focal length and 2 for radial distortion. The
// principal point is not modeled (i.e. it is assumed be located at
// the image center).
struct SnavelyReprojectionError
{
    // (u, v): the position of the observation with respect to the image
    // center point.
    SnavelyReprojectionError(
        double observed_x,
        double observed_y,
        double weight):
        observed_x(observed_x),
        observed_y(observed_y),
        weight(weight)
    {}

    template <typename T>
    bool operator()(
        const T* const camera,
        const T* const point_input,
        T* residuals) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 4>> extrinsics(camera);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> intrinsics(camera + 12);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> point(point_input);

        Eigen::Matrix<T, 3, 1> p = extrinsics.template leftCols<3>().transpose() * (point - extrinsics.col(3));

        // Noah Snavely's Bundler assumes, the camera coordinate system has a negative z axis.
        // In dataset.cu measurement were negated so change of sign is not needed here.
        const T xp = p[0] / p[2];
        const T yp = p[1] / p[2];

        // Apply second and fourth order radial distortion.
        const T& k1 = intrinsics[1];
        const T& k2 = intrinsics[2];
        const T r2 = xp * xp + yp * yp;
        const T distortion = 1.0 + r2 * (k1 + k2 * r2);

        // Compute final projected point position.
        const T& focal = intrinsics[0];
        const T predicted_x = focal * distortion * xp;
        const T predicted_y = focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;
        residuals[0] *= T(weight);
        residuals[1] *= T(weight);

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(
        const double observed_x,
        const double observed_y,
        const double weight )
    {
        return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 15, 3>(
                        new SnavelyReprojectionError(observed_x, observed_y, weight));
    }

    double observed_x;
    double observed_y;
    double weight;
};