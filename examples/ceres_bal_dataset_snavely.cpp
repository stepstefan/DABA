// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <ceres/ceres.h>
#include <examples/ceres/camera.h>
#include <examples/ceres/reprojection.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sfm/ba/dataset.h>
#include <sfm/math/SO3.h>

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  boost::program_options::options_description desc("Program options");
  desc.add_options()                   // solver options
      ("help", "produce help message") // produce help message
      ("dataset", boost::program_options::value<std::string>(),
       "path to BAL dataset") // path to BAL dataset
      ("loss",
       boost::program_options::value<std::string>()->default_value("trivial"),
       "loss type (\"trivial\" or \"huber\")"); // loss types

  boost::program_options::variables_map program_options;
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc),
      program_options);

  if (program_options.count("help")) {
    std::cout << desc << "\n";
    exit(1);
  }

  if (program_options.count("dataset") == false) {
    LOG(ERROR) << "No dataset has been specfied." << std::endl;
    exit(-1);
  }

  std::string filename = program_options["dataset"].as<std::string>();
  std::string robust_loss_info = program_options["loss"].as<std::string>();

  sfm::ba::BALDataset<Ceres::Scalar> ba_dataset(filename, true);
  const auto &measurements = ba_dataset.Measurements();
  const auto &extrinsics = ba_dataset.Extrinsics();
  const auto &intrinsics = ba_dataset.Intrinsics();

  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << ba_dataset.Extrinsics().size() << " cameras, "
            << ba_dataset.Points().size() << " points, "
            << ba_dataset.Measurements().size() << " measurements."
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;

  // Set all parameters for SnavelyReprojectionErrorMatrix
  std::vector<Ceres::Matrix<3, 5>> cameras(extrinsics.size());
  for (int i = 0; i < extrinsics.size(); i++)
  {
    cameras[i].leftCols(3) = extrinsics[i].template leftCols<3>();
    cameras[i].col(3) = extrinsics[i].col(3);
    cameras[i].col(4) = intrinsics[i];
  }
  std::vector<Ceres::Vector<3>> points(ba_dataset.Points().size());
  for (int i = 0; i < ba_dataset.Points().size(); i++)
  {
    points[i] = ba_dataset.Points()[i];
  }
  std::cout << "Measurement: " << measurements[0].measurement.transpose() << std::endl;

  ceres::Problem problem;
  ceres::LossFunction *loss = nullptr;
  if (robust_loss_info == "huber") {
    loss = new ceres::HuberLoss(32);
  } else if (robust_loss_info == "trivial") {
    loss = nullptr;
  } else {
    LOG(ERROR) << "The loss type can only be \"trivial\" and "
                  "\"huber\"."
               << std::endl;
    exit(-1);
  }

  for (const auto &measurement : measurements) {
    auto edge = SnavelyReprojectionError::Create(measurement.measurement[0], measurement.measurement[1], measurement.sqrt_weight);

    problem.AddResidualBlock(edge, loss,
      cameras[measurement.extrinsics_index].data(),
      points[measurement.point_index].data());
  }

  ceres::Manifold *manifold = new Ceres::Camera();
  for (auto &camera : cameras)
  {
    problem.SetManifold(camera.data(), manifold);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;
  options.preconditioner_type = ceres::PreconditionerType::SCHUR_JACOBI;
  options.trust_region_strategy_type =
      ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 40;
  options.num_threads = 8;
  options.parameter_tolerance = 0;
  options.function_tolerance = 0;
  options.gradient_tolerance = 0;
  options.max_solver_time_in_seconds = 14400;
  options.eta = 1e-02;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

//_______________________________________________________________________________________________________________________________________________________________
  std::vector<Eigen::Matrix<Ceres::Scalar, 3, 4>> optimized_extrinsics_merged;
  std::vector<Eigen::Matrix<Ceres::Scalar, 3, 1>> optimized_intrinsics_merged;
  std::vector<Eigen::Matrix<Ceres::Scalar, 3, 1>> optimized_points_merged;

  for (const auto& camera : cameras)
  {
    optimized_extrinsics_merged.push_back(camera.leftCols(4));
    optimized_intrinsics_merged.push_back(camera.col(4));
  }

  for (const auto& point : points)
  {
    optimized_points_merged.push_back(point);
  }

  sfm::ba::BALDataset<Ceres::Scalar> ba_optimized_dataset(
    ba_dataset.Measurements(),
    optimized_extrinsics_merged,
    optimized_intrinsics_merged,
    optimized_points_merged);
  std::string outfile = filename.substr(filename.rfind("problem-"));
  std::string dir_path = filename.substr(0, filename.rfind("problem-"));
  outfile = outfile.substr(0, outfile.find(".txt"));
  outfile = dir_path + "ceres_snavely-result-" + robust_loss_info + "-" + outfile + ".txt";

  ba_optimized_dataset.Write(outfile, ba_dataset.Scales());

  return 0;
}