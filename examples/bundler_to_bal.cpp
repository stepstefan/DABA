// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <sfm/ba/utils/utils.h>

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  boost::program_options::options_description desc("Program options");
  desc.add_options()                   // solver options
      ("help", "produce help message") // produce help message
      ("bundler_dataset", boost::program_options::value<std::string>(),
       "path to bundler dataset")
      ("bal_dataset",
       boost::program_options::value<std::string>()->default_value("trivial"),
       "path to BAL dataset");

  boost::program_options::variables_map program_options;
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc),
      program_options);

  if (program_options.count("help")) {
    std::cout << desc << "\n";
    exit(1);
  }

  if (program_options.count("bundler_dataset") == false) {
    LOG(ERROR) << "No bundler_dataset has been specfied." << std::endl;
    exit(-1);
  }

  std::string bundler_filename = program_options["bundler_dataset"].as<std::string>();
  std::string bal_filename = program_options["bal_dataset"].as<std::string>();

  int num_cameras, num_points, num_measurements;
  sfm::ba::BundlerDatasetToBALDataset(bundler_filename, bal_filename, num_cameras, num_points, num_measurements);

  return 0;
}