// Portable command-line front end for the 2012 DDE Lab SRM implementation.
// The SRM feature computation itself is unchanged and lives in SRM_src/.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "SRM_src/SRMclass.h"
#include "SRM_src/submodel.h"

namespace fs = std::filesystem;

struct Options {
    fs::path input_dir;
    fs::path output_dir;
    int threshold = 2;
    int order = 4;
    bool verbose = false;
    bool merge_spams = true;
    bool symm_sign = true;
    bool symm_reverse = true;
    bool symm_minmax = true;
    bool erase_lsb = false;
    bool parity = false;
};

static void print_help() {
    std::cout
        << "Portable SRM feature extractor\n\n"
        << "Usage: srm_extract --input-dir DIR --output-dir DIR [options]\n\n"
        << "Options:\n"
        << "  --T N                  residual truncation threshold (default 2)\n"
        << "  --order N              co-occurrence order (default 4)\n"
        << "  --no-merge-spams       do not merge horizontal/vertical SPAMs\n"
        << "  --no-symm-sign         disable sign symmetry\n"
        << "  --no-symm-reverse      disable reverse-order symmetry\n"
        << "  --no-symm-minmax       disable min/max symmetry\n"
        << "  --eraseLSB             erase image LSBs before extraction\n"
        << "  --parity               apply the parity residual\n"
        << "  --verbose              print progress\n";
}

static Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto value = [&](const char *name) -> std::string {
            if (++i >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
            return argv[i];
        };
        if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else if (arg == "--input-dir" || arg == "-I") {
            options.input_dir = value("--input-dir");
        } else if (arg == "--output-dir" || arg == "-O") {
            options.output_dir = value("--output-dir");
        } else if (arg == "--T") {
            options.threshold = std::stoi(value("--T"));
        } else if (arg == "--order") {
            options.order = std::stoi(value("--order"));
        } else if (arg == "--no-merge-spams") {
            options.merge_spams = false;
        } else if (arg == "--no-symm-sign") {
            options.symm_sign = false;
        } else if (arg == "--no-symm-reverse") {
            options.symm_reverse = false;
        } else if (arg == "--no-symm-minmax") {
            options.symm_minmax = false;
        } else if (arg == "--eraseLSB") {
            options.erase_lsb = true;
        } else if (arg == "--parity") {
            options.parity = true;
        } else if (arg == "--verbose" || arg == "-V") {
            options.verbose = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (options.input_dir.empty()) throw std::runtime_error("--input-dir is required");
    if (options.output_dir.empty()) throw std::runtime_error("--output-dir is required");
    if (options.threshold < 1) throw std::runtime_error("--T must be positive");
    if (options.order < 1) throw std::runtime_error("--order must be positive");
    return options;
}

static void write_features(SRMclass &srm, const fs::path &output_dir) {
    for (Submodel *submodel : srm.GetSubmodels()) {
        const fs::path path = output_dir / (submodel->GetName() + ".fea");
        std::ofstream output(path);
        if (!output) throw std::runtime_error("Could not open output file: " + path.string());
        const std::vector<float *> features = submodel->ReturnFea();
        for (std::size_t image_index = 0; image_index < features.size(); ++image_index) {
            for (int feature_index = 0; feature_index < submodel->symmDim; ++feature_index) {
                output << features[image_index][feature_index] << ' ';
            }
            output << srm.imageNames[image_index] << '\n';
        }
    }
}

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (!fs::is_directory(options.input_dir)) {
            throw std::runtime_error("Input directory does not exist: " + options.input_dir.string());
        }
        fs::create_directories(options.output_dir);

        std::vector<std::string> images;
        for (const auto &entry : fs::directory_iterator(options.input_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".pgm") {
                images.push_back(entry.path().string());
            }
        }
        std::sort(images.begin(), images.end());
        if (images.empty()) throw std::runtime_error("No .pgm images found");

        Config config(
            options.verbose,
            options.threshold,
            options.order,
            options.symm_sign,
            options.symm_reverse,
            options.symm_minmax,
            options.merge_spams,
            options.erase_lsb,
            options.parity);
        SRMclass srm(&config);
        srm.ComputeFeatures(images);
        write_features(srm, options.output_dir);
        if (options.verbose) {
            std::cout << "Extracted " << images.size() << " image(s) into "
                      << options.output_dir << '\n';
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "SRM extraction failed: " << error.what() << '\n';
        return 1;
    }
}
