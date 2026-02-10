///////////////////////////////////////////////////
// Low Level Parallel Programming 2017.
//
// The main starting point for the crowd simulation.
//
#undef max
#include "ped_model.h"
#include "ParseScenario.h"

#include <thread>

#include "Simulation.h"
#include "TimingSimulation.h"
#include "ExportSimulation.h"
#ifndef NOQT
#include "QTSimulation.h"
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QApplication>
#include <QTimer>
#include "MainWindow.h"
#endif
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstring>

#pragma comment(lib, "libpedsim.lib")

#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>

void print_usage(char *command) {
    printf("Usage: %s [--timing-mode|--export-trace[=export_trace.bin]] [--max-steps=100] [--help] [--cuda|--simd|--omp|--pthread|--seq] [scenario filename]\n", command);
    printf("There are three modes of execution:\n");
#ifndef NOQT
    printf("\t the QT window mode (default if no argument is provided. But this is also deprecated. Please opt to use the --export-trace mode instead)\n");
#endif
    printf("\t the --export-trace mode: where the agent movement are stored in a trace file and can be visualized by a separate python tool.\n");
    printf("\t the --timing-mode: the mode where no visualization is done and can be used to measure the performance of your implementation/optimization\n");
    printf("\nIf you need visualization, please try using the --export-trace mode. You can even copy the trace file to your computer and locally run the python visualizer. (You'll need to fork the assignment repository on your local machine too.)\n");
}

int main(int argc, char*argv[]) {
    bool timing_mode = true;
#ifndef NOQT
    bool export_trace = false; // If no QT, export_trace is default
#else
    bool export_trace = true;
#endif
    std::string scenefile = std::string("hugeScenario.xml");
    int max_steps = 1000;
    Ped::IMPLEMENTATION implementation_to_test = Ped::SEQ;
    std::string export_trace_file = "";
    int max_threads = 0; // default thread count; can also set via --max-threads CLI arg

    // Parsing command line arguments
    while (1) {
        static struct option long_options[] = {
            {"timing-mode", no_argument, NULL, 't'},
            {"export-trace", optional_argument, NULL, 'e'},
            {"max-steps", required_argument, NULL, 'm'},
            {"help", no_argument, NULL, 'h'},
            {"cuda", no_argument, NULL, 'c'},
            {"simd", no_argument, NULL, 's'},
            {"omp", no_argument, NULL, 'o'},
            {"pthread", no_argument, NULL, 'p'},
            {"seq", no_argument, NULL, 'q'},
            {0, 0, 0, 0}  // End of options
        };

        int option_index = 0;
        int long_opt = getopt_long(argc, argv, "", long_options, &option_index);

        if (long_opt == -1) break;  // No more options

        switch (long_opt) {
            case 't':
                std::cout << "Option --timing-mode activated\n";
                timing_mode = true;
                export_trace = false;
                break;
            case 'e':
                export_trace = true;
                export_trace_file = (optarg != NULL) ? optarg : "export_trace.bin";
                std::cout << "Option --export-trace set to: " << export_trace_file << std::endl;
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            case 'c':
                std::cout << "Option --cuda activated\n";
                implementation_to_test = Ped::CUDA;
                break;
            case 's':
                std::cout << "Option --simd activated\n";
                implementation_to_test = Ped::VECTOR;
                break;
            case 'o':
                std::cout << "Option --omp activated\n";
                implementation_to_test = Ped::OMP;
                break;
            case 'p':
                std::cout << "Option --pthread activated\n";
                implementation_to_test = Ped::PTHREAD;
                break;
            case 'q':
                std::cout << "Option --seq activated\n";
                implementation_to_test = Ped::SEQ;
                break;
            case 'm':
                max_steps = std::stoi(optarg);
                std::cout << "Option --max-steps set to: " << max_steps << std::endl;
                break;
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }

    if (optind < argc) {
        scenefile = argv[optind];  // Scenario file argument
    }

    if (export_trace && max_steps > 200) {
        std::cout << "Reducing max_steps to 200 for tracing run." << std::endl;
        max_steps = 200;
    }

    int retval = 0;
    { // scope to avoid false memory leak positives

        if (timing_mode) {
            // Timing version (repeats for stability)
            int repeats = 10;
            double avg_seq_ms = 0.0;
            double avg_target_ms = 0.0;

            // --- SEQ ---
            for (int i = 0; i < repeats; ++i) {
                Ped::Model model;
                ParseScenario parser(scenefile);
                model.setup(parser.getAgents(), parser.getWaypoints(), Ped::SEQ, max_threads);
                Simulation* sim = new TimingSimulation(model, max_steps);

                auto start = std::chrono::steady_clock::now();
                sim->runSimulation();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start);
                avg_seq_ms += duration.count();
                delete sim;
            }
            avg_seq_ms /= repeats;
            std::cout << "SEQ average time: " << avg_seq_ms << " ms" << std::endl;

            // --- Target Implementation ---
            for (int i = 0; i < repeats; ++i) {
                Ped::Model model;
                ParseScenario parser(scenefile);
                model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test, max_threads);
                Simulation* sim = new TimingSimulation(model, max_steps);

                auto start = std::chrono::steady_clock::now();
                sim->runSimulation();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start);
                avg_target_ms += duration.count();
                delete sim;
            }
            avg_target_ms /= repeats;
            std::cout << "Target average time: " << avg_target_ms << " ms" << std::endl;

            std::cout << "Speedup: " << avg_seq_ms / avg_target_ms << std::endl;
        }
        else if (export_trace) {
            Ped::Model model;
            ParseScenario parser(scenefile);
            model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test, max_threads);

            Simulation* simulation = new ExportSimulation(model, max_steps, export_trace_file);

            std::cout << "Running Export Tracer...\n";
            auto start = std::chrono::steady_clock::now();
            simulation->runSimulation();
            auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
            float fps = ((float)simulation->getTickCount()) / ((float)duration_target.count())*1000.0;
            std::cout << "Time: " << duration_target.count() << " milliseconds, " << fps << " Frames Per Second." << std::endl;

            delete simulation;
        }
#ifndef NOQT
        else {
            // Graphics version
            Ped::Model model;
            ParseScenario parser(scenefile);
            model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test, max_threads);

            QApplication app(argc, argv);
            MainWindow mainwindow(model);

            QTSimulation simulation(model, max_steps, &mainwindow);

            std::cout << "Demo setup complete, running ..." << std::endl;

            auto start = std::chrono::steady_clock::now();
            mainwindow.show();
            simulation.runSimulation();
            retval = app.exec();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
            float fps = ((float)simulation.getTickCount()) / ((float)duration.count())*1000.0;
            std::cout << "Time: " << duration.count() << " milliseconds, " << fps << " Frames Per Second." << std::endl;
        }
#endif
    }

    std::cout << "Done" << std::endl;
    return retval;
}
