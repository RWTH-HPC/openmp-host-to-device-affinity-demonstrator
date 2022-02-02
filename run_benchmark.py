#!/bin/env python3

import sys
import json
import os
import subprocess
import numpy as np

def main():
    config_path, output_path = parse_cmd()
    
    config = {}
    with open(config_path, "r") as config_file:
        config = json.loads(config_file.read())

    if not os.path.exists(output_path + "raw"):
        os.mkdir(output_path + "raw")

    #generate data

    for test in config["tests"]:
        test_output_name = "_".join([str(test[key]).replace(" ", "-") for key in test.keys() if not key == "OMP_PLACES"])
        best_output = b""
        worst_output = b""

        print("Executing test", test_output_name)
        test["best_result"] = []
        for i in range(config["repetitions"]):
            print(i, end=" ")
            sys.stdout.flush()
            cmd_best = ["no_numa_balancing"] + ["benchmark/build/app/distanceBenchmark_best"] + test["parameters"].split(" ")
            cmd_worst = ["no_numa_balancing"] + ["benchmark/build/app/distanceBenchmark_worst"] + test["parameters"].split(" ")
            env = os.environ
            for key in test.keys():
                if "OMP" in key:
                    env[key] = str(test[key])
            best_output += subprocess.check_output(cmd_best, env=env)
            worst_output += subprocess.check_output(cmd_worst, env=env)
        print("")

        test["best_result"] = best_output.decode("UTF-8");
        test["worst_result"] = worst_output.decode("UTF-8");
        with open(output_path + "raw/" + config["name"] + test_output_name + "_best.output", "w") as output_file:
            output_file.write(test["best_result"])
        with open(output_path + "raw/" + config["name"] + test_output_name + "_worst.output", "w") as output_file:
            output_file.write(test["worst_result"])

    print("Raw output files can be found at", output_path + "raw/" + config["name"] + "_*.output");
    
    # parse data
    try:
        results = {}
        for test in config["tests"]:
            test_output_name = "_".join([str(test[key]) for key in test.keys() if not key == "OMP_PLACES" and not "result" in key ])
            results[test_output_name] = { "best" : parse_results(test["best_result"]), "worst" : parse_results(test["worst_result"]) }
            results[test_output_name]["config"] = {item[0] : item[1] for item in test.items() if not "result" in item[0]}

        with open(output_path + config["name"] + "_parsed.json", "w") as result_file:
            result_file.write(json.dumps(results, indent=4))
        print("Parsed json output file can be found at", output_path + config["name"] + "_parsed.json");
    except:
        pass
        


def parse_results(results):
    parsed_data = {}
    known_keys = {
            "CUDA device distance initalization was successful and took" : "init",
            "Computations with normal tasking took" : "computation",
            "Waiting times of thread" : "wait",
            "Ratio longest waiting time / shortest waiting time" : "ratio"
    }

    for value in known_keys.values():
        if value == "wait":
            parsed_data[value] = {}
        else:
            parsed_data[value] = {"data" : [], "average" : 0., "derivation" : 0.}

    for line in results.split("\n"):
        words = line.split(" ")
        value = 0
        try:
            value = float(words[-1])
        except ValueError:
            continue

        found = False
        for kkey in known_keys.keys():
            if line.startswith(kkey):
                if known_keys[kkey] == "wait":
                    thread = int(words[4])
                    if not thread in parsed_data["wait"].keys():
                        parsed_data["wait"][thread] = {"data" : [], "average" : 0., "derivation" : 0.}
                    parsed_data["wait"][thread]["data"].append(value)
                else:
                    parsed_data[known_keys[kkey]]["data"].append(value)
                found = True

        if not found:
            print("Unknown key: ", line)

    for value in known_keys.values():
        if value == "wait":
            for thread in parsed_data["wait"].keys():
                parsed_data["wait"][thread]["average"] = np.average(parsed_data["wait"][thread]["data"])
                parsed_data["wait"][thread]["derivation"] = np.sqrt(np.var(parsed_data["wait"][thread]["data"]))
        else:
            parsed_data[value]["average"] = np.average(parsed_data[value]["data"])
            parsed_data[value]["derivation"] = np.sqrt(np.var(parsed_data[value]["data"]))

    
    return parsed_data



def parse_cmd():
    config_path = ""
    output_path = ""

    for i, arg in enumerate(sys.argv):
        if i < len(sys.argv) - 1:
            if arg == "--config":
                config_path = sys.argv[i+1]
            elif arg == "--output":
                output_path = sys.argv[i+1].strip("/")

    if config_path == "" or output_path == "":
        print("Error: config or output dir not specified: Use --config config/parameters.json and --output output_dir")
        sys.exit(1)

    output_path += "/"

    return config_path, output_path


if __name__ == "__main__":
    main()
