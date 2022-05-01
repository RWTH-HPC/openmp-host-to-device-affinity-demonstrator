#!/bin/env python3

import sys
import json
import os
from time import time
import subprocess
import numpy as np

def main():
    config_path, output_path, binary_path, numa_balancing = parse_cmd()
    
    config = {}
    with open(config_path, "r") as config_file:
        config = json.loads(config_file.read())

    if not os.path.exists(output_path + "raw"):
        os.mkdir(output_path + "raw")

    #generate data

    guess = 90
    for test in config["tests"]:
        test_output_name = "_".join([str(test[key]).replace(" ", "-") for key in test.keys() if not key == "OMP_PLACES"])
        best_output = b""
        worst_output = b""

        print("Executing test", test_output_name)
        test["best_result"] = []

        guess *= 2

        for i in range(config["repetitions"]):
            print(i, end="")
            sys.stdout.flush()
            cmd_best = [binary_path + "distanceBenchmark_best"] + test["parameters"].split(" ")
            cmd_worst = [binary_path + "distanceBenchmark_worst"] + test["parameters"].split(" ")
            if not numa_balancing:
                cmd_best = ["no_numa_balancing"] + cmd_best
                cmd_worst = ["no_numa_balancing"] + cmd_worst
            env = os.environ
            for key in test.keys():
                if "OMP" in key:
                    env[key] = str(test[key])

            is_stuck = True
            while is_stuck:
                try:
                    start = time()
                    tmp_output = subprocess.check_output(cmd_best, env=env, timeout=guess)
                    guess = int(min(guess, 2*(time()-start))) + 1
                    best_output += tmp_output
                    is_stuck = False
                except subprocess.TimeoutExpired:
                    print("+", end="")

            is_stuck = True
            while is_stuck:
                try:
                    start = time()
                    tmp_output = subprocess.check_output(cmd_worst, env=env, timeout=guess)
                    guess = int(min(guess, 2*(time()-start))) + 1
                    worst_output += tmp_output
                    is_stuck = False
                except subprocess.TimeoutExpired:
                    print("-", end="")
            print(" ", end="")
        print("")

        test["best_result"] = best_output.decode("UTF-8");
        test["worst_result"] = worst_output.decode("UTF-8");
        with open(output_path + "raw/" + config["name"] + test_output_name + "_best.output", "w") as output_file:
            output_file.write(test["best_result"])
        with open(output_path + "raw/" + config["name"] + test_output_name + "_worst.output", "w") as output_file:
            output_file.write(test["worst_result"])

    print("Raw output files can be found at", output_path + "raw/" + config["name"] + "_*.output");
    
    results = {}
    for test in config["tests"]:
        test_output_name = "_".join([str(test[key]) for key in test.keys() if not key == "OMP_PLACES" and not "result" in key ])
        results[test_output_name] = { "best" : parse_results(test["best_result"]), "worst" : parse_results(test["worst_result"]) }
        results[test_output_name]["config"] = {item[0] : item[1] for item in test.items() if not "result" in item[0]}

    with open(output_path + config["name"] + "_parsed.json", "w") as result_file:
        result_file.write(json.dumps(results, indent=4))
    print("Parsed json output file can be found at", output_path + config["name"] + "_parsed.json");
        


def parse_results(results):
    parsed_data = {}
    known_keys = {
            "CUDA device distance initalization was successful and took" : "init",
            "Memory Allocation duration" : "allocation",
            "Computations with normal tasking took" : "computation",
            "Invocation latency of thread" : "wait",
            #"Ratio longest waiting time / shortest waiting time" : "ratio"
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
    binary_path = ""
    numa_balancing = True

    for i, arg in enumerate(sys.argv):
        if i < len(sys.argv) - 1:
            if arg == "--config":
                config_path = sys.argv[i+1]
            elif arg == "--output":
                output_path = sys.argv[i+1].strip("/")
            elif arg == "--binary":
                binary_path = sys.argv[i+1].strip("/")
            elif arg == "--no_numa_balancing":
                numa_balancing = False

    if config_path == "" or output_path == "" or binary_path == "":
        print("Error: config or output dir not specified: Use --config config/parameters.json and --output output_dir and --binary binary_path")
        sys.exit(1)

    output_path += "/"
    binary_path += "/"

    return config_path, output_path, binary_path, numa_balancing


if __name__ == "__main__":
    main()
