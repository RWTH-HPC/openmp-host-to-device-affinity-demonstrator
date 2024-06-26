import argparse
import os, sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import statistics as st

class TestCase:
    def __init__(self, line: str) -> None:
        self.name = line
        spl1 = line.split(" ")
        spl2 = spl1[0].split("_")

        self.n_tasks    = int(spl1[-1])
        self.proc_bind  = spl2[0]
        self.n_threads  = int(spl2[1])
        self.m_size     = int(spl2[2])

class CVersion:
    def __init__(self) -> None:
        self.data       = None
        self.test_cases = []

    def parse_data(self, folder_path, folder_name, args) -> bool:
        path_benchmark_res = os.path.join(folder_path, f"{args.configuration}_parsed.json")
        if not os.path.exists(path_benchmark_res):
            return False

        spl = folder_name.split("_")
        self.exe_id             = "-".join(spl[1:])
        self.has_numa_balancing = int(spl[0][-1])==1
        self.has_compute        = int(spl[1][-1])==1
        self.is_async           = int(spl[2][-1])==1
        self.has_pinned_mem     = int(spl[3][-1])==1
        self.has_unified_mem    = int(spl[4][-1])==1
        
        with open(path_benchmark_res, 'r') as f:
            self.data       = json.loads(f.read())
            self.test_cases = [TestCase(x) for x in list(self.data.keys())]

        return True

    def create_title(self):
        title = ""
        title += "w/ computation, " if self.has_compute else "w/o computation, "
        title += "w/ async, " if self.is_async else "w/o async, "
        title += "w/ pinned memory, " if self.has_pinned_mem else "w/o pinned memory, "
        title += "w/ unified memory " if self.has_unified_mem else "w/o unified memory "
        return title

def plot_data(grp: list[CVersion], args, threshold_limits):
    min_m_size  = threshold_limits[0]
    max_m_size  = threshold_limits[1]
    tmp_ext     = "large-sizes" if max_m_size == sys.maxsize else "small-sizes"
    tmp_max_val = "inf" if max_m_size == sys.maxsize else max_m_size

    cases = sorted(grp[0].test_cases, key=lambda x: x.m_size, reverse=False)
    cases = [x for x in cases if x.m_size >= min_m_size and x.m_size < max_m_size]
    mb_per_thr = [x.m_size*x.m_size*8/(1000**2)*3 for x in cases]

    # TODO: current assuming fixed n_thr
    n_thr = grp[0].test_cases[0].n_threads
        
    fig         = plt.figure(constrained_layout=True, figsize=(12,9))
    gs          = GridSpec(3, 2, figure=fig)
    ax_all      = fig.add_subplot(gs[:2, :2])
    ax_allocate = fig.add_subplot(gs[2,0])
    ax_compute  = fig.add_subplot(gs[2,1])

    path_stats = os.path.join(args.destination, f"stats.log")

    for ver in grp:
        cur_txt_ext         = "w/ numa_balancing" if ver.has_numa_balancing else ""
        cur_linestyle       = "-" if ver.has_numa_balancing else "--"
        allocation_best     = np.array([ver.data[c.name]["best"]["allocation"]["average"] for c in cases])
        execution_best      = np.array([ver.data[c.name]["best"]["computation"]["average"] for c in cases])
        u_allocation_best   = np.array([ver.data[c.name]["best"]["allocation"]["derivation"] for c in cases])/np.sqrt(2)
        u_exection_best     = np.array([ver.data[c.name]["best"]["computation"]["derivation"] for c in cases])/np.sqrt(2)

        allocation_worst    = np.array([ver.data[c.name]["worst"]["allocation"]["average"] for c in cases])
        execution_worst     = np.array([ver.data[c.name]["worst"]["computation"]["average"] for c in cases])
        u_allocation_worst  = np.array([ver.data[c.name]["worst"]["allocation"]["derivation"] for c in cases])/np.sqrt(2)
        u_exection_worst    = np.array([ver.data[c.name]["worst"]["computation"]["derivation"] for c in cases])/np.sqrt(2)

        ax_allocate.errorbar(mb_per_thr, allocation_best/n_thr,  yerr=u_allocation_best/n_thr,  marker='o', linestyle=cur_linestyle, label=f'optimal distribution {cur_txt_ext}')
        ax_allocate.errorbar(mb_per_thr, allocation_worst/n_thr, yerr=u_allocation_worst/n_thr, marker='o', linestyle=cur_linestyle, label=f'suboptimal distribution {cur_txt_ext}')
        ax_compute.errorbar( mb_per_thr, execution_best/n_thr,   yerr=u_exection_best/n_thr,    marker='o', linestyle=cur_linestyle, label=f'optimal distribution {cur_txt_ext}')
        ax_compute.errorbar( mb_per_thr, execution_worst/n_thr,  yerr=u_exection_worst/n_thr,   marker='o', linestyle=cur_linestyle, label=f'suboptimal distribution {cur_txt_ext}')
        ax_all.errorbar(     mb_per_thr, (allocation_best + execution_best)/n_thr,   yerr=np.sqrt(u_allocation_best**2+u_exection_best**2)/n_thr,   marker='o', linestyle=cur_linestyle, label=f'Allocation + Computation time closest {cur_txt_ext}')
        ax_all.errorbar(     mb_per_thr, (allocation_worst + execution_worst)/n_thr, yerr=np.sqrt(u_allocation_worst**2+u_exection_worst**2)/n_thr, marker='o', linestyle=cur_linestyle, label=f'Allocation + Computation time furthest {cur_txt_ext}')

        # writing statistics to file
        # TODO: create proper csv file with all data (absolute, speedups, statistics)
        with open(path_stats, mode="a") as f_stats:
            f_stats.write(f"Statistics (Computation): {ver.exe_id} {cur_txt_ext} ({tmp_ext})\n")
            tmp_improv  = [float((execution_worst[x]-execution_best[x])/execution_best[x]) for x in range(len(execution_worst))]
            improv_mean = improv_min = improv_max = 0
            if len(tmp_improv) > 0:
                improv_mean = st.mean(tmp_improv)
                improv_min = min(tmp_improv); i_min = tmp_improv.index(improv_min)
                improv_max = max(tmp_improv); i_max = tmp_improv.index(improv_max)
                f_stats.write(f'Min  relative difference {"{:8.2f}".format(mb_per_thr[i_min])} MB, {"{:8.3f}".format(improv_min*100)} %\n')
                f_stats.write(f'Mean relative difference {"{:8.2f}".format(-1)} MB, {"{:8.3f}".format(improv_mean*100)} %\n')
                f_stats.write(f'Max  relative difference {"{:8.2f}".format(mb_per_thr[i_max])} MB, {"{:8.3f}".format(improv_max*100)} %\n')
            else:
                f_stats.write(f'Min  relative difference {"{:8.2f}".format(-1)} MB, {"{:8.3f}".format(improv_min*100)} %\n')
                f_stats.write(f'Mean relative difference {"{:8.2f}".format(-1)} MB, {"{:8.3f}".format(improv_mean*100)} %\n')
                f_stats.write(f'Max  relative difference {"{:8.2f}".format(-1)} MB, {"{:8.3f}".format(improv_max*100)} %\n')
            f_stats.write("\n")
    
    fig.suptitle(ver.create_title() + f", m_sizes [{min_m_size}, {tmp_max_val}]")
    ax_all.set_title("Allocation + Computation")
    ax_all.set_xlabel("Copied data per task [MB]")
    # ax_allocate.set_ylabel("Duration per task / s") # TODO: ist das richtig???
    ax_all.set_ylabel("Avg duration [sec]")
    ax_all.grid()
    ax_all.legend()
    
    ax_allocate.set_title("Allocation only")
    ax_allocate.set_xlabel("Copied data per task [MB]")
    # ax_allocate.set_ylabel("Duration per task / s") # TODO: ist das richtig???
    ax_allocate.set_ylabel("Avg duration [sec]")
    ax_allocate.grid()
    
    ax_compute.set_title("Workphase only")
    ax_compute.set_xlabel("Copied data per task [MB]")
    # ax_allocate.set_ylabel("Duration per task / s") # TODO: ist das richtig???
    ax_compute.set_ylabel("Avg duration [sec]")
    ax_compute.grid()

    if args.interactive:
        plt.show()
    fig.savefig(os.path.join(args.destination, f"plot_{grp[0].exe_id}_{tmp_ext}.png"), dpi='figure', format="png", bbox_inches="tight", pad_inches=0.1, facecolor='w', edgecolor='w', transparent=False)
    plt.close(fig)

def group_versions_by_id(versions):
    ret = []
    unique_ids = list(set([x.exe_id for x in versions]))
    for id in unique_ids:
        tmp_list = [x for x in versions if x.exe_id == id]
        ret.append(tmp_list)
    return ret

def main(args):
    versions = []
    for item in os.listdir(args.source):
        tmp_path = os.path.join(args.source, item)
        if os.path.isdir(tmp_path):
            print(tmp_path)
            obj = CVersion()
            if obj.parse_data(tmp_path, item, args):
                versions.append(obj)

    # group version results with same ID (w/o and w/ numa balancing)
    v_groups = group_versions_by_id(versions)

    for grp in v_groups:
        # plot small sizes
        plot_data(grp, args, (0, args.plot_threshold))
        # plot larger sizes
        plot_data(grp, args, (args.plot_threshold, sys.maxsize))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parses and summarizes execution time measurement results")
    parser.add_argument("-s", "--source",         required=True,  type=str, metavar="<folder path>", help=f"source folder containing outputs")
    parser.add_argument("-d", "--destination",    required=False, type=str, metavar="<folder path>", default=None, help=f"destination folder where resulting plots will be stored")
    parser.add_argument("-c", "--configuration",  required=False, type=str, default="memory_benchmark", help=f"name of the configuration")
    parser.add_argument("-i", "--interactive",    required=False, type=int, default=0, help=f"Whether open interactive plots. Requires graphical interface.")
    parser.add_argument("-t", "--plot_threshold", required=False, type=int, metavar="<threshold>", default=1024, help=f"threshold for plots to split between large and small sizes")
    args = parser.parse_args()

    # ========== DEBUGGING ==========
    # args.source = "C:\\J.Klinkenberg.Local\\repos\\hpc-research\\openmp\\host-to-device-affinity\\data-demonstrator-host-to-device-affinity\\results"
    # ===============================

    if not os.path.exists(args.source):
        print(f"Source folder path \"{args.source}\" does not exist")
        sys.exit(1)

    if args.destination is None:
        args.destination = os.path.join(args.source, "plots")

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)

    main(args)