import ioh
import argparse
from strategies import firefly_algorithm, greedy_mls, mls, pso


if __name__ == "__main__":
    ndim = 10
    niter = 10
    # parse arg that indicate the optimization algorithgm
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PSO")
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()
    for fid in range(1, 25):
        prob = ioh.get_problem(fid, 1, ndim, ioh.ProblemClass.BBOB)
        if args.algo == "pso":
            l = ioh.logger.Analyzer(
                root="BBOB/PSO/{}_{}".format(fid, ndim),
                folder_name="BBOB_PSO",
                algorithm_name="PSO",
                algorithm_info="PSO from kernel_tuner"
            )
            prob.attach_logger(l)
            for _ in range(niter):
                pso.tune(prob, verbose=args.verbose)
                print("Problem ID: {}, {}/{}".format(fid, _, niter))
                print(prob.state)
                prob.reset()
            del l
        elif args.algo == "firefly_algorithm":
            l = ioh.logger.Analyzer(
                root="BBOB/firefly_algorithm/{}_{}".format(fid, ndim),
                folder_name="BBOB_firefly_algorithm",
                algorithm_name="firefly_algorithm",
                algorithm_info="firefly_algorithm from kernel_tuner"
            )
            prob.attach_logger(l)
            for _ in range(niter):
                firefly_algorithm.tune(prob, verbose=args.verbose)
                print("Problem ID: {}, {}/{}".format(fid, _, niter))
                print(prob.state)
                prob.reset()
            del l
        elif args.algo == "greedy_mls":
            l = ioh.logger.Analyzer(
                root="BBOB/greedy_mls/{}_{}".format(fid, ndim),
                folder_name="BBOB_greedy_mls",
                algorithm_name="greedy_mls",
                algorithm_info="greedy_mls from kernel_tuner"
            )
            prob.attach_logger(l)
            for _ in range(niter):
                greedy_mls.tune(prob, verbose=args.verbose)
                print("Problem ID: {}, {}/{}".format(fid, _, niter))
                print(prob.state)
                prob.reset()
            del l
        elif args.algo == "mls":
            l = ioh.logger.Analyzer(
                root="BBOB/mls/{}_{}".format(fid, ndim),
                folder_name="BBOB_mls",
                algorithm_name="mls",
                algorithm_info="mls from kernel_tuner"
            )
            prob.attach_logger(l)
            for _ in range(niter):
                mls.tune(prob, verbose=args.verbose)
                print("Problem ID: {}, {}/{}".format(fid, _, niter))
                print(prob.state)
                prob.reset()
            del l
