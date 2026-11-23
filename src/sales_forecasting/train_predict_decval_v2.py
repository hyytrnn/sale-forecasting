import sys

from .train_predict_decval import main


DEFAULT_ARGS = [
    "--run-name",
    "decval_v2",
    "--validation-months",
    "12",
    "--regressor-target",
    "raw",
    "--regressor-loss",
    "MAPE",
    "--no-use-bucket-model",
    "--threshold-grid",
    "0.0",
    "--scale-grid",
    "0.52",
    "0.55",
    "0.58",
    "0.61",
    "0.64",
    "0.67",
    "--blend-grid",
    "0.0",
    "--bucket-blend-grid",
    "0.0",
    "--proba-power-grid",
    "0.0",
    "--final-threshold",
    "0.0",
    "--final-scale",
    "0.67",
    "--final-blend",
    "0.0",
    "--final-bucket-blend",
    "0.0",
    "--final-proba-power",
    "0.0",
]


def has_flag(flag: str) -> bool:
    return flag in sys.argv[1:]


if __name__ == "__main__":
    # Keep user-provided flags authoritative. These defaults define the v2 run.
    injected = []
    i = 0
    while i < len(DEFAULT_ARGS):
        flag = DEFAULT_ARGS[i]
        values = []
        i += 1
        while i < len(DEFAULT_ARGS) and not DEFAULT_ARGS[i].startswith("--"):
            values.append(DEFAULT_ARGS[i])
            i += 1
        if not has_flag(flag):
            injected.extend([flag, *values])
    sys.argv = [sys.argv[0], *injected, *sys.argv[1:]]
    main()