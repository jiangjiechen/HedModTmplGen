"""Module defining various utilities."""
from monmt.utils.misc import aeq, use_gpu
from monmt.utils.report_manager import ReportMgr, build_report_manager
from monmt.utils.statistics import Statistics
from monmt.utils.optimizers import build_optim, MultipleOptimizer, \
	Optimizer, AdaFactor

__all__ = ["aeq", "use_gpu", "ReportMgr", "loss",
           "build_report_manager", "Statistics",
           "build_optim", "MultipleOptimizer", "Optimizer", "AdaFactor"]
