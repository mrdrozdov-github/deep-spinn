"""
logging.py

Log format convenience methods for training spinn.

"""

import numpy as np
from spinn.util.blocks import flatten
from spinn.util.misc import time_per_token


def train_accumulate(model, data_manager, A, batch):

    X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = batch


def train_metrics(M, stats_args, step):
    metric_stats = ['class_acc', 'total_loss', 'transition_acc', 'transition_loss']
    for key in metric_stats:
        M.write(key, stats_args[key], step)


def train_stats(model, optimizer, A, step):

    time_metric = time_per_token(A.get('total_tokens'), A.get('total_time'))

    ret = dict(
        step=step,
        class_acc=A.get_avg('class_acc'),
        transition_acc=A.get_avg('transition_acc'),
        xent_loss=A.get_avg('xent_loss'),  # not actual mean
        transition_loss=model.transition_loss.data[0],
        total_loss=A.get_avg('total_loss'),
        auxiliary_loss=A.get_avg('auxiliary_loss'),
        l2_loss=A.get_avg('l2_loss'),  # not actual mean
        learning_rate=optimizer.lr,
        time=time_metric,
    )

    return ret


def train_format(model):

    stats_str = "Step: {step}"

    # Accuracy Component.
    stats_str += " Acc: {class_acc:.5f} {transition_acc:.5f}"

    # Cost Component.
    stats_str += " Cost: {total_loss:.5f} {xent_loss:.5f} {transition_loss:.5f} {l2_loss:.5f}"

    # Time Component.
    stats_str += " Time: {time:.5f}"

    return stats_str


def train_extra_format(model):

    # Extra Component.
    extra_str = "Train Extra:"
    extra_str += " lr{learning_rate:.7f}"

    return extra_str


def eval_accumulate(model, data_manager, A, batch):

    X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = batch


def eval_format(model):
    eval_str = "Step: {step} Eval acc: {class_acc:.5f} {transition_acc:.5f} {filename} Time: {time:.5f}"

    return eval_str


def eval_extra_format(model):
    extra_str = "Eval Extra:"

    return extra_str


def eval_metrics(M, stats_args, step):
    metric_stats = ['class_acc', 'transition_acc']
    for key in metric_stats:
        M.write("eval_" + key, stats_args[key], step)


def eval_stats(model, A, step):

    class_correct = A.get('class_correct')
    class_total = A.get('class_total')
    class_acc = sum(class_correct) / float(sum(class_total))

    time_metric = time_per_token(A.get('total_tokens'), A.get('total_time'))

    ret = dict(
        step=step,
        class_acc=class_acc,
        transition_acc=A.get_avg('transition_acc'),
        time=time_metric,
    )

    return ret
