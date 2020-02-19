"""
A variety of visualizations for the output of models. In general, these functions
shouldn't do any heavy lifting in series manipulation and should concern themselves
with how to construct and arrange visualizations on the screen. Series manipulation or
stats functions should be offloaded to ml.stats or sklearn.

Building block level functions should take (preds|probs)/rels pairs, and functions
operating on lists of results objects can take the raw lists (for now).
"""

from collections import defaultdict
import math

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow as tf

import ml.data.conversion as conversion
import ml.stats.curves
import ml.stats.binary_metrics
import ml.stats.metrics

MULTI_RESULT_STAT_RANGE = 0.3
MULTI_RESULT_STAT_MIN = 0.4

ROC_CURVE_LINES = [
    ['aqua', 'solid', 3],
    ['darkorange', 'solid', 3],
    ['cornflowerblue', 'solid', 3],
    ['aqua', 'dotted', 6],
    ['darkorange', 'dotted', 6],
    ['cornflowerblue', 'dotted', 6],
]

# Baseball cards for WorkSpec runs.

def regressor_baseball_card(results, dataset_name='test'):
    """
    A simple one-page graphical summary of a regressor result. Includes:
    - Confusion Matrix
    - Cost/ACC/PPV/NPV/TPR/TNR table
    - Acc over time series
    - Acc over time hist
    - Raw time series predictions/reality

    Todo:
      - Cost vs. baseline
      - R2 and other error vs. variance measures
    """

    rels = conversion.decimal_labels_to_binary_categorical(
        results[dataset_name]['series']['reality'],
    )

    preds = conversion.decimal_labels_to_binary_categorical(
        results[dataset_name]['series']['predictions'],
    )

    reset_current_plot()

    plt.subplot(231)
    plot_confusion_matrix(preds, rels, normalize=True)

    plt.subplot(234)
    plot_basic_results_table(preds, rels)

    plt.subplot(232)
    plot_acc_over_time_chart(preds, rels, 60)

    plt.subplot(235)
    plot_acc_over_time_hist(preds, rels, 60)

    plt.subplot(133)
    pd.Series(results[dataset_name]['series']['reality'], name='Reality').plot()
    pd.Series(results[dataset_name]['series']['predictions'], name='Predictions').plot()
    plt.legend()

    if 'spec_info' in results:
        card_title = '%s %s %s' % (
            results['spec_info']['work_spec_name'],
            results['spec_info']['work_unit_name'],
            results['spec_info']['model_spec_name'],
        )

        plt.suptitle(card_title, size='large')


def classifier_baseball_card(results, dataset_name='test'):
    """
    A simple one-page graphical summary of a classifier result. Includes:
    - Confusion Matrix
    - ACC/PPV/NPV/TPR/TNR table
    - Precision curves
    - ROC Curve
    - Acc over time series
    - Acc over time hist

    Todo:
      - underlying series
    """

    rels = results[dataset_name]['series']['reality']
    preds = results[dataset_name]['series']['predictions']
    probs = results[dataset_name]['series']['probabilities']

    reset_current_plot()

    plt.subplot(231)
    plot_confusion_matrix(preds, rels, normalize=True)

    plt.subplot(234)
    plot_basic_results_table(probs, preds, rels)

    plt.subplot(232)
    plot_acc_over_time_chart(preds, rels, 60)

    plt.subplot(235)
    plot_acc_over_time_hist(preds, rels, 60)

    plt.subplot(233)
    plot_likelihood_curves(probs, rels)
    plt.grid()

    plt.subplot(236)
    plot_roc_curve(probs, rels)

    """
    plt.subplot(221)
    plot_acc_over_time_chart(preds, rels, 60)
    plt.title('')

    plt.subplot(223)
    plot_acc_over_time_hist(preds, rels, 60)
    plt.title('')

    plt.subplot(222)
    plot_likelihood_curves(probs, rels)
    plt.grid()
    plt.title('')

    plt.subplot(224)
    plot_roc_curve(probs, rels)
    plt.title('')
    """
    if 'spec_info' in results:
        card_title = '%s %s %s' % (
            results['spec_info']['work_spec_name'],
            results['spec_info']['work_unit_name'],
            results['spec_info']['model_spec_name'],
        )

        plt.suptitle(card_title, size='large')


def step_through_classifier_cards(multi_results, dataset_name='test'):
    """
    Little helper function that lets us just plot one baseball card after the other
    in a list.
    """

    i = 0;

    while True:
        plt.subplot(111)
        plt.cla()

        classifier_baseball_card(multi_results[i], dataset_name)

        option = raw_input('Next (n)/Back (b)/Quit (q): ')

        if option == 'n':
            i += 1
            continue
        elif option == 'b':
            i -= 1
            continue
        elif option == 'q':
            break

        if i == len(multi_results):
            break


# Multi results functions.

def get_summary_metrics_for_classifier_result(probs, preds, rels):
    """
    Take a single classifier result and get a list of summary results for it with
    heading names.
    """

    num_classes = probs.shape[1]
    class_ids = range(num_classes)

    summary_metrics = []
    col_labels = []

    col_labels.append('ACC')
    summary_metrics.append(ml.stats.metrics.acc(preds, rels))

    # Precisions (PPV, etc.)
    for class_id in class_ids:
        class_pv = ml.stats.metrics.predictive_value_for_class(
            preds,
            rels,
            class_id,
        )

        summary_metrics.append(class_pv)

        col_labels.append('PV%s' % class_id)

    # Sensitivities (TPR, etc.)
    for class_id in class_ids:
        class_pr = ml.stats.metrics.positive_rate_for_class(
            preds,
            rels,
            class_id,
        )

        summary_metrics.append(class_pr)

        col_labels.append('PR%s' % class_id)


    # TODO: F-score
    # this_f = sklearn.metrics.f1_score(rels, preds)

    # LRPs
    for class_id in class_ids:
        class_lr = ml.stats.metrics.class_lr(
            preds,
            rels,
            class_id,
        )

        summary_metrics.append(class_lr)

        col_labels.append('LR%s' % class_id)

    return summary_metrics, col_labels


def plot_multi_basic_results_table(multi_results, dataset_name='test', col_headings=True, row_headings=False):
    """
    Simple function just plots the basic classification metrics in a table.
    """

    row_data = []
    row_labels = []
    col_labels = []

    for results in multi_results:
        row_label = results['spec_info']['model_spec_name']
        row_labels.append(row_label)

        rels = results[dataset_name]['series']['reality']
        preds = results[dataset_name]['series']['predictions']
        probs = results[dataset_name]['series']['probabilities']

        summary_stats, col_labels = get_summary_metrics_for_classifier_result(
            probs,
            preds,
            rels,
        )

        row_data.append(summary_stats)

    celldata = np.array(row_data)
    celldata = np.around(celldata, 3)
    cell_colours = plt.cm.viridis(celldata)

    table = plt.table(
        cellText=celldata,
        colColours=np.zeros((len(col_labels), 3)),
        colLabels=col_labels if col_headings is True else None,
        rowLabels=row_labels if row_headings is True else None,
        cellColours=cell_colours,
        bbox=[0, 0, 1, 1],
        loc='center',
    )

    for key, cell in table.get_celld().items():
        cell.set_text_props(fontproperties=FontProperties(size='large'), color='white')
        cell.set_linewidth(0)

    plt.xticks([])
    plt.yticks([])


def plot_work_unit_results_table(multi_results, dataset_name='test'):
    reset_current_plot()

    work_unit_name = multi_results[0]['spec_info']['work_unit_name']

    plot_multi_basic_results_table(
        multi_results,
        dataset_name,
        col_headings=True,
        row_headings=True,
    )

    plt.suptitle(work_unit_name, size='large')


def plot_work_spec_basic_results_table(multi_results, dataset_name='test'):
    reset_current_plot()

    super_title = multi_results.values()[0][0]['spec_info']['work_spec_name']

    row_data = []
    num_units = len(multi_results) + 1

    i = 1

    for work_unit_name, rs in sorted(multi_results.items(), key=lambda x: x[0]):
        ax = plt.subplot(num_units, 1, i)
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')

        plot_multi_basic_results_table(rs, col_headings=(i == 1))
        plt.ylabel(work_unit_name.replace('_', '\n'), rotation=0, fontsize='x-large')
        plt.gca().yaxis.set_label_coords(-0.05, 0.15)

        i = i + 1

    plt.suptitle(super_title, size='large')


def plot_probability_histograms(multi_results, dataset_name='test'):
    """
    Quick look at the distribution of negative probabilities in a binary multi_result.
    """

    num_results = len(multi_results)
    columns = 1
    rows = num_results

    for i, result in enumerate(multi_results):
        probs = result['test']['series']['probabilities']
        rels = result['test']['series']['reality']

        plt.subplot(rows, columns, i + 1)
        plt.grid()
        pd.Series(probs[:, 0]).hist(bins=100)
        plt.xlim(0, 1)


def plot_multi_likelihood_curves(multi_results, dataset_name='test', idx=None):
    """
    A quick way to look at all the likelihood curves from a work unit on a given index.
    Quick as in simple, not fast, takes minutes still.
    """

    num_results = len(multi_results)
    columns = math.ceil(num_results ** 0.5)
    rows = columns

    for i, result in enumerate(multi_results):
        probs = result['test']['series']['probabilities']
        rels = result['test']['series']['reality']

        likelihoods = ml.stats.curves.likelihoods_at_freq(probs, rels, idx)

        plt.subplot(rows, columns, i + 1)
        likelihoods.plot(ax=plt.gca())
        plt.grid()


def plot_multi_precision_curves(multi_results, dataset_name='test'):
    for result in multi_results:
        rels = result[dataset_name]['series']['reality']
        probs = result[dataset_name]['series']['probabilities']

        accs, ppvs, npvs, num_preds = ml.stats.curves.precisions_and_frequency_at_z(
            probs,
            rels,
        )

        plt.subplot(221)
        accs.plot()
        plt.subplot(222)
        ppvs.plot()
        plt.subplot(223)
        npvs.plot()
        plt.subplot(224)
        num_preds.plot()

    plt.subplot(221)
    plt.title('ACC')
    plt.subplot(222)
    plt.title('PPV')
    plt.subplot(223)
    plt.title('NPV')
    plt.subplot(224)
    plt.title('NUM')


def plot_multi_ensemble_results_table(multi_results, dataset_name='test'):
    """
    Plot a heatmap of the performance of the different nets in the ensemble.
    """

    accs = pd.Series([r[dataset_name]['summary']['acc'] for r in multi_results])
    tprs = pd.Series([r[dataset_name]['summary']['tpr'] for r in multi_results])
    tnrs = pd.Series([r[dataset_name]['summary']['tnr'] for r in multi_results])
    ppvs = pd.Series([r[dataset_name]['summary']['ppv'] for r in multi_results])
    npvs = pd.Series([r[dataset_name]['summary']['npv'] for r in multi_results])

    acc_colours = (accs - MULTI_RESULT_STAT_MIN) / MULTI_RESULT_STAT_RANGE
    tpr_colours = (tprs - MULTI_RESULT_STAT_MIN) / MULTI_RESULT_STAT_RANGE
    tnr_colours = (tnrs - MULTI_RESULT_STAT_MIN) / MULTI_RESULT_STAT_RANGE
    ppv_colours = (ppvs - MULTI_RESULT_STAT_MIN) / MULTI_RESULT_STAT_RANGE
    npv_colours = (npvs - MULTI_RESULT_STAT_MIN) / MULTI_RESULT_STAT_RANGE

    max_acc = accs.max()
    max_tpr = tprs.max()
    max_tnr = tnrs.max()
    max_ppv = ppvs.max()
    max_npv = npvs.max()

    cell_colours = plt.cm.RdYlGn(zip(
        acc_colours,
        tpr_colours,
        tnr_colours,
        ppv_colours,
        npv_colours,
    ))

    acc_str = ['> %.4f' % a if a == max_acc else '%.4f' % a for a in accs]
    tpr_str = ['> %.4f' % a if a == max_tpr else '%.4f' % a for a in tprs]
    tnr_str = ['> %.4f' % a if a == max_tnr else '%.4f' % a for a in tnrs]
    ppv_str = ['> %.4f' % a if a == max_ppv else '%.4f' % a for a in ppvs]
    npv_str = ['> %.4f' % a if a == max_npv else '%.4f' % a for a in npvs]

    celldata = zip(acc_str, tpr_str, tnr_str, ppv_str, npv_str)

    column_labels = ['ACC', 'TPR', 'TNR', 'PPV', 'NPV']

    plt.ion()
    plt.cla()

    plt.subplot(231)

    plt.title('Absolute')

    table = plt.table(
        cellText=celldata,
        cellColours=cell_colours,
        colLabels=column_labels,
        bbox=[0, 0, 1, 1],
        loc='center',
    )

    for key, cell in table.get_celld().items():
        cell.set_text_props(fontproperties=FontProperties(size='xx-large'))

    plt.xticks([])
    plt.yticks([])

    plt.subplot(232)

    acc_colours = (accs - accs.min()) / (accs.max() - accs.min())
    tpr_colours = (tprs - tprs.min()) / (tprs.max() - tprs.min())
    tnr_colours = (tnrs - tnrs.min()) / (tnrs.max() - tnrs.min())
    ppv_colours = (ppvs - ppvs.min()) / (ppvs.max() - ppvs.min())
    npv_colours = (npvs - npvs.min()) / (npvs.max() - npvs.min())

    cell_colours = plt.cm.RdYlGn(zip(
        acc_colours,
        tpr_colours,
        tnr_colours,
        ppv_colours,
        npv_colours,
    ))

    plt.title('Relative')

    table = plt.table(
        cellText=celldata,
        cellColours=cell_colours,
        colLabels=column_labels,
        bbox=[0, 0, 1, 1],
        loc='center',
    )

    for key, cell in table.get_celld().items():
        cell.set_text_props(fontproperties=FontProperties(size='xx-large'))

    plt.xticks([])
    plt.yticks([])

    plt.subplot(234)
    plt.title('ACC Curve')
    plt.hlines(0.60, 0.5, 0.7)
    plt.subplot(235)
    plt.title('Signal Freq')
    plt.grid()

    plt.subplot(5, 3, 3)
    plt.title('ACC')
    plt.xlim((0,1))
    accs.hist(bins=20, color='peru')

    plt.subplot(5, 3, 6)
    plt.title('TPR')
    plt.xlim((0,1))
    tprs.hist(bins=20, color='peru')

    plt.subplot(5, 3, 9)
    plt.title('TNR')
    plt.xlim((0,1))
    tnrs.hist(bins=20, color='peru')

    plt.subplot(5, 3, 12)
    plt.title('PPV')
    plt.xlim((0,1))
    ppvs.hist(bins=20, color='peru')

    plt.subplot(5, 3, 15)
    plt.title('NPV')
    plt.xlim((0,1))
    npvs.hist(bins=20, color='peru')

    for result in multi_results:
        accs, ppvs, npvs, num_preds = ml.stats.curves.precisions_and_frequency_at_z(
            result,
            dataset_name,
        )

        plt.subplot(234)
        accs.plot()
        plt.subplot(235)
        num_preds.plot()

    plt.subplot(234)
    plt.grid()
    plt.subplot(235)
    plt.grid()


# Single results functions.

def plot_basic_results_table(probs, preds, rels):
    """
    Simple function just plots the basic classification metrics in a table.
    TODO: Probably make the row/column labels to be cells themselves so they don't
    jump out of the subplot's container.
    """

    summary_stats, row_labels = get_summary_metrics_for_classifier_result(
        probs,
        preds,
        rels,
    )

    celldata = np.array([summary_stats]).T
    celldata = np.around(celldata, 3)

    table = plt.table(
        cellText=celldata,
        rowLabels=row_labels,
        bbox=[0, 0, 1, 1],
        loc='center',
    )

    for key, cell in table.get_celld().items():
        cell.set_text_props(fontproperties=FontProperties(size='large'))

    plt.xticks([])
    plt.yticks([])


def plot_confusion_matrix(preds, rels, normalize=False):
    """
    Plots the confusion matrix of a classification result in a table. Our plotted
    confusion matrix has an extra column on the end for the prevalence of each class in
    the datset, and an extra row for the prevalence of each class in our predictions.

    TODO: Probably make the row/column labels to be cells themselves so they don't
    jump out of the subplot's container.
    """

    confusion_matrix = sklearn.metrics.confusion_matrix(rels, preds)

    num_classes = confusion_matrix.shape[1]
    class_ids = range(num_classes)
    total_samples = len(rels)

    celldata = confusion_matrix

    # Sum the rows and add them as a column.
    row_sums = np.array([[row.sum()] for row in confusion_matrix])
    celldata = np.hstack([celldata, row_sums])

    # Sum the columns and add them as a row.
    col_sums = np.array([col.sum() for col in celldata.T])
    celldata = np.vstack([celldata, col_sums])

    cell_colours = np.around(celldata / float(total_samples), 2)
    cell_colours[num_classes: ] = 0
    cell_colours[:, num_classes] = 0
    cell_colours = plt.cm.plasma(cell_colours)

    if normalize is True:
        celldata = np.around(celldata / float(total_samples), 2)

    column_labels = ['Pred %s' % class_id for class_id in class_ids] + ['Prevalence']
    row_labels = ['Real %s' % class_id for class_id in class_ids] + ['Prediction Bias']

    table = plt.table(
        cellText=celldata,
        colLabels=column_labels,
        rowLabels=row_labels,
        #cellColours=cell_colours,
        bbox=[0, 0, 1, 1],
        loc='center',
    )

    for key, cell in table.get_celld().items():
        cell.set_text_props(fontproperties=FontProperties(size='xx-large'))

    plt.xticks([])
    plt.yticks([])


def plot_acc_over_time_chart(preds, rels, bucket_size=24*60):
    """
    Plots the output of acc_over_time in a simple time plot.
    """
    accs = ml.stats.curves.acc_over_time(preds, rels, bucket_size)

    accs.plot()
    plt.hlines(0.5, 0, len(accs), color='red', lw=3)
    plt.title('Interval Accuracy over time')


def plot_acc_over_time_hist(preds, rels, bucket_size=24*60):
    """
    Displays the output of acc_overt_time in a histogram, highlighting the percentage
    of the buckets which are "win intervals", meaning ACC > 50%.
    """
    accs = ml.stats.curves.acc_over_time(preds, rels, bucket_size)

    accs.hist(bins=100)
    interval_win_rate = (accs >= 0.5).sum() / float(len(accs))
    plt.title('%.4f Interval Win Rate' % interval_win_rate)

    plt.vlines(0.5, 0, plt.ylim()[1], color='red', lw=3)


def plot_roc_curve(probs, rels):
    """
    Plots the receiver operator characteristic for a binary classification result. This
    is the parameterized curve [(TP[z], FP[z])] for z the classification threshold.
    """
    num_classes = probs.shape[1]
    class_ids = range(num_classes)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    for class_id, lines in zip(class_ids, ROC_CURVE_LINES[:num_classes]):
        pos_probs = probs[:, class_id]

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            rels,
            pos_probs,
            pos_label=class_id,
        )

        auc_pos = sklearn.metrics.auc(fpr, tpr)

        label = 'Class %s AUC = %.4f' % (class_id, auc_pos)

        plt.plot(
            fpr,
            tpr,
            color=lines[0],
            linestyle=lines[1],
            lw=lines[2],
            label=label,
        )

        plt.title('ROC Curves')
        plt.legend(loc=4)


def plot_likelihood_curves(probs, rels):
    lrs = ml.stats.curves.likelihoods_at_freq(probs, rels)

    lrs.plot(ax=plt.gca())


def plot_simple_precision_curves(probs, rels):
    pvs_and_freq_df = ml.stats.curves.precisions_and_frequency_at_z(probs, rels)

    ax = plt.gca()

    pvs_and_freq_df[['ACC', 'PPV', 'NPV']].plot(ax=ax)
    plt.legend()
    plt.grid()


def plot_precision_curves(probs, rels):
    pvs_and_freq_df = ml.stats.curves.precisions_and_frequency_at_z(probs, rels)

    ax = plt.subplot(211)
    pvs_and_freq_df[['ACC', 'PPV', 'NPV']].plot(ax=ax)
    plt.legend()
    plt.grid()

    ax = plt.subplot(212)
    pvs_and_freq_df[['Freq']].plot(ax=ax)
    plt.grid()


def plot_regression_fit(predictions, reality, subplots=True, window=None):
    if window is not None:
        predictions = predictions[window[0]: window[1]]
        reality = reality[window[0]: window[1]]

    if subplots is True:
        plt.subplot(211)
        plt.cla()
        plt.plot(predictions, label='Predictions')
        plt.legend()

        min_y, max_y = plt.ylim()

        plt.subplot(212)
        plt.cla()
        plt.plot(reality, label='Reality')
        plt.legend()

        min_y = min(min_y, plt.ylim()[0])
        max_y = max(max_y, plt.ylim()[1])

        plt.subplot(211)
        plt.ylim(min_y, max_y)
        plt.subplot(212)
        plt.ylim(min_y, max_y)
    else:
        plt.subplot(111)
        plt.plot(predictions, label='Prediction')
        plt.plot(reality, label='Reality')
        plt.legend()

    plt.show()


# Random helper functions.

def reset_current_plot():
    plt.subplot(111)
    plt.cla()

