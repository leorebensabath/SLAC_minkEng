import numpy as np
import os
from mlreco.utils import utils
import scipy
from scipy.spatial.distance import cdist

def deghosting_metrics(cfg, data_blob, res, logdir, iteration):#, idx):
    """
    Some useful metrics to measure deghosting performance

    Parameters
    ----------
    data_blob: dict
        Input dictionary returned by iotools
    res: dict
        Results from the network, dictionary using `analysis_keys`
    cfg: dict
        Configuration
    idx: int
        Iteration number

    Input
    -----
    Requires the following analysis keys:
    - `segmentation`
    - `ghost` only if 5+2 types architecture for GhostNet
    Requires the following input keys:
    - `input_data`
    - `segment_label`
    Assumes no minibatching

    Output
    ------
    Writes to a CSV file `deghosting_metrics-*`
    """

    method_cfg = cfg['post_processing']['deghosting_metrics']

    csv_logger = utils.CSVData(os.path.join(logdir,"deghosting_metrics-iter-%.07d.csv" % iteration))
    csv_logger2 = utils.CSVData(os.path.join(logdir,"deghosting_metrics-true-ghost-iter-%.07d.csv" % iteration))
    csv_logger3 = utils.CSVData(os.path.join(logdir,"deghosting_metrics-true-nonghost-iter-%.07d.csv" % iteration))

    for data_idx, tree_idx in enumerate(data_blob['index']):

        deghosting_type = method_cfg['method']
        assert(deghosting_type in ['5+2','6','2'])

        pcluster = None
        if 'pcluster' in data_blob:
            pcluster = data_blob['pcluster'][data_idx][:, -1]
        data = data_blob['input_data'][data_idx]
        label = data_blob['segment_label'][data_idx][:,-1]
        segmentation = res['segmentation'][data_idx]  # (N, 5)
        predictions  = np.argmax(segmentation, axis=1)
        softmax_predictions = scipy.special.softmax(segmentation, axis=1)

        num_classes = segmentation.shape[1]
        num_ghost_points = np.count_nonzero(label == 5)
        num_nonghost_points = np.count_nonzero(label < 5)

        csv_logger.record(('num_ghost_points', 'num_nonghost_points', 'idx'),
                          (num_ghost_points, num_nonghost_points, tree_idx))

        if deghosting_type == '5+2':
            # Accuracy for ghost prediction for 5+2
            ghost_predictions = np.argmax(res['ghost'][data_idx], axis=1)
            ghost_softmax = scipy.special.softmax(res['ghost'][data_idx], axis=1)
            mask = ghost_predictions == 0
            # 0 = non ghost, 1 = ghost
            # Fraction of true points predicted correctly
            ghost_acc = ((ghost_predictions == 1) == (label == 5)).sum() / float(label.shape[0])
            # Fraction of ghost points predicted as ghost points
            ghost2ghost = (ghost_predictions[label == 5] == 1).sum() / float(num_ghost_points)
            # Fraction of true non-ghost points predicted as true non-ghost points
            nonghost2nonghost = (ghost_predictions[label < 5] == 0).sum() / float(num_nonghost_points)
            csv_logger.record(("ghost2ghost", "nonghost2nonghost"),
                              (ghost2ghost, nonghost2nonghost))

            # # Looking at mistakes: true ghost predicted as nonghost
            # # distance from a true ghost point predicted as nonghost, to closest true nonghost point
            # d = cdist(data[(ghost_predictions == 0) & (label == 5), :3], data[label < 5, :3])
            # closest_true_nonghost = d.argmin(axis=1)
            # for d_idx in range(d.shape[0]):
            #     csv_logger2.record(("distance_to_closest_true_nonghost", "semantic_of_closest_true_nonghost", "predicted_semantic",
            #                         "nonghost_softmax"),
            #                        (d[d_idx, closest_true_nonghost[d_idx]], label[label<5][closest_true_nonghost[d_idx]], predictions[(ghost_predictions == 0) & (label == 5)][d_idx],
            #                        ghost_softmax[(ghost_predictions == 0) & (label == 5)][d_idx][0]))
            #     for c in range(num_classes):
            #         csv_logger2.record(("softmax_class%d" %c,),
            #                             (softmax_predictions[(ghost_predictions == 0) & (label == 5)][d_idx][c],))
            #     csv_logger2.write()
            #
            # # Looking at mistakes: true nonghost predicted as ghost
            # d = cdist(data[(ghost_predictions == 1) & (label < 5), :3], data[label == 5, :3])
            # closest_true_ghost = d.argmin(axis=1)
            # for d_idx in range(d.shape[0]):
            #     csv_logger3.record(("distance_to_closest_true_ghost", "semantic",
            #                         "ghost_softmax", "predicted_semantic"),
            #                         (d[d_idx, closest_true_ghost[d_idx]], label[(ghost_predictions == 1) & (label < 5)][d_idx],
            #                         ghost_softmax[(ghost_predictions == 1) & (label < 5)][d_idx][1],
            #                         predictions[(ghost_predictions == 1) & (label < 5)][d_idx]))
            #     for c in range(num_classes):
            #         csv_logger3.record(("softmax_class%d" % c,),
            #                             (softmax_predictions[(ghost_predictions == 1) & (label < 5)][d_idx][c],))
            #     csv_logger3.write()

            # Accuracy for 5 types, global
            uresnet_acc = (label[label < 5] == predictions[label < 5]).sum() / float(np.count_nonzero(label < 5))
            csv_logger.record(('ghost_acc', 'uresnet_acc'),
                              (ghost_acc, uresnet_acc))
            # Class-wise nonzero accuracy for 5 types, based on true mask
            acc, num_true_pix, num_pred_pix = [], [], []
            num_pred_pix_true, num_true_pix_pred = [], []
            num_true_deghost_pix, num_original_pix = [], []
            ghost_false_positives, ghost_true_positives = [], []
            for c in range(num_classes):
                class_mask = label == c
                class_predictions = predictions[class_mask]
                # Fraction of pixels in this class predicted correctly
                acc.append((class_predictions == c).sum() / float(class_predictions.shape[0]))
                # Pixel counts
                # Pixels in sparse3d_semantics_reco
                num_true_pix.append(np.count_nonzero(class_mask))
                # Pixels in sparse3d_semantics_reco predicted as nonghost
                num_true_deghost_pix.append(np.count_nonzero(class_mask & mask))
                # Pixels in original pcluster
                if pcluster is not None:
                    num_original_pix.append(np.count_nonzero(pcluster == c))
                # Pixels in predictions + nonghost
                num_pred_pix.append(np.count_nonzero(predictions[mask] == c))
                # Pixels in predictions + nonghost that are correctly classified
                num_pred_pix_true.append(np.count_nonzero(class_predictions == c))
                num_true_pix_pred.append(np.count_nonzero(predictions[mask & class_mask] == c))
                # Fraction of pixels in this class (wrongly) predicted as ghost
                ghost_false_positives.append(np.count_nonzero(ghost_predictions[class_mask] == 1))
                # Fraction of pixels in this class (correctly) predicted as nonghost
                ghost_true_positives.append(np.count_nonzero(ghost_predictions[class_mask] == 0))
                # confusion matrix
                # pixels predicted as nonghost + should be in class c, but predicted as c2
                for c2 in range(num_classes):
                    csv_logger.record(('confusion_%d_%d' % (c, c2),),
                                      (((class_predictions == c2) & (ghost_predictions[class_mask] == 0)).sum(),))
            csv_logger.record(['acc_class%d' % c for c in range(num_classes)],
                              acc)
            csv_logger.record(['num_true_pix_class%d' % c for c in range(num_classes)],
                              num_true_pix)
            csv_logger.record(['num_true_deghost_pix_class%d' % c for c in range(num_classes)],
                              num_true_deghost_pix)
            if pcluster is not None:
                csv_logger.record(['num_original_pix_class%d' % c for c in range(num_classes)],
                                  num_original_pix)
            csv_logger.record(['num_pred_pix_class%d' % c for c in range(num_classes)],
                              num_pred_pix)
            csv_logger.record(['num_pred_pix_true_class%d' % c for c in range(num_classes)],
                              num_pred_pix_true)
            csv_logger.record(['num_true_pix_pred_class%d' % c for c in range(num_classes)],
                              num_true_pix_pred)
            csv_logger.record(['ghost_false_positives_class%d' % c for c in range(num_classes)],
                              ghost_false_positives)
            csv_logger.record(['ghost_true_positives_class%d' % c for c in range(num_classes)],
                              ghost_true_positives)

        elif deghosting_type == '6':
            ghost2ghost = (predictions[label == 5] == 5).sum() / float(num_ghost_points)
            nonghost2nonghost = (predictions[label < 5] < 5).sum() / float(num_nonghost_points)
            csv_logger.record(("ghost2ghost", "nonghost2nonghost"),
                              (ghost2ghost, nonghost2nonghost))
            # 6 types confusion matrix
            for c in range(num_classes):
                for c2 in range(num_classes):
                    # Fraction of points of class c, predicted as c2
                    x = (predictions[label == c] == c2).sum() / float(np.count_nonzero(label == c))
                    csv_logger.record(('confusion_%d_%d' % (c, c2),), (x,))
        elif deghosting_type == '2':
            ghost2ghost = (predictions[label == 5] == 1).sum() / float(num_ghost_points)
            nonghost2nonghost = (predictions[label < 5] == 0).sum() / float(num_nonghost_points)
            csv_logger.record(("ghost2ghost", "nonghost2nonghost"),
                              (ghost2ghost, nonghost2nonghost))
        else:
            print('Invalid "deghosting_type" config parameter value:',deghosting_type)
            raise ValueError
        csv_logger.write()
    csv_logger.close()
    csv_logger2.close()
    csv_logger3.close()
