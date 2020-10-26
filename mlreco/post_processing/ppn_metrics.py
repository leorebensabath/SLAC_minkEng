import numpy as np
from scipy.spatial.distance import cdist
import scipy
import os
from mlreco.utils import CSVData
from mlreco.utils.dbscan import dbscan_points
from mlreco.utils.ppn import uresnet_ppn_point_selector, uresnet_ppn_type_point_selector

def ppn_metrics(cfg, data_blob, res, logdir, iteration):
    # UResNet prediction
    if not 'segmentation' in res: return
    if not 'points' in res: return

    method_cfg = cfg['post_processing']['ppn_metrics']

    index        = data_blob['index']
    segmentation = res['segmentation']
    points       = res['points']
    attention    = res['mask_ppn2']
    input_data   = data_blob.get('input_data' if method_cfg is None else method_cfg.get('input_data', 'input_data'), None)
    segment_label = data_blob.get('segment_label' if method_cfg is None else method_cfg.get('segment_label', 'segment_label'), None)
    num_classes = 5 if method_cfg is None else method_cfg.get('num_classes', 5)
    points_label = data_blob.get('particles_label' if method_cfg is None else method_cfg.get('particles_label', 'particles_label'), None)
    particles    = data_blob.get('particles' if method_cfg is None else method_cfg.get('particles', 'particles'), None)

    store_per_iteration = True
    if method_cfg is not None and method_cfg.get('store_method',None) is not None:
        assert(method_cfg['store_method'] in ['per-iteration','per-event'])
        store_per_iteration = method_cfg['store_method'] == 'per-iteration'
    fout=None
    if store_per_iteration:
        fout_gt=CSVData(os.path.join(logdir, 'ppn-metrics-gt-iter-%07d.csv' % iteration))
        fout_pred=CSVData(os.path.join(logdir, 'ppn-metrics-pred-iter-%07d.csv' % iteration))
        fout_all=CSVData(os.path.join(logdir, 'ppn-metrics-iter-%07d.csv' % iteration))

    pool_op = np.max
    if method_cfg is not None and method_cfg.get('pool_op', 'max') == 'mean':
        pool_op = np.mean
    for data_idx, tree_idx in enumerate(index):

        if not store_per_iteration:
            fout_gt=CSVData(os.path.join(logdir, 'ppn-metrics-gt-event-%07d.csv' % tree_idx))
            fout_pred=CSVData(os.path.join(logdir, 'ppn-metrics-pred-event-%07d.csv' % tree_idx))
            fout_all=CSVData(os.path.join(logdir, 'ppn-metrics-event-%07d.csv' % tree_idx))

        # Store PPN metrics
        # fout_all.record(('idx', 'acc_ppn1', 'acc_ppn2'),
        #                 (data_idx, res['acc_ppn1'][data_idx], res['acc_ppn2'][data_idx]))
        # fout_all.write()

        # UResNet output
        predictions = np.argmax(segmentation[data_idx],axis=1)
        label = segment_label[data_idx][:, -1]


        # Remove deltas from true points
        delta = 3
        #points_label_idx = points_label[data_idx][points_label[data_idx][:, -1] != delta]
        points_label_idx = points_label[data_idx]
        # type and idx in this order = -2, -1
        # print(np.unique(points_label_idx[:, -2]), np.unique(points_label_idx[:, -1]))

        ppn_voxels = points[data_idx][:, :3] + 0.5 + input_data[data_idx][:, :3]
        ppn_score  = scipy.special.softmax(points[data_idx][:, 3:5], axis=1)[:, 1]
        ppn_type   = scipy.special.softmax(points[data_idx][:, 5:], axis=1)

        ppn_mask = (attention[data_idx][:, 0]==1) & (ppn_score > 0.5)

        mode = 'select'
        #mode = 'simple'
        if mode == 'simple':
            ppn_voxels = ppn_voxels[ppn_mask]
            ppn_score  = ppn_score[ppn_mask]
            ppn_type   = ppn_type[ppn_mask]

            all_voxels, all_scores, all_types, all_ocupancy = [], [], [], []
            clusts = dbscan_points(ppn_voxels, epsilon=1.99, minpts=1)
            for c in clusts:
                all_voxels.append(np.mean(ppn_voxels[c], axis=0))
                all_scores.append(pool_op(ppn_score[c], axis=0))
                all_types.append(pool_op(ppn_type[c], axis=0))
                all_occupancy.append(len(c))
            ppn_voxels = np.stack(all_voxels, axis=0)
            ppn_score = np.stack(all_scores)
            ppn_type = np.stack(all_types)
            ppn_occupancy = np.stack(all_occupancy)
        else:
            #ppn = uresnet_ppn_point_selector(input_data[data_idx], res, entry=data_idx, score_threshold=0.6, window_size=10, nms_score_threshold=0.99 )
            ppn = uresnet_ppn_type_point_selector(input_data[data_idx], res, entry=data_idx, score_threshold=0.5, window_size=3, type_threshold=2)
            # Remove delta from predicted points
            #ppn = ppn[ppn[:, -1] != delta]
            #ppn = ppn[ppn[:, -3] < 0.1]
            print(ppn.shape, ppn[:5])
            ppn_voxels = ppn[:, :3]
            ppn_score = ppn[:, 5]
            ppn_occupancy = ppn[:, 6]
            ppn_type = ppn[:, 7:-1]#np.repeat(ppn[:, -1][:, None], num_classes, axis=1)
        #print(ppn_voxels.shape, ppn_score.shape, ppn_type.shape)

        # Metrics now
        # Distance to closest true point (regardless of type)
        # Ignore points predicted as delta for this part
        no_delta = ppn[:, -3] < 0.1
        d = cdist(ppn_voxels, points_label_idx[:, :3])
        distance_to_closest_true_point = d.min(axis=1)
        #print(d.shape, ppn_type.shape, points_label_idx.shape)
        distance_to_closest_true_point_nodelta = d[:, points_label_idx[:, -2] != 3].min(axis=1)

        distance_to_closest_true_point_type = []
        distance_to_closest_true_pix_type = []
        distance_to_closest_pred_pix_type = []
        for c in range(num_classes):
            true_mask = points_label_idx[:, -2] == c
            d = cdist(ppn_voxels, points_label_idx[true_mask][:, :3])
            #print(d.shape)
            if d.shape[1] > 0:
                distance_to_closest_true_point_type.append(d.min(axis=1))
            else:
                distance_to_closest_true_point_type.append(-1 * np.ones(ppn_voxels.shape[0],))
            d = cdist(ppn_voxels, input_data[data_idx][segment_label[data_idx][:, -1] == c][:, :3])
            if d.shape[1] > 0:
                distance_to_closest_true_pix_type.append(d.min(axis=1))
            else:
                distance_to_closest_true_pix_type.append(-1 * np.ones(ppn_voxels.shape[0],))
            d = cdist(ppn_voxels, input_data[data_idx][predictions == c][:, :3])
            if d.shape[1] > 0:
                distance_to_closest_pred_pix_type.append(d.min(axis=1))
            else:
                distance_to_closest_pred_pix_type.append(-1 * np.ones(ppn_voxels.shape[0],))
        distance_to_closest_true_point_type = np.array(distance_to_closest_true_point_type)
        distance_to_closest_true_pix_type = np.array(distance_to_closest_true_pix_type)
        distance_to_closest_pred_pix_type = np.array(distance_to_closest_pred_pix_type)

        for i in range(ppn_voxels.shape[0]):
            fout_pred.record(('idx', 'distance_to_closest_true_point', 'distance_to_closest_true_point_nodelta', 'score', 'x', 'y', 'z', 'type', 'occupancy') + tuple(['distance_to_closest_true_point_type_%d' % c for c in range(num_classes)]) + tuple(['score_type_%d' % c for c in range(num_classes)]) + tuple(['distance_to_closest_true_pix_type_%d' % c for c in range(num_classes)]) + tuple(['distance_to_closest_pred_pix_type_%d' % c for c in range(num_classes)]),
                            (tree_idx, distance_to_closest_true_point[i], distance_to_closest_true_point_nodelta[i], ppn_score[i], ppn_voxels[i, 0], ppn_voxels[i, 1], ppn_voxels[i, 2], np.argmax(ppn_type[i]), ppn_occupancy[i]) + tuple(distance_to_closest_true_point_type[:, i]) + tuple(ppn_type[i]) + tuple(distance_to_closest_true_pix_type[:, i]) + tuple(distance_to_closest_pred_pix_type[:, i]))
            fout_pred.write()

        # Distance to closest pred point (regardless of type)
        d = cdist(ppn_voxels, points_label_idx[:, :3])
        #print(d.shape)
        distance_to_closest_pred_point = d.min(axis=0)
        distance_to_closest_pred_point_nodelta = d[ppn_type[:, 3] < 0.1, :].min(axis=0)
        score_of_closest_pred_point = ppn_score[d.argmin(axis=0)]
        types_of_closest_pred_point = ppn_type[d.argmin(axis=0)]

        # closest pred point with type score >0.9

        distance_to_closest_true_pix_type = []
        distance_to_closest_pred_pix_type = []
        for c in range(num_classes):
            d2 = cdist(points_label_idx[:, :3], input_data[data_idx][segment_label[data_idx][:, -1] == c][:, :3])
            if d2.shape[1] > 0:
                distance_to_closest_true_pix_type.append(d2.min(axis=1))
            else:
                distance_to_closest_true_pix_type.append(-1 * np.ones(points_label_idx.shape[0],))
            d3 = cdist(points_label_idx[:, :3], input_data[data_idx][predictions == c][:, :3])
            if d3.shape[1] > 0:
                distance_to_closest_pred_pix_type.append(d3.min(axis=1))
            else:
                distance_to_closest_pred_pix_type.append(-1 * np.ones(points_label_idx.shape[0],))
        distance_to_closest_true_pix_type = np.array(distance_to_closest_true_pix_type)
        distance_to_closest_pred_pix_type = np.array(distance_to_closest_pred_pix_type)

        one_pixel = 2.
        for i in range(points_label_idx.shape[0]):
            #print(d.shape, ppn_voxels.shape,  ppn_type.shape, ppn_type[:, int(points_label_idx[i, -2])].shape)
            local_d = d[ppn_type[:, int(points_label_idx[i, -2])] > 0.9, i]
            closest_pred_point_same_type = 1000000
            if local_d.shape[0] > 0:
                closest_pred_point_same_type = local_d.min()

            num_voxels, energy_deposit = -1, -1
            if particles is not None:
                num_voxels = particles[data_idx][int(points_label_idx[i, -1])].num_voxels()
                energy_deposit = particles[data_idx][int(points_label_idx[i, -1])].energy_deposit()
            # Whether this point is already missed in mask_ppn2 or not
            #print(input_data[data_idx][ppn_mask].shape, points_label[data_idx].shape)
            is_in_attention = cdist(input_data[data_idx][ppn_mask][:, :3], [points_label_idx[i, :3]]).min(axis=0) < one_pixel
            fout_gt.record(('idx', 'distance_to_closest_pred_point', 'distance_to_closest_pred_point_nodelta', 'type', 'score_of_closest_pred_point', 'x', 'y', 'z',
                            'attention', 'particle_idx', 'num_voxels', 'energy_deposit', 'distance_to_closest_pred_point_same_type') + tuple(['type_of_closest_pred_point_%d' % c for c in range(num_classes)]) + tuple(['distance_to_closest_true_pix_type_%d' % c for c in range(num_classes)]) + tuple(['distance_to_closest_pred_pix_type_%d' % c for c in range(num_classes)]),
                    (tree_idx, distance_to_closest_pred_point[i], distance_to_closest_pred_point_nodelta[i], points_label_idx[i, -2], score_of_closest_pred_point[i],
                        points_label_idx[i, 0], points_label_idx[i, 1], points_label_idx[i, 2],
                            int(is_in_attention), points_label_idx[i, -1], num_voxels, energy_deposit, closest_pred_point_same_type) + tuple(types_of_closest_pred_point[i]) + tuple(distance_to_closest_true_pix_type[:, i]) + tuple(distance_to_closest_pred_pix_type[:, i]))
            fout_gt.write()

        if not store_per_iteration:
            fout_gt.close()
            fout_pred.close()
            fout_all.close()

    if store_per_iteration:
        fout_gt.close()
        fout_pred.close()
        fout_all.close()
