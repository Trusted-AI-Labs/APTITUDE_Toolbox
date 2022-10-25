"""
Simple IOU based tracker with Kalman filter.
This tracker is based on the original IOU Tracker.

Copyright (c) 2017 TU Berlin, Communication Systems Group
Licensed under The MIT License [see LICENSE for details]
Written by Erik Bochinski

See https://github.com/siyuanc2/kiout/ for more information

Copyright (c) 2021 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Adapted for online tracking by Jonathan Samelson (2021)
"""
from pykalman import KalmanFilter

from pytb.utils.util import iou
import numpy as np


class KIOU:

    def __init__(self, sigma_len, sigma_iou, sigma_p):
        """
        Parameters
        ----------
             sigma_iou (float): minimum IOU threshold to associate two bounding boxes.
             sigma_len (float): minimum track length in frames before a track is created.
             sigma_p (int): maximum frames a track remains pending before termination.
        """
        self.sigma_p = sigma_p
        self.sigma_iou = sigma_iou
        self.sigma_len = sigma_len
        self.next_id = 1

        self.tracks_active = []
        self.tracks_pending = []

    def update(self, dets):
        """
        Args:
             dets (list): list of detections, field names: ['bbox': ('x1', 'y1', 'x2', 'y2'), 'score', 'class']
        Returns:
            list (list): list of tracks, field names: ['bboxes', 'scores', 'classes'].
        """

        updated_tracks = []
        for tracks in self.tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match_idx, best_match = max(enumerate(dets), key=lambda x: active_criteria(x[1], tracks))

                if active_criteria(best_match, tracks) >= self.sigma_iou:
                    filtered_state_mean, filtered_state_cov = tracks[1].filter_update(
                        tracks[0][-1]['cur_state'], tracks[0][-1]['cur_covar'], best_match['centroid'])
                    best_match['cur_state'] = filtered_state_mean
                    best_match['cur_covar'] = filtered_state_cov
                    best_match['pred_state'], best_match['pred_covar'] = tracks[1].filter_update(
                        filtered_state_mean, filtered_state_cov)
                    best_match['time_pending'] = 0
                    tracks[0].append(best_match)
                    updated_tracks.append(tracks)

                    # remove from best matching detection from detections
                    del dets[best_match_idx]

            # if track was not updated
            if len(updated_tracks) == 0 or tracks is not updated_tracks[-1]:
                # keep track in tracks_pending, where tracks will be kept for sigma_p frames before track termination
                self.tracks_pending.append(tracks)

        tracks_to_keep = []
        for tracks in self.tracks_pending:
            if tracks[0][-1]['time_pending'] >= self.sigma_p:
                # finish long tracks that have been inactive for more than sigma_p frames
                continue

            elif len(dets) == 0:
                # if track is fresh enough but no detections in this frame are available for matching,
                # keep the track pending and extrapolate for one time step
                tracks[0][-1]['pred_state'], tracks[0][-1]['pred_covar'] = \
                    tracks[1].filter_update(tracks[0][-1]['pred_state'], tracks[0][-1]['pred_covar'])
                tracks[0][-1]['time_pending'] += 1
                tracks_to_keep.append(tracks)

            else:
                # replicating the process in tracks_active
                # get det with highest iou
                best_match_idx, best_match = max(enumerate(dets), key=lambda x: active_criteria(x[1], tracks))

                if active_criteria(best_match, tracks) >= self.sigma_iou:
                    filtered_state_mean, filtered_state_cov = tracks[1].filter_update(
                        tracks[0][-1]['cur_state'], tracks[0][-1]['cur_covar'], best_match['centroid'])
                    best_match['cur_state'] = filtered_state_mean
                    best_match['cur_covar'] = filtered_state_cov
                    best_match['pred_state'], best_match['pred_covar'] = \
                        tracks[1].filter_update(filtered_state_mean, filtered_state_cov)
                    best_match['time_pending'] = 0
                    tracks[0].append(best_match)
                    updated_tracks.append(tracks)

                    del dets[best_match_idx]
                else:
                    # if the proposed match does not pass the threshold, keep the track pending
                    # tracks[0][-1]['pred_state'], tracks[0][-1]['pred_covar'] =
                    #       tracks[1].filter_update(tracks[0][-1]['pred_state'], tracks[0][-1]['pred_covar'])
                    tracks[0][-1]['time_pending'] += 1
                    tracks_to_keep.append(tracks)

        # form pending tracks for next frame
        self.tracks_pending = tracks_to_keep

        # create new tracks
        cur_covar = [[100, 0, 25, 0], [0, 100, 0, 25], [0, 0, 25, 0], [0, 0, 0, 25]]

        new_tracks = []
        for det in dets:
            track = [[{'id': self.next_id, 'bbox': det['bbox'], 'score': det['score'], 'class': det['class'],
                       'cur_state': [*det['centroid'], 0, 0], 'cur_covar': cur_covar, 'time_pending': 0}],
                     setup_kf(det['centroid'])]
            track[0][0]['pred_state'], track[0][0]['pred_covar'] = track[1].filter_update(track[0][0]['cur_state'],
                                                                                          track[0][0]['cur_covar'])
            new_tracks.append(track)
            self.next_id += 1

        self.tracks_active = updated_tracks + new_tracks

        # TODO Equivalent of interp_tracks at the tracking manager level to skip frames
        # tracks_trimmed = interp_tracks(tracks_finished)

        return [track[0] for track in self.tracks_active if len(track[0]) >= self.sigma_len]

    def reset_state(self, reset_id):
        """
        Resets to the initial state
        Params:
        reset_id - whether the object id counter is reset
        """
        self.tracks_active = []
        self.tracks_pending = []
        if reset_id:
            self.next_id = 1


def setup_kf(imean, a=None, o=None):
    """
    Initialize Kalman filter object for each new tracks.
    The transfermation matrix (a) and observation matrix (o) can be tuned to better suit with a specific motion pattern,
    but to preserve generality we are using a simple constant speed model here.
    Args:
        imean (2x1 array): 1x2 array or list of the location of centroid.
        a (array): transformation matrix that governs state transition to the next time step. Size varies with model.
        o (array): observation matrix that defines the observable states. Size varies with model.
    """
    if o is None:
        o = [[1, 0, 0, 0], [0, 1, 0, 0]]
    if a is None:
        a = [[1, 0, 0.5, 0], [0, 1, 0, 0.5], [0, 0, 1, 0], [0, 0, 0, 1]]
    return KalmanFilter(transition_matrices=a, observation_matrices=o, initial_state_mean=[*imean, 0, 0])


def active_criteria(x, tracks):
    """
    Take matching candidate and track, offset the track's last bounding box by the predicted offset, and calculate IOU.
    Args:
        x (list [roi, bbox, score]): a detection from this frame.
        tracks (list [[frames], Kalman_filter]): a track containing all frames and a Kalman filter associated with it.
    """
    ofdx, ofdy, _, _ = tracks[0][-1]['pred_state'] - tracks[0][-1]['cur_state']
    offset_vector = np.array([ofdy, ofdx, ofdy, ofdx])
    offset_bbox = tracks[0][-1]['bbox'] + offset_vector

    return iou(x['bbox'], offset_bbox)
