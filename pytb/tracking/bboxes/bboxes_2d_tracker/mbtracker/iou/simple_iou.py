"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""


from pytb.utils.util import iou


class SimpleIOU:

    def __init__(self, t_min, sigma_iou):
        """
        Parameters
        ----------
             sigma_iou (float): minimum IOU threshold to associate two bounding boxes.
             t_min (float): minimum track length in frames before a track is created.
        """
        self.t_min = t_min
        self.sigma_iou = sigma_iou
        self.next_id = 1

        self.tracks_active = []

    def update(self, dets):
        """
        Simple IOU based tracker, adapted for online tracking
        See "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora"
        for more information.
        Args:
             dets (list): list of detections, field names: ['bbox': ('x1', 'y1', 'x2', 'y2'), 'score', 'class']
        Returns:
            list (list): list of tracks, field names: ['bboxes', 'scores', 'classes'].
        """
        updated_tracks = []
        for track in self.tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match_idx, best_match = max(enumerate(dets), key=lambda x: iou(track['bboxes'][-1], x[1]['bbox']))
                if iou(track['bboxes'][-1], best_match['bbox']) >= self.sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['scores'].append(best_match['score'])
                    track['classes'].append(best_match['class'])
                    updated_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[best_match_idx]

        # create new tracks with remaining detections
        new_tracks = []
        for det in dets:
            new_tracks.append({'id': self.next_id, 'bboxes': [det['bbox']],
                               'scores': [det['score']], 'classes': [det['class']]})
            self.next_id += 1
        self.tracks_active = new_tracks + updated_tracks
        return [track for track in self.tracks_active if len(track['bboxes']) >= self.t_min]

    def reset_state(self, reset_id):
        """
        Resets to the initial state
        Params:
        reset_id - whether the object id counter is reset
        """
        self.tracks_active = []
        if reset_id:
            self.next_id = 1
