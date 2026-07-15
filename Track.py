class TrackState:
    """Enum-style track lifecycle states."""
    Tentative  = 1   # Newly created, not yet confirmed
    Confirmed  = 2   # Confirmed after N_INIT consecutive matches
    Deleted    = 3   # Marked for removal (too many misses)


class Track:
    """
    A single tracked object with Kalman Filter state, appearance history,
    and a lifecycle state machine (Tentative → Confirmed → Deleted).
    """

    N_INIT    = 3    # hits needed to confirm a new track
    MAX_AGE   = 30   # max frames a track survives without a match

    def __init__(self, track_id, state, covariance, cls, cls_name, conf, feature):
        self.track_id   = track_id
        self.state      = state            # Kalman state vector [x,y,w,h,vx,vy,vw,vh]
        self.covariance = covariance
        self.cls        = cls
        self.cls_name   = cls_name
        self.conf       = conf
        self.features   = [feature] if feature is not None else []
        self.hits       = 1
        self.age        = 1
        self.time_since_update = 0
        self.track_state = TrackState.Tentative

    def predict(self, kf):
        """Advance Kalman Filter prediction by one frame."""
        self.state, self.covariance = kf.predict(self.state, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Update track with a matched detection."""
        self.state, self.covariance = kf.update(
            self.state, self.covariance, detection.to_xywh()
        )
        if detection.feature is not None:
            self.features.append(detection.feature)
            if len(self.features) > 10:
                self.features.pop(0)
        self.cls        = detection.cls
        self.cls_name   = detection.cls_name
        self.conf       = detection.conf
        self.hits      += 1
        self.time_since_update = 0

        if self.track_state == TrackState.Tentative and self.hits >= self.N_INIT:
            self.track_state = TrackState.Confirmed

    def mark_missed(self):
        """Called when no detection matched this track in current frame."""
        if self.track_state == TrackState.Tentative:
            self.track_state = TrackState.Deleted
        elif self.time_since_update > self.MAX_AGE:
            self.track_state = TrackState.Deleted

    def is_tentative(self):
        return self.track_state == TrackState.Tentative

    def is_confirmed(self):
        return self.track_state == TrackState.Confirmed

    def is_deleted(self):
        return self.track_state == TrackState.Deleted

    def to_xyxy(self):
        """Convert Kalman state [cx,cy,w,h,...] back to [x1,y1,x2,y2]."""
        cx, cy, w, h = self.state[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [int(x1), int(y1), int(x2), int(y2)]

    def __repr__(self):
        return (
            f"Track(id={self.track_id}, cls={self.cls_name}, "
            f"hits={self.hits}, age={self.age}, state={self.track_state})"
        )
