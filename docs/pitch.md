# Pitch notes

## Three-minute pitch (with timings)

1. (20 s) Problem: a 10-euro dampener IMU instead of a motion-capture lab. One forehand, 400 samples, 0.96 s.
2. (40 s) Method: the gyro drives orientation (exact quaternion integration). Video is never an input, only the referee.
   Show the 3D viewer playing at 25% speed.
3. (40 s) Robustness: the accelerometer saturates for 43 ms. We rebuild it from the unclipped gyro with rigid-body physics.
   Proof: we hide good data on purpose and recover it within 1.6 g, where curve fitting misses by 7.4 g (figure 2).
   Noise and bias move the result by about 1 degree.
4. (40 s) Validation: overlay video. Forward swing matches both cameras to about 2 degrees median against our clicked
   points (figure 3), and 4 to 5 degrees against an independent SAM 2 segmentation that saw our clicks only on the
   first frame (figure 4). Our hand-clicked reference itself agrees with SAM 2 to about 3 degrees.
   We are honest about the rest: the error rises in the last ~70 ms before contact, and the impact shock disturbs
   the gyro, so the follow-through is drawn as lower confidence.
5. (40 s) Toward 6-DoF: the rigid-body fit already finds the rotation centre at the hand, so the racket can be placed in space
   as hand position plus rotation, without double-integrating acceleration. Next steps: a short rest period at the start of
   each recording for gravity, and pose tracking of the wrist to validate the rotation centre.

