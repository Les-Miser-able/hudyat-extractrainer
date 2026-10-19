"""Exercise the real live predictor with deterministic camera landmarks."""
import unittest
from unittest.mock import patch
import numpy as np
from fsl_trainer.dashboard.live_worker import LivePredictor


class Detector:
    def detect_for_video(self, image, timestamp):
        return image

    def detect(self, image):
        return image

    def close(self):
        pass


class Extractor:
    swap_hands = False
    image_detector = Detector()

    def detector(self, video):
        return Detector()

    def image(self, frame):
        return frame

    def close(self):
        pass


class SequenceModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, sequence, training):
        self.inputs.append(sequence.copy())
        # A pose-only tail loses the distinguishing motion in this fixture.
        return np.array([[.9, .1] if np.ptp(sequence[0, :, 0]) > .2 else [.1, .9]])


def hand(x):
    frame = np.zeros(128, dtype=np.float32)
    frame[:63] = np.tile([x, .4, 0], 21)
    frame[126] = 1
    return frame


class MovingGestureTests(unittest.TestCase):
    def setUp(self):
        self.packer = patch('fsl_trainer.dashboard.live_worker.pack_result', side_effect=lambda result, swap: result)
        self.packer.start()
        self.addCleanup(self.packer.stop)
        self.model = SequenceModel()
        self.predictor = LivePredictor(self.model, ['afternoon', 'evening'], Extractor(), 'motion')
        self.addCleanup(self.predictor.close)

    def test_frame_results_identify_hand_presence_for_automatic_trial_timing(self):
        missing = self.predictor.frame(np.zeros(128, dtype=np.float32), 0)
        self.assertFalse(missing['hand_detected'])
        present = self.predictor.frame(hand(.1), 200)
        self.assertTrue(present['hand_detected'])
        pose = LivePredictor(self.model, ['afternoon', 'evening'], Extractor(), 'pose')
        self.addCleanup(pose.close)
        self.assertFalse(pose.frame(np.zeros(128, dtype=np.float32), 0)['hand_detected'])
        self.assertTrue(pose.frame(hand(.1), 200)['hand_detected'])

    def test_holding_ending_does_not_replace_motion_with_pose_only_prediction(self):
        results = []
        for timestamp in range(0, 6401, 200):
            x = .1 + .6 * min(timestamp, 3000) / 3000
            result = self.predictor.frame(hand(x), timestamp)
            if result['status'] == 'prediction':
                results.append((timestamp, result))
        self.assertEqual(results[-1][1]['probabilities']['afternoon'], .9,
                         'Holding the ending replaced the movement with an ending-pose prediction')
        self.assertEqual(len(self.model.inputs), 1, 'Classify one complete movement, not sliding tails')
        sequence = self.model.inputs[0][0]
        self.assertEqual(sequence.shape, (32, 128))
        self.assertAlmostEqual(float(sequence[:, 0].min()), .1, places=5)
        self.assertAlmostEqual(float(sequence[:, 0].max()), .7, places=5)
        self.assertGreaterEqual(results[0][0], 3600)

    def test_missing_hands_finalize_retained_movement_after_grace_period(self):
        for timestamp in range(0, 3201, 200):
            self.predictor.frame(hand(.1 + .6 * timestamp / 3200), timestamp)
        calls_before_missing = len(self.model.inputs)
        absent = np.zeros(128, dtype=np.float32)
        for timestamp in (3400, 3500, 3600, 3700):
            result = self.predictor.frame(absent, timestamp)
            self.assertNotEqual(result['status'], 'no_hand', 'Missing hands discarded the movement buffer')
            self.assertEqual(len(self.model.inputs), calls_before_missing)
        result = self.predictor.frame(absent, 3800)
        self.assertEqual(result['status'], 'prediction')
        self.assertTrue(result['complete'])
        sequence = self.model.inputs[-1][0]
        detected_x = sequence[sequence[:, 126] == 1, 0]
        self.assertLess(float(detected_x.min()), .15)
        self.assertGreater(float(detected_x.max()), .65)
        call_count = len(self.model.inputs)
        for timestamp in range(4200, 7001, 200):
            result = self.predictor.frame(absent, timestamp)
            self.assertTrue(result.get('keep_result'), 'Finished gesture result disappeared with the hands')
        self.assertEqual(len(self.model.inputs), call_count)

    def test_brief_missing_hands_resume_same_capture(self):
        for timestamp in range(0, 1801, 200):
            self.predictor.frame(hand(.1 + timestamp / 10000), timestamp)
        absent = np.zeros(128, dtype=np.float32)
        self.predictor.frame(absent, 2000)
        self.predictor.frame(absent, 2200)
        for timestamp in range(2400, 4001, 200):
            self.predictor.frame(hand(.4 + (timestamp - 2400) / 10000), timestamp)
        for timestamp in range(4200, 5001, 200):
            result = self.predictor.frame(hand(.56), timestamp)
        self.assertEqual(len(self.model.inputs), 1)
        sequence = self.model.inputs[0][0]
        self.assertLess(float(sequence[sequence[:, 126] == 1, 0].min()), .15)
        self.assertTrue((sequence[:, 126] == 0).any(), 'Brief missing frames should stay inside the sequence')

    def test_continuous_mode_captures_next_movement_without_reloading(self):
        for timestamp in range(0, 4001, 200):
            self.predictor.frame(hand(.1 + .6 * min(timestamp, 3000) / 3000), timestamp)
        self.assertEqual(len(self.model.inputs), 1)
        for timestamp in range(4200, 7201, 200):
            self.predictor.frame(hand(.7 - .5 * min(timestamp - 4200, 2000) / 2000), timestamp)
        self.assertEqual(len(self.model.inputs), 2)
        self.assertGreater(float(self.model.inputs[1][0, :, 0].max()), .65)
        self.assertLess(float(self.model.inputs[1][0, :, 0].min()), .25)

    def test_perform_once_stays_held_until_explicit_reset(self):
        self.predictor = LivePredictor(self.model, ['afternoon', 'evening'], Extractor(), 'gesture')
        for timestamp in range(0, 4001, 200):
            self.predictor.frame(hand(.1 + .6 * min(timestamp, 3000) / 3000), timestamp)
        for timestamp in range(4200, 7201, 200):
            result = self.predictor.frame(hand(.7 - .5 * min(timestamp - 4200, 2000) / 2000), timestamp)
        self.assertEqual(len(self.model.inputs), 1)
        self.assertEqual(result['status'], 'held')
        self.assertEqual(self.predictor.reset()['status'], 'waiting')

    def test_still_ending_finishes_within_600_ms(self):
        for timestamp in range(0, 1001, 50):
            self.predictor.frame(hand(.1 + .5 * timestamp / 1000), timestamp)
        for timestamp in range(1050, 1601, 50):
            result = self.predictor.frame(hand(.6), timestamp)
        self.assertEqual(len(self.model.inputs), 1, 'Stillness deadline exceeded 600 ms')
        self.assertEqual(result['status'], 'prediction')
        self.assertLess(float(self.model.inputs[0][0, :, 0].min()), .15)

    def test_missing_hands_finish_within_400_ms(self):
        for timestamp in range(0, 1001, 50):
            self.predictor.frame(hand(.1 + .5 * timestamp / 1000), timestamp)
        absent = np.zeros(128, dtype=np.float32)
        for timestamp in range(1050, 1451, 50):
            result = self.predictor.frame(absent, timestamp)
        self.assertEqual(len(self.model.inputs), 1, 'Missing-hand deadline exceeded 400 ms')
        self.assertEqual(result['status'], 'prediction')
        present = self.model.inputs[0][0, :, 126] == 1
        self.assertLess(float(self.model.inputs[0][0, present, 0].min()), .15)
        self.assertGreater(float(self.model.inputs[0][0, present, 0].max()), .55)

    def test_live_stillness_change_keeps_captured_movement(self):
        for timestamp in range(0, 1001, 50):
            self.predictor.frame(hand(.1 + .5 * timestamp / 1000), timestamp)
        capture = self.predictor.capture
        for timestamp in range(1050, 1201, 50):
            result = self.predictor.frame(hand(.6), timestamp, {'settle_ms': 200, 'missing_ms': 250})
        self.assertEqual(result['status'], 'prediction')
        self.assertIs(self.predictor.capture, capture)
        self.assertEqual(capture.settle_ms, 200)
        self.assertEqual(capture.missing_ms, 250)
        self.assertAlmostEqual(float(self.model.inputs[0][0, :, 0].min()), .1, places=5)
        self.assertAlmostEqual(float(self.model.inputs[0][0, :, 0].max()), .6, places=5)
        self.predictor.reset()
        self.assertEqual(capture.settle_ms, 200, 'Next gesture should keep slider settings')

    def test_live_missing_delay_change_uses_existing_gap_and_buffer(self):
        for timestamp in range(0, 1001, 50):
            self.predictor.frame(hand(.1 + .5 * timestamp / 1000), timestamp)
        absent = np.zeros(128, dtype=np.float32)
        for timestamp in range(1050, 1301, 50):
            result = self.predictor.frame(absent, timestamp, {'missing_ms': 1000})
            self.assertEqual(result['status'], 'capturing')
        result = self.predictor.frame(absent, 1350, {'missing_ms': 200})
        self.assertEqual(result['status'], 'prediction')
        sequence = self.model.inputs[0][0]
        self.assertLess(float(sequence[sequence[:, 126] == 1, 0].min()), .15)
        self.assertEqual(self.predictor.frame(absent, 1400)['keep_result'], True)

    def test_fastest_frame_interval_preserves_a_five_second_movement(self):
        for timestamp in range(0, 5001, 20):
            result = self.predictor.frame(hand(.1 + .6 * timestamp / 5000), timestamp)
            self.assertNotEqual(result['status'], 'incomplete')
        for timestamp in range(5020, 5421, 20):
            result = self.predictor.frame(np.zeros(128, dtype=np.float32), timestamp)
        self.assertEqual(result['status'], 'prediction')
        self.assertGreater(result['frames'], 256)
        self.assertEqual(len(self.model.inputs), 1)

    def test_held_pose_still_uses_current_frame(self):
        self.predictor = LivePredictor(self.model, ['afternoon', 'evening'], Extractor(), 'pose')
        result = self.predictor.frame(hand(.7), 0)
        self.assertEqual(result['status'], 'prediction')
        np.testing.assert_allclose(self.model.inputs[0][0, :, 0], .7)
        result = self.predictor.frame(np.zeros(128, dtype=np.float32), 200)
        self.assertEqual(result['status'], 'no_hand')


if __name__ == '__main__':
    unittest.main()
