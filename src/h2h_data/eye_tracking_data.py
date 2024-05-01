from typing import Optional, List, Dict, Tuple, Union, Any, ClassVar
from skspatial.objects import Point, Vector, Line, Cylinder
from collections.abc import Sequence
import numpy as np


from h2h_data.re_sample import SamplingMethod, re_sample
from h2h_data import SessionData


class EyeTrackingData(Sequence):
    """
    Represents a fast and convenient Python interface for the additional calculated eye tracking features for the H2H
    session data.

    The object is accessed as a sequence of trials. Note the original files index starting with trial 1, and we
    continue that here - there is no trial 0.

    Each element of the sequence looks as follows:
    SessionData[trial] = {
        'force': {subject: [(timestamp, 5D vector force)]},
        'imu': {subject: [(timestamp, 3D vector)]},
        'mocap': {marker: [(timestamp, 3D point)]},
        'object': [(timestamp, cylinder)],
        'role': str(Role),
        'start_side': str(StartSide),
        'position': str(HandoverPosition),
        'trial_type': str(TrialType),
        'object_type': str(ObjectType),
        'gaze2d': {subject: [(timestamp, 3D Line, 3D Point)]},
        'gaze3d': {subject: [(timestamp, 3D Line, 3D Point)]},
        'handover': float
    }
    The enums in () come from session_data.py. These describe the possible values the strings take on. The actual data
    being output is strings, but refer to session_data.py for their possible values.

    Attributes:
        fps: (class attribute) The framerate the data has been sampled to.
        object_markers: (class attribute) The markers needed to generate the object cylinder.
        sampling_method: The sampling method used to re sample the data to the gaze fps.
                         Options are listed in SamplingMethod in re_sample.py.
        loaded: True if the data has been loaded, False otherwise.
        session: The id string for the session.
        date: The date the session data was collected on.
        _session_data: The session data object we use for the composition pattern.
        _force_data: A dictionary of the force data for the current session.
            Structured: {trial: {subject: [(timestamp, 5d_vec force)]}}.
        _imu_data: A dictionary of the imu data for the current session.
            Structured: {trial: {subject: [(timestamp, Point)]}}.
        _mocap_data: A dictionary of the mocap data for the current session.
            Structured: {trial: {marker: [(timestamp, Point)]}}.
        _head_pose_data: A dictionary of the head pose data for the current session.
            Structured: {trial: {subject: [(timestamp, 3d_vec point, 4d_vec quaternion)]}}.
        _object_data: A dictionary of the object data for the current session.
            Structured: {trial: {marker: [(timestamp, Cylinder)]}}.
        _gaze2d_data: A dictionary of the gaze 2d data for the current session.
            Structured: {trial: {subject: [(timestamp, Line, Point)]}}.
        _gaze3d_data: A dictionary of the gaze 3d data for the current session.
            Structured: {trial: {subject: [(timestamp, Line, Point)]}}.
        _handover_data: A dictionary of the computed handover data for the current session.
            Structured: {trial: timestamp}.
    """
    fps: ClassVar[float] = 60.0
    object_markers: ClassVar[List[str]] = ['OB1', 'OB2', 'OB3']
    sampling_method: str
    loaded: bool
    session: str
    date: str
    _session_data: SessionData
    _force_data: Dict[int, Dict[int, List[Tuple[float, np.ndarray]]]]
    _imu_data: Dict[int, Dict[int, List[Tuple[float, Vector]]]]
    _mocap_data: Dict[int, Dict[str, List[Tuple[float, Point]]]]
    _head_pose_data: Dict[int, Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]]]
    _object_data: Dict[int, List[Tuple[float, Cylinder]]]
    _gaze2d_data: Dict[int, Dict[int, List[Tuple[float, Line, Point]]]]
    _gaze3d_data: Dict[int, Dict[int, List[Tuple[float, Line, Point]]]]
    _handover_data: Dict[int, float]

    def __init__(self, session_data: SessionData, sampling_method: Optional[str] = 'timestamps'):
        """
        Initializes the object with the given session data object.

        Args:
            session_data: The session data object containing the H2H session data.
            sampling_method: The sampling method used to re sample the data to the gaze fps.

        Raises:
            ValueError: If the provided session_data does not include the object markers.
        """
        # Check that object markers are part of the session_data.
        for object_marker in self.object_markers:
            if object_marker not in session_data.target_markers:
                raise ValueError('Object marker "' + object_marker + '" must be included in session data.')
        # Check that the sampling method is valid, and set self.sampling_method.
        self.sampling_method = SamplingMethod(sampling_method).value
        # Init public attributes.
        self.loaded = False
        self.session = None
        self.date = None
        # Init private attributes.
        self._session_data = session_data
        self._force_data = None
        self._imu_data = None
        self._mocap_data = None
        self._head_pose_data = None
        self._object_data = None
        self._gaze2d_data = None
        self._gaze3d_data = None
        self._handover_data = None

    def __len__(self) -> int:
        """
        Gets the number of trials in this session.

        Returns:
            The number of trials in this session.
        """
        return len(self._session_data)

    def __getitem__(self, k: int) -> Dict[str, Any]:
        """
        Gets the data for trial k. Note the original files index starts with trial one, and this function follows
        the same paradigm, so there is no trial 0.

        Args:
            k: The trial we want to access, starting with 1.

        Returns:
            A dict with the data for each trial. Refer to the class documentation for the format.

        Raises:
            KeyError when a non-integer key, or one not in self._trials is entered.
        """
        # We check if k is an integer that is in self.trials().
        if not isinstance(k, int) or k not in self.trials():
            raise KeyError()
        # Use the internal session data object to get the trial data we will be re-using.
        old_trial_data = self._session_data[k]
        # Construct each new element of the trial data, accounting for missing data.
        force_data = self._getitem_copy(self._force_data, k)
        imu_data = self._getitem_copy(self._imu_data, k)
        mocap_data = self._getitem_copy(self._mocap_data, k)
        object_data = self._getitem_copy(self._object_data, k)
        gaze2d_data = self._getitem_copy(self._gaze2d_data, k)
        gaze3d_data = self._getitem_copy(self._gaze3d_data, k)
        handover_data = self._getitem_copy(self._handover_data, k)
        # Return a dict with all the data.
        return {
            'force': force_data,
            'imu': imu_data,
            'mocap': mocap_data,
            'object': object_data,
            'role': old_trial_data['role'],
            'start_side': old_trial_data['start_side'],
            'handover_position': old_trial_data['handover_position'],
            'trial_type': old_trial_data['trial_type'],
            'object_size': old_trial_data['object_size'],
            'gaze2d': gaze2d_data,
            'gaze3d': gaze3d_data,
            'handover': handover_data
        }

    @staticmethod
    def _getitem_copy(data: Optional[Dict[int, Union[str, Any]]], k: int) -> Optional[Any]:
        """
        Gets the copy of the item at trail k for the given private attribute.
        Returns None for the item if the attribute is None.
        For dicts of strings, just returns the string since it is immutable.

        Args:
            data: An attribute of self, that might be None.
            k: The trial we want to get the data for.

        Returns:
            The data for the given trial for the given attribute, or None.
        """
        if data is None:
            return None
        item = data[k]
        if isinstance(item, str):
            return item
        else:
            return item.copy()

    def pop(self, k: int) -> Dict[str, Any]:
        """
        Pops the data for trial k. Note the original files index starts with trial one, and this function follows
        the same paradigm, so there is no trial 0. Also note this removes the trial k.

        Modifies the private attributes:
            - self._session_data
            - self._force_data
            - self._imu_data
            - self._mocap_data
            - self._head_pose_data
            - self._object_data
            - self._gaze2d_data
            - self._gaze3d_data
            - self._handover_data

        Args:
            k: The trial we want to pop, starting with 1.

        Returns:
            A dict with the data for each trial. Refer to the class documentation for the format.

        Raises:
            KeyError when a non-integer key, or one not in self._trials is entered.
        """
        # Get the data for this trial. Automatically throws a KeyError.
        trial_data = self[k]
        # Carefully remove k from each internal dict, and the session data.
        self._session_data.pop(k)
        self._force_data = self._remove_safe(self._force_data, k)
        self._imu_data = self._remove_safe(self._imu_data, k)
        self._mocap_data = self._remove_safe(self._mocap_data, k)
        self._head_pose_data = self._remove_safe(self._head_pose_data, k)
        self._object_data = self._remove_safe(self._object_data, k)
        self._gaze2d_data = self._remove_safe(self._gaze2d_data, k)
        self._gaze3d_data = self._remove_safe(self._gaze3d_data, k)
        self._handover_data = self._remove_safe(self._handover_data, k)
        # Return the trial data for pop.
        return trial_data

    @staticmethod
    def _remove_safe(data: Optional[Dict], k: int) -> Optional[Any]:
        """
        Removes the item at trial k for the given private attribute.
        Returns the modified private attribute.
        Returns None for the item if the attribute is None.

        Args:
            data: An attribute of self, that might be None.
            k: The trial we want to remove the data for.

        Returns:
            The modified attribute.
        """
        if data is None or not isinstance(data, dict):
            return data
        else:
            data.pop(k)
            return data

    def trials(self) -> List[int]:
        """
        Gets the trials in this session.

        Returns:
            A list of trial numbers.
        """
        return self._session_data.trials()

    def load(self) -> None:
        """
        Loads the session data if needed. Then, re-samples everything to the gaze fps. Finally, creates the skspatial
        object as needed.


        Modifies the public attributes:
            - self.session
            - self.date
            - self.loaded
        Modifies the private attributes:
            - self._session_data
            - self._force_data
            - self._imu_data
            - self._mocap_data
            - self._head_pose_data
            - self._object_data
            - self._gaze2d_data
            - self._gaze3d_data
            - self._handover_data
        """
        # Load the session data if needed.
        if not self._session_data.loaded:
            self._session_data.load()
        # Get the session, date, and gaze data.
        self.session = self._session_data.session
        self.date = self._session_data.date
        self._gaze2d_data = {trial: self._session_data[trial]['gaze2d'] for trial in self.trials()}
        self._gaze3d_data = {trial: self._session_data[trial]['gaze3d'] for trial in self.trials()}
        # Re-sample everything else to be the same fps as gaze.
        self._re_sample_non_gaze()
        self._re_sample_handover({trial: self._session_data[trial]['handover'] for trial in self.trials()})
        # Convert the data to skspatial objects, and generate the object cylinder.
        self._imu_data = self._get_imu()
        self._mocap_data = self._get_mocap()
        self._object_data = self._get_object()
        self._gaze2d_data = self._get_gaze2d()
        self._gaze3d_data = self._get_gaze3d()
        # Mark the data as loaded.
        self.loaded = True

    def _re_sample_non_gaze(self) -> None:
        """
        Resamples all the data that is not gaze to be at gaze fps. Uses self.sampling_method to do so.

        Modifies the private attributes:
            - self._force_data
            - self._imu_data
            - self._mocap_data
            - self._head_pose_data
        """
        self._force_data = {}
        self._imu_data = {}
        self._mocap_data = {}
        self._head_pose_data = {}
        for trial in self.trials():
            re_sample_args = self._get_re_sample_args(trial)
            self._force_data[trial] = self._re_sample_subjects(self._session_data[trial]['force'], re_sample_args[0])
            self._imu_data[trial] = self._re_sample_subjects(self._session_data[trial]['imu'], re_sample_args[1])
            self._mocap_data[trial] = self._re_sample_mocap(self._session_data[trial]['mocap'], re_sample_args[2])
            self._head_pose_data[trial] = self._re_sample_subjects(self._session_data[trial]['head_pose'],
                                                                   re_sample_args[3])

    def _get_re_sample_args(self, trial: int) -> List[Dict[str, Any]]:
        """
        Gets a list of arguments for re-sampling each of:
            0. self._force_data
            1. self._imu_data
            2. self._mocap_data
            3. self._head_pose_data

        Args:
            The trial we want to re-sample.

        Returns:
            A list of arguments for resampling each internal variable.

        Raises:
            RuntimeError: If the re-sampling method is not supported.
        """
        if self.sampling_method == 'timestamps':
            # Time everything to subject 1's 3D gaze.
            timestamps = [element[0] for element in self._session_data[trial]['gaze3d'][1]]
            output = [{'timestamps': timestamps}] * 4
        elif self.sampling_method == 'simple':
            # Get the appropriate sets of FPS.
            output = [{'source_fps': self._session_data.force_fps, 'target_fps': self._session_data.gaze_fps},
                      {'source_fps': self._session_data.imu_fps,   'target_fps': self._session_data.gaze_fps},
                      {'source_fps': self._session_data.mocap_fps, 'target_fps': self._session_data.gaze_fps},
                      {'source_fps': self._session_data.mocap_fps, 'target_fps': self._session_data.gaze_fps}]
        else:
            raise RuntimeError(f'Sampling method {self.sampling_method} is not supported.')
        return output

    def _re_sample_subjects(self, subjects_data: Dict[int, List[Tuple[float, Any]]], args: Dict[str, Any]) \
            -> Dict[int, List[Tuple[float, Any]]]:
        """
        Re-samples subjects data, which has a list of frames for each subject.

        Args:
            subjects_data: Data frames for each subject, {subject: [(timestamp, ...)]}.
            args: The re-sampling args. These are the same for each subject.

        Returns:
            The re-sampled subjects data.
        """
        return {1: re_sample(subjects_data[1], self.sampling_method, args),
                2: re_sample(subjects_data[2], self.sampling_method, args)}

    def _re_sample_mocap(self, mocap_data: Dict[int, List[Tuple[float, Any]]], args: Dict[str, Any]) \
            -> Dict[int, List[Tuple[float, Any]]]:
        """
        Re-samples mocap data, which has a list of frames for each marker.

        Args:
            mocap_data: Data frames for each marker, {marker: [(timestamp, ...)]}.
            args: The re-sampling args. These are the same for each marker.

        Returns:
            The re-sampled subjects data.
        """
        output = {}
        for marker in mocap_data:
            output[marker] = re_sample(mocap_data[marker], self.sampling_method, args)
        return output

    def _re_sample_handover(self, handover_data: Dict[int, float]) -> Dict[int, float]:
        """
        Re-samples the handover for each trial. Does this by selecting the gaze timestamp closest to the handover
        timestamp.

        Args:
            handover_data: The handover timestamp for each trial.

        Returns:
            The re-sampled handover data.
        """
        for trial in handover_data:
            # Grab the list of gaze timestamps from subject 1's 3D gaze.
            gaze_timestamps = [element[0] for element in self._gaze3d_data[trial][1]]
            # Choosing the closest timestamp to handover.
            min_error = float('inf')
            handover = gaze_timestamps[0]
            for timestamp in gaze_timestamps:
                error = abs(timestamp - handover)
                if error < min_error:
                    handover = timestamp
            # Setting the output data to our answer.
            handover_data[trial] = handover
        return handover_data
