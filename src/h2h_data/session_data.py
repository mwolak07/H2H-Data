from typing import ClassVar, List, Dict, Tuple, Optional, Set, Any, Union
from collections.abc import Sequence
from scipy import signal
from enum import Enum
import numpy as np
import traceback
import mat73


# TODO: Integrate a more flexible reader that can read only specific trials (performance).
# TODO: Replace lists with numpy fixed memory allocation (performance).


class HandoverMethod(Enum):
    """
    Represents the method by which we calculate handover.
    """
    VELOCITY = 'velocity'
    DISTANCE = 'distance'


class Role(Enum):
    """
    Represents the possible roles for each trial.
    """
    SUB1_IG = 'Sub1_IG'
    SUB1_IR = 'Sub1_IR'
    SUB2_IG = 'Sub2_IG'
    SUB2_IR = 'Sub2_IR'


class StartSide(Enum):
    """
    Represents the possible start sides for each trial.
    """
    LEFT = 'L'
    RIGHT = 'R'


class HandoverPosition(Enum):
    """
    Represents the possible final handover position instructions for each trial.
    """
    BOTTOM_LEFT = 'BL'
    BOTTOM_RIGHT = 'BR'
    TOP_LEFT = 'TL'
    TOP_RIGHT = 'TR'


class TrialType(Enum):
    """
    Represents the type of each trial. 1-5 are standard trials, P is a perturbation.
    """
    ONE = '1'
    TWO = '2'
    THREE = '3'
    FOUR = '4'
    FIVE = '5'
    PERTURBATION = 'P'


class ObjectSize(Enum):
    """
    Represents the size of the object for each trial.
    """
    LARGE = 'LO'
    SMALL = 'SO'


class SessionData(Sequence):
    """
    Represents a fast and convenient Python interface for the H2H session data, stored in Matlab 7.3 files.
    Each SessionData object corresponds to one session, and contains the data gathered for each trial, except
    for images and video. (Currently, we only include mocap).
    Handover time is also provided, computed from the velocity profile of the follower wrist marker.

    The object is accessed as a sequence of trials. Note the original files index starting with trial 1, and we
    continue that here - there is no trial 0.

    Each element of the sequence looks as follows:
    SessionData[trial] = {
        'force': {subject: [(timestamp, 5d_vec force)]},
        'imu': {subject: [(timestamp, 3d_vec imu)]},
        'mocap': {marker: [(timestamp, point)]},
        'head_pose': {subject: [(timestamp, 3d_vec point, 4d_vec quaternion)]},
        'object_pose': [(timestamp, 3d_vec point, 4d_vec quaternion)],
        'role': str(Role),
        'start_side': str(StartSide),
        'position': str(HandoverPosition),
        'trial_type': str(TrialType),
        'object_type': str(ObjectType),
        'gaze2d': {subject: [(timestamp, 2d_vec point)]},
        'gaze3d': {subject: [(timestamp, 2d_vec point)]},
        'handover': float
    }
    We export the enums as strings, to avoid complicated representations. The enum in () aids in understanding the
    expected options for each string.

    The mocap will only have the markers in the target_markers dict, if it was provided. All of them will be there
    otherwise.

    The object also supports the len() function, and can be used as an iterator in loops.

    Attributes:
        force_fps: (class attribute) The framerate the force data was captured at.
        imu_fps: (class attribute) The framerate the imu data was captured at.
        mocap_fps: (class attribute) The framerate the mocap data was captured at.
        gaze_fps: (class attribute) The framerate the gaze data was captured at.
        wrist_marker_name: (class attribute) The name of the wrist marker, this is always loaded to compute handover.
        sub_1_tag: (class attribute) The preceding "tag" for subject 1's markers (used for handover).
        sub_2_tag: (class attribute) The preceding "tag" for subject 2's markers (used for handover).
        session_file: The path to the Matlab 7.3 file containing the session's data.
        target_markers: The list of markers to include. (If None, include all).
        loaded: True if the data has already been loaded.
        session: The id string for the session.
        date: The date the session data was collected on.
        _trials: A list of trials in this session (1, 2, 3, etc.).
        _force_data: A dictionary of the force data for the current session.
            Structured: {trial: {subject: [(timestamp, 5d_vec force)]}}.
        _imu_data: A dictionary of the imu data for the current session.
            Structured: {trial: {subject: [(timestamp, 3d_vec imu)]}}.
        _mocap_data: A dictionary of the mocap data for the current session.
            Structured: {trial: {marker: [(timestamp, 3d_vec point)]}}.
        _wrist_data: A dictionary of the wrist marker data for the current session.
            Structured: {trial: {subject: [(timestamp, 3d_vec point)]}}.
        _head_pose_data: A dictionary of the head pose data for the current session.
            Structured: {trial: {subject: [(timestamp, 3d_vec point, 4d_vec quaternion)]}}.
        _object_pose_data: A dictionary of the object pose data for the current session.
            Structured: {trial: [(timestamp, 3d_vec point, 4d_vec quaternion)]}.
        _role_data: A dictionary of the role data for the current session.
            Structured: {trial: role}.
        _start_side_data: A dictionary of the start side for the current session.
            Structured: {trial: start side}.
        _handover_position_data: A dictionary of the handover position data for the current session.
            Structured: {trial: handover position}.
        _trial_type_data: A dictionary of the trial type data for the current session.
            Structured: {trial: trial type}.
        _object_size_data: A dictionary of the object type data for the current session.
            Structured: {trial: object size}.
        _gaze2d_data: A dictionary of the gaze 2d data for the current session.
            Structured: {trial: {subject: [(timestamp, 2d_vec point)]}}.
        _gaze3d_data: A dictionary of the gaze 3d data for the current session.
            Structured: {trial: {subject: [(timestamp, 3d_vec point)]}}.
        _handover_data: A dictionary of the computed handover data for the current session.
            Structured: {trial: timestamp}.
    """
    force_fps: ClassVar[float] = 120.0
    imu_fps: ClassVar[float] = 45.0
    mocap_fps: ClassVar[float] = 100.0
    gaze_fps: ClassVar[float] = 60.0
    wrist_marker_name: ClassVar[str] = 'RUSP'
    sub_1_tag: ClassVar[str] = 'Sub1_'
    sub_2_tag: ClassVar[str] = 'Sub2_'
    session_file: str
    target_markers: Set[str]
    handover_method: HandoverMethod
    loaded: bool
    session: str
    date: str
    _trials: List[int]
    _force_data: Dict[int, Dict[int, List[Tuple[float, np.ndarray]]]]
    _imu_data: Dict[int, Dict[int, List[Tuple[float, np.ndarray]]]]
    _mocap_data: Dict[int, Dict[str, List[Tuple[float, np.ndarray]]]]
    _wrist_data: Dict[int, Dict[int, List[Tuple[float, np.ndarray]]]]
    _head_pose_data: Dict[int, Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]]]
    _object_pose_data: Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]]
    _role_data: Dict[int, Role]
    _start_side_data: Dict[int, StartSide]
    _handover_position_data: Dict[int, HandoverPosition]
    _trial_type_data: Dict[int, TrialType]
    _object_size_data: Dict[int, ObjectSize]
    _gaze2d_data: Dict[int, Dict[int, List[Tuple[float, np.ndarray]]]]
    _gaze3d_data: Dict[int, Dict[int, List[Tuple[float, np.ndarray]]]]
    _handover_data: Dict[int, float]

    def __init__(self, session_file: str, target_markers: Optional[List[str]] = None,
                 handover_method: Optional[str] = 'velocity', print_warnings: Optional[bool] = False,
                 debug: Optional[bool] = False):
        """
        Initializes the object with the given session file and target markers.
        Does not read any data until .load() is called.

        Args:
            session_file: The path to the Matlab 7.3 file containing the session's data.
            target_markers: The list of markers to include. (If None, include all). The names should have the subject
                            tags included, like: 'Sub1_FH'.
            handover_method: The method by which we calculate the handover timestamp.
            print_warnings: If True, we will print warnings to the console.
            debug: If True, we are in debug mode.
        """
        # Initialize public attributes.
        self.session_file = session_file
        self.target_markers = set(target_markers) if target_markers is not None else None
        self.handover_method = HandoverMethod(handover_method)
        self.print_warnings = print_warnings
        self.debug = debug
        self.loaded = False
        self.session = None
        self.date = None
        # Initialize private attributes.
        self._trials: None
        self._force_data: None
        self._imu_data: None
        self._mocap_data: None
        self._wrist_data: None
        self._head_pose_data: None
        self._object_pose_data: None
        self._role_data: None
        self._start_side_data: None
        self._handover_position_data: None
        self._trial_type_data: None
        self._object_size_data: None
        self._gaze2d_data: None
        self._gaze3d_data: None
        self._handover_data: None

    def __len__(self) -> int:
        """
        Gets the number of trials in this session.

        Returns:
            The number of trials in this session.
        """
        return len(self._trials)

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
        # We check if k is an integer that is in self._trials.
        if not isinstance(k, int) or k not in self._trials:
            raise KeyError()
        # Construct each element of the trial data, accounting for missing data.
        force_data = self._getitem_copy(self._force_data, k)
        imu_data = self._getitem_copy(self._imu_data, k)
        mocap_data = self._getitem_copy(self._mocap_data, k)
        head_pose_data = self._getitem_copy(self._head_pose_data, k)
        object_pose_data = self._getitem_copy(self._object_pose_data, k)
        role_data = self._getitem_copy(self._role_data, k)
        start_side_data = self._getitem_copy(self._start_side_data, k)
        handover_position_data = self._getitem_copy(self._handover_position_data, k)
        trial_type_data = self._getitem_copy(self._trial_type_data, k)
        object_size_data = self._getitem_copy(self._object_size_data, k)
        gaze2d_data = self._getitem_copy(self._gaze2d_data, k)
        gaze3d_data = self._getitem_copy(self._gaze3d_data, k)
        handover_data = self._getitem_copy(self._handover_data, k)
        # Return a dict with all the data.
        return {
            'force': force_data,
            'imu': imu_data,
            'mocap': mocap_data,
            'head_pose': head_pose_data,
            'object_pose': object_pose_data,
            'role': role_data,
            'start_side': start_side_data,
            'handover_position': handover_position_data,
            'trial_type': trial_type_data,
            'object_size': object_size_data,
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
            - self._trials
            - self._force_data
            - self._imu_data
            - self._mocap_data
            - self._wrist_data
            - self._head_pose_data
            - self._object_pose_data
            - self._role_data
            - self._start_side_data
            - self._handover_position_data
            - self._trial_type_data
            - self._object_size_data
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
        # Carefully remove k from each internal dict.
        self._trials = self._remove_safe(self._trials, k)
        self._force_data = self._remove_safe(self._force_data, k)
        self._imu_data = self._remove_safe(self._imu_data, k)
        self._mocap_data = self._remove_safe(self._mocap_data, k)
        self._wrist_data = self._remove_safe(self._wrist_data, k)
        self._head_pose_data = self._remove_safe(self._head_pose_data, k)
        self._object_pose_data = self._remove_safe(self._object_pose_data, k)
        self._role_data = self._remove_safe(self._role_data, k)
        self._start_side_data = self._remove_safe(self._start_side_data, k)
        self._handover_position_data = self._remove_safe(self._handover_position_data, k)
        self._trial_type_data = self._remove_safe(self._trial_type_data, k)
        self._object_size_data = self._remove_safe(self._object_size_data, k)
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
        return self._trials.copy()

    def load(self) -> None:
        """
        Loads the data from the session file to this object. Parses the data into the output format we want.

        Modifies the public attributes:
            - self.session
            - self.date
            - self.loaded
        Modifies the private attributes:
            - self._trials
            - self._force_data
            - self._imu_data
            - self._mocap_data
            - self._wrist_data
            - self._head_pose_data
            - self._object_pose_data
            - self._role_data
            - self._start_side_data
            - self._handover_position_data
            - self._trial_type_data
            - self._object_size_data
            - self._gaze2d_data
            - self._gaze3d_data
            - self._handover_data
        """
        # Read the session file.
        session_data = mat73.loadmat(self.session_file)
        # Parsing the data for which trials are valid.
        self._trials = self._parse_trials(session_data)
        # Parse the data for each internal dict.
        self.session = self._parse_session(session_data)
        self.date = self._parse_date(session_data)
        self._force_data = self._parse_force(session_data)
        self._imu_data = self._parse_imu(session_data)
        self._mocap_data = self._parse_mocap(session_data)
        self._wrist_data = self._parse_wrist(session_data)
        self._head_pose_data = self._parse_head_pose(session_data)
        self._object_pose_data = self._parse_object_pose(session_data)
        self._role_data = self._parse_role(session_data)
        self._start_side_data = self._parse_start_side(session_data)
        self._handover_position_data = self._parse_handover_position(session_data)
        self._trial_type_data = self._parse_trial_type(session_data)
        self._object_size_data = self._parse_object_size(session_data)
        self._gaze2d_data = self._parse_gaze2d(session_data)
        self._gaze3d_data = self._parse_gaze3d(session_data)
        self._handover_data = self._get_handover()
        # Mark the data as loaded.
        self.loaded = True

    def _parse_trials(self, session_data: Dict[str, Any]) -> List[int]:
        """
        Parses the number of each trial from the session data

        We derive this from the role data. It is important to take into account missing trials - some have an empty
        role, and these should be skipped.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The list of trials in this session.
        """
        # Get the role list.
        role_list = session_data['cropped_data']['role']
        # For each trial with a valid role, add it to the internal list of trials in this session.
        trials = []
        for i, role in enumerate(role_list):
            if role is not None:
                # Make sure to offset i by 1.
                trials.append(i + 1)
            elif self.print_warnings:
                print(f'Warning: Trial {i + 1} empty.')
        return trials

    @staticmethod
    def _parse_session(session_data: Dict[str, Any]) -> str:
        """
        Parses the session id string from the session data.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The session id string.
        """
        return session_data['cropped_data']['session']

    @staticmethod
    def _parse_date(session_data: Dict[str, Any]) -> str:
        """
        Parses the date string from the session data.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The date string.
        """
        return session_data['cropped_data']['date']

    def _parse_force(self, session_data: Dict[str, Any]) -> Dict[int, Dict[int, List[Tuple[float, np.ndarray]]]]:
        """
        Parses the force data from the session data.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The force data, represented as: {trial: {subject: [(timestamp, 5d_vec force)]}}.
        """
        return self._parse_subject_frames(session_data, 'cropped_force', self.force_fps)

    def _parse_subject_frames(self, session_data: Dict[str, Any], field: str, fps: float) \
            -> Dict[int, Dict[int, List[Tuple[float, np.ndarray]]]]:
        """
        Parses the data from the session data. This data must be split by subject, then into multiple frames for each
        subject.

        Details:
            - We extract elements along: session_data['cropped_data'][field][trial][subject].
            - We assume each element is a 2D numpy array of shape (frames, d).

        Args:
            session_data: The data dict loaded from the session file.
            field: The field for the specific data we want to extract.
            fps: The frames per second to use to generate timestamps.

        Returns:
            The field data, represented as: {trial: {subject: [(timestamp, 2D numpy array)]}}.
        """
        # Get the list of force data for each trial.
        data_list = session_data['cropped_data'][field]
        # Parse the force data for each trial.
        output = {}
        for trial in self._trials:
            # self._trials uses trial number, trial index is trial - 1.
            i = trial - 1
            # Get the data for each subject.
            output[trial] = {}
            for subject_i in range(2):
                # data_list[i] with have element 0 and 1, but we store as subject 1 and 2.
                subject = subject_i + 1
                # Get the data for the subject.
                output[trial][subject] = self._add_frames_timestamps(data_list[i][subject_i], fps)
        return output

    @staticmethod
    def _add_frames_timestamps(frames_data: np.ndarray, fps: float) -> List[Tuple[float, np.ndarray]]:
        """
        Adds timestamps to the numpy array of data for each frame.

        Args:
            frames_data: A shape (n_frames, d) numpy array of data for each frame.
            fps: The frames per second to use to generate timestamps.

        Returns:
            A list of pairs of (timestamp, 1D numpy array).
        """
        # Get the frames data from a 2D array into a list of 1D arrays.
        frames_data_list = []
        for frame_data in frames_data:
            frames_data_list.append(np.array(frame_data))
        # Get the list of timestamps.
        frame_time = 1 / fps
        max_time = frames_data.shape[0] * frame_time
        timestamp_list = np.arange(0, max_time, frame_time, dtype=float).tolist()
        # Combine the lists.
        return list(zip(timestamp_list, frames_data_list))

    def _parse_imu(self, session_data: Dict[str, Any]) -> Dict[int, Dict[int, List[Tuple[float, np.ndarray]]]]:
        """
        Parses the imu data from the session data.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The imu data, represented as: {trial: {subject: [(timestamp, 3d_vec imu)]}}.
        """
        return self._parse_subject_frames(session_data, 'cropped_imu', self.imu_fps)

    def _parse_mocap(self, session_data: Dict[str, Any]) -> Dict[int, Dict[str, List[Tuple[float, np.ndarray]]]]:
        """
        Parses the mocap data from the session data

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The mocap data, represented as: {trial: {marker: [(timestamp, Point)]}}.
        """
        # Get the list of mocap data for each trial.
        mocap_list = session_data['cropped_data']['cropped_mocap']
        # Parse the mocap data for each trial.
        mocap_data = {}
        for trial in self._trials:
            # self._trials uses trial number, trial index is trial - 1.
            i = trial - 1
            # Get the corresponding lists of marker names and marker points.
            label_list = mocap_list[i]['Labels']
            point_list = mocap_list[i]['Loc']
            # Get the list of target labels. If self._target_markers is None, we take all the labels.
            target_labels = self.target_markers if self.target_markers is not None else label_list
            # Get the mocap data for each target marker for each subject.
            mocap_data[trial] = {}
            for j in range(len(label_list)):
                label = label_list[j]
                if label in target_labels:
                    mocap_data[trial][label] = self._add_frames_timestamps(point_list[j], self.mocap_fps)
        return mocap_data

    def _parse_wrist(self, session_data: Dict[str, Any]) -> Dict[int, Dict[int, List[Tuple[float, np.ndarray]]]]:
        """
        Parses the wrist data from the session data

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The mocap data, represented as: {trial: {subject: [(timestamp, Point)]}}.
        """
        # Get the names of each subject's wrist marker.
        sub_1_wrist_marker = self.sub_1_tag + self.wrist_marker_name
        sub_2_wrist_marker = self.sub_2_tag + self.wrist_marker_name
        # Get the list of mocap data for each trial.
        mocap_list = session_data['cropped_data']['cropped_mocap']
        # Get the wrist data for each trial.
        wrist_data = {}
        for trial in self._trials:
            # self._trials uses trial number, trial index is trial - 1.
            i = trial - 1
            # Get the index of each wrist marker in the marker list based on the label list.
            label_list = mocap_list[i]['Labels']
            sub_1_i = label_list.index(sub_1_wrist_marker)
            sub_2_i = label_list.index(sub_2_wrist_marker)
            # Parse the mocap data for the wrist marker for each subject.
            point_list = mocap_list[i]['Loc']
            wrist_data[trial] = {
                1: self._add_frames_timestamps(point_list[sub_1_i], self.mocap_fps),
                2: self._add_frames_timestamps(point_list[sub_2_i], self.mocap_fps)
            }
        return wrist_data

    def _parse_head_pose(self, session_data: Dict[str, Any]) \
            -> Dict[int, Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]]]:
        """
        Parses the head position and orientation data from the session data.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The head pose data, represented as: {trial: {subject: [(timestamp, 3d_vec point, 4d_vec quaternion)]}}.
        """
        # Get the list of pose data for each trial.
        head_pose_list = session_data['cropped_data']['cropped_head_pose']
        # Parse the pose data for each trial.
        head_pose_data = {}
        for trial in self._trials:
            # self._trials uses trial number, trial index is trial - 1.
            i = trial - 1
            # Get the corresponding lists of positions and orientations.
            position_list = head_pose_list[i]['Position']
            orientation_list = head_pose_list[i]['Orientation']
            # Get the head pose data for each subject.
            head_pose_data[trial] = {}
            for subject_i in range(2):
                # pose_list[i] will have element 0 and 1, but we store as subject 1 and 2.
                subject = subject_i + 1
                # Add timestamps to the position data, then add the orientation data using some zip logic.
                position_data = self._add_frames_timestamps(position_list[subject_i], self.mocap_fps)
                orientation_data = [orientation for orientation in orientation_list[subject_i]]
                pose_data = list(zip(position_data, orientation_data))
                head_pose_data[trial][subject] = [(pose[0][0], pose[0][1], pose[1]) for pose in pose_data]
        return head_pose_data

    def _parse_object_pose(self, session_data: Dict[str, Any]) -> Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]]:
        """
        Parses the object position and orientation data from the session data.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The object pose data, represented as: {trial: [(timestamp, 3d_vec point, 4d_vec quaternion)]}.
        """
        # Get the list of pose data for each trial.
        object_pose_list = session_data['cropped_data']['cropped_object_pose']
        # Parse the pose data for each trial.
        object_pose_data = {}
        for trial in self._trials:
            # self._trials uses trial number, trial index is trial - 1.
            i = trial - 1
            # Get the corresponding lists of positions and orientations.
            position_list = [position for position in object_pose_list[i]['Position']]
            orientation_list = [orientation for orientation in object_pose_list[i]['Orientation']]
            # Add timestamps to the position list, then add the orientation list using some zip logic.
            position_list = self._add_frames_timestamps(position_list, self.mocap_fps)
            pose_data = list(zip(position_list, orientation_list))
            object_pose_data[trial] = [(pose[0][0], pose[0][1], pose[1]) for pose in pose_data]
        return object_pose_data

    def _parse_role(self, session_data: Dict[str, Any]) -> Dict[int, Role]:
        """
        Parses the role data from the session data

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The role data, represented as: {trial: role}.
        """
        # Get the dict of string data.
        role_data = self._parse_trial_strings(session_data, 'role')
        # Convert each element to a Role.
        for trial in self._trials:
            role_data[trial] = Role(role_data[trial])
        return role_data

    def _parse_trial_strings(self, session_data: Dict[str, Any], field: str) -> Dict[int, str]:
        """
        Parses the string list data from the session data.

        Details:
            - We extract elements along: session_data['cropped_data'][field].
            - We assume each element is a string.

        Args:
            session_data: The data dict loaded from the session file.
            field: The field for the specific data we want to extract.

        Returns:
            The string for each trial, represented as {trial: string}.
        """
        # Get the string list.
        string_list = session_data['cropped_data'][field]
        # For each trial, record the string.
        string_data = {}
        # For each trial with a valid string, add it to the internal list of trials in this session.
        for trial in self._trials:
            # self._trials uses trial number, trial index is trial - 1.
            i = trial - 1
            string_data[trial] = string_list[i][0]
        return string_data

    def _parse_start_side(self, session_data: Dict[str, Any]) -> Dict[int, StartSide]:
        """
        Parses the start side data from the session data

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The mocap data, represented as: {trial: start side}.
        """
        # Get the dict of string data.
        start_side_data = self._parse_trial_strings(session_data, 'start_side')
        # Convert each element to a StartSide.
        for trial in self._trials:
            start_side_data[trial] = StartSide(start_side_data[trial])
        return start_side_data

    def _parse_handover_position(self, session_data: Dict[str, Any]) -> Dict[int, HandoverPosition]:
        """
        Parses the handover position data from the session data

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The mocap data, represented as: {trial: handover position}.
        """
        # Get the dict of string data.
        handover_position_data = self._parse_trial_strings(session_data, 'position')
        # Convert each element to a HandoverPosition.
        for trial in self._trials:
            handover_position_data[trial] = HandoverPosition(handover_position_data[trial])
        return handover_position_data

    def _parse_trial_type(self, session_data: Dict[str, Any]) -> Dict[int, TrialType]:
        """
        Parses the trial type data from the session data

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The mocap data, represented as: {trial: trial type}.
        """
        # Get the dict of string data.
        trial_type_data = self._parse_trial_strings(session_data, 'trial')
        # Convert each element to a TrialType.
        for trial in self._trials:
            trial_type_data[trial] = TrialType(trial_type_data[trial])
        return trial_type_data

    def _parse_object_size(self, session_data: Dict[str, Any]) -> Dict[int, ObjectSize]:
        """
        Parses the object size data from the session data

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The mocap data, represented as: {trial: object size}.
        """
        # Get the dict of string data.
        object_size_data = self._parse_trial_strings(session_data, 'object')
        # Convert each element to a TrialType.
        for trial in self._trials:
            object_size_data[trial] = ObjectSize(object_size_data[trial])
        return object_size_data

    def _parse_gaze2d(self, session_data: Dict[str, Any]) -> Dict[int, Dict[int, List[Tuple[float, np.ndarray]]]]:
        """
        Parses the 2D gaze data from the session data.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The 2D gaze data, represented as: {trial: {subject: [(timestamp, 2d_vec point)]}}.
        """
        return self._parse_subject_frames(session_data, 'cropped_gaze', self.gaze_fps)

    def _parse_gaze3d(self, session_data: Dict[str, Any]) -> Dict[int, Dict[int, List[Tuple[float, np.ndarray]]]]:
        """
        Parses the 3D gaze data from the session data.

        Args:
            session_data: The data dict loaded from the session file.

        Returns:
            The 3D gaze data, represented as: {trial: {subject: [(timestamp, 3d_vec point)]}}.
        """
        return self._parse_subject_frames(session_data, 'cropped_gaze3d', self.gaze_fps)

    def _get_handover(self) -> Dict[int, float]:
        """
        Gets the handover using the method defined by self._handover_method.

        Modifies the private attributes:
            - self._trials
            - self._force_data
            - self._imu_data
            - self._mocap_data
            - self._wrist_data
            - self._head_pose_data
            - self._object_pose_data
            - self._role_data
            - self._start_side_data
            - self._handover_position_data
            - self._trial_type_data
            - self._object_size_data
            - self._gaze2d_data
            - self._gaze3d_data
            - self._handover_data

        Returns:
            The handover data, represented as: {trial: handover}.
        """
        if self.handover_method == HandoverMethod.VELOCITY:
            return self._get_handover_from_velocity()
        else:
            return self._get_handover_from_distance()

    def _get_handover_from_velocity(self) -> Dict[int, float]:
        """
        Gets the handover data from the already loaded wrist and role data. Does this by looking at the butterworth
        filtered velocity curve for the wrist marker of the follower.

        Modifies the private attributes:
            - self._trials
            - self._force_data
            - self._imu_data
            - self._mocap_data
            - self._wrist_data
            - self._head_pose_data
            - self._object_pose_data
            - self._role_data
            - self._start_side_data
            - self._handover_position_data
            - self._trial_type_data
            - self._object_size_data
            - self._gaze2d_data
            - self._gaze3d_data

        Returns:
            The handover data, represented as: {trial: timestamp}.
        """
        bad_trials = []
        # Defining thresholds for picking peaks and minimums.
        peak_threshold = 0.2
        min_threshold = 0.1
        handover_data = {}
        for trial in self._trials:
            try:
                follower = 2 if self._role_data[trial] in ['Sub1_IG', 'Sub1_IR'] else 1
                follower_velocity = self._get_follower_velocity(trial, follower)
                # Finding the local maxima using the peak_threshold.
                peaks_i, _ = signal.find_peaks([elem[1] for elem in follower_velocity])
                maxs_i = [i for i in peaks_i if follower_velocity[i][1] > peak_threshold]
                # Finding the local minima by inverting the graph and using min_threshold.
                follower_velocity_inv = [-1 * elem[1] for elem in follower_velocity]
                peaks_i, _ = signal.find_peaks(follower_velocity_inv)
                # Getting the candidate mins, that appear between the maxes we found.
                candidate_mins_i = [i for i in peaks_i if maxs_i[0] < i < maxs_i[-1]]
                # Filtering out the mins that do not have values below our min threshold.
                mins_i = [i for i in candidate_mins_i if follower_velocity[i][1] < min_threshold]
                # If the list is not empty after filtering with the min threshold, we use the end of the list.
                if len(mins_i) != 0:
                    contact_pt = mins_i[-1]
                # If the list is empty, we revert to the candidate mins list and use the end of that list.
                else:
                    contact_pt = candidate_mins_i[-1]
                # Our final output is the timestamp at our contact point.
                handover_data[trial] = follower_velocity[contact_pt][0]
            # When there's an issue finding handover, we remove the trial.
            except Exception as e:
                handover_data[trial] = None
                bad_trials.append(trial)
                if self.print_warnings:
                    print(f'Warning: problem finding handover in trial {trial}: {type(e).__name__}: {e}')
                    if self.debug:
                        print(traceback.format_exc())
        # Remove the bad trials.
        for trial in bad_trials:
            self.pop(trial)
        return handover_data

    def _get_follower_velocity(self, trial: int, follower: int) -> List[Tuple[float, float]]:
        """
        Gets a list of the instantaneous velocity at each point for the given follower subject.
        This does not include the velocity for the first point, since there is no prior point. Velocity is defined as
        the distance between the consecutive coordinates over the frame time.

        Also applies a low-pass butterworth filter with order 4 and cutoff frequency 10 / (fps / 2).

        Args:
            trial: The trial to get the follower velocity for.
            follower: 1 if subject 1 is the follower, 2 if subject 2 is the follower.

        Returns:
            A list of (timestamp, instantaneous velocity), not including the first point.
        """
        # Getting the points with the appropriate trial and marker.
        points = [frame[1] for frame in self._wrist_data[trial][follower]]
        # Filtering out points where we have nan values, as this messes with the butterworth filter.
        points = [point for point in points if not np.any(np.isnan(point))]
        # Getting the velocities from the points.
        velocities = []
        for i in range(1, len(points)):
            d = np.linalg.norm(points[i][1] - points[i - 1][1])
            dt = points[i][0] - points[i - 1][0]
            v = d / dt if dt != 0 else 0
            velocities.append(v)
        # Setting up butterworth filter params.
        filt_order = 4
        cutoff_freq = 10
        filt_cutoff = cutoff_freq / (self.mocap_fps / 2)
        filt_type = 'low'
        # Applying the butterworth filter.
        result = signal.butter(N=filt_order, Wn=filt_cutoff, btype=filt_type, output='ba')
        velocities = signal.filtfilt(result[0], result[1], velocities)
        # Adding the timestamps back in.
        output = [(points[i + 1][0], velocities[i]) for i in range(len(velocities))]
        return output

    def _get_handover_from_distance(self) -> Dict[int, float]:
        """
        Gets the handover from the already loaded wrist data. Does this by looking at the minimum distance
        between the wrist markers.

        Returns:
            The handover data, represented as: {trial: timestamp}.
        """
        handover_data = {}
        for trial in self._trials:
            # Record the minimum distance and handover for this trial.
            min_dist = float('inf')
            handover = 0.0
            # Separate the wrist data for each subject.
            for frame in self._wrist_data[trial][1]:
                sub_1_point = self._wrist_data[trial][1][frame][1]
                sub_2_point = self._wrist_data[trial][2][frame][1]
                dist = np.linalg.norm(sub_1_point - sub_2_point)
                if dist < min_dist:
                    handover = self._wrist_data[trial][1][frame][0]
                    min_dist = dist
            # Record the handover for this trial.
            handover_data[trial] = handover
        return handover_data

    def crop_to_handover(self) -> None:
        """
        Crops all the trials so that they end at object handover.

        If handover is after the end of the trial, we leave the trial alone.
        If we crop anything to empty, remove that trial.

        Modifies the private attributes:
            - self._trials
            - self._force_data
            - self._imu_data
            - self._mocap_data
            - self._wrist_data
            - self._head_pose_data
            - self._object_pose_data
            - self._role_data
            - self._start_side_data
            - self._handover_position_data
            - self._trial_type_data
            - self._object_size_data
            - self._gaze2d_data
            - self._gaze3d_data
            - self._handover_data

        Modifies the private attributes:
            - self._mocap_data
            - self._wrist_data
        """
        bad_trials = []
        # Iterate over each trial.
        for trial in self._trials:
            # Get the handover time from the internal list.
            handover_time = self._handover_data[trial]
            # Crop the trial from 0.0 to handover.
            # If handover > the last timestamp, it will pick the last one as the closest.
            success = self._crop_trial(trial, 0.0, handover_time)
            if not success:
                bad_trials.append(trial)
        # Remove the bad trials.
        for trial in bad_trials:
            self.pop(trial)

    def _crop_trial(self, trial: int, start: float, end: float) -> bool:
        """
        Crops the frames of the trial data to the given start and end timestamps.

        Modifies the private attributes:
            - self._force_data
            - self._imu_data
            - self._mocap_data
            - self._wrist_data
            - self._head_pose_data
            - self._object_pose_data
            - self._gaze2d_data
            - self._gaze3d_data

        Args:
            trial: The trial to crop data for.
            start: The start timestamp.
            end: The end timestamp.

        Returns:
            True if there were no problems, False otherwise.
        """
        # Crop all the internal data that has lists of frames.
        self._force_data[trial], force_ok = self._crop_subjects_trial(self._force_data[trial], start, end)
        self._imu_data[trial], imu_ok = self._crop_subjects_trial(self._imu_data[trial], start, end)
        self._mocap_data[trial], mocap_ok = self._crop_mocap_trial(self._mocap_data[trial], start, end)
        self._wrist_data[trial], wrist_ok = self._crop_subjects_trial(self._wrist_data[trial], start, end)
        self._head_pose_data[trial], head_pose_ok = self._crop_subjects_trial(self._head_pose_data[trial], start, end)
        self._object_pose_data[trial], object_pose_ok = self._crop_frames(self._object_pose_data[trial], start, end)
        self._gaze2d_data[trial], gaze2d_ok = self._crop_subjects_trial(self._gaze2d_data[trial], start, end)
        self._gaze3d_data[trial], gaze3d_ok = self._crop_subjects_trial(self._gaze3d_data[trial], start, end)
        # Check if everything was ok for our final return.
        return (force_ok and imu_ok and mocap_ok and wrist_ok and head_pose_ok and object_pose_ok and gaze2d_ok
                and gaze3d_ok)

    def _crop_subjects_trial(self, subjects_data: Dict[int, List[Tuple[float, Any]]], start: float, end: float) \
            -> Tuple[Dict[int, List[Tuple[float, Any]]], bool]:
        """
        Crops the dict of frames data for each subject for a single trial to the given start and end timestamps.

        Args:
            subjects_data: The dict of frames data lists for each subject for a single trial
                           {subject: [(timestamp, data)]}.
            start: The start timestamp.
            end: The end timestamp.

        Returns:
            The cropped subjects_data.
            True if there were no problems, False otherwise.
        """
        subjects_data[1], sub1_ok = self._crop_frames(subjects_data[1], start, end)
        subjects_data[2], sub2_ok = self._crop_frames(subjects_data[2], start, end)
        return subjects_data, sub1_ok and sub2_ok

    def _crop_frames(self, frames_data: List[Tuple[float, Any]], start: float, end: float) \
            -> Tuple[List[Tuple[float, Any]], bool]:
        """
        Crops the list of frames data to the given start and end timestamps.

        Args:
            frames_data: The list of frames data [(timestamp, data)] to crop.
            start: The start timestamp.
            end: The end timestamp.

        Returns:
            The cropped frames_data.
            True if there were no problems, False otherwise.
        """
        # Convert our timestamps to indices.
        start_i = self._timestamp_to_index(frames_data, start)
        end_i = self._timestamp_to_index(frames_data, end)
        # Ensure the crop is valid. Return an empty list if start_i <= end_i.
        if start_i <= end_i:
            return [], False
        else:
            # Perform the crop. We do end + 1, since we want to include the data for the closest end frame.
            return frames_data[start_i: end_i + 1], True

    @staticmethod
    def _timestamp_to_index(data: List[Tuple[float, Any]], timestamp: float) -> int:
        """
        Gets the index in data that corresponds to the closest timestamp to the given timestamp.

        Args:
            data: The data to find the index for.
            timestamp: The timestamp to get the closest element for.

        Returns:
            The index of the element from data with the closest timestamp.
        """
        # Get the list of timestamps.
        timestamps = [element[0] for element in data]
        # Get the index of the closest timestamp.
        min_error = float('inf')
        index = 0
        for i, t in enumerate(timestamps):
            error = abs(t - timestamp)
            if error < min_error:
                min_error = error
                index = i
        return index

    def _crop_mocap_trial(self, mocap_data: Dict[str, List[Tuple[float, Any]]], start: float, end: float) \
            -> Tuple[Dict[str, List[Tuple[float, Any]]], bool]:
        """
        Crops the dict of frames data for marker a single trial to the given start and end timestamps.

        Args:
            mocap_data: The dict of frames data lists for each marker for a single trial {marker: [(timestamp, data)]}.
            start: The start timestamp.
            end: The end timestamp.

        Returns:
            The cropped mocap_data.
            True if there is an issue cropping the data for either subject, false otherwise.
        """
        mocap_ok = True
        for marker in mocap_data:
            mocap_data[marker], ok = self._crop_frames(mocap_data[marker], start, end)
            mocap_ok = mocap_ok and ok
        return mocap_data, mocap_ok

    def crop_nan(self) -> None:
        """
        Crops all the nan values out of each trial. This is done by taking the smallest crop window over all
        the markers in self._mocap_data and applying it to the rest of the per-frame data.

        Modifies the private attributes:
            - self._trials
            - self._force_data
            - self._imu_data
            - self._mocap_data
            - self._wrist_data
            - self._head_pose_data
            - self._object_pose_data
            - self._role_data
            - self._start_side_data
            - self._handover_position_data
            - self._trial_type_data
            - self._object_size_data
            - self._gaze2d_data
            - self._gaze3d_data
            - self._handover_data

        Modifies the private attributes:
            - self._mocap_data
            - self._wrist_data
            - self._role_data
            - self._handover_data
        """
        bad_trials = []
        # Iterate over each trial.
        for trial in self._trials:
            # Set up the pose data to concatenate position and orientation for NaN checking.
            sub1_head_pose_concat = self._concat_pose_frames(self._head_pose_data[trial][1])
            sub2_head_pose_concat = self._concat_pose_frames(self._head_pose_data[trial][2])
            object_pose_concat = self._concat_pose_frames(self._object_pose_data[trial])
            # Get the crop parameters for each internal variable.
            crops = [self._get_subjects_nan_crop(self._force_data[trial]),
                     self._get_subjects_nan_crop(self._imu_data[trial]),
                     self._get_subjects_nan_crop(self._wrist_data[trial]),
                     self._get_frames_nan_crop(sub1_head_pose_concat),
                     self._get_frames_nan_crop(sub2_head_pose_concat),
                     self._get_frames_nan_crop(object_pose_concat),
                     self._get_subjects_nan_crop(self._gaze2d_data[trial]),
                     self._get_subjects_nan_crop(self._gaze3d_data[trial])]
            # Add the mocap crops if we have any markers.
            if len(self._mocap_data[trial]) != 0:
                crops.append(self._get_mocap_nan_crop(self._mocap_data[trial]))
            # Choose the latest start and earliest end.
            start_crop = max([crop[0] for crop in crops])
            end_crop = min([crop[1] for crop in crops])
            # If the crops eliminate all the frames, throw out the trial.
            if end_crop <= start_crop:
                bad_trials.append(trial)
            # Otherwise, crop the internal data.
            else:
                self._crop_trial(trial, start_crop, end_crop)
        # Remove the bad trials.
        for trial in bad_trials:
            self.pop(trial)

    @staticmethod
    def _concat_pose_frames(pose_frames: List[Tuple[float, np.ndarray, np.ndarray]]) -> List[Tuple[float, np.ndarray]]:
        """
        Concatenates the position and orientation data for each frame in the given pose frames.

        Args:
            pose_frames: The list of frames pose data: [(timestamp, position, orientation)] to concatenate.

        Returns:
            pose_frames with the position and orientation concatenated.
        """
        output = []
        for pose_frame in pose_frames:
            concat_data = np.concatenate((pose_frame[1], pose_frame[2]), axis=0)
            output.append((pose_frame[0], concat_data))
        return output

    @staticmethod
    def _get_subjects_nan_crop(subjects_data: Dict[int, List[Tuple[float, np.ndarray]]]) -> Tuple[float, float]:
        """
        Gets the start and end timestamps for the start and end timestamps of the strictest NaN crop for both subjects.

        Args:
            subjects_data: The dict of subjects data {subject: [(timestamp, data)]}, to look for NaN values in.

        Returns:
            The timestamps [start, end] to use to crop out NaN values.
        """
        # Get the crops for each subject.
        sub1_start, sub1_end = SessionData._get_frames_nan_crop(subjects_data[1])
        sub2_start, sub2_end = SessionData._get_frames_nan_crop(subjects_data[2])
        # Use the max start and min end.
        return max(sub1_start, sub1_end), min(sub1_end, sub2_end)

    @staticmethod
    def _get_frames_nan_crop(frames_data: List[Tuple[float, np.ndarray]]) -> Tuple[float, float]:
        """
        Gets the start and end timestamps for the last leading NaN and first trailing NaN value.

        Note, these timestamps are both inclusive.

        Args:
            frames_data: The list of frames data [(timestamp, data)] to look for NaN values in.

        Returns:
            The timestamps [start, end] to use to crop out NaN values.
        """
        # Convert to a numpy array of shape (n, d) for data_arrays.
        data_array = np.array([frame[1] for frame in frames_data])
        # Start crop is the first index where the values are not Nan.
        start = 0
        for data_row in data_array:
            isnan = np.isnan(data_row)
            if not np.any(isnan):
                break
            start += 1
        # End crop is the first index where the values are not Nan, in the reversed list of frames data.
        end = data_array.shape[0] - 1
        for data_row in np.flip(data_array, axis=0):
            isnan = np.isnan(data_row)
            if not np.any(isnan):
                break
            end -= 1
        # Convert the indices to timestamps.
        return frames_data[start][0], frames_data[end][0]

    @staticmethod
    def _get_mocap_nan_crop(mocap_data: Dict[str, List[Tuple[float, np.ndarray]]]) -> Tuple[float, float]:
        """
        Gets the start and end timestamps for the start and end timestamps of the strictest NaN crop for each marker.

        Args:
            mocap_data: The dict of mocap data {marker: [(timestamp, data)]}, to look for NaN values in.

        Returns:
            The timestamps [start, end] to use to crop out NaN values.
        """
        starts = []
        ends = []
        # Get the crops for each marker.
        for marker in mocap_data:
            start, end = SessionData._get_frames_nan_crop(mocap_data[marker])
            starts.append(start)
            ends.append(end)
        # Use the max start and min end.
        return max(starts), min(ends)

    def drop_nan(self) -> None:
        """
        Drops any trials with nan values in self._mocap_data.

        Modifies the private attributes:
            - self._trials
            - self._force_data
            - self._imu_data
            - self._mocap_data
            - self._wrist_data
            - self._head_pose_data
            - self._object_pose_data
            - self._role_data
            - self._start_side_data
            - self._handover_position_data
            - self._trial_type_data
            - self._object_size_data
            - self._gaze2d_data
            - self._gaze3d_data
            - self._handover_data
        """
        # Keep track of trials to remove.
        bad_trials = []
        # Iterate over each trial.
        for trial in self._trials:
            # Set up the pose data to concatenate position and orientation for NaN checking.
            sub1_head_pose_concat = self._concat_pose_frames(self._head_pose_data[trial][1])
            sub2_head_pose_concat = self._concat_pose_frames(self._head_pose_data[trial][2])
            object_pose_concat = self._concat_pose_frames(self._object_pose_data[trial])
            # Check if each internal variable contains nans.
            isnans = [self._get_subjects_isnan(self._force_data[trial]),
                      self._get_subjects_isnan(self._imu_data[trial]),
                      self._get_mocap_isnan(self._mocap_data[trial]),
                      self._get_subjects_isnan(self._wrist_data[trial]),
                      self._get_frames_isnan(sub1_head_pose_concat),
                      self._get_frames_isnan(sub2_head_pose_concat),
                      self._get_frames_isnan(object_pose_concat),
                      self._get_subjects_isnan(self._gaze2d_data[trial]),
                      self._get_subjects_isnan(self._gaze3d_data[trial])]
            # If any nans were present, mark the trial as bad.
            if any(isnans):
                bad_trials.append(trial)
        # Remove the bad trials.
        for trial in bad_trials:
            self.pop(trial)

    @staticmethod
    def _get_subjects_isnan(subjects_data: Dict[int, List[Tuple[float, np.ndarray]]]) -> bool:
        """
        Tells us if there are any NaN values for both subjects.

        Args:
            subjects_data: The dict of subjects data {subject: [(timestamp, data)]}, to look for NaN values in.

        Returns:
            True if there are any NaN values in the data for either subject, False otherwise.
        """
        # Get the crops for each subject.
        sub1_isnan = SessionData._get_frames_isnan(subjects_data[1])
        sub2_isnan = SessionData._get_frames_isnan(subjects_data[2])
        # Use the max start and min end.
        return sub1_isnan or sub2_isnan

    @staticmethod
    def _get_frames_isnan(frames_data: List[Tuple[float, np.ndarray]]) -> bool:
        """
        Tells us if there are any NaN values in the given frames data.

        Args:
            frames_data: The list of frames data [(timestamp, data)] to look for NaN values in.

        Returns:
            True if there are any NaN values in frames_data, False otherwise.
        """
        isnan = False
        for frame in frames_data:
            isnan = isnan or np.isnan(frame[1])
        return isnan

    @staticmethod
    def _get_mocap_isnan(mocap_data: Dict[str, List[Tuple[float, np.ndarray]]]) -> Tuple[float, float]:
        """
        Tells us if there are any NaN values for each marker.

        Args:
            mocap_data: The dict of mocap data {marker: [(timestamp, data)]}, to look for NaN values in.

        Returns:
            True if there are any NaN values for any marker, False otherwise.
        """
        isnan = False
        # Get the crops for each marker.
        for marker in mocap_data:
            isnan = isnan or SessionData._get_frames_isnan(mocap_data[marker])
        return isnan

    def drop_small_trials(self, min_duration: float = 0.1) -> None:
        """
        Drops all trials with less than min_length duration.

        Args:
            min_duration: The minimum duration a trial has to be (in seconds) to not get dropped.

        Modifies the private attributes:
            - self._trials
            - self._force_data
            - self._imu_data
            - self._mocap_data
            - self._wrist_data
            - self._head_pose_data
            - self._object_pose_data
            - self._role_data
            - self._start_side_data
            - self._handover_position_data
            - self._trial_type_data
            - self._object_size_data
            - self._gaze2d_data
            - self._gaze3d_data
            - self._handover_data
        """
        # Keep track of trials to remove.
        bad_trials = []
        # Iterate over each trial.
        for trial in self._trials:
            # Add mocap duration if we have any markers to look at.
            # Get the durations of each internal variable. We know each subject and marker has the same duration.
            durations = [len(self._force_data[trial][1]) * self.force_fps,
                         len(self._imu_data[trial][1]) * self.imu_fps,
                         len(self._wrist_data[trial][1]) * self.mocap_fps,
                         len(self._head_pose_data[trial][1]) * self.mocap_fps,
                         len(self._object_pose_data[trial]) * self.mocap_fps,
                         len(self._gaze2d_data[trial][1]) * self.gaze_fps,
                         len(self._gaze3d_data[trial][1]) * self.gaze_fps]
            # Add the mocap duration if we have any markers.
            if len(self._mocap_data[trial]) != 0:
                marker = list(self._mocap_data[trial].keys())[0]
                durations.append(len(self._mocap_data[trial][marker]) * self.mocap_fps)
            # Make our decision based on the durations.
            for duration in durations:
                if duration < min_duration:
                    bad_trials.append(trial)
                    break
        # Remove the bad trials.
        for trial in bad_trials:
            self.pop(trial)


def main() -> None:
    """
    Main function that runs when this file is invoked, to test this implementation.
    """
    short_test()
    long_test()


def short_test() -> None:
    """
    Short test for SessionData.
    """
    session_file = 'E:/Datasets/CS 4440 Final Project/mat_files_full/test_data.mat'
    session_data = SessionData(session_file=session_file, debug=True)
    session_data.load()
    # session_data.crop_to_handover()  # TODO: This is likely wrong, I do not like the method here.
    session_data.crop_nan()
    session_data.drop_nan()
    h2h_session_data_isnan(session_data)
    print(f'All markers trials: {session_data.trials()}')
    target_markers = ['Sub1_C7', 'Sub1_Th7', 'Sub1_SXS', 'Sub1_LPSIS', 'Sub1_RPSIS',
                      'Sub1_LAC', 'Sub1_LHME', 'Sub1_LRSP',
                      'Sub1_RAC', 'Sub1_RHME', 'Sub1_RRSP',
                      'Sub1_LFT', 'Sub1_LFLE', 'Sub1_LLM', 'Sub1_L2MH',
                      'Sub1_RFT', 'Sub1_RFLE', 'Sub1_RLM', 'Sub1_R2MH',
                      'Sub2_C7', 'Sub2_Th7', 'Sub2_SXS', 'Sub2_LPSIS', 'Sub2_RPSIS',
                      'Sub2_LAC', 'Sub2_LHME', 'Sub2_LRSP',
                      'Sub2_RAC', 'Sub2_RHME', 'Sub2_RRSP',
                      'Sub2_LFT', 'Sub2_LFLE', 'Sub2_LLM', 'Sub2_L2MH',
                      'Sub2_RFT', 'Sub2_RFLE', 'Sub2_RLM', 'Sub2_R2MH']
    session_data = SessionData(session_file=session_file, target_markers=target_markers, debug=True)
    session_data.load()
    # session_data.crop_to_handover()  # TODO: This is likely wrong, I do not like the method here.
    session_data.crop_nan()
    session_data.drop_nan()
    session_data.drop_small_trials()
    h2h_session_data_isnan(session_data)
    print(f'Some markers trials: {session_data.trials()}')


def long_test() -> None:
    """
    Long test for SessionData.
    """
    session_file = 'E:/Datasets/CS 4440 Final Project/mat_files_full/26_M4_F5_cropped_data_v2.mat'
    session_data = SessionData(session_file=session_file, debug=True)
    session_data.load()
    # session_data.crop_to_handover()  # TODO: This is likely wrong, I do not like the method here.
    session_data.crop_nan()
    session_data.drop_nan()
    session_data.drop_small_trials()
    h2h_session_data_isnan(session_data)
    print(f'All markers trials: {session_data.trials()}')


def h2h_session_data_isnan(session_data: SessionData) -> bool:
    """
    Checks if the SessionData object contains any NaN values.

    Args:
        session_data: The SessionData object to be checked.

    Returns:
        True if there were nans, False if there were not.
    """
    output = False
    for trial in session_data.trials():
        # Get the trial data.
        trial_data = session_data[trial]
        # Check the mocap data.
        mocap_data = trial_data['mocap']
        for marker in mocap_data.keys():
            for frame in range(len(mocap_data[marker])):
                timestamp, point = mocap_data[marker][frame]
                if np.isnan(timestamp):
                    print(f'NaN found in mocap timestamp, trial: {trial}, marker: {marker}, frame: {frame}')
                    output = True
                for i in range(len(point)):
                    value = point[i]
                    if np.isnan(value):
                        print(f'NaN found in mocap point, trial: {trial}, marker: {marker}, frame: {frame}, i: {i}')
                        output = True
    return output


if __name__ == '__main__':
    main()
