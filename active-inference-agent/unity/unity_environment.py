import enum
import os

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel


class Condition(enum.Enum):
    """
    Enumeration used to configure the experimental condition of the environment (i.e. the position of the rubber arm)
    """
    Left = 0.0
    Center = 1.0
    Right = 2.0
    RandReachClose = 3.0
    RandReachFar = 4.0
    Break = 5.0


class VisibleArm(enum.Enum):
    """
    Enumeration used to configure the arm visibility of the environment's camera
    """
    RealArm = 0.0
    RubberArm = 1.0


class Stimulation(enum.Enum):
    """
    Enumeration used to configure the type of visuo-tactile stimulation the agent receives
    """
    Synchronous = 0.0
    Asynchronous = 1.0


class UnityContainer:
    """
    Class encapsulating the ML-Agents connection with Unity allowing easy interaction with the simulated environment
    """
    VISUAL_OBSERVATION_INDEX = 0
    VECTOR_OBSERVATIONS_INDEX = 1

    # Path of the pre-build environment
    BUILD_PATH = os.path.join(os.path.dirname(__file__),
                              "../../Unity Environment/build/deep active inference agent environment")

    def __init__(self, use_editor, time_scale=1):
        """
        Set up the Unity environment
        :param use_editor: Set to true to connect directly to the Unity editor, set to false to use the pre-build
        environment at BUILD_PATH
        :param time_scale: time_scale of the environment (1 is normal time)
        """
        if use_editor:
            self.env_path = None
        else:
            self.env_path = self.BUILD_PATH
        self.time_scale = time_scale
        self.env = None
        self.float_properties_channel = None
        self.group_name = None
        self.group_spec = None

    def initialise_environment(self):
        """Initialise and reset unity environment"""
        engine_configuration_channel = EngineConfigurationChannel()
        self.float_properties_channel = FloatPropertiesChannel()
        self.env = UnityEnvironment(file_name=self.env_path, base_port=5004,
                                    side_channels=[engine_configuration_channel, self.float_properties_channel])

        # Reset the environment
        self.env.reset()

        # Set the default brain to work with
        self.group_name = self.env.get_agent_groups()[0]
        self.group_spec = self.env.get_agent_group_spec(self.group_name)

        # Set the time scale of the engine
        engine_configuration_channel.set_configuration_parameters(time_scale=self.time_scale)

    def set_condition(self, condition: Condition):
        """Sets the experimental condition setting"""
        self.float_properties_channel.set_property("condition", condition.value)

    def set_visible_arm(self, visible_arm: VisibleArm):
        """Sets the visible arm setting"""
        self.float_properties_channel.set_property("visiblearm", visible_arm.value)

    def set_stimulation(self, stimulation: Stimulation):
        """Sets the stimulation setting"""
        self.float_properties_channel.set_property("stimulation", stimulation.value)

    def get_joint_observation(self):
        """:returns joint angles of the agent"""
        return self.env.get_step_result(self.group_name).obs[self.VECTOR_OBSERVATIONS_INDEX][:, :2]

    def get_touch_observation(self):
        """:returns last visual and tactile touch events"""
        return self.env.get_step_result(self.group_name).obs[self.VECTOR_OBSERVATIONS_INDEX][:, 2:4]

    def get_current_env_time(self):
        """:returns current env time"""
        return self.env.get_step_result(self.group_name).obs[self.VECTOR_OBSERVATIONS_INDEX][:, 4]

    def get_cartesian_distance(self):
        """:returns cartesian euclidean (absolute) distance between real hand and rubber hand"""
        return self.env.get_step_result(self.group_name).obs[self.VECTOR_OBSERVATIONS_INDEX][0, 5]

    def get_horizontal_distance(self):
        """:returns horizontal euclidean distance between real hand and rubber hand"""
        return self.env.get_step_result(self.group_name).obs[self.VECTOR_OBSERVATIONS_INDEX][0, 6]

    def get_rubber_joint_observation(self):
        """:returns joint angles of the rubber arm"""
        return self.env.get_step_result(self.group_name).obs[self.VECTOR_OBSERVATIONS_INDEX][:, 7:9]

    def get_visual_observation(self):
        """:returns visual perception of the agent"""
        return self.env.get_step_result(self.group_name).obs[self.VISUAL_OBSERVATION_INDEX][0]

    def act(self, action):
        """Make the agent perform an action (velocity) in the environment"""
        self.env.set_actions(self.group_name, np.append([[0]], action, axis=1))
        self.env.step()

    def set_rotation(self, rotation):
        """Manually set the joint angles to a particular rotation"""
        self.env.set_actions(self.group_name, np.append([[1]], rotation,  axis=1))
        self.env.step()

    def set_rubber_arm_rotation(self, rotation):
        """Manually set the joint angles of the rubber arm to a particular rotation"""
        self.env.set_actions(self.group_name, np.append([[2]], rotation,  axis=1))
        self.env.step()

    def reset(self):
        """Reset the environment"""
        self.env.reset()

    def close(self):
        """Gracefully close the environment"""
        self.env.close()
