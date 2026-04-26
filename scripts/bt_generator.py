#!/usr/bin/env python3
"""
BTGenerator - YAML to Behavior Tree XML Generator for LangChain Integration

This module provides a comprehensive tool for generating BehaviorTree.CPP v4 XML files
from YAML configuration files. It's designed to work as a LangChain tool for automated
behavior tree generation in robotics applications.

Usage:
    # As a LangChain tool
    from bt_generator import BTGenerator
    generator = BTGenerator()
    xml_output = generator.generate_from_yaml("/path/to/config.yaml")

    # Or standalone
    python bt_generator.py /path/to/input.yaml /path/to/output.xml

Features:
    - Complete XML generation from YAML configs
    - Support for single and multi-object pick-and-place operations
    - Customizable task flows with ReturnHome and other operations
    - Built-in XML validation and formatting
    - Template-based generation with fallback support
    - LangChain-compatible interface with structured output
"""

import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Custom exception for BT generation errors"""
    pass


@dataclass
class ScriptConfig:
    """Script configuration for robot setup"""
    end_effector: str
    planning_group: str
    gripper_cmd_action_name: str
    arm_joint_names: List[str] = field(default_factory=list)
    allowed_touch_links: List[str] = field(default_factory=list)


@dataclass
class PickOperation:
    """Pick operation configuration"""
    object_id: str
    pick_frame_id: str
    pre_grasp_frame: str
    post_grasp_frame: str


@dataclass
class PlaceOperation:
    """Place operation configuration"""
    object_id: str
    place_frame_id: str
    preplace_frame: str
    postplace_frame: str


@dataclass
class ObjectOperation:
    """Combined pick and place operation for an object"""
    pick: PickOperation
    place: PlaceOperation
    operation_index: int
    pick_key: str = ""  # Original YAML key (e.g., "PickAndMove_1")
    place_key: str = ""  # Original YAML key (e.g., "MoveAndPlace_1")


class BTGenerator:
    """
    Behavior Tree Generator for converting YAML configs to XML.

    This class provides a comprehensive interface for generating BehaviorTree.CPP
    XML files from YAML configuration files. It supports various operation types
    and can be easily integrated with LangChain as a tool.

    Attributes:
        validate_xml (bool): Whether to validate generated XML
        auto_format (bool): Whether to auto-format XML output
        indent_size (int): XML indentation size
    """

    def __init__(self, validate_xml: bool = True, auto_format: bool = True, indent_size: int = 2):
        """
        Initialize the BT Generator.

        Args:
            validate_xml: Whether to validate XML after generation
            auto_format: Whether to format XML with indentation
            indent_size: Number of spaces for XML indentation
        """
        self.validate_xml = validate_xml
        self.auto_format = auto_format
        self.indent_size = indent_size

        # Default parameter values for behavior tree nodes
        self.default_params = {
            'approach_distance': 0.1,
            'approach_direction': [0.0, 0.0, -1.0],
            'goal_tolerances': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'velocity_scaling_factor': 1.0,
            'acceleration_scaling_factor': 1.0,
            'pick_velocity': 0.05,
            'pick_goal_tolerances': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'place_velocity': 0.05,
            'place_goal_tolerances': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'retreat_acceleration': 0.01,
        }

    def generate_from_yaml(self, yaml_path: str, output_path: Optional[str] = None) -> str:
        """
        Generate Behavior Tree XML from YAML configuration file.

        This is the main entry point for generating behavior trees. It reads
        a YAML file, parses the configuration, generates the XML, and optionally
        saves it to a file.

        Args:
            yaml_path: Path to the input YAML configuration file
            output_path: Optional path to save the generated XML file

        Returns:
            The generated XML string

        Raises:
            GenerationError: If YAML parsing or XML generation fails
            FileNotFoundError: If YAML file doesn't exist
        """
        logger.info(f"Loading YAML configuration from: {yaml_path}")

        # Load and parse YAML
        config = self._load_yaml(yaml_path)

        # Generate XML
        xml_content = self._generate_xml(config)

        # Validate if enabled
        if self.validate_xml:
            self._validate_xml(xml_content)

        # Format if enabled
        if self.auto_format:
            xml_content = self._format_xml(xml_content)

        # Save to file if path provided
        if output_path:
            self._save_xml(xml_content, output_path)
            logger.info(f"XML saved to: {output_path}")

        return xml_content

    def generate_from_dict(self, config: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate Behavior Tree XML from a configuration dictionary.

        This method allows direct generation from a dictionary, useful when
        working with parsed YAML or dynamically created configurations.

        Args:
            config: Configuration dictionary matching YAML structure
            output_path: Optional path to save the generated XML file

        Returns:
            The generated XML string
        """
        logger.info("Generating XML from configuration dictionary")

        # Generate XML
        xml_content = self._generate_xml(config)

        # Validate if enabled
        if self.validate_xml:
            self._validate_xml(xml_content)

        # Format if enabled
        if self.auto_format:
            xml_content = self._format_xml(xml_content)

        # Save to file if path provided
        if output_path:
            self._save_xml(xml_content, output_path)
            logger.info(f"XML saved to: {output_path}")

        return xml_content

    def _load_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """
        Load and parse YAML configuration file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Parsed configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            GenerationError: If YAML parsing fails
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)

            if not config:
                raise GenerationError("YAML file is empty or invalid")

            logger.info("YAML loaded successfully")
            return config

        except yaml.YAMLError as e:
            raise GenerationError(f"YAML parsing error: {e}")

    def _generate_xml(self, config: Dict[str, Any]) -> str:
        """
        Generate complete Behavior Tree XML from configuration.

        This method orchestrates the generation of all XML components including
        the main tree, subtrees, and node model definitions.

        Args:
            config: Parsed YAML configuration

        Returns:
            Complete XML string
        """
        # Extract configuration sections
        bt_config = config.get('behavior_tree', {})
        script_config = config.get('script', {})
        scene_config = config.get('scene', {})
        task_flow = config.get('TaskFlow', [])

        # Get main tree ID
        bt_id = bt_config.get('name', 'GeneratedTree')

        # Update default params from config
        self._update_default_params(bt_config)

        # Parse script configuration
        script = ScriptConfig(
            end_effector=script_config.get('end_effector', ''),
            planning_group=script_config.get('planning_group', ''),
            gripper_cmd_action_name=script_config.get('gripper_cmd_action_name', ''),
            arm_joint_names=script_config.get('arm_joint_names', []),
            allowed_touch_links=script_config.get('allowed_touch_links', [])
        )

        # Parse object operations
        operations = self._parse_operations(config)

        # Build XML
        root = ET.Element('root')
        root.set('BTCPP_format', '4')

        # Generate main behavior tree
        main_bt = self._generate_main_tree(bt_id, script, operations, task_flow)
        root.append(main_bt)

        # Generate generic subtrees for MainPick and MainPlace (reusable for all operations)
        if operations:
            main_pick = self._generate_main_pick_subtree()
            main_place = self._generate_main_place_subtree()
            root.append(main_pick)
            root.append(main_place)

        # Generate reusable subtrees for each operation
        for i, operation in enumerate(operations):
            pick_and_move = self._generate_pick_and_move_subtree(i + 1)
            root.append(pick_and_move)

            move_and_place = self._generate_move_and_place_subtree(i + 1)
            root.append(move_and_place)

        picking_sequence = self._generate_picking_sequence_subtree()
        root.append(picking_sequence)

        placing_sequence = self._generate_placing_sequence_subtree()
        root.append(placing_sequence)

        # Generate spawn objects subtree
        if operations:
            spawn_objects = self._generate_spawn_objects_subtree(scene_config, operations)
            root.append(spawn_objects)

        # Generate TreeNodesModel
        tree_model = self._generate_tree_nodes_model()
        root.append(tree_model)

        # Convert to string
        xml_str = ET.tostring(root, encoding='unicode')

        # Add XML declaration
        xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str

        return xml_str

    def _update_default_params(self, bt_config: Dict[str, Any]) -> None:
        """Update default parameters from behavior tree configuration"""
        if 'approach_distance' in bt_config:
            self.default_params['approach_distance'] = bt_config['approach_distance']
        if 'approach_direction' in bt_config:
            self.default_params['approach_direction'] = bt_config['approach_direction']
        if 'goal_tolerances' in bt_config:
            self.default_params['goal_tolerances'] = bt_config['goal_tolerances']
        if 'pick_goal_tolerances' in bt_config:
            self.default_params['pick_goal_tolerances'] = bt_config['pick_goal_tolerances']
        if 'place_goal_tolerances' in bt_config:
            self.default_params['place_goal_tolerances'] = bt_config['place_goal_tolerances']

    def _parse_operations(self, config: Dict[str, Any]) -> List[ObjectOperation]:
        """
        Parse pick and place operations from configuration.

        Handles both numbered (PickAndMove_1, PickAndMove_2) and non-numbered formats.

        Args:
            config: Configuration dictionary

        Returns:
            List of ObjectOperation instances
        """
        operations = []
        index = 1

        # Try to find PickAndMove and MoveAndPlace pairs
        while True:
            # First try numbered keys (e.g., PickAndMove_1, MoveAndPlace_1)
            pick_key = f'PickAndMove_{index}'
            place_key = f'MoveAndPlace_{index}'

            pick_data = config.get(pick_key)
            place_data = config.get(place_key)

            # For index 1, also try non-numbered as fallback
            if index == 1 and (not pick_data or not place_data):
                if not pick_data:
                    pick_data = config.get('PickAndMove')
                if not place_data:
                    place_data = config.get('MoveAndPlace')

            # If no data found for current index, stop
            if not pick_data or not place_data:
                break

            pick = PickOperation(
                object_id=pick_data['object_id'],
                pick_frame_id=pick_data['pick_frame_id'],
                pre_grasp_frame=pick_data.get('pre_grasp_frame', pick_data['pick_frame_id']),
                post_grasp_frame=pick_data.get('post_grasp_frame', pick_data['pick_frame_id'])
            )

            place = PlaceOperation(
                object_id=place_data['object_id'],
                place_frame_id=place_data['place_frame_id'],
                preplace_frame=place_data.get('preplace_frame', place_data['place_frame_id']),
                postplace_frame=place_data.get('postplace_frame', place_data['place_frame_id'])
            )

            # Store actual key names from YAML (e.g., "PickAndMove_1", "MoveAndPlace_1")
            actual_pick_key = pick_key if config.get(pick_key) else 'PickAndMove'
            actual_place_key = place_key if config.get(place_key) else 'MoveAndPlace'

            operations.append(ObjectOperation(
                pick=pick,
                place=place,
                operation_index=index,
                pick_key=actual_pick_key,
                place_key=actual_place_key
            ))
            index += 1

        logger.info(f"Parsed {len(operations)} pick-and-place operations")
        return operations

    def _generate_main_tree(self, bt_id: str, script: ScriptConfig,
                           operations: List[ObjectOperation], task_flow: List[str]) -> ET.Element:
        """
        Generate the main behavior tree.

        Args:
            bt_id: Behavior tree ID
            script: Script configuration
            operations: List of operations
            task_flow: Task flow sequence

        Returns:
            Main BehaviorTree XML element
        """
        bt = ET.Element('BehaviorTree')
        bt.set('ID', bt_id)

        # Root sequence
        root_seq = ET.SubElement(bt, 'Sequence')
        root_seq.set('name', 'root')

        # Script node for initialization
        script_code = self._build_script_code(script)
        script_elem = ET.SubElement(root_seq, 'Script')
        script_elem.set('code', script_code)

        # Spawn scene
        spawn = ET.SubElement(root_seq, 'SubTree')
        spawn.set('ID', 'SpawnObjects')
        spawn.set('_autoremap', 'true')

        # Process task flow
        for task in task_flow:
            if task == 'SpawnScene':
                continue  # Already added
            elif task == 'Pick' or task.startswith('Pick_'):
                # Handle both "Pick" and "Pick_N" formats
                if task == 'Pick':
                    idx = 1
                else:
                    idx = int(task.split('_')[1]) if len(task.split('_')) > 1 else 1
                if idx <= len(operations):
                    op = operations[idx - 1]
                    subtree = ET.SubElement(root_seq, 'SubTree')
                    # Always use generic MainPick (it's reusable with different parameters)
                    subtree.set('ID', 'MainPick')
                    # Use the original YAML key name (e.g., "PickAndMove_1")
                    subtree.set('name', op.pick_key)
                    subtree.set('object_id', op.pick.object_id)
                    subtree.set('pick_frame_id', op.pick.pick_frame_id)
                    subtree.set('pregrasp_frame', op.pick.pre_grasp_frame)
                    subtree.set('postgrasp_frame', op.pick.post_grasp_frame)
                    subtree.set('_autoremap', 'true')
            elif task == 'Place' or task.startswith('Place_'):
                # Handle both "Place" and "Place_N" formats
                if task == 'Place':
                    idx = 1
                else:
                    idx = int(task.split('_')[1]) if len(task.split('_')) > 1 else 1
                if idx <= len(operations):
                    op = operations[idx - 1]
                    subtree = ET.SubElement(root_seq, 'SubTree')
                    # Always use generic MainPlace (it's reusable with different parameters)
                    subtree.set('ID', 'MainPlace')
                    # Use the original YAML key name (e.g., "MoveAndPlace_1")
                    subtree.set('name', op.place_key)
                    subtree.set('object_id', op.place.object_id)
                    subtree.set('place_frame_id', op.place.place_frame_id)
                    subtree.set('preplace_frame', op.place.preplace_frame)
                    subtree.set('postplace_frame', op.place.postplace_frame)
                    subtree.set('_autoremap', 'true')
            elif task == 'ReturnHome':
                # Add return home logic if needed
                pass

        return bt

    def _build_script_code(self, script: ScriptConfig) -> str:
        """Build script initialization code with actual characters (not HTML entities)"""
        # Convert lists to strings
        joint_names = str(script.arm_joint_names)
        allowed_links = str(script.allowed_touch_links)

        # Build code with actual quotes and newlines (XML library will handle escaping)
        code = f'end_effector:="{script.end_effector}";\n'
        code += f'planning_group:="{script.planning_group}";\n'
        code += f'gripper_cmd_action_name:="{script.gripper_cmd_action_name}";\n'
        code += f'default_arm_joint_names:="{joint_names}";\n'
        code += f'allowed_touch_links:="{allowed_links}"'

        return code

    def _generate_main_pick_subtree(self) -> ET.Element:
        """Generate generic MainPick subtree (reusable for all pick operations)"""
        bt = ET.Element('BehaviorTree')
        bt.set('ID', 'MainPick')

        seq = ET.SubElement(bt, 'Sequence')

        script_code = 'gripper_tcp:=end_effector;\n'
        script_code += 'gripper_cmd_action_name:=gripper_cmd_action_name;\n'
        script_code += 'allowed_touch_links:=allowed_touch_links;\n'
        script_code += 'arm_joint_names:=default_arm_joint_names;\n'
        script_code += 'pg_arm:=planning_group'

        script_elem = ET.SubElement(seq, 'Script')
        script_elem.set('code', script_code)

        subtree = ET.SubElement(seq, 'SubTree')
        subtree.set('ID', 'PickAndMove')
        subtree.set('object_id', '{object_id}')
        subtree.set('pick_frame_id', '{pick_frame_id}')
        subtree.set('postgrasp_frame', '{postgrasp_frame}')
        subtree.set('pregrasp_frame', '{pregrasp_frame}')
        subtree.set('_autoremap', 'true')

        return bt

    def _generate_main_place_subtree(self) -> ET.Element:
        """Generate generic MainPlace subtree (reusable for all place operations)"""
        bt = ET.Element('BehaviorTree')
        bt.set('ID', 'MainPlace')

        seq = ET.SubElement(bt, 'Sequence')

        script_code = 'gripper_tcp:=end_effector;\n'
        script_code += 'gripper_cmd_action_name:=gripper_cmd_action_name;\n'
        script_code += 'allowed_touch_links:=allowed_touch_links;\n'
        script_code += 'arm_joint_names:=default_arm_joint_names;\n'
        script_code += 'pg_arm:=planning_group'

        script_elem = ET.SubElement(seq, 'Script')
        script_elem.set('code', script_code)

        subtree = ET.SubElement(seq, 'SubTree')
        subtree.set('ID', 'MoveAndPlace')
        subtree.set('object_id', '{object_id}')
        subtree.set('place_frame_id', '{place_frame_id}')
        subtree.set('postplace_frame', '{postplace_frame}')
        subtree.set('preplace_frame', '{preplace_frame}')
        subtree.set('_autoremap', 'true')

        return bt

    def _generate_pick_and_move_subtree(self, index: int = 1) -> ET.Element:
        """Generate PickAndMove reusable subtree"""
        bt = ET.Element('BehaviorTree')
        bt.set('ID', 'PickAndMove')

        seq = ET.SubElement(bt, 'Sequence')

        # Release any held object
        release = ET.SubElement(seq, 'Release')
        release.set('object_id', '{object_id}')
        release.set('gripper_cmd_action_name', '{gripper_cmd_action_name}')
        release.set('place_frame_id', '{@place_frame_id}')
        release.set('parent_object_id_after_release', '{@parent_object_id_after_release}')
        release.set('joint_states', '')
        release.set('disable_scene_handling', 'False')

        # Read current joint state
        retry_joint_state = ET.SubElement(seq, 'RetryUntilSuccessful')
        retry_joint_state.set('num_attempts', '-1')
        read_joint = ET.SubElement(retry_joint_state, 'ReadCurrentJointState')
        read_joint.set('joint_names_filter', '{arm_joint_names}')
        read_joint.set('joint_names', '{joint_names_initial}')
        read_joint.set('joint_positions', '{joint_positions_initial}')

        # Picking sequence with fallback
        retry_pick = ET.SubElement(seq, 'RetryUntilSuccessful')
        retry_pick.set('num_attempts', '2')

        fallback = ET.SubElement(retry_pick, 'Fallback')

        # Try direct picking sequence first
        direct_pick = ET.SubElement(fallback, 'SubTree')
        direct_pick.set('ID', 'PickingSequence')
        direct_pick.set('_autoremap', 'true')

        # Fallback to OMPL-based approach
        fallback_seq = ET.SubElement(fallback, 'Sequence')

        reach_pre = ET.SubElement(fallback_seq, 'ReachPreGraspOmpl')
        reach_pre.set('object_id', '{object_id}')
        reach_pre.set('end_effector_link', '{gripper_tcp}')
        reach_pre.set('pick_frame_id', '{pregrasp_frame}')
        reach_pre.set('pick_position_xyz', '[0.0, 0.0, 0.0]')
        reach_pre.set('pick_rotation_quat', '[0.0, 0.0, 0.0, 1.0]')
        reach_pre.set('pick_approach_direction', '[0.0, 0.0, 0.0]')
        reach_pre.set('pick_approach_distance', str(self.default_params['approach_distance']))
        reach_pre.set('planning_group', '{pg_arm}')
        reach_pre.set('controller_names', '')
        reach_pre.set('velocity_scaling_factor', str(self.default_params['velocity_scaling_factor']))
        reach_pre.set('use_joint_constraint', 'False')
        reach_pre.set('use_joint_target', 'True')
        reach_pre.set('joint_names', '{joint_names_initial}')
        reach_pre.set('joint_states', '{joint_positions_initial}')
        reach_pre.set('goal_tolerances', str(self.default_params['goal_tolerances']))
        reach_pre.set('constraint_tolerances_above', '')
        reach_pre.set('constraint_tolerances_below', '')
        reach_pre.set('acceleration_scaling_factor', str(self.default_params['acceleration_scaling_factor']))

        retry_subtree = ET.SubElement(fallback_seq, 'SubTree')
        retry_subtree.set('ID', 'PickingSequence')
        retry_subtree.set('_autoremap', 'true')

        # Grasp the object
        grasp = ET.SubElement(seq, 'Grasp')
        grasp.set('object_id', '{object_id}')
        grasp.set('end_effector_link', '{gripper_tcp}')
        grasp.set('gripper_cmd_action_name', '{gripper_cmd_action_name}')
        grasp.set('pick_allowed_touch_objects', '{allowed_touch_links}')
        grasp.set('joint_states', '')
        grasp.set('disable_scene_handling', 'False')

        # Post-grasp retreat
        retry_retreat = ET.SubElement(seq, 'RetryUntilSuccessful')
        retry_retreat.set('num_attempts', '2')

        switch = ET.SubElement(retry_retreat, 'Switch2')
        switch.set('case_1', '{gripper_tcp}')
        switch.set('case_2', '{postgrasp_frame}')
        switch.set('variable', '{postgrasp_frame}')

        # Case 1: Linear retreat relative to gripper
        retreat1 = ET.SubElement(switch, 'MoveGraspLin')
        retreat1.set('object_id', '{object_id}')
        retreat1.set('pick_allowed_touch_objects', '{allowed_touch_links}')
        retreat1.set('end_effector_link', '{gripper_tcp}')
        retreat1.set('pick_frame_id', '{gripper_tcp}')
        retreat1.set('pick_position_xyz', '[0.0, 0.0, -0.1]')
        retreat1.set('pick_rotation_quat', '[0.0, 0.0, 0.0, 1.0]')
        retreat1.set('planning_group', '{pg_arm}')
        retreat1.set('controller_names', '')
        retreat1.set('velocity_scaling_factor', str(self.default_params['velocity_scaling_factor']))
        retreat1.set('use_joint_constraint', 'False')
        retreat1.set('joint_names', '')
        retreat1.set('goal_tolerances', '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]')
        retreat1.set('constraint_tolerances_above', '')
        retreat1.set('constraint_tolerances_below', '')
        retreat1.set('acceleration_scaling_factor', str(self.default_params['retreat_acceleration']))
        retreat1.set('blend_radius', '0.0')
        retreat1.set('add_to_blending_queue', 'False')

        # Case 2: Move to post-grasp frame
        retreat2 = ET.SubElement(switch, 'MoveGraspLin')
        retreat2.set('object_id', '{object_id}')
        retreat2.set('pick_allowed_touch_objects', '{allowed_touch_links}')
        retreat2.set('end_effector_link', '{gripper_tcp}')
        retreat2.set('pick_frame_id', '{postgrasp_frame}')
        retreat2.set('pick_position_xyz', '[0.0, 0.0, 0.0]')
        retreat2.set('pick_rotation_quat', '[0.0, 0.0, 0.0, 1.0]')
        retreat2.set('planning_group', '{pg_arm}')
        retreat2.set('controller_names', '')
        retreat2.set('velocity_scaling_factor', str(self.default_params['velocity_scaling_factor']))
        retreat2.set('use_joint_constraint', 'False')
        retreat2.set('joint_names', '')
        retreat2.set('goal_tolerances', '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]')
        retreat2.set('constraint_tolerances_above', '')
        retreat2.set('constraint_tolerances_below', '')
        retreat2.set('acceleration_scaling_factor', str(self.default_params['retreat_acceleration']))
        retreat2.set('blend_radius', '0.0')
        retreat2.set('add_to_blending_queue', 'False')

        # Default: Point-to-point retreat
        retreat3 = ET.SubElement(switch, 'ReachPreGraspPtp')
        retreat3.set('object_id', '{object_id}')
        retreat3.set('end_effector_link', '{gripper_tcp}')
        retreat3.set('pick_frame_id', '{gripper_tcp}')
        retreat3.set('pick_position_xyz', '[0.0, 0.0, 0.0]')
        retreat3.set('pick_rotation_quat', '[0.0, 0.0, 0.0, 1.0]')
        retreat3.set('pick_approach_direction', '[0.0, 0.0, -1.0]')
        retreat3.set('pick_approach_distance', str(self.default_params['approach_distance']))
        retreat3.set('planning_group', '{pg_arm}')
        retreat3.set('controller_names', '')
        retreat3.set('velocity_scaling_factor', str(self.default_params['velocity_scaling_factor']))
        retreat3.set('use_joint_constraint', 'false')
        retreat3.set('joint_names', '')
        retreat3.set('goal_tolerances', '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]')
        retreat3.set('constraint_tolerances_above', '')
        retreat3.set('constraint_tolerances_below', '')
        retreat3.set('acceleration_scaling_factor', str(self.default_params['acceleration_scaling_factor']))
        retreat3.set('blend_radius', '0.0')
        retreat3.set('add_to_blending_queue', 'False')

        return bt

    def _generate_move_and_place_subtree(self, index: int = 1) -> ET.Element:
        """Generate MoveAndPlace reusable subtree"""
        bt = ET.Element('BehaviorTree')
        bt.set('ID', 'MoveAndPlace')

        seq = ET.SubElement(bt, 'Sequence')

        # Read current joint state
        read_joint = ET.SubElement(seq, 'ReadCurrentJointState')
        read_joint.set('joint_names_filter', '{arm_joint_names}')
        read_joint.set('joint_names', '{joint_names_initial}')
        read_joint.set('joint_positions', '{joint_positions_initial}')

        # Placing sequence with fallback
        retry_place = ET.SubElement(seq, 'RetryUntilSuccessful')
        retry_place.set('num_attempts', '2')

        fallback = ET.SubElement(retry_place, 'Fallback')

        # Try direct placing sequence first
        direct_place = ET.SubElement(fallback, 'SubTree')
        direct_place.set('ID', 'PlacingSequence')
        direct_place.set('_autoremap', 'true')

        # Fallback to OMPL-based approach
        fallback_seq = ET.SubElement(fallback, 'Sequence')

        reach_pre = ET.SubElement(fallback_seq, 'ReachPreGraspOmpl')
        reach_pre.set('object_id', '{object_id}')
        reach_pre.set('end_effector_link', '{gripper_tcp}')
        reach_pre.set('pick_frame_id', '{preplace_frame}')
        reach_pre.set('pick_position_xyz', '[0.0, 0.0, 0.0]')
        reach_pre.set('pick_rotation_quat', '[0.0, 0.0, 0.0, 1.0]')
        reach_pre.set('pick_approach_direction', '[0.0, 0.0, 0.0]')
        reach_pre.set('pick_approach_distance', str(self.default_params['approach_distance']))
        reach_pre.set('planning_group', '{pg_arm}')
        reach_pre.set('controller_names', '')
        reach_pre.set('velocity_scaling_factor', str(self.default_params['velocity_scaling_factor']))
        reach_pre.set('use_joint_constraint', 'False')
        reach_pre.set('use_joint_target', 'True')
        reach_pre.set('joint_names', '{joint_names_initial}')
        reach_pre.set('joint_states', '{joint_positions_initial}')
        reach_pre.set('goal_tolerances', str(self.default_params['goal_tolerances']))
        reach_pre.set('constraint_tolerances_above', '')
        reach_pre.set('constraint_tolerances_below', '')
        reach_pre.set('acceleration_scaling_factor', str(self.default_params['acceleration_scaling_factor']))

        retry_subtree = ET.SubElement(fallback_seq, 'SubTree')
        retry_subtree.set('ID', 'PlacingSequence')
        retry_subtree.set('_autoremap', 'true')

        # Release the object
        release = ET.SubElement(seq, 'Release')
        release.set('object_id', '{object_id}')
        release.set('gripper_cmd_action_name', '{gripper_cmd_action_name}')
        release.set('place_frame_id', '{place_frame_id}')
        release.set('parent_object_id_after_release', '{@parent_object_id_after_release}')
        release.set('joint_states', '')
        release.set('disable_scene_handling', 'False')

        # Post-place retreat
        retry_retreat = ET.SubElement(seq, 'RetryUntilSuccessful')
        retry_retreat.set('num_attempts', '2')

        switch = ET.SubElement(retry_retreat, 'Switch2')
        switch.set('case_1', '{gripper_tcp}')
        switch.set('case_2', '{postplace_frame}')
        switch.set('variable', '{postplace_frame}')

        # Case 1: Linear retreat
        retreat1 = ET.SubElement(switch, 'MoveGraspLin')
        retreat1.set('object_id', '{object_id}')
        retreat1.set('pick_allowed_touch_objects', '{allowed_touch_links}')
        retreat1.set('end_effector_link', '{gripper_tcp}')
        retreat1.set('pick_frame_id', '{gripper_tcp}')
        retreat1.set('pick_position_xyz', '[0.0, 0.0, -0.1]')
        retreat1.set('pick_rotation_quat', '[0.0, 0.0, 0.0, 1.0]')
        retreat1.set('planning_group', '{pg_arm}')
        retreat1.set('controller_names', '')
        retreat1.set('velocity_scaling_factor', str(self.default_params['velocity_scaling_factor']))
        retreat1.set('use_joint_constraint', 'False')
        retreat1.set('joint_names', '')
        retreat1.set('goal_tolerances', '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]')
        retreat1.set('constraint_tolerances_above', '')
        retreat1.set('constraint_tolerances_below', '')
        retreat1.set('acceleration_scaling_factor', str(self.default_params['retreat_acceleration']))
        retreat1.set('blend_radius', '0.0')
        retreat1.set('add_to_blending_queue', 'False')

        # Case 2: Move to post-place frame
        retreat2 = ET.SubElement(switch, 'MoveGraspLin')
        retreat2.set('object_id', '{object_id}')
        retreat2.set('pick_allowed_touch_objects', '{allowed_touch_links}')
        retreat2.set('end_effector_link', '{gripper_tcp}')
        retreat2.set('pick_frame_id', '{postplace_frame}')
        retreat2.set('pick_position_xyz', '[0.0, 0.0, 0.0]')
        retreat2.set('pick_rotation_quat', '[0.0, 0.0, 0.0, 1.0]')
        retreat2.set('planning_group', '{pg_arm}')
        retreat2.set('controller_names', '')
        retreat2.set('velocity_scaling_factor', str(self.default_params['velocity_scaling_factor']))
        retreat2.set('use_joint_constraint', 'False')
        retreat2.set('joint_names', '')
        retreat2.set('goal_tolerances', '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]')
        retreat2.set('constraint_tolerances_above', '')
        retreat2.set('constraint_tolerances_below', '')
        retreat2.set('acceleration_scaling_factor', str(self.default_params['retreat_acceleration']))
        retreat2.set('blend_radius', '0.0')
        retreat2.set('add_to_blending_queue', 'False')

        # Default: Point-to-point retreat
        retreat3 = ET.SubElement(switch, 'ReachPreGraspPtp')
        retreat3.set('object_id', '{object_id}')
        retreat3.set('end_effector_link', '{gripper_tcp}')
        retreat3.set('pick_frame_id', '{gripper_tcp}')
        retreat3.set('pick_position_xyz', '[0.0, 0.0, 0.0]')
        retreat3.set('pick_rotation_quat', '[0.0, 0.0, 0.0, 1.0]')
        retreat3.set('pick_approach_direction', '[0.0, 0.0, -1.0]')
        retreat3.set('pick_approach_distance', str(self.default_params['approach_distance']))
        retreat3.set('planning_group', '{pg_arm}')
        retreat3.set('controller_names', '')
        retreat3.set('velocity_scaling_factor', str(self.default_params['velocity_scaling_factor']))
        retreat3.set('use_joint_constraint', 'false')
        retreat3.set('joint_names', '')
        retreat3.set('goal_tolerances', '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]')
        retreat3.set('constraint_tolerances_above', '')
        retreat3.set('constraint_tolerances_below', '')
        retreat3.set('acceleration_scaling_factor', str(self.default_params['acceleration_scaling_factor']))
        retreat3.set('blend_radius', '0.0')
        retreat3.set('add_to_blending_queue', 'False')

        return bt

    def _generate_picking_sequence_subtree(self) -> ET.Element:
        """Generate PickingSequence reusable subtree"""
        bt = ET.Element('BehaviorTree')
        bt.set('ID', 'PickingSequence')

        seq = ET.SubElement(bt, 'Sequence')

        # Reach pre-grasp
        reach_pre = ET.SubElement(seq, 'ReachPreGraspOmpl')
        reach_pre.set('object_id', '{object_id}')
        reach_pre.set('end_effector_link', '{gripper_tcp}')
        reach_pre.set('pick_frame_id', '{pregrasp_frame}')
        reach_pre.set('pick_position_xyz', '[0.0, 0.0, 0.0]')
        reach_pre.set('pick_rotation_quat', '[0.0, 0.0, 0.0, 1.0]')
        reach_pre.set('pick_approach_direction', '[0.0, 0.0, 0.0]')
        reach_pre.set('pick_approach_distance', str(self.default_params['approach_distance']))
        reach_pre.set('planning_group', '{pg_arm}')
        reach_pre.set('controller_names', '')
        reach_pre.set('velocity_scaling_factor', str(self.default_params['velocity_scaling_factor']))
        reach_pre.set('use_joint_constraint', 'False')
        reach_pre.set('use_joint_target', 'False')
        reach_pre.set('joint_names', '')
        reach_pre.set('joint_states', '')
        reach_pre.set('goal_tolerances', str(self.default_params['goal_tolerances']))
        reach_pre.set('constraint_tolerances_above', '')
        reach_pre.set('constraint_tolerances_below', '')
        reach_pre.set('acceleration_scaling_factor', str(self.default_params['acceleration_scaling_factor']))

        # Linear grasp motion
        grasp = ET.SubElement(seq, 'MoveGraspLin')
        grasp.set('object_id', '{object_id}')
        grasp.set('pick_allowed_touch_objects', '{allowed_touch_links}')
        grasp.set('end_effector_link', '{gripper_tcp}')
        grasp.set('pick_frame_id', '{pick_frame_id}')
        grasp.set('pick_position_xyz', '[0.0, 0.0, 0.0]')
        grasp.set('pick_rotation_quat', '[0.0, 0.0, 0.0, 1.0]')
        grasp.set('planning_group', '{pg_arm}')
        grasp.set('controller_names', '')
        grasp.set('velocity_scaling_factor', str(self.default_params['pick_velocity']))
        grasp.set('use_joint_constraint', 'False')
        grasp.set('joint_names', '')
        grasp.set('goal_tolerances', str(self.default_params['pick_goal_tolerances']))
        grasp.set('constraint_tolerances_above', '')
        grasp.set('constraint_tolerances_below', '')
        grasp.set('acceleration_scaling_factor', str(self.default_params['retreat_acceleration']))
        grasp.set('blend_radius', '0.0')
        grasp.set('add_to_blending_queue', 'False')

        return bt

    def _generate_placing_sequence_subtree(self) -> ET.Element:
        """Generate PlacingSequence reusable subtree"""
        bt = ET.Element('BehaviorTree')
        bt.set('ID', 'PlacingSequence')

        seq = ET.SubElement(bt, 'Sequence')

        # Reach pre-place
        reach_pre = ET.SubElement(seq, 'ReachPreGraspOmpl')
        reach_pre.set('object_id', '{object_id}')
        reach_pre.set('end_effector_link', '{gripper_tcp}')
        reach_pre.set('pick_frame_id', '{preplace_frame}')
        reach_pre.set('pick_position_xyz', '[0.0, 0.0, 0.0]')
        reach_pre.set('pick_rotation_quat', '[0.0, 0.0, 0.0, 1.0]')
        reach_pre.set('pick_approach_direction', '[0.0, 0.0, 0.0]')
        reach_pre.set('pick_approach_distance', str(self.default_params['approach_distance']))
        reach_pre.set('planning_group', '{pg_arm}')
        reach_pre.set('controller_names', '')
        reach_pre.set('velocity_scaling_factor', str(self.default_params['velocity_scaling_factor']))
        reach_pre.set('use_joint_constraint', 'False')
        reach_pre.set('use_joint_target', 'False')
        reach_pre.set('joint_names', '')
        reach_pre.set('joint_states', '')
        reach_pre.set('goal_tolerances', str(self.default_params['goal_tolerances']))
        reach_pre.set('constraint_tolerances_above', '')
        reach_pre.set('constraint_tolerances_below', '')
        reach_pre.set('acceleration_scaling_factor', str(self.default_params['acceleration_scaling_factor']))

        # Linear place motion
        place = ET.SubElement(seq, 'MoveGraspLin')
        place.set('object_id', '{object_id}')
        place.set('pick_allowed_touch_objects', '{allowed_touch_links}')
        place.set('end_effector_link', '{gripper_tcp}')
        place.set('pick_frame_id', '{place_frame_id}')
        place.set('pick_position_xyz', '[0.0, 0.0, 0.0]')
        place.set('pick_rotation_quat', '[0.0, 0.0, 0.0, 1.0]')
        place.set('planning_group', '{pg_arm}')
        place.set('controller_names', '')
        place.set('velocity_scaling_factor', str(self.default_params['place_velocity']))
        place.set('use_joint_constraint', 'False')
        place.set('joint_names', '')
        place.set('goal_tolerances', str(self.default_params['place_goal_tolerances']))
        place.set('constraint_tolerances_above', '')
        place.set('constraint_tolerances_below', '')
        place.set('acceleration_scaling_factor', str(self.default_params['retreat_acceleration']))
        place.set('blend_radius', '0.0')
        place.set('add_to_blending_queue', 'False')

        return bt

    def _generate_spawn_objects_subtree(self, scene_config: Dict[str, Any],
                                       operations: List[ObjectOperation]) -> ET.Element:
        """Generate SpawnObjects subtree"""
        bt = ET.Element('BehaviorTree')
        bt.set('ID', 'SpawnObjects')

        seq = ET.SubElement(bt, 'Sequence')

        # Use the first operation's pick frame for frame availability check
        if operations:
            first_frame = operations[0].pick.pick_frame_id

            # Fallback structure
            fallback = ET.SubElement(seq, 'Fallback')

            # Check if frame is available
            is_frame = ET.SubElement(fallback, 'IsFrameAvailableSrv')
            is_frame.set('service_name', '/is_frame_available')
            is_frame.set('frame', first_frame)

            # If not available, load workspace
            retry_load = ET.SubElement(fallback, 'RetryUntilSuccessful')
            retry_load.set('num_attempts', '2')

            load_seq = ET.SubElement(retry_load, 'Sequence')

            # Load workspace service
            load_ws = ET.SubElement(load_seq, 'LoadWorkspaceService')
            load_ws.set('service_name', '/load_workspace')
            load_ws.set('file_path', scene_config.get('file_path', ''))

            # Retry checking frame availability
            retry_check = ET.SubElement(load_seq, 'RetryUntilSuccessful')
            retry_check.set('num_attempts', '2')

            check_seq = ET.SubElement(retry_check, 'Sequence')

            # Delay and check
            delay = ET.SubElement(check_seq, 'Delay')
            delay.set('delay_msec', '2000')

            check_inner = ET.SubElement(delay, 'IsFrameAvailableSrv')
            check_inner.set('service_name', '/is_frame_available')
            check_inner.set('frame', first_frame)

        return bt

    def _generate_tree_nodes_model(self) -> ET.Element:
        """Generate TreeNodesModel with action definitions"""
        model = ET.Element('TreeNodesModel')

        # Grasp action
        grasp = ET.SubElement(model, 'Action')
        grasp.set('ID', 'Grasp')
        grasp.set('editable', 'true')
        for port in ['object_id', 'end_effector_link', 'gripper_cmd_action_name',
                     'pick_allowed_touch_objects', 'joint_states', 'disable_scene_handling']:
            inp = ET.SubElement(grasp, 'input_port')
            inp.set('name', port)

        # Release action
        release = ET.SubElement(model, 'Action')
        release.set('ID', 'Release')
        release.set('editable', 'true')
        for port in ['object_id', 'gripper_cmd_action_name', 'place_frame_id',
                     'parent_object_id_after_release', 'joint_states', 'disable_scene_handling']:
            inp = ET.SubElement(release, 'input_port')
            inp.set('name', port)

        # IsFrameAvailableSrv action
        is_frame = ET.SubElement(model, 'Action')
        is_frame.set('ID', 'IsFrameAvailableSrv')
        is_frame.set('editable', 'true')
        for port in ['service_name', 'frame']:
            inp = ET.SubElement(is_frame, 'input_port')
            inp.set('name', port)

        # LoadWorkspaceService action
        load_ws = ET.SubElement(model, 'Action')
        load_ws.set('ID', 'LoadWorkspaceService')
        load_ws.set('editable', 'true')
        for port in ['service_name', 'file_path']:
            inp = ET.SubElement(load_ws, 'input_port')
            inp.set('name', port)

        # MoveGraspLin action
        move_grasp = ET.SubElement(model, 'Action')
        move_grasp.set('ID', 'MoveGraspLin')
        move_grasp.set('editable', 'true')
        for port in ['object_id', 'pick_allowed_touch_objects', 'end_effector_link',
                     'pick_frame_id', 'pick_position_xyz', 'pick_rotation_quat',
                     'planning_group', 'controller_names', 'velocity_scaling_factor',
                     'use_joint_constraint', 'joint_names', 'goal_tolerances',
                     'constraint_tolerances_above', 'constraint_tolerances_below',
                     'acceleration_scaling_factor', 'blend_radius', 'add_to_blending_queue']:
            inp = ET.SubElement(move_grasp, 'input_port')
            inp.set('name', port)

        # ReachPreGraspOmpl action
        reach_ompl = ET.SubElement(model, 'Action')
        reach_ompl.set('ID', 'ReachPreGraspOmpl')
        reach_ompl.set('editable', 'true')
        for port in ['object_id', 'end_effector_link', 'pick_frame_id',
                     'pick_position_xyz', 'pick_rotation_quat', 'pick_approach_direction',
                     'pick_approach_distance', 'planning_group', 'controller_names',
                     'velocity_scaling_factor', 'use_joint_constraint', 'use_joint_target',
                     'joint_names', 'joint_states', 'goal_tolerances',
                     'constraint_tolerances_above', 'constraint_tolerances_below',
                     'acceleration_scaling_factor']:
            inp = ET.SubElement(reach_ompl, 'input_port')
            inp.set('name', port)

        # ReachPreGraspPtp action
        reach_ptp = ET.SubElement(model, 'Action')
        reach_ptp.set('ID', 'ReachPreGraspPtp')
        reach_ptp.set('editable', 'true')
        for port in ['object_id', 'end_effector_link', 'pick_frame_id',
                     'pick_position_xyz', 'pick_rotation_quat', 'pick_approach_direction',
                     'pick_approach_distance', 'planning_group', 'controller_names',
                     'velocity_scaling_factor', 'use_joint_constraint', 'joint_names',
                     'goal_tolerances', 'constraint_tolerances_above',
                     'constraint_tolerances_below', 'acceleration_scaling_factor',
                     'blend_radius', 'add_to_blending_queue']:
            inp = ET.SubElement(reach_ptp, 'input_port')
            inp.set('name', port)

        # ReadCurrentJointState action
        read_joint = ET.SubElement(model, 'Action')
        read_joint.set('ID', 'ReadCurrentJointState')
        read_joint.set('editable', 'true')
        for port in ['joint_names_filter', 'joint_names', 'joint_positions']:
            inp = ET.SubElement(read_joint, 'input_port')
            inp.set('name', port)

        return model

    def _validate_xml(self, xml_content: str) -> bool:
        """
        Validate generated XML.

        Args:
            xml_content: XML string to validate

        Returns:
            True if valid

        Raises:
            GenerationError: If XML is invalid
        """
        try:
            ET.fromstring(xml_content)
            logger.info("XML validation successful")
            return True
        except ET.ParseError as e:
            raise GenerationError(f"XML validation failed: {e}")

    def _format_xml(self, xml_content: str) -> str:
        """
        Format XML with proper indentation.

        Args:
            xml_content: Raw XML string

        Returns:
            Formatted XML string
        """
        try:
            # Remove XML declaration for parsing
            xml_body = xml_content.replace('<?xml version="1.0" encoding="UTF-8"?>\n', '')

            # Parse and format
            dom = minidom.parseString(xml_body)
            formatted = dom.toprettyxml(indent=" " * self.indent_size)

            # Clean up extra whitespace
            lines = [line for line in formatted.split('\n') if line.strip()]

            # Re-add XML declaration
            result = '<?xml version="1.0" encoding="UTF-8"?>\n'
            result += '\n'.join(lines[1:])  # Skip the first line (xml declaration from minidom)

            return result

        except Exception as e:
            logger.warning(f"XML formatting failed: {e}, returning unformatted")
            return xml_content

    def _save_xml(self, xml_content: str, output_path: str) -> None:
        """
        Save XML content to file.

        Args:
            xml_content: XML string
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(xml_content)


class BTLangChainTool:
    """
    LangChain-compatible wrapper for BTGenerator.

    This class provides a tool interface for LangChain agents to generate
    behavior trees from YAML configurations.

    Example:
        from langchain.agents import Tool

        bt_tool = BTLangChainTool()
        tool = Tool(
            name="BehaviorTreeGenerator",
            func=bt_tool.generate,
            description="Generates Behavior Tree XML from YAML configuration"
        )
    """

    def __init__(self, validate_xml: bool = True, auto_format: bool = True):
        """
        Initialize the LangChain tool.

        Args:
            validate_xml: Whether to validate generated XML
            auto_format: Whether to auto-format output
        """
        self.generator = BTGenerator(
            validate_xml=validate_xml,
            auto_format=auto_format
        )

    def generate(self, yaml_path: str, output_path: Optional[str] = None) -> str:
        """
        Generate behavior tree XML from YAML.

        This method provides a simple interface for LangChain integration.

        Args:
            yaml_path: Path to YAML configuration file
            output_path: Optional output path for XML file

        Returns:
            Generated XML string or error message
        """
        try:
            return self.generator.generate_from_yaml(yaml_path, output_path)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"

    def generate_from_string(self, yaml_content: str, output_path: Optional[str] = None) -> str:
        """
        Generate behavior tree XML from YAML string.

        Args:
            yaml_content: YAML content as string
            output_path: Optional output path

        Returns:
            Generated XML string or error message
        """
        try:
            config = yaml.safe_load(yaml_content)
            return self.generator.generate_from_dict(config, output_path)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"


def main():
    """CLI entry point for standalone usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate Behavior Tree XML from YAML configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Generate XML and print to stdout
    python bt_generator.py input.yaml

    # Generate and save to file
    python bt_generator.py input.yaml -o output.xml

    # Generate with validation disabled
    python bt_generator.py input.yaml -o output.xml --no-validate

    # Generate without formatting
    python bt_generator.py input.yaml -o output.xml --no-format
        '''
    )

    parser.add_argument('input', help='Input YAML configuration file')
    parser.add_argument('-o', '--output', help='Output XML file path (optional)')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip XML validation')
    parser.add_argument('--no-format', action='store_true',
                       help='Skip XML formatting')
    parser.add_argument('--indent', type=int, default=2,
                       help='XML indentation size (default: 2)')

    args = parser.parse_args()

    # Create generator
    generator = BTGenerator(
        validate_xml=not args.no_validate,
        auto_format=not args.no_format,
        indent_size=args.indent
    )

    try:
        # Generate XML
        xml_output = generator.generate_from_yaml(args.input, args.output)

        # Print to stdout if no output file specified
        if not args.output:
            print(xml_output)
        else:
            print(f"Successfully generated: {args.output}")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        exit(1)


if __name__ == '__main__':
    main()
