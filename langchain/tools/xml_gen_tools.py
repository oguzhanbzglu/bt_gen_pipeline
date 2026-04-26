"""
Behavior Tree XML Generation and Validation Tools

This module provides LangChain-compatible tools for:
1. Generating BehaviorTree.CPP v4 XML from YAML configurations
2. Validating XML structure and syntax

Uses langchain_core for modern LangChain/LangGraph integration.
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import xml.etree.ElementTree as ET
import yaml
import os
import sys
from typing import Optional, Dict, List
from pathlib import Path

# Get package root dynamically (no hardcoded paths)
PACKAGE_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
sys.path.insert(0, str(PACKAGE_ROOT.parent))
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from bt_generator import BTGenerator, GenerationError
except ImportError:
    BTGenerator = None
    GenerationError = Exception


class XMLGeneratorInput(BaseModel):
    """Input schema for XML generator tool"""
    yaml_content: str = Field(
        description="YAML configuration as a string OR path to YAML file. If it's a path (starts with / or contains .yaml), the file will be read automatically."
    )
    yaml_path: Optional[str] = Field(
        None,
        description="Optional path to save the YAML config file (for debugging)"
    )
    output_path: Optional[str] = Field(
        None,
        description="Optional path to save the generated XML file. If not provided, auto-saves to behavior_trees/"
    )


class XMLValidatorInput(BaseModel):
    """Input schema for XML validator tool"""
    xml_content: str = Field(
        description="The XML content to validate (full XML string)"
    )

def xml_generator_func(
    yaml_content: str,
    yaml_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict:
    """
    Generate Behavior Tree XML from YAML configuration.

    Args:
        yaml_content: YAML configuration as string OR path to YAML file
        yaml_path: Optional path to save YAML temporarily
        output_path: Optional path to save generated XML.
                    If None, auto-generates path in behavior_trees/ folder

    Returns:
        Dict with:
        - success (bool): Whether generation succeeded
        - xml_content (str): Generated XML content
        - message (str): Human-readable status message
        - output_path (str): Where XML was saved
        - error (str): Error message if failed
    """
    try:
        if BTGenerator is None:
            return {
                'success': False,
                'xml_content': None,
                'message': 'BTGenerator not available',
                'output_path': None,
                'error': 'Failed to import BTGenerator module. Check if bt_generator.py is in scripts/'
            }

        generator = BTGenerator(validate_xml=True, auto_format=True)

        # Check if yaml_content is a file path (if it looks like a path, read it)
        if yaml_content.strip().startswith('/') or '.yaml' in yaml_content or '.yml' in yaml_content:
            # It's likely a file path, try to read it
            try:
                with open(yaml_content.strip(), 'r') as f:
                    yaml_content = f.read()
            except FileNotFoundError:
                return {
                    'success': False,
                    'xml_content': None,
                    'message': f'YAML file not found: {yaml_content}',
                    'output_path': None,
                    'error': f'File not found: {yaml_content}'
                }
            except Exception as e:
                return {
                    'success': False,
                    'xml_content': None,
                    'message': f'Failed to read YAML file: {str(e)}',
                    'output_path': None,
                    'error': str(e)
                }

        # Parse YAML to get BT name for auto-naming
        config = yaml.safe_load(yaml_content)
        bt_name = config.get('behavior_tree', {}).get('name', 'UnnamedBT')

        # Auto-generate output path if not provided
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            behavior_trees_dir = PACKAGE_ROOT / 'behavior_trees'
            behavior_trees_dir.mkdir(exist_ok=True)
            output_path = str(behavior_trees_dir / f'{bt_name}_{timestamp}.xml')

        # If yaml_path provided, save content to file first
        if yaml_path:
            dir_name = os.path.dirname(yaml_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)
            xml_content = generator.generate_from_yaml(yaml_path, output_path)
        else:
            # Generate from dict directly
            xml_content = generator.generate_from_dict(config, output_path)

        return {
            'success': True,
            'xml_content': xml_content,
            'message': f'Successfully generated XML ({len(xml_content)} characters)',
            'output_path': output_path,
            'error': None
        }

    except GenerationError as e:
        return {
            'success': False,
            'xml_content': None,
            'message': 'BT generation failed',
            'output_path': None,
            'error': f'GenerationError: {str(e)}'
        }
    except yaml.YAMLError as e:
        return {
            'success': False,
            'xml_content': None,
            'message': 'Invalid YAML format',
            'output_path': None,
            'error': f'YAML parsing error: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'xml_content': None,
            'message': 'Unexpected error during generation',
            'output_path': None,
            'error': f'{type(e).__name__}: {str(e)}'
        }


def xml_validator_func(xml_content: str) -> Dict:
    """
    Validate Behavior Tree XML for syntax and structural errors.

    Args:
        xml_content: The XML content to validate

    Returns:
        Dict with:
        - valid (bool): Whether XML is valid
        - message (str): Summary of validation result
        - errors (List[str]): List of errors found
        - warnings (List[str]): List of warnings
    """
    errors = []
    warnings = []

    try:
        # Check 1: Basic XML syntax
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            return {
                'valid': False,
                'message': f'XML syntax error: {str(e)}',
                'errors': [f'XML Parse Error: {str(e)}'],
                'warnings': []
            }

        # Check 2: BTCPP_format attribute
        if root.tag != 'root':
            errors.append(f'Root element should be <root>, found <{root.tag}>')

        bt_format = root.get('BTCPP_format')
        if not bt_format:
            errors.append('Missing BTCPP_format attribute on root element')
        elif bt_format != '4':
            warnings.append(f'BTCPP_format is "{bt_format}", expected "4"')

        # Check 3: Required BehaviorTree elements
        behavior_trees = root.findall('BehaviorTree')
        if not behavior_trees:
            errors.append('No BehaviorTree elements found')
        else:
            for i, bt in enumerate(behavior_trees):
                bt_id = bt.get('ID')
                if not bt_id:
                    errors.append(f'BehaviorTree #{i+1} missing ID attribute')
                else:
                    # Check for Sequence
                    sequence = bt.find('Sequence')
                    if sequence is None:
                        errors.append(f'BehaviorTree "{bt_id}" missing root Sequence')

                    # Check for Script node
                    scripts = bt.findall('.//Script')
                    if not scripts:
                        warnings.append(f'BehaviorTree "{bt_id}" has no Script nodes')

        # Check 4: TreeNodesModel
        tree_model = root.find('TreeNodesModel')
        if tree_model is None:
            warnings.append('No TreeNodesModel found (needed for Groot visualization)')

        # Check 5: Common syntax issues
        if '&amp;amp;' in xml_content:
            errors.append('Double-encoded ampersands found (&amp;amp;)')

        if xml_content.count('<root') != xml_content.count('</root>'):
            errors.append('Unmatched root tags')

        # Determine overall validity
        is_valid = len(errors) == 0

        if is_valid and len(warnings) == 0:
            message = 'XML is valid and well-formed'
        elif is_valid:
            message = f'XML is valid with {len(warnings)} warning(s)'
        else:
            message = f'XML has {len(errors)} error(s) and {len(warnings)} warning(s)'

        return {
            'valid': is_valid,
            'message': message,
            'errors': errors,
            'warnings': warnings
        }

    except Exception as e:
        return {
            'valid': False,
            'message': f'Validation failed: {str(e)}',
            'errors': [str(e)],
            'warnings': []
        }

xml_generator = StructuredTool.from_function(
    func=xml_generator_func,
    name="xml_generator",
    description="""Generate Behavior Tree XML from YAML configuration.

Input: yaml_content can be EITHER:
  1. A file path (e.g., "/path/to/config.yaml") - the file will be read automatically
  2. Raw YAML content as a string

The tool will:
  - Read the file if a path is provided
  - Generate BehaviorTree.CPP v4 XML
  - Auto-save to behavior_trees/{BT_NAME}_{TIMESTAMP}.xml
  - Return the XML content

Use this when user asks to "generate", "create", or "make" a BT from a YAML file.
""",
    args_schema=XMLGeneratorInput,
    return_direct=False
)


xml_validator = StructuredTool.from_function(
    func=xml_validator_func,
    name="xml_validator",
    description="""Validate Behavior Tree XML for syntax and structural errors.

NOTE: xml_generator already validates during generation. Use this tool when:
- You have XML from another source
- You want to re-check existing XML files
- You need detailed validation feedback

Checks performed:
- XML syntax validity (well-formed)
- BTCPP_format="4" attribute presence
- Required BehaviorTree elements
- Proper tree structure
- Script node syntax
- TreeNodesModel for Groot visualization
""",
    args_schema=XMLValidatorInput,
    return_direct=False
)


# Export tools for easy import
__all__ = ['xml_generator', 'xml_validator']
