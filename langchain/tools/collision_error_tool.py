"""
Collision Error Tool

This tool fixes collision errors by modifying the YAML configuration.
Instead of subscribing to rosout (which has timing issues), 
it takes the collision info directly from the LLM.
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import yaml


class CollisionErrorInput(BaseModel):
    """Input schema for collision error tool"""
    yaml_path: str = Field(
        description="Path to the YAML configuration file to fix"
    )
    colliding_objects: str = Field(
        default="",
        description="Comma-separated colliding objects (e.g., 'cell, part_2'). If empty, will try to parse from bt_executor error."
    )
    attempt: int = Field(
        default=1,
        description="Recovery attempt number (1=add allowed_touch_links, 2+=also increase approach_distance)"
    )


def collision_error_func(yaml_path: str, colliding_objects: str = "", attempt: int = 1) -> str:
    """
    Fix collision error by modifying YAML.
    
    Input:
        - yaml_path: Path to YAML file to modify
        - colliding_objects: Comma-separated objects (e.g., "cell, part_2")
        - attempt: Recovery attempt (1=allowed_touch_links, 2+=also increase approach_distance)
    
    Recovery strategy:
        - Attempt 1: Add colliding objects to script.allowed_touch_links
        - Attempt 2+: Also increase behavior_tree.approach_distance by 1.5x
    
    Returns JSON with success status and what was changed.
    """
    try:
        # Parse colliding objects
        objects_list = []
        if colliding_objects:
            objects_list = [obj.strip() for obj in colliding_objects.split(',') if obj.strip()]
        
        # If no objects provided, we can't fix
        if not objects_list:
            return json.dumps({
                'success': False,
                'message': 'No colliding objects provided. Please specify colliding_objects parameter.'
            })
        
        # Read YAML
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        changes = []
        
        # Fix 1: Add colliding objects to allowed_touch_links
        current_links = config.get('script', {}).get('allowed_touch_links', [])
        if isinstance(current_links, str):
            current_links = [current_links]
        elif not current_links:
            current_links = []
        
        new_links = list(set(current_links + objects_list))
        
        if new_links != current_links:
            if 'script' not in config:
                config['script'] = {}
            config['script']['allowed_touch_links'] = new_links
            changes.append(f"added {objects_list} to allowed_touch_links")
        
        # Fix 2: Increase approach_distance (attempt 2+)
        if attempt >= 2:
            current_dist = config.get('behavior_tree', {}).get('approach_distance', 0.1)
            new_dist = round(current_dist * 1.5, 3)
            
            if 'behavior_tree' not in config:
                config['behavior_tree'] = {}
            
            config['behavior_tree']['approach_distance'] = new_dist
            changes.append(f"increased approach_distance from {current_dist} to {new_dist}")
        
        # Save modified YAML
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return json.dumps({
            'success': True,
            'message': f"Recovery applied ({attempt}): {'; '.join(changes)}",
            'colliding_objects': objects_list,
            'yaml_path': yaml_path,
            'changes': changes
        })
        
    except FileNotFoundError:
        return json.dumps({
            'success': False,
            'message': f'YAML file not found: {yaml_path}'
        })
    except Exception as e:
        return json.dumps({
            'success': False,
            'message': f'Error: {str(e)}'
        })


collision_error = StructuredTool.from_function(
    func=collision_error_func,
    name="collision_error",
    description="""Fix collision error by modifying YAML.

Input:
- yaml_path: Path to YAML file to modify
- colliding_objects: Comma-separated colliding objects (e.g., "cell, part_2")
- attempt: Recovery attempt number (1=add allowed_touch_links, 2+=also increase approach_distance)

Recovery strategy:
- Attempt 1: Add colliding objects to script.allowed_touch_links
- Attempt 2+: Also increase behavior_tree.approach_distance by 1.5x

Returns JSON with success status and what was changed.
""",
    args_schema=CollisionErrorInput,
    return_direct=False
)


__all__ = ['collision_error']
