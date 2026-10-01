"""Blender MCP wrappers and project-level tool adapters."""
from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from datetime import datetime

from langchain_mcp_adapters.client import MultiServerMCPClient


BLENDER_MCP_CONFIG = {
    "blender": {
        "command": "uvx",
        "args": ["blender-mcp"],
        "transport": "stdio",
    }
}

IMPORT_OBJECT_PREFIX = "IMPORTED_OBJECT:"
IMPORT_OBJECTS_LIST_PREFIX = "IMPORTED_OBJECTS_LIST:"
DEFAULT_SCENE_INFO_PROMPT = (
    "Summarize the current Blender scene. "
    "List objects, counts, transforms, and composition notes."
)


class BlenderMCP:
    """Thin wrapper around Blender MCP tools."""

    def __init__(self):
        self._client: MultiServerMCPClient | None = None
        self._tools: list = []

    async def connect(self):
        """Create the MCP connection."""
        self._client = MultiServerMCPClient(BLENDER_MCP_CONFIG)
        self._tools = await self._client.get_tools()
        print(f"  Blender MCP connected, tools: {[t.name for t in self._tools]}")
        return self._tools

    @property
    def tools(self):
        return self._tools

    async def execute_command(self, tool_name: str, args: dict) -> str:
        """Execute a single Blender MCP tool call."""
        matching_tool = next((t for t in self._tools if t.name == tool_name), None)
        if not matching_tool:
            return f"ERROR: tool not found: {tool_name}"

        try:
            result = await matching_tool.ainvoke(args)
            return str(result)
        except Exception as exc:
            err = str(exc)
            if "user_prompt" in err and "Field required" in err and "user_prompt" not in args:
                retry_args = dict(args)
                retry_args["user_prompt"] = DEFAULT_SCENE_INFO_PROMPT
                try:
                    result = await matching_tool.ainvoke(retry_args)
                    return str(result)
                except Exception as retry_exc:
                    return f"ERROR: {retry_exc}"
            return f"ERROR: {exc}"

    async def get_scene_info(self, user_prompt: str = DEFAULT_SCENE_INFO_PROMPT) -> str:
        """Get current Blender scene info."""
        return await self.execute_command("get_scene_info", {"user_prompt": user_prompt})

    async def get_screenshot(self) -> str:
        """Get a stable Blender viewport screenshot path."""
        return await self._capture_screenshot_via_code()

    async def execute_code(self, code: str) -> str:
        """Execute bpy code inside Blender."""
        return await self.execute_command("execute_blender_code", {"code": code})

    async def _capture_screenshot_via_code(self) -> str:
        """Capture a Blender window screenshot via execute_blender_code."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(tempfile.gettempdir(), f"blender_fallback_{timestamp}.png")
        safe_path = filepath.replace("\\", "\\\\")
        code = f"""
import bpy
import os

path = r"{safe_path}"
try:
    bpy.ops.screen.screenshot(filepath=path)
    print(path)
    print(os.path.exists(path))
except Exception as exc:
    print(f"ERROR: {{exc}}")
"""
        await self.execute_code(code)
        if os.path.exists(filepath):
            return filepath
        return ""

    async def import_model(self, file_path: str, label: str = "") -> str:
        """Import one 3D model into Blender and print deterministic markers."""
        ext = file_path.rsplit(".", 1)[-1].lower()
        path_escaped = file_path.replace("\\", "/")

        if ext in {"glb", "gltf"}:
            import_line = f'bpy.ops.import_scene.gltf(filepath="{path_escaped}")'
        elif ext == "obj":
            import_line = f'bpy.ops.wm.obj_import(filepath="{path_escaped}")'
        elif ext == "fbx":
            import_line = f'bpy.ops.import_scene.fbx(filepath="{path_escaped}")'
        else:
            return f"ERROR: unsupported model format: {ext}"

        import_code = f"""
import bpy

before = set(bpy.data.objects.keys())
{import_line}
after = set(bpy.data.objects.keys())
new_objects = after - before

for obj_name in new_objects:
    obj = bpy.data.objects[obj_name]
    print(f"{IMPORT_OBJECT_PREFIX} {{obj_name}}, type: {{obj.type}}, location: {{list(obj.location)}}")

print(f"{IMPORT_OBJECTS_LIST_PREFIX} {{list(new_objects)}}")
"""
        result = await self.execute_code(import_code)
        print(f"  Imported model [{label}]: {file_path}")
        return result


class BlenderSceneAssembler:
    """Project-level tool adapters for the ReAct agent."""

    def __init__(self, blender: BlenderMCP):
        self.blender = blender

    async def connect(self):
        """Ensure the Blender MCP connection is ready."""
        return await self.blender.connect()

    async def get_scene_info(self, user_prompt: str = DEFAULT_SCENE_INFO_PROMPT) -> dict:
        """Return structured scene info for the agent."""
        scene_info = await self.blender.get_scene_info(user_prompt=user_prompt)
        scene_info = self.sanitize_scene_info(scene_info)
        return {
            "success": not str(scene_info).startswith("ERROR:"),
            "scene_info": scene_info,
        }

    @staticmethod
    def sanitize_scene_info(scene_info: str) -> str:
        """Remove import-container EMPTY nodes from scene summaries shown to the agent."""
        if not isinstance(scene_info, str) or not scene_info.strip():
            return scene_info

        try:
            parsed = json.loads(scene_info)
        except Exception:
            return scene_info

        if not isinstance(parsed, dict):
            return scene_info

        objects = parsed.get("objects", [])
        if not isinstance(objects, list):
            return scene_info

        filtered_objects: list[dict] = []
        for item in objects:
            if not isinstance(item, dict):
                filtered_objects.append(item)
                continue

            obj_type = str(item.get("type", "")).strip().upper()
            name = str(item.get("name", "")).strip()
            is_import_container = obj_type == "EMPTY" and re.fullmatch(r"world(?:\.\d+)?", name, flags=re.IGNORECASE)
            if is_import_container:
                continue
            filtered_objects.append(item)

        parsed["objects"] = filtered_objects
        parsed["object_count"] = len(filtered_objects)
        return json.dumps(parsed, ensure_ascii=False, indent=2)

    async def get_blender_screenshot(self) -> dict:
        """Return the latest Blender screenshot filepath for the agent."""
        screenshot_path = await self.blender.get_screenshot()
        return {
            "success": bool(screenshot_path and os.path.exists(screenshot_path)),
            "screenshot_path": str(screenshot_path) if screenshot_path else "",
        }

    @staticmethod
    def parse_imported_objects(result: str) -> list[dict]:
        """Parse imported object records from tool output."""
        objects: list[dict] = []
        for line in result.splitlines():
            if IMPORT_OBJECT_PREFIX not in line:
                continue
            payload = line.split(IMPORT_OBJECT_PREFIX, 1)[1].strip()
            parts = [part.strip() for part in payload.split(",")]
            name = parts[0] if parts else ""
            obj_type = ""
            for part in parts[1:]:
                if "type:" in part:
                    obj_type = part.split("type:", 1)[1].strip()
            if name:
                objects.append({"name": name, "type": obj_type})
        return objects

    @staticmethod
    def parse_blender_object_name(result: str) -> str:
        """Pick the primary mesh name that the agent should manipulate."""
        imported_objects = BlenderSceneAssembler.parse_imported_objects(result)
        if imported_objects:
            for item in imported_objects:
                if item.get("type") == "MESH":
                    return str(item.get("name") or "").strip()
            for item in imported_objects:
                if item.get("type") == "EMPTY":
                    return str(item.get("name") or "").strip()
            return str(imported_objects[0].get("name") or "").strip()

        for line in result.splitlines():
            if IMPORT_OBJECTS_LIST_PREFIX not in line:
                continue
            raw = line.split(IMPORT_OBJECTS_LIST_PREFIX, 1)[1].strip()
            try:
                names = ast.literal_eval(raw)
            except Exception:
                names = []
            if names:
                return str(list(names)[0]).strip()
        return ""

    async def import_model(self, file_path: str, label: str = "") -> dict:
        """Import one model and return a structured observation."""
        if not os.path.exists(file_path):
            return {
                "success": False,
                "label": label,
                "source_file": file_path,
                "root_object_name": "",
                "imported_objects": [],
                "raw_result": f"ERROR: file does not exist: {file_path}",
            }

        result = await self.blender.import_model(file_path, label)
        imported_objects = self.parse_imported_objects(result)
        blender_name = self.parse_blender_object_name(result)
        primary_mesh_name = ""
        parent_object_name = ""
        for item in imported_objects:
            if item.get("type") == "MESH":
                primary_mesh_name = str(item.get("name") or "").strip()
                if primary_mesh_name:
                    break
        for item in imported_objects:
            if item.get("type") == "EMPTY":
                parent_object_name = str(item.get("name") or "").strip()
                if parent_object_name:
                    break
        return {
            "success": bool(blender_name),
            "label": label,
            "source_file": file_path,
            "root_object_name": blender_name,
            "primary_mesh_name": primary_mesh_name,
            "parent_object_name": parent_object_name,
            "imported_objects": imported_objects,
            "raw_result": result,
        }

    async def execute_blender_code(self, code: str) -> dict:
        """Execute bpy code and return a structured observation."""
        result = await self.blender.execute_code(code)
        success = not str(result).startswith("ERROR:") and "Traceback" not in str(result)
        return {
            "success": success,
            "stdout": result if success else "",
            "error": "" if success else result,
        }

    async def get_object_summary(self, object_name: str) -> dict:
        """Collect spatial data for one object root."""
        if not object_name:
            return {"success": False, "found": False, "name": ""}

        safe_name = object_name.replace("\\", "\\\\").replace('"', '\\"')
        code = f"""
import bpy
import json
from mathutils import Vector

name = "{safe_name}"
obj = bpy.data.objects.get(name)

if obj is None:
    print(json.dumps({{"found": False, "name": name}}))
else:
    targets = []
    if obj.type == "EMPTY":
        targets = [child for child in obj.children_recursive if child.type == "MESH"]
    if not targets and obj.type == "MESH":
        targets = [obj]

    corners = []
    for target in targets:
        for corner in target.bound_box:
            world_corner = target.matrix_world @ Vector(corner)
            corners.append([round(world_corner.x, 4), round(world_corner.y, 4), round(world_corner.z, 4)])

    if corners:
        mins = [round(min(c[i] for c in corners), 4) for i in range(3)]
        maxs = [round(max(c[i] for c in corners), 4) for i in range(3)]
        dims = [round(maxs[i] - mins[i], 4) for i in range(3)]
    else:
        loc = [round(float(v), 4) for v in obj.location]
        mins = list(loc)
        maxs = list(loc)
        dims = [0.0, 0.0, 0.0]

    payload = {{
        "success": True,
        "found": True,
        "name": obj.name,
        "type": obj.type,
        "location": [round(float(v), 4) for v in obj.location],
        "scale": [round(float(v), 4) for v in obj.scale],
        "dimensions": dims,
        "bbox_min": mins,
        "bbox_max": maxs,
        "children": [child.name for child in obj.children],
    }}
    print(json.dumps(payload, ensure_ascii=False))
"""
        result = await self.blender.execute_code(code)
        parsed = self._extract_json_from_text(result)
        if parsed:
            return parsed
        return {
            "success": False,
            "found": False,
            "name": object_name,
            "raw_result": result,
        }

    @staticmethod
    def _extract_json_from_text(text: str) -> dict | None:
        """Extract a JSON object from Blender stdout."""
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            candidates = [line]
            first_brace = line.find("{")
            last_brace = line.rfind("}")
            if 0 <= first_brace < last_brace:
                candidates.append(line[first_brace : last_brace + 1])
            for candidate in candidates:
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    try:
                        parsed = ast.literal_eval(candidate)
                    except Exception:
                        continue
                if isinstance(parsed, dict):
                    return parsed
        return None
