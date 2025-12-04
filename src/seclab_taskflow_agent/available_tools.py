# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

import importlib.resources
from enum import Enum

import yaml


class BadToolNameError(Exception):
    pass


class VersionException(Exception):
    pass


class FileTypeException(Exception):
    pass


class AvailableToolType(Enum):
    Personality = "personality"
    Taskflow = "taskflow"
    Prompt = "prompt"
    Toolbox = "toolbox"
    ModelConfig = "model_config"


class AvailableTools:
    """
    This class is used for storing dictionaries of all the available
    personalities, taskflows, and prompts.
    """

    def __init__(self):
        self.__yamlcache = {}

    def get_personality(self, name: str):
        return self.get_tool(AvailableToolType.Personality, name)

    def get_taskflow(self, name: str):
        return self.get_tool(AvailableToolType.Taskflow, name)

    def get_prompt(self, name: str):
        return self.get_tool(AvailableToolType.Prompt, name)

    def get_toolbox(self, name: str):
        return self.get_tool(AvailableToolType.Toolbox, name)

    def get_model_config(self, name: str):
        return self.get_tool(AvailableToolType.ModelConfig, name)

    def get_tool(self, tooltype: AvailableToolType, toolname: str):
        """for example: available_tools.get_tool("personality", "personalities/fruit_expert")
        This method first checks whether the tool has already been loaded. If not, it
        finds the yaml file and parses it. It also checks that the filetype in the header
        matches the expected tooltype.
        """
        try:
            return self.__yamlcache[tooltype][toolname]
        except KeyError:
            pass
        # Split the string to get the package and filename.
        components = toolname.rsplit(".", 1)
        if len(components) != 2:
            raise BadToolNameError(
                f'Not a valid toolname: "{toolname}". It should be something like: "packagename.filename"'
            )
        package = components[0]
        filename = components[1]
        try:
            d = importlib.resources.files(package)
            if not d.is_dir():
                raise BadToolNameError(f"Cannot load {toolname} because {d} is not a valid directory.")
            f = d.joinpath(filename + ".yaml")
            with open(f) as s:
                y = yaml.safe_load(s)
                header = y["seclab-taskflow-agent"]
                version = header["version"]
                if version != 1:
                    raise VersionException(str(version))
                filetype = header["filetype"]
                if filetype != tooltype.value:
                    raise FileTypeException(f"Error in {f}: expected filetype to be {tooltype}, but it's {filetype}.")
                if tooltype not in self.__yamlcache:
                    self.__yamlcache[tooltype] = {}
                self.__yamlcache[tooltype][toolname] = y
                return y
        except ModuleNotFoundError as e:
            raise BadToolNameError(f"Cannot load {toolname}: {e}")
        except FileNotFoundError:
            # deal with editor temp files etc. that might have disappeared
            raise BadToolNameError(f"Cannot load {toolname} because {f} is not a valid file.")
        except ValueError as e:
            raise BadToolNameError(f"Cannot load {toolname}: {e}")
