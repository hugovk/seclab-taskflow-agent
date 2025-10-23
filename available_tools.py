import logging

class VersionException(Exception):
    pass

class FileIDException(Exception):
    pass

class FileTypeException(Exception):
    pass

def add_yaml_to_dict(table, key, yaml):
    """Add the yaml to the table, but raise an error if the id isn't unique """
    if key in table:
        raise FileIDException(str(key))
    table.update({key: yaml})

class AvailableTools:
    """
    This class is used for storing dictionaries of all the available
    personalities, taskflows, and prompts.
    """
    def __init__(self, yamls: dict):
        self.personalities = {}
        self.taskflows = {}
        self.prompts = {}
        self.toolboxes = {}
        self.model_config = {}
        self.namespace_config = {}

        # Iterate through all the yaml files and divide them into categories.
        # Each file should contain a header like this:
        #
        #   seclab-taskflow-agent:
        #     type: taskflow
        #     version: 1
        #
        for path, yaml in yamls.items():
            try:
                header = yaml['seclab-taskflow-agent']
                version = header['version']
                if version != 1:
                    raise VersionException(str(version))
                filekey = header['filekey']
                filetype = header['filetype']
                if filetype == 'personality':
                    add_yaml_to_dict(self.personalities, filekey, yaml)
                elif filetype == 'taskflow':
                    add_yaml_to_dict(self.taskflows, filekey, yaml)
                elif filetype == 'prompt':
                    add_yaml_to_dict(self.prompts, filekey, yaml)
                elif filetype == 'toolbox':
                    add_yaml_to_dict(self.toolboxes, filekey, yaml)
                elif filetype == 'model_config':
                    add_yaml_to_dict(self.model_config, filekey, yaml)
                elif filetype == 'namespace_config':
                    add_yaml_to_dict(self.namespace_config, filekey, yaml)
                else:
                    raise FileTypeException(str(filetype))
            except KeyError as err:
                logging.error(f'{path} does not contain the key {err.args[0]}')
            except VersionException as err:
                logging.error(f'{path}: seclab-taskflow-agent version {err.args[0]} is not supported')
            except FileIDException as err:
                logging.error(f'{path}: file ID {err.args[0]} is not unique')
            except FileTypeException as err:
                logging.error(f'{path}: seclab-taskflow-agent file type {err.args[0]} is not supported')

    def copy_with_alias(self, alias_dict : dict) -> dict:
        def _copy_add_alias_to_dict(original_dict : dict, alias_dict : dict) -> dict:
            new_dict = dict(original_dict)
            alias_keys = alias_dict.keys()
            for k,v in original_dict.items():
                for ak in alias_keys:
                    if k.startswith(ak) and k[len(ak)] == '/':
                        new_key = alias_dict[ak] + k[len(ak):]
                        new_dict[new_key] = v
            return new_dict
        new_available_tools = AvailableTools({})
        new_available_tools.personalities = _copy_add_alias_to_dict(self.personalities, alias_dict)
        new_available_tools.taskflows = _copy_add_alias_to_dict(self.taskflows, alias_dict)
        new_available_tools.prompts = _copy_add_alias_to_dict(self.prompts, alias_dict)
        #toolboxes are looked up after canonicalized
        new_available_tools.toolboxes = dict(self.toolboxes)
        new_available_tools.model_config = _copy_add_alias_to_dict(self.model_config, alias_dict)
        new_available_tools.namespace_config = _copy_add_alias_to_dict(self.namespace_config, alias_dict)
        return new_available_tools

def canonicalize_toolboxes(toolboxes : list, alias_dict : dict) -> list:
    """
    Toolboxes needs to be canonicalize because both personalities and taskflows can use toolboxes with potentially different aliases
    """
    out = set()
    if not alias_dict:
        return toolboxes
    for tb in toolboxes:
        found_alias = False
        for k,v in alias_dict.items():
            if tb.startswith(v) and tb[len(v)] == '/':
                out.add(k + tb[len(v):])
                found_alias = True
        if not found_alias:
            out.add(tb)
    return list(out)