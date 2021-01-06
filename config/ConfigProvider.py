from Utils.yaml_utils.ConfigParser import ConfigParser
from Testing.Research.config.paths import config_path


class ConfigProvider(object):
    _config = None

    @staticmethod
    def get_config():
        if ConfigProvider._config is None:
            ConfigProvider._config = ConfigParser(config_path).parse()
        return ConfigProvider._config
