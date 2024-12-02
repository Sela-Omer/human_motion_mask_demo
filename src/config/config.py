import configparser
import os.path
import sys

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the config.ini file
config.read('src/config/config.ini')

# Scan command line arguments for the APP_overwrite_config_path argument
for arg in sys.argv:
    # Check if the argument contains the APP_overwrite_config_path string
    if '--APP_overwrite_config_path=' in arg:
        # Set the OVERWRITE_CONFIG_PATH in the config object
        config['APP']['OVERWRITE_CONFIG_PATH'] = arg.replace('--APP_overwrite_config_path=', '')

if 'OVERWRITE_CONFIG_PATH' in config['APP']:
    if os.path.isfile(config['APP']['OVERWRITE_CONFIG_PATH']):
        print(f'overwriting config with {config["APP"]["OVERWRITE_CONFIG_PATH"]}')
        config.read(config['APP']['OVERWRITE_CONFIG_PATH'])
    else:
        print(
            f"cannot overwrite config file as it does not exist: {config['APP']['OVERWRITE_CONFIG_PATH']}... from path {os.getcwd()}")
        print('no OVERWRITE_CONFIG_PATH found. continuing with main config.')
else:
    print('no OVERWRITE_CONFIG_PATH found. continuing with main config.')