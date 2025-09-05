import os


def check_env_variables(required_env_vars = []):
    for var in required_env_vars:
        if not os.environ.get(var):
            return False
    return True
