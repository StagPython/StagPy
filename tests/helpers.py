import pathlib
import stagpy

repo_dir = pathlib.Path(__file__).parent.parent.resolve()
ra1e5_dir = repo_dir / 'Examples' / 'ra-100000'

def reset_config():
    """Reset stagpy.conf to default values."""
    stagpy.init_config(None)
