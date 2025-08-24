from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fouriever")
except PackageNotFoundError:
    pass  # package is not installed
