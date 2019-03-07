"""Prysm, a python optics module."""

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup

setup(use_scm_version=True)
