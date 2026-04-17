from setuptools import setup, find_packages

setup(
    name         = 'project',
    version      = '1.0',
    packages     = find_packages(),
    scripts      = ['bin/pp5_venues_merge.py'],
    entry_points = {'scrapy': ['settings = showrunner_backend.settings']},
)
