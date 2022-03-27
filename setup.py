# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['brunnermunzel_from_count_data']

package_data = \
{'': ['*']}

install_requires = \
['numpy>=1.21.1,<2.0.0', 'scipy>=1.2.0,<2.0.0']

setup_kwargs = {
    'name': 'brunnermunzel-from-count-data',
    'version': '1.0.0',
    'description': '',
    'long_description': None,
    'author': 'dmingn',
    'author_email': 'dmingn@users.noreply.github.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7,<4.0',
}


setup(**setup_kwargs)
