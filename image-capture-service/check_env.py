#!/usr/bin/env python
import importlib
import importlib.util

pkgs = ['yaml', 'cv2', 'numpy', 'pika', 'coloredlogs', 'prometheus_client', 'psutil', 'PIL']

for p in pkgs:
    spec = importlib.util.find_spec(p)
    if spec:
        try:
            mod = importlib.import_module(p)
            version = getattr(mod, '__version__', 'OK')
            print(f'{p}: {version}')
        except Exception as e:
            print(f'{p}: IMPORT ERROR - {e}')
    else:
        print(f'{p}: NOT INSTALLED')
