from setuptools import setup

setup(
    name='keras_gan',
    version='0.1',
    description='Several GAN implementations',
    author='Erik Lindernoren',
    license='MIT',
    packages=['keras_gan'],
    zip_safe=False,
    install_requires=[
        "numpy",
        "keras"
    ],
)
