# knimeVizLab

![alt text](LogoProject.png)

## Description
This repository serves as the official [UNIBZ](www.unibz.it) project for creating custom KNIME nodes dedicated to image processing using Python. The goal is to provide seamless integration between KNIME's workflow analytics platform and Python's advanced capabilities for image analysis and processing.

With this project, users can:

- Develop custom KNIME nodes tailored to specific image processing requirements.
- Leverage Python libraries and tools for advanced image manipulation and analysis.
- Simplify workflows by combining KNIME's no-code/low-code environment with Python's scripting flexibility.


## Installation

To start developing KNIME nodes with Python, follow these steps:
1. **Clone this repository**:
    ```bash
    git clone ...
    ```

2. **Install the KNIME Analytics Platform**
    Install [KNIME Analytics Platform](https://docs.knime.com/2024-12/analytics_platform_installation_guide/index.html#_installing_knime_analytics_platform) version 4.6.0 or higher.
    Make sure to also install the KNIME Python Extension Development (Labs). In case of problem follow the offical guide line on how to develope pure [python nod in KNIME](https://docs.knime.com/latest/pure_python_node_extensions_guide/index.html#extension-bundling)

3. **Set up a Conda/Python environment**
    Create a conda/Python environment containing the [knime-python-base metapackage](https://anaconda.org/knime/knime-python-base), together with the node development API [knime-extension](https://anaconda.org/knime/knime-extension) for the KNIME Analytics Platform you are using

    ```bash
        cd KnimeVisLab
        conda env create -f env.yml
    ```

4. **Modify the config.yml file**
    Update the following fields in the `config.yml` file:
        - Set the absolute path to point to the src directory of this repository.
        - Set the path to the Conda environment you just created.

5. **Update the knime.ini file**
    Locate the `knime.ini` file in the KNIME installation directory and sdd the following line to the end of the file:

    ```-Dknime.python.extension.config=path/to/config/config.yml```


## Usage
Once you have completed the installation, you can begin developing and using custom KNIME nodes for image processing. To integrate your Python-based node into KNIME, follow these steps:

1. Create your custom node in Python.
2. Integrate the node into your KNIME workflow by using the custom node created in the previous step.
3. Execute the workflow, and the image processing will be handled by Python scripts through KNIME's interface.

## Support
For any request about this software, please refer to the authors.

## Roadmap
This is a preliminary version. Further improvements and nodes will come soon. Stay tuned.

## Contributing
We welcome contributions to this project! To contribute:

- Fork this repository.
- Create a new branch.
- Make your changes and commit them.
- Submit a pull request.

Please make sure your code follows the existing style and includes appropriate tests.

## Authors and acknowledgment
Thanks to the collaboration with KNIME.

## License
The nodes are released under GPLv3.

## Project status
We are currently developing new nodes and improving the existing ones. Please feel free to contribute.
