# Robotic Manipulation Inside a Simulation Environment Through Reinforcement Learning

This repository contains all the necessary code for training a robot to perform manipulation inside a physics simulation environment, V-REP, with the use of Reinforcement Learning. More specifically, the Reinforcement Learning algorithm that was chosen for training the robot is the [Soft Actor Critic](https://arxiv.org/pdf/1801.01290.pdf) algorithm. This choice was motivated by the fact that the Soft Actor Critic algorithm entails the merits of off-policy training, storing transition samples experienced through past policies in an off-policy buffer and using these to update the later policies, offering stability. Additionally, the algorithm draws also from the merits of on-policy algorithms since it directly optimizes for the policy though performing backpropagation over it.



## Getting Started

### Prerequisites

1. Linux Operating System, preferably Ubuntu 18.04.

2. Please [download the V-REP](https://coppeliarobotics.com/downloads) physics simulation engine. The repository was

   created using the Education Version. **Add the version that it was developed under**

3. sudo apt-get install libxkbcommon-x11-dev

   export PATH=$PATH:~/Qt/Tools/QtCreator/bin

### Installation

1.  If you don't already have it, [install Python](https://www.python.org/downloads/).

    This repository was developed is compatible with Python 2.7, 3.3, 3.4, 3.5 and 3.6.

2.  General recommendation for Python development is to use a Virtual Environment.
    For more information, see https://docs.python.org/3/tutorial/venv.html

    Install and initialize the virtual environment with the "venv" module on Python 3 (you must install [virtualenv](https://pypi.python.org/pypi/virtualenv) for Python 2.7):

    ```
    python -m venv mytestenv # Might be "python3" or "py -3.6" depending on your Python installation
    cd mytestenv
    source bin/activate      
    ```

### Quickstart

1.  Clone the repository.

    ```
    git clone https://github.com/gthd/Reinforcement_Learning.git
    ```

2.  Install the dependencies using pip.

    ```
    cd Reinforcement_Learning
    pip install -r requirements.txt
    ```

## Demo

A demo app is included to show how to use the project.

To run the soft actor critic algorithm:

1. `python Reinforcement_Learning/sac_vrep.py`

To show the policy that has been learned run:
2. `python Reinforcement_Learning/show_policy.py`

## Contributing

Please read [CONTRIBUTING.md](Contributing.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* [**Georgios Theodorou**](https://github.com/gthd)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* I want to acknowledge the help and guidance I received from my supervisor [Edward Johns](https://www.imperial.ac.uk/people/e.johns).
