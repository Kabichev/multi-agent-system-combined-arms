# CombinedArms

##Project Description
This is a project for the AASMA course at Instituto Superior Tecnico. This project used an enviorment called CombinedArms from PettingZoo.ml.

##File structure
The project contains two pyhton files.

-`agents.py`: Contains the code for the different types of agents

-`combined_arms.py`: that contains the enviorment and data collection from different episodes ran.

##Technologies
Project is created with:
* Python 3.10.5 

##Installation
Use the package manager [pip] to install

-`numpy`

-`ma-gym`

-`pygame`

-`matplotlib`

-`pandas`

##Running the project
Write the following command in the Terminal:

```bash
python combined_arms.py
```
Default is that a clingy policy (RED TEAM) fights greedy agents (BLUE TEAM).

The policy can be changed in line 54 for blue team, and 56 for red team.

The different values they can take is:

-`agents.GreedyAgent(args, agent_name)`: A greedy policy

-`agents.RandomAgent(args, agent_name)`: A random policy

-`agents.ClingyGreedyAgent(args, agent_name)`: A Clingy policy

-`agents.GreedyAgent(args, agent_name, True)`: A Safe and greedy agents

-`agents.ClingyGreedyAgent(args, agent_name, True)`: A Safe and Clingy Policy

## License
[MIT](https://choosealicense.com/licenses/mit/)
