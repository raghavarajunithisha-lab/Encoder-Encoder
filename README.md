Running the Project

Install the required dependencies:

pip install -r requirements.txt


Open main.py and select the desired model configuration by uncommenting the corresponding line:

from configs import config_use as cfg
// from configs import config_bert as cfg
// from configs import config_dpr as cfg


After choosing the model, update the corresponding configuration file in the configs/ directory as needed.

Finally, run the main script from the terminal:

python main.py
