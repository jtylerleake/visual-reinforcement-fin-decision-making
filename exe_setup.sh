
# --- 1. update and upgrade the system --- # 
# echo "Starting system update and essential package installation..."
# sudo apt update && sudo apt upgrade -y

# --- 2. install the correct version of python --- #  
echo "Installing python 3..."
sudo apt install -y git python3-pip python3-venv build-essential
python3 --version
pip3 --version

# --- 3. clone the github repository --- #
echo "Cloning the project repository..."
cd ~
git clone https://github.com/jtylerleake/visual-reinforcement-fin-decision-making.git
cd visual-reinforcement-fin-decision-making

# --- 4. make the python virtual environment --- # 
echo "Initializing the virtual environment and upgrading pip inside venv..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# --- 5. install the correct version of pytorch --- #
echo "Installing pytorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"

# --- 6. install project dependencies from requirements --- #
echo "Installing the project dependencies..."
cd visual-reinforcement-fin-decision-making
pip install -r requirements.txt
