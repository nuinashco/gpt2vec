# git
# git clone https://nuinashco:$GITHUB_TOKEN@github.com/nuinashco/GPT2Vec.git; cd GPT2Vec; source setup_vast.sh

git config --global user.name "Ivan Havlytskyi"
git config --global user.email "ivan.havlytskyi@gmail.com"

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate

# login
huggingface-cli login --token $HUGGINGFACE_TOKEN
wandb login