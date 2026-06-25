# WandB / Slack configuration template.
# Copy this file to `secrete_project_config.py` and fill in your own values.
# `secrete_project_config.py` is gitignored and must NOT be committed.

# Get your API key from: https://wandb.ai/authorize
api_token = "YOUR_WANDB_API_KEY"   # Replace with your WandB API key
project = "DeepfakeDetection"        # WandB project name
entity = "YOUR_WANDB_ENTITY"         # Replace with your WandB username or team name
slack_api_url = "https://hooks.slack.com/services/[YOUR_SLACK_WEBHOOK_URL]"
