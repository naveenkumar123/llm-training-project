# LLM Project
Different LLM model usage projects, building multiple llm applicaions using opensource model and frontier models 

# Prerequisites
### Install Anaconda:
Install the Anaconda form the from https://docs.anaconda.com/anaconda/install/mac-os
- **Note**: It takes sometime to install the conda in your machine.

### Environment setup
  - Move to project root directory
  - Run the commanda **conda env create -f environment.yml**
    - **Note**: You can change the environment name and python version, add if any dependecies required.
  - Activate the Environment
     - Run the command to activate the environment **conda activate llms** .Here ***llms*** is the environment name mentioned in the environment.yml
  - Update the .env file with the Open API Key, this key is required to connect to the OpenAPI api's to run the model. You can create the API key [here](https://platform.openai.com/settings/organization/api-keys) and also check the model pricing details [here](https://platform.openai.com/docs/pricing)

 **Note:**  Well you do not need the above step, if you running the model in local using ollama. All the examples are written using multi model, you can choose the Ollama while running. 

### Launch the applicaion
- Move to project root directory and execute the command mentioned in the examples below based the on the application you want to lunch.
   - **Example 1**: python src/llmchat/Chat_ui.py  (App will be running on Gradio local server)
   - **Example 2:** python src/image_to_text/DigitalImage.py [App will be lunched on localhost server]
   - **Example 3:** python src/hugging_face_models/common_tasks/Common.py [App will be lunched on localhost server]
   - **Example 4:** python src/hugging_face_models/tokenizer/TextAutotokenizer.py 
