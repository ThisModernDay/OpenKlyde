# OpenKlyde - A Self-Hosted AI Discord Bot

OpenKlyde is a powerful, self-hosted AI Discord bot that integrates with large language models (LLMs) and image generation models to provide an interactive and engaging experience for users. With OpenKlyde, you can chat with an AI, generate images based on prompts, and even enjoy text-to-speech functionality.

## Table of Contents

- [OpenKlyde - A Self-Hosted AI Discord Bot](#openklyde---a-self-hosted-ai-discord-bot)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [To Do](#to-do)
  - [Prerequisites](#prerequisites)
    - [Koboldcpp](#koboldcpp)
    - [Stable Diffusion (Optional)](#stable-diffusion-optional)
    - [XTTS (Optional)](#xtts-optional)
    - [LLAVA Image Recognition (Optional)](#llava-image-recognition-optional)
  - [Setup](#setup)
    - [Installation](#installation)
    - [Configuration](#configuration)
      - [Discord API Key](#discord-api-key)
      - [LLM Server Configuration](#llm-server-configuration)
        - [Koboldcpp](#koboldcpp-1)
        - [OpenAI API](#openai-api)
        - [Mistral API](#mistral-api)
        - [Ollama](#ollama)
      - [Character Prompt](#character-prompt)
      - [Dialogue Examples (Optional)](#dialogue-examples-optional)
  - [Running the Bot](#running-the-bot)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- Integration with LLMs such as Koboldcpp and Oobabooga for chat functionality
- Image generation using Stable Diffusion via the Automatic1111 web UI
- Text-to-speech support using XTTS
- Image recognition and description using LLAVA
- Customizable character prompts and dialogue examples
- Support for OpenAI and Mistral API backends

## To Do

- [x] Make a better README.
- [x] Enable XTTS Support
- [x] Enable LLaVa Image recognition
- [ ] Make switching between Koboldcpp or Oobabooga Textgen-ui more mainstream.
- [ ] Enable support for Character.ai, TavernAI, SillyTavern, etc. character formats.
- [ ] Add more standard Discord Bot features. (music, games, moderation, etc.)

## Prerequisites

Before setting up OpenKlyde, ensure you have the following prerequisites installed and running:

### Koboldcpp

Koboldcpp is required for the bot to function. You can download Koboldcpp from the [official repository](https://github.com/LostRuins/koboldcpp).

### Stable Diffusion (Optional)

If you want to generate images, you'll need to have the Automatic1111 Stable Diffusion web UI running with the `--listen` and `--api` options enabled. Download Automatic1111 from the [official repository](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

For best results, it is recommended to use the SDXL model, as previous Stable Diffusion models (like SD1.5) may struggle with the natural language prompts sent by the bot.

### XTTS (Optional)

To enable text-to-speech functionality, follow these steps:

1. Install the required dependencies:
   - Run `pip install pillow` and `pip install TTS`
   - Ensure that FFmpeg is in your system's PATH environment variable

2. Create an `xtts` folder in the project directory and place a short (5-15 seconds) clean voice sample WAV file named `scarlett24000.wav` inside it.

3. Download the XTTS model files from [here](https://huggingface.co/coqui/XTTS-v2/tree/main) and place them in the `xtts` folder.

   Your `xtts` directory should have the following structure:
   ```
   xtts/
   ├── config.json
   ├── model/
   │   ├── dvae.pth
   │   ├── mel_stats.pth
   │   ├── model.pth
   │   └── vocab.json
   └── scarlett24000.wav
   ```

   Note: You may need to have the CUDA toolkit installed for XTTS to work properly.

### LLAVA Image Recognition (Optional)

To enable image recognition and description using LLAVA, follow these steps:

1. Download the llamacpp portable binaries and the required model files from the LLAVA repository or ShareGPT.

2. Run the following command in a terminal:
   ```
   ./server.exe -c 2048 -ngl 43 -nommq -m ./models/ggml-model-q4_k.gguf --host 0.0.0.0 --port 8007 --mmproj ./models/mmproj-model-f16.gguf
   ```

   Note: If you don't plan to use the image recognition feature, you don't need to have LLAVA running.

## Setup

### Installation

***It is recommended to use a virtual environment like Anaconda or Miniconda to manage the dependencies.***

1. Clone the OpenKlyde repository:
   ```
   git clone https://github.com/badgids/OpenKlyde.git
   ```

2. Navigate to the project directory:
   ```
   cd OpenKlyde
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

### Configuration

#### Discord API Key

Open the `bot.py` file and replace `API_KEY` with your Discord bot's API key:

```python
discord_api_key = "YOUR_DISCORD_API_KEY_HERE"
```

#### LLM Server Configuration

OpenKlyde supports various LLM servers, including Koboldcpp, OpenAI API, Mistral API, and Ollama. To configure the bot to use a specific LLM server, you need to provide a JSON configuration file in the `configurations` directory.

##### Koboldcpp

To use Koboldcpp as the LLM server, create a file named `text-default.json` in the `configurations` directory with the following content:

```json
{
  "name": "koboldcpp",
  "address": "http://localhost:5001/api/v1/",
  "model": "model",
  "generation": "generate",
  "headers": {
    "Accept": "application/json",
    "Content-Type": "application/json"
  },
  "instruction-user": "### Instruction",
  "instruction-response": "### Response",
  "parameters": {
    "prompt": "",
    "stop_sequence": [],
    "add_bos_token": "True",
    "ban_eos_token": "True",
    "do_sample": "False",
    "max_length": 400,
    "max_tokens": 400,
    "max_context_length": 2048,
    "genamt": 1095,
    "temperature": 0.95,
    "top_k": 0,
    "top_p": 0.75,
    "top_a": 0,
    "typical": 1,
    "tfs": 1.0,
    "rep_pen": 1.1,
    "rep_pen_range": 128,
    "rep_pen_slope": 0.9,
    "use_default_badwordsids": "True",
    "newline_as_stopseq": "True",
    "early_stopping": "True",
    "sampler_order": [6, 0, 1, 3, 4, 2, 5]
  }
}
```

Make sure to update the `address` field with the correct URL and port where your Koboldcpp server is running.

**Note:**
- We use the `generate` endpoint with `basic_api_flag = False` or in newer versions with `api_format=2`.
- The `koboldcpp.py` file handles the API format in the `generate_text()` function (the `else` block).
- Here are the parameters defined by the Koboldcpp API:

```python
return generate(
    prompt=genparams.get('prompt', ""),
    memory=genparams.get('memory', ""),
    max_context_length=genparams.get('max_context_length', maxctx),
    max_length=genparams.get('max_length', 100),
    temperature=genparams.get('temperature', 0.7),
    top_k=genparams.get('top_k', 100),
    top_a=genparams.get('top_a', 0.0),
    top_p=genparams.get('top_p', 0.92),
    min_p=genparams.get('min_p', 0.0),
    typical_p=genparams.get('typical', 1.0),
    tfs=genparams.get('tfs', 1.0),
    rep_pen=genparams.get('rep_pen', 1.1),
    rep_pen_range=genparams.get('rep_pen_range', 256),
    mirostat=genparams.get('mirostat', 0),
    mirostat_tau=genparams.get('mirostat_tau', 5.0),
    mirostat_eta=genparams.get('mirostat_eta', 0.1),
    sampler_order=genparams.get('sampler_order', [6,0,1,3,4,2,5]),
    seed=genparams.get('sampler_seed', -1),
    stop_sequence=genparams.get('stop_sequence', []),
    use_default_badwordsids=genparams.get('use_default_badwordsids', False),
    stream_sse=stream_flag,
    grammar=genparams.get('grammar', ''),
    grammar_retain_state = genparams.get('grammar_retain_state', False),
    genkey=genparams.get('genkey', ''),
    trimstop=genparams.get('trim_stop', False),
    quiet=is_quiet)
```

##### OpenAI API

To use the OpenAI API as the LLM server, create a file named `text-default.json` in the `configurations` directory with the following content:

```json
{
  "name": "openai",
  "address": "https://api.openai.com/v1/",
  "generation": "chat/completions",
  "headers": {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_OPENAI_API_KEY_HERE"
  },
  "parameters": {
    "model": "gpt-3.5-turbo-16k",
    "messages": {},
    "max_tokens": 240,
    "temperature": 0.95,
    "top_p": 0.80
  }
}
```

Replace `YOUR_OPENAI_API_KEY_HERE` with your actual OpenAI API key.

##### Mistral API

To use the Mistral API as the LLM server, create a file named `text-default.json` in the `configurations` directory with the following content:

```json
{
  "name": "openai",
  "address": "https://api.mistral.ai/v1/",
  "generation": "chat/completions",
  "headers": {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_MISTRAL_API_KEY_HERE"
  },
  "parameters": {
    "model": "mistral-medium",
    "messages": {},
    "max_tokens": 240,
    "temperature": 0.95,
    "top_p": 0.80
  }
}
```

Replace `YOUR_MISTRAL_API_KEY_HERE` with your actual Mistral API key.

##### Ollama

To use an Ollama server as the LLM server, create a file named `text-default.json` in the `configurations` directory with the following content:

```json
{
  "name": "ollama",
  "address": "http://localhost:11434/",
  "generation": "generate",
  "headers": {
    "Accept": "application/json",
    "Content-Type": "application/json"
  },
  "parameters": {
    "prompt": "",
    "max_new_tokens": 250,
    "do_sample": true,
    "temperature": 0.7,
    "top_p": 0.1,
    "typical_p": 1,
    "repetition_penalty": 1.18,
    "top_k": 40,
    "min_length": 0,
    "no_repeat_ngram_size": 0,
    "num_beams": 1,
    "penalty_alpha": 0,
    "length_penalty": 1,
    "early_stopping": false,
    "seed": -1,
    "add_bos_token": true,
    "truncation_length": 2048,
    "ban_eos_token": false,
    "skip_special_tokens": true,
    "stopping_strings": []
  }
}
```

Make sure to update the `address` field with the correct URL and port where your Ollama server is running.

You can customize the `parameters` section in each JSON file to adjust the generation settings according to your preferences and the capabilities of the chosen LLM server.

#### Character Prompt

Open the `characters/default.json` file and fill in the character prompt according to your preferences.

#### Dialogue Examples (Optional)

In the `functions.py` file, locate the `get_character()` function and fill out the `examples` array with example dialogues you want the bot to follow:

```python
examples = [
    "Example dialogue 1",
    "Example dialogue 2",
    ...
]
```

## Running the Bot

1. Start the LLM model of your choice in Koboldcpp.

2. Run the bot using the following command:
   ```
   python bot.py
   ```

   If you want to enable text-to-speech functionality, use the `--tts_xtts` flag:
   ```
   python bot.py --tts_xtts
   ```

## Contributing

Contributions to OpenKlyde are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/badgids/OpenKlyde).

## License

OpenKlyde is released under the [MIT License](https://github.com/badgids/OpenKlyde/blob/main/LICENSE).
