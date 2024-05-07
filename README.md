# chroma-chord-generator

This repo contains:

- A pre-trained chroma histogram generation model
- Its surrounding infrastructure for real-time chord generation
- Utilities for extracting chroma histogram datasets from datasets of raw MIDI files
- Utilities for training models on chroma histogram datasets

The cleaned subset of the Lakh MIDI Dataset was used to generate chroma histograms used to train the included chord generation model: [https://www.kaggle.com/datasets/imsparsh/lakh-midi-clean]

## Running the real-time chord generation system

The real-time system included in this repo consists of two components.

### Python script

Install the required packages in your Python environment:

``
pip install -r requirements.txt
``

Run the system with the default IP/ports:

``
python ./src/main.py
``

By default, the real-time system expects to receive `list[pitch (0-127), intensity (0-1)]` on port `10000`, and sends out each MIDI note as a `list[pitch (0-127), velocity (0-127)]` on `127.0.0.1:11000`.

Add command line arguments to change the IP/ports:

``
python ./src/main.py <ip> <receive_port> <send_port>
``

in which the IP address is of format `x.x.x.x` and the ports are integers.

### Max patch

Open `./src/chroma-chord-generator_starter_patch.maxpat` (requires [Max 8](https://cycling74.com/products/max)).