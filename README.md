# CSE-251B-Melody-Voice-VAE
To learn the representation of polyphonic music

The model folder contains the core files for the model. However, this model currently has a series of minor issues, and it does not support dynamic padding. We will need to fix this later.

The recon_demo and result folders store some results and visualization components. Most of them only need to be read and executed by @Ke.

The loader folder contains code for processing data and hierarchical melody. This also lacks dynamic padding, and we will need to fix it later.

merge and midi_decode are responsible for merging monophonic outputs into polyphonic format and exporting them as playable MIDI files.

The main areas we plan to modify next:

Model files in the model folder: add dynamic padding and fix some small bugs.

Modify training and testing code: add dynamic padding and optimize some hyperparameters.

We hope to handle 2-bar music instead of just 1 bar, as the interpolation of 1-bar melodies doesn't seem very impressive right now.

Comparison with PianoTree: this is mostly handled by @Ke, and both the code and models are already available.

Output of visualizations like loss curves, etc.

About the four types of data:
The approximate ratio of training/validation/test data is 29000:3500:3500. (We decided not to use overlapping data in the end.)

fix: Fixed-length padded to 10 bars, size = [N, 320]

dynamic: Variable length to reduce redundancy, size = [N, variable length]

dynamic_sub: Variable length + subgraph segmentation; segmentation is based on the highest and lowest note in each bar

dynamic_sub_pr: Variable length + subgraph segmentation; segmentation is based on fixed note ranges [24, 48, 60, 108]


