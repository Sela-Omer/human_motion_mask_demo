# Human Motion Mask Demo

## Overview

This project attempts to reconstruct masked human motion data.
The masking algorithm used is similar to the one used in MaskedMimic.

## Preliminary Results

The following are the preliminary results of the project.
Plots of real(blue) vs. predicted(orange) joint positions over time are shown below.
Red markers indicate the non-masked samples. More images are available
in `model/UNET_1D-config-unet_1d/version_0/checkpoints/`.

![Alt](model/UNET_1D-config-unet_1d/version_0/checkpoints/UNET_1D-config-unet_1d-epoch=18-train_loss=0.00661-sample-1-0.png)
![Alt](model/UNET_1D-config-unet_1d/version_0/checkpoints/UNET_1D-config-unet_1d-epoch=18-train_loss=0.00661-sample-2-0.png)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the Repository**: First, clone this repository to your local machine using Git:

   ```bash
   git clone git@github.com:Sela-Omer/human_motion_mask_demo.git
   cd human_motion_mask_demo
   ```

2. **Install Dependencies**: Install all the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the project, you can execute the `__init__.py` file directly from the command line:

```bash
python __init__.py
```

### Training

To train the model execute it with the APP_mode config option set to FIT.

### Evaluation

To evaluate the model execute it with the APP_mode config option set to EVAL.

### Configuration

The `config.ini` parameters can be dynamically overridden at runtime by passing them as command line arguments to
the `__init__.py` file. This allows for flexible adjustments of parameters without permanently changing the
configuration file.\
Example Command to Override Settings:

```bash
python __init__.py --APP_mode FIT --APP_arch UNET_1D
```