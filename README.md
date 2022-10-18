# Land cover predictive modeling from satellite images

## Setup 

Create a new conda environment and activate it:
```bash
conda create --name lcc python==3.10.4
conda activate lcc
```
Install torch 1.12.1 with or without cuda depending on your hardware (for example here I have GPU with cuda 11.6 installed, refer to [this page](https://pytorch.org/) for installation)
``` 
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
Install `requirements.txt`
```
pip install -r requirements.txt
```
Install the `lcc` package
```
pip install -e .
```


