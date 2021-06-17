
You can also use [W&B sweeps](https://docs.wandb.ai/) to optimize hyper parameters:

* define your sweep configuration file

* create a sweep -> this will return a sweep id

  `wandb sweep sweep.yaml`

* launch an agent against the sweep

  `wandb agent my_sweep_id`

# Usage:
In terminal do:
`bash BuildDocker.sh`

Then:
`0_RunExperiment0_GPU0.sh`

