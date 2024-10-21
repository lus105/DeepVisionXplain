## Model Training Environment

#### Environment Description
The setup is designed to streamline experimentation, foster modularity, and simplify tracking and reproducibility:

✅ Minimal boilerplate code (easily add new models, datasets, tasks, experiments, and different accelerator configurations).

✅ Logging experiments to one place for easier comparison of performance metrics.

✅ Hyperparameter search integration.

#### Working principle:

<p align="center">
  <img src="res/principle_diagram.svg" width="700"/>
</p>

*Configuration*

This part of the diagram illustrates how configuration files (train.yaml, eval.yaml, model.yaml, etc.) are used to manage different aspects of the project, such as data preprocessing, model parameters, and training settings.

*Hydra Loader*

The diagram shows how Hydra loads all configuration files and combines them into a single configuration object (DictConfig). This unified configuration object simplifies the management of settings across different modules and aspects of the project, such as data handling, model specifics, callbacks, logging, and the training process.

*Train/test Script*

 This section represents the operational part of the project. Scripts train.py and eval.py are required for training and evaluatging the model. DictConfig: The combined configuration object passed to these scripts, guiding the instantiation of the subsequent components.

  * LightningDataModule: manages data loading and processing specific to training, validation, testing and predicting phases.

  * LightningModule (model): defines the model, including the computation that transforms inputs into outputs, loss computation, and metrics.

  *	Callbacks: provide a way to insert custom logic into the training loop, such as model checkpointing, early stopping, etc.

  * Logger: handles the logging of training, testing, and validation metrics for monitoring progress.

  *	Trainer: the central object in PyTorch Lightning that orchestrates the training process, leveraging all the other components.

  *	The trainer uses the model, data module, logger, and callbacks to execute the training/evaluating process through the trainer.fit/test/predict methods, integrating all the configuration settings specified through Hydra.

#### Workflow steps:
<p align="center">
  <img src="res/workflow_diagram.svg" width="350"/>
</p>