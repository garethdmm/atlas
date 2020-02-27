<p align=center><img src='https://github.com/garethdmm/gryphon-atlas/raw/master/img/gryphon-logo-blue.png' height=150/></p>

<h1 align=center>Gryphon Atlas</h1>

<p align=center>Atlas is an end-to-end machine learning workflow designed for use with the <a href='https://github.com/garethdmm/gryphon'>Gryphon Trading Framework</a>.
<img src='https://github.com/garethdmm/gryphon-atlas/raw/master/img/summary.png' height=500/>
</p>
<p align=center><i>Some pieces of the Atlas model pipeline. Build featuresets of millions of datapoints (top), train hundreds of models at once and (bottom left) digest the results quickly so you can (bottom right) zero in on promising individual models.</i></p>


<br/>

## Overview

Atlas is a model development workflow built on Tensorflow and TFLearn. It's purpose is to allow individual model developers to iterate quickly. Atlas consists of roughly four modules:

- `infra` - Classes for defining training runs of thousands of parallel models, initializing the run on remote machines, and gathering results.
- `data` - Classes and tools for working with large timeseries datasets. Includes a feature library for defining featuresets easily, data cleaning and conversion functions, and tools for handling unbalanced datasets for classification.
- `models` - The atlas model zoo, including implementations of the most common machine learning model types.
- `stats` - A library that includes implementations of missing evaluation stats in the scikit/tensorflow libraries, as well as visual summaries of trained model performance and comparisons of arbitrary numbers of models at once.

## Workflow

Before using Atlas, you should be running the Gryphon Data Service and building a database of market data on which to train models.

The workflow itself is roughly as follows:

1. Create a featureset you wish to train models against.
    - The [feature library](ml/data/feature.py) has dozens of built-in features which can be generated from the GDS database. Each of these features can be built/referenced in code using a human-readable syntax. For example, this is how you would create a feature for the one-minute future log-returns on the bitstamp btc_usd pair.
      ```python
        LogReturns().bitstamp_btc_usd().one_min().lookforward(1)
      ```
    - Features are grouped together with a prediction target into a [FeatureLabelSet](ml/data/feature_label_set.py). The following example uses the top bid/ask volume on bitstamp, the past one-minute log return, and the midpoint spread between bitstamp and itbit, to predict the next one-minute log return of bitstamp.
      ```python
        example_set = ml.data.feature_label_set.FeatureLabelSet(
            features=[
                LogReturns().bitstamp().one_min().lookback(1),
                BidStrength().bitstamp().one_min().slippage(0),
                AskStrength().bitstamp().one_min().slippage(0),
                InterExchangeSpread(
                    Midpoint().bitstamp().one_min(),
                    Midpoint().itbit().one_min(),
                ),       
            ],       
            labels=[
                LogReturns().bitstamp().one_min().lookforward(1),
            ],       
        )
      ```

    - Before starting a run, you can build this featureset in the atlas console and examine to data visually
      ```python
        simple_prices.plot_data(datetime(2019, 1, 1), datetime(2019, 3, 1), subplots=True)
      ```
      The output should look something like this:
      <p align='center'><img src='https://github.com/garethdmm/gryphon-atlas/raw/master/img/featureset_visual.png'/></p>

2. Create a [WorkUnit](ml/infra/work_unit.py). WorkUnits group a single featureset with a set of models and hyperparameters we think might perform well when trained on this featureset. For example, we might use the above `example_set` and want to try training single-layer DNNs with each of 10, 100, and 1000 neurons. A related class to WorkUnit is a [ModelSpec](ml/infra/model_spec.py), which just describes how to instantiate a model with a particular set of hyperparameters.
3. Combine many WorkUnits into a single [WorkSpec](ml/infra/work_spec.py). WorkSpecs are a grouping of WorkUnits that we want to run all at the same time. This class also tells atlas how to split the work between many GPUs. You can see a full example of a WorkSpec [here](ml/infra/specs/june_1_2017.py)
4. Use general_trainer to run the work spec on remote machines.
    - Presently, each pipeline needs to be started independently. For simplicity, a tool like screen can be used to achieve this parallelism. This is done with this command:
      ```shell
        ./general_trainer ...
      ```
5. Use [Tensorboard](https://www.tensorflow.org/tensorboard) to monitor training progress during the run.
6. On completion, use [harvest.py](http://harvest.py) to download the training run results
    - This can be done with the command
      ```shell
        python harvest.py ...
      ```
    - The result of this is to retreive a blob of python objects onto your machine.
7. Use batch overview visualizations to find the best models from your training run. You can load the results object from the analysis console as follows:
    ```python
        # Code to get the analysis object.
    ```
    - With a results object in hand, you can see a total overview of the training run with this command
    ```python
        # Code to show the stuff.
    ```
      <p align='center'><img src='https://github.com/garethdmm/gryphon-atlas/raw/master/img/batch_run.png'/></p>

- This example is from a training run where the goal was to create a trinary classifier for price movement into "strong up, about the same, strong down" categories. The viridis colour scheme tells us that purple is low, green/yellow is very high, and teals/blues are mostly about the average. We can see immediately that the fourth and fifth models have some very extreme results, so they are most likely degenerate cases not worth our time. The sixth model is questionable. Of the first three however, the first and third at least seem to have a little signal, in particular with predicting the "about the same" case. Those might be worth a closer look.

8. Use individual model baseball cards and other visualizations to evaluate what worked and what didn't. 
    - Good results on the summary page do not necessarily mean the model is good. It may, for example, have simply performed extremely well in one period of the training data and fallen off completely after.
    - For this we can dig into individual model results using Baseball Cards. Baseball cards are a quick readout of several visualizations that help us interpret the results of a model. Here's an example.

      <p align='center'><img src='https://github.com/garethdmm/gryphon-atlas/raw/master/img/baseball_card.png' height=400/></p>

- In this baseball card, from left to right and top to bottom, the first pane is accuracy over time. We can see it's accuracy does appear to come in waves.
9. Run production models with *model_runner*, and write Gryphon strategies that can read their outputs through redis.
  ```shell
    ./gryphon-atlas run-model [model_name]
  ```
  
## Limitations

- Atlas was built on tensorflow RC 0.12. For future usage it will need to be adapted to current tensorflow releases.
- Atlas was built on TFLearn, which has had substantial API changes. For future usage it would be good to move the model library to keras or some other library that is in consistent use.
- This release of Atlas does not include RNN models, which are a standard in time-series prediction.

## Enterprise Support

Enterprise support, custom deployments, strategy development, and other services are available through [Gryphon Labs](http://www.http://www.gryphonlabs.co). If you're a firm interested in using Gryphon, you can schedule a chat with us or contact one of the maintainers directly.
