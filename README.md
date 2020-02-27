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

    - Before starting a run, you can build and inspect parts of this featureset in the [Atlas Console](ml/console.py). Start the console from the root directory with `make console`, and you can plot a pre-defined featureset like this:
      ```python
        simple_prices.plot_data(datetime(2019, 1, 1), datetime(2019, 3, 1), subplots=True)
      ```
      The output should look something like this:
      <p align='center'><img src='https://github.com/garethdmm/gryphon-atlas/raw/master/img/featureset_visual.png'/></p>

2. Create a [WorkUnit](ml/infra/work_unit.py). WorkUnits group a single featureset with a set of models and hyperparameters we think might perform well when trained on this featureset. For example, we might use the above `example_set` and want to try training single-layer DNNs with each of 10, 100, and 1000 neurons. A related class to WorkUnit is a [ModelSpec](ml/infra/model_spec.py), which just describes how to instantiate a model with a particular set of hyperparameters.
3. Combine many WorkUnits into a single [WorkSpec](ml/infra/work_spec.py). WorkSpecs are a grouping of WorkUnits that we want to run all at the same time. This class also tells atlas how to split the work between many GPUs. You can see a full example of a WorkSpec [here](ml/infra/specs/june_1_2017.py)
4. Use [General Trainer](ml/infra/general_trainer.py) to run the work spec on remote machines. This is done with this command:
      ```shell
        python general_trainer.py [work_spec_name] [pipline number] [--execute] 
      ```
    Presently, each pipeline needs to be started independently. For simplicity, a tool like screen can be used to achieve this parallelism.
5. Use [Tensorboard](https://www.tensorflow.org/tensorboard) to monitor training progress during the run.
6. On completion, use the [Harvester](ml/infra/harvest.py) to download the training run results. Run results are kept in a format called a `ResultsObject` which is pickled and written to disk on the training machine at the end of a run. The Harvester simply moves this file to your local machine, and can be run as follows:
      ```shell
        python harvest.py [work_spec_name] [host] [--execute]
      ```
7. Use batch overview visualizations to find the best models from your training run. To do this, we import the workspec in the atlas console, and pass the results object into the functions in the [visualizations](ml/visualize.py) library (which is already pre-imported in the console). Here's an example.
    ```python
        from ml.infra.work_specs import june_1_2017 as spec
        results = spec.get_all_results_objs()
        visualize.plot_multi_basic_results_table(results)
    ```
    This will give you a visual output something like this:
      <p align='center'><img src='https://github.com/garethdmm/gryphon-atlas/raw/master/img/batch_run.png'/></p>

    This example is from a training run where the goal was to create a trinary classifier for price movement into "strong up, about the same, strong down" categories. The viridis colour scheme tells us that purple is low, green/yellow is very high, and teals/blues are mostly about the average. We can see immediately that the fourth and fifth models have some very extreme results, so they are most likely degenerate cases not worth our time. The sixth model is questionable. Of the first three however, the first and third at least seem to have a little signal, in particular with predicting the "about the same" case. Those might be worth a closer look.

8. Use individual model Baseball Cards and other visualizations to dig into particular models' results. Baseball cards are a quick readout of several visualizations for a single model that help us interpret its results. Here's an example.
    ```python
    import ml.visualize
    visualize.classifier_baseball_card(results_obj[0])
    ```
      <p align='center'><img src='https://github.com/garethdmm/gryphon-atlas/raw/master/img/baseball_card.png' height=400/></p>

    In this baseball card, from left to right and top to bottom, the first pane is accuracy over time. We can see it's accuracy does appear to come in waves. The next is Likelihood ratios for all three classes at different confidence values (z-values in some traditions). The third is a histogram of accuracy values in different time periods. The fourth is the ROC curves for all three classes. 
    
    Examining individual results in detail is important because summary statistic may hide an underlying degenerancy. For example, a model may show 60% accuracy, but that may be split between 80% accuracy in the first half of the time period and 40% in the second half. This might lead to zero or worse overall revenue if traded against naively.
    
    Of course, baseball cards themselves are still summaries. To gain confidence in a model we recommend using many of the visualizations in the [Visualize](ml/visualize.py) library, as well as examining the models' predictions series' itself directly.
    
9. Once you've digested results of this run, return to step 1. and iterate on your model hyperparameters and featureset until you've found a model that performs well enough to move into production.
10. Run production models with *model_runner*. The runner generates the values of the relevant features for the current moment, feeds it into your production model, and places the output in redis. To trade against these predictions using a Gryphon strategy, all you have to do is read the predictions out of redis and write your trading behaviour accordingly.
  ```shell
    ./gryphon-atlas run-model [model_name]
  ```
  
## Limitations

- Atlas was built on tensorflow RC 0.12. For future usage it will need to be adapted to current tensorflow releases.
- Atlas was built on TFLearn, which has had substantial API changes. For future usage it would be good to move the model library to keras or some other library that is in consistent use.
- This release of Atlas does not include RNN models, which are a standard in time-series prediction.

## Enterprise Support

Enterprise support, custom deployments, strategy development, and other services are available through [Gryphon Labs](http://www.http://www.gryphonlabs.co). If you're a firm interested in using Gryphon, you can schedule a chat with us or contact one of the maintainers directly.
