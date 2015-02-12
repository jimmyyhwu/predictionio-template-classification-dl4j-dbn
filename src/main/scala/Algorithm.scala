package org.template.vanilla

import grizzled.slf4j.Logger
import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params
import org.apache.commons.math3.random.{MersenneTwister, RandomGenerator}
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.distributions.Distributions
import org.deeplearning4j.models.classifiers.dbn.DBN
import org.deeplearning4j.models.featuredetectors.rbm.RBM
import org.deeplearning4j.nn.WeightInit
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.api.activation.Activations
import org.nd4j.linalg.lossfunctions.LossFunctions

case class AlgorithmParams(mult: Int) extends Params

class Algorithm(val ap: AlgorithmParams)
  // extends PAlgorithm if Model contains RDD[]
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(data: PreparedData): Model = {

    logger.info("logging test")

    // ADDED BELOW LINES

    val gen: RandomGenerator = new MersenneTwister(123)
    val conf: NeuralNetConfiguration = new NeuralNetConfiguration.Builder().iterations(100).weightInit(WeightInit.DISTRIBUTION).dist(Distributions.normal(gen, 1e-3)).constrainGradientToUnitNorm(false).lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).activationFunction(Activations.tanh).rng(gen).visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED).dropOut(0.3f).learningRate(1e-3f).nIn(4).nOut(3).build
    val d: DBN = new DBN.Builder().configure(conf).hiddenLayerSizes(Array[Int](3)).build

    NeuralNetConfiguration.setClassifier(d.getOutputLayer.conf)

    val iter: DataSetIterator = new IrisDataSetIterator(150, 150)

    //fetch first
    val next: DataSet = iter.next(110)
    next.normalizeZeroMeanZeroUnitVariance

    d.fit(next)

    val eval: Evaluation = new Evaluation
    val output: INDArray = d.output(next.getFeatureMatrix)
    eval.eval(next.getLabels, output)
    //log.info("Score " + eval.stats)
    logger.info("Score " + eval.stats)
    // END OF ADDED LINES


    // Simply count number of events
    // and multiple it by the algorithm parameter
    // and store the number as model
    val count = data.events.count().toInt * ap.mult
    new Model(mc = count)
  }

  def predict(model: Model, query: Query): PredictedResult = {
    logger.error(s"Prediction")



    // Prefix the query with the model data
    val result = s"${model.mc}-${query.q}"
    PredictedResult(p = result)
  }
}

class Model(val mc: Int) extends Serializable {
  override def toString = s"mc=${mc}"
}
