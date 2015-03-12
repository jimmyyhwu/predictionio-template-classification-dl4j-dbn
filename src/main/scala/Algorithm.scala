package org.template.classification

import grizzled.slf4j.Logger
import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params
import org.apache.commons.math3.random.{MersenneTwister, RandomGenerator}
import org.apache.spark.SparkContext
import org.deeplearning4j.distributions.Distributions
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.classifiers.dbn.DBN
import org.deeplearning4j.models.featuredetectors.rbm.RBM
import org.deeplearning4j.nn.WeightInit
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.api.activation.Activations
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

case class AlgorithmParams(mult: Int) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {

    // deeplearning4j IRIS example with DBN
    val gen: RandomGenerator = new MersenneTwister(123)
    val conf: NeuralNetConfiguration = new NeuralNetConfiguration.Builder().iterations(100).weightInit(WeightInit.DISTRIBUTION).dist(Distributions.normal(gen, 1e-3)).constrainGradientToUnitNorm(false).lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).activationFunction(Activations.tanh).rng(gen).visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED).dropOut(0.3f).learningRate(1e-3f).nIn(4).nOut(3).build
    val dbn: DBN = new DBN.Builder().configure(conf).hiddenLayerSizes(Array[Int](3)).build
    NeuralNetConfiguration.setClassifier(dbn.getOutputLayer.conf)
    dbn.fit(data.data)

    // Evaluate F1 score on trained DBN
    val eval: Evaluation = new Evaluation
    val output: INDArray = dbn.output(data.data.getFeatureMatrix)
    eval.eval(data.data.getLabels, output)
    logger.info("Score " + eval.stats)

    new Model(dbn)
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val features = Array(
      query.sepal_length,
      query.sepal_width,
      query.petal_length,
      query.petal_width
    )
    logger.info(Nd4j.create(features))
    val output = model.dbn.output(Nd4j.create(features))
    PredictedResult(Array(
      output.getDouble(0),
      output.getDouble(1),
      output.getDouble(2)
    ))
  }
}

class Model(val dbn: DBN) extends Serializable {
  override def toString = s"dbn=${dbn}"
}
