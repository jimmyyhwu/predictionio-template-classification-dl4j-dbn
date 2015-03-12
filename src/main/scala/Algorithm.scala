package org.template.classification

import grizzled.slf4j.Logger
import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params
import org.apache.commons.math3.random.{MersenneTwister, RandomGenerator}
import org.apache.spark.SparkContext
import org.deeplearning4j.distributions.Distributions
import org.deeplearning4j.models.classifiers.dbn.DBN
import org.deeplearning4j.models.featuredetectors.rbm.RBM
import org.deeplearning4j.nn.WeightInit
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.api.activation.Activations
import org.nd4j.linalg.lossfunctions.LossFunctions

case class AlgorithmParams(mult: Int) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    logger.info(data.records)

    // deeplearning4j IRIS example
    val gen: RandomGenerator = new MersenneTwister(123)
    val conf: NeuralNetConfiguration = new NeuralNetConfiguration.Builder().iterations(100).weightInit(WeightInit.DISTRIBUTION).dist(Distributions.normal(gen, 1e-3)).constrainGradientToUnitNorm(false).lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).activationFunction(Activations.tanh).rng(gen).visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED).dropOut(0.3f).learningRate(1e-3f).nIn(4).nOut(3).build
    val d: DBN = new DBN.Builder().configure(conf).hiddenLayerSizes(Array[Int](3)).build
    NeuralNetConfiguration.setClassifier(d.getOutputLayer.conf)
    d.fit(data.records)

    /*val eval: Evaluation = new Evaluation
    val output: INDArray = d.output(data.records.getFeatureMatrix)
    eval.eval(data.records.getLabels, output)
    logger.info("Score " + eval.stats)*/

    new Model(d = d)
  }

  def predict(model: Model, query: Query): PredictedResult = {
    // Prefix the query with the model data
    //val result = s"${model.mc}-${query.features}"
    //PredictedResult(label = result)
    PredictedResult(0)
  }
}

class Model(val d: DBN) extends Serializable {
  override def toString = s"d=${d}"
}
